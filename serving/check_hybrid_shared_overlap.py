#!/usr/bin/env python3
"""Idle-GPU fork/join probe using real FP8 shared MLP and EXL3 routed weights.

Eight real layer1 routed experts use their rank0 TP2 slices (N768), top8.
This is a concurrency/storage/graph probe, not full256 routing or transport.
"""
import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from safetensors import safe_open
from vllm.config import VllmConfig,set_current_vllm_config
from vllm.distributed.parallel_state import (init_distributed_environment,
    initialize_model_parallel,destroy_model_parallel,destroy_distributed_environment)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config,Fp8LinearMethod
from b12x.moe import fused_moe
from b12x.preparation import PreparedCall,PreparationSession
from hybrid_shared_overlap import launch_shared,prepared_stream


def checks(args):
    torch.manual_seed(447);torch.set_default_dtype(torch.bfloat16)
    index=json.loads((args.source/'model.safetensors.index.json').read_text())['weight_map']
    def load(name):
        with safe_open(args.source/index[name],framework='pt',device='cpu') as f:return f.get_tensor(name)
    prefix='model.layers.1.mlp.'
    def linear(projections):
        weights=[load(prefix+'shared_experts.'+p+'.weight') for p in projections]
        scales=[load(prefix+'shared_experts.'+p+'.weight_scale_inv') for p in projections]
        n,k=sum(w.shape[0] for w in weights),weights[0].shape[1]
        module=torch.nn.Module();module.tp_size=1
        method=Fp8LinearMethod(Fp8Config(is_checkpoint_fp8_serialized=True,
            activation_scheme='dynamic',weight_block_size=[128,128]))
        method.create_weights(module,input_size_per_partition=k,output_partition_sizes=[w.shape[0] for w in weights],
                              input_size=k,output_size=n,params_dtype=torch.bfloat16)
        module.cuda();module.weight.data.copy_(torch.cat(weights).cuda())
        module.weight_scale_inv.data.copy_(torch.cat(scales).cuda())
        method.process_weights_after_loading(module)
        return module,method
    gate,gate_method=linear(('gate_proj','up_proj'))
    down,down_method=linear(('down_proj',))
    from vllm.model_executor.layers.activation import SiluAndMul
    activation=SiluAndMul()
    def shared(x):return down_method.apply(down,activation(gate_method.apply(gate,x)))
    def stack(projection,field,dim=None,length=None):
        values=[]
        for e in range(8):
            value=load(prefix+f'experts.{e}.{projection}.{field}')
            if dim is not None:value=value.narrow(dim,0,length)
            values.append(value)
        return torch.stack(values).cuda().contiguous()
    weights=fused_moe.Exl3TrellisWeights(
        w13=torch.stack([stack(p,'trellis',1,48) for p in ('gate_proj','up_proj')]),
        w2=stack('down_proj','trellis',0,48),gate_suh=stack('gate_proj','suh'),up_suh=stack('up_proj','suh'),
        intermediate_rotations=torch.cat([stack('gate_proj','svh',0,768),stack('up_proj','svh',0,768),stack('down_proj','suh',0,768)],dim=1),
        down_svh=stack('down_proj','svh'),mcg=0xcbac1fed)
    wp=fused_moe.plan_weights(source=fused_moe.Exl3TrellisSource(bits=4),
        activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A16,nonlinearity='silu',io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(num_experts=8,hidden_size=5120,intermediate_size=768))
    experts=fused_moe.prepare_weights(plan=wp,weights=weights)
    plan=fused_moe.plan_execution(experts=experts,capacity=fused_moe.ExecutionCapacity(max_tokens=512,top_k=8))
    device=torch.device('cuda:0');scratch=[]
    primer=torch.randn((512,5120),device=device)*.1
    ids=torch.arange(8,device=device,dtype=torch.int32).expand(512,8).contiguous()
    route_weights=torch.full((512,8),.125,device=device,dtype=torch.float32)
    output=torch.empty_like(primer,dtype=torch.float32)
    def prepare(state):
        scratch.extend(torch.empty(spec.shape,dtype=spec.dtype,device=spec.device) for spec in state.scratch.scratch_specs())
        binding=state.bind(scratch=tuple(scratch),a=primer,experts=experts,topk_ids=ids,topk_weights=route_weights,output=output)
        return PreparedCall(run=binding.run,output=output,owners=(binding,))
    session=PreparationSession(device=device,autotune=False,compile_workers=2)
    session.prepare((plan.request(name='hybrid-shared-overlap',prepare_call=prepare),))
    prepared_stream(device)
    records=[]
    for rows in (1,4,16,512):
        x=primer[:rows]
        binding=fused_moe.bind(plan,scratch=tuple(scratch),a=x,experts=experts,
            topk_ids=ids[:rows],topk_weights=route_weights[:rows],output=output[:rows])
        def run(overlap):
            pending=launch_shared(shared,x) if overlap else None
            aux=None if overlap else shared(x)
            routed=fused_moe.run(binding=binding)
            return routed.to(torch.bfloat16)+(pending.join() if overlap else aux)
        graphs=[]
        for overlap in (False,True):
            for _ in range(4):run(overlap)
            torch.cuda.synchronize()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):result=run(overlap)
            graphs.append((g,result))
        for change in range(3):
            x.copy_(torch.randn_like(x)*(.1+.1*change))
            route_weights[:rows].copy_(torch.rand_like(route_weights[:rows]))
            route_weights[:rows].div_(route_weights[:rows].sum(dim=-1,keepdim=True))
            reference=run(False).clone()
            for g,result in graphs:
                g.replay();torch.cuda.synchronize()
                if not torch.equal(result,reference):
                    raise AssertionError(('shared overlap mismatch',rows,change,float((result-reference).abs().max())))
        timings={}
        for mode,(g,result) in zip(('serial','overlap'),graphs):
            samples=[]
            for _ in range(5):
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(40):g.replay()
                end.record();end.synchronize();samples.append(begin.elapsed_time(end)*1000/40)
            timings[mode]=samples
            g.reset()
        records.append({'rows':rows,'changed_input_graph_equal':True,'timings_us':timings})
    return {'passed':True,'scope':__doc__,'cases':records,
        'native_shared_kernels':[type(gate_method.fp8_linear).__name__,type(down_method.fp8_linear).__name__]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent,prefix='shared-dist-') as tmp:
        torch.cuda.set_device(0)
        try:
            init_distributed_environment(world_size=1,rank=0,local_rank=0,distributed_init_method=(Path(tmp)/'store').resolve().as_uri(),backend='nccl')
            cfg=VllmConfig();cfg.model_config=NS(dtype=torch.bfloat16,hf_text_config=NS(model_type='dots3_note'),head_dtype=None)
            with set_current_vllm_config(cfg):
                initialize_model_parallel(tensor_model_parallel_size=1,backend='nccl')
                result=checks(args)
            args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
        finally:destroy_model_parallel();destroy_distributed_environment()

if __name__=='__main__':main()
