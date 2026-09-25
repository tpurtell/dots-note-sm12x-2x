#!/usr/bin/env python3
"""Real-checkpoint EP route-map oracle; CPU-only loader audit unless --gpu.

GPU mode emulates two rank-local partials on one idle GPU. It tests no
collective transport or full128-expert-per-rank load and reports that scope.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from dots3_ep2 import validate_placement,make_global_loader,route_plan

GLOBAL_IDS = tuple(range(8)) + tuple(range(248,256))
# Deliberately noncontiguous ownership and scrambled local slots. Include the
# largest valid global ID to exercise the full256-entry router namespace.
OWNERS = ((255,0,253,2,251,4,249,6),(1,254,3,252,5,250,7,248))
TOP_K = 8


def route_cases():
    return [
        [list(GLOBAL_IDS[offset:offset+TOP_K]) for offset in (0,2,4,8)],
        [list(OWNERS[0])]*4, # eight distinct routes, rank1 entirely remote
        [list(OWNERS[1])]*4, # eight distinct routes, rank0 entirely remote
        [[global_id]*TOP_K for global_id in (0,255,1,248)],
        [list(OWNERS[0][:7])+[OWNERS[1][i]] for i in range(4)],
    ]


def mapping(ids):
    m = torch.full((256,),-1,dtype=torch.int32)
    for local,global_id in enumerate(ids):
        m[global_id] = local
    return m


def checkpoint(source,layer):
    index = json.loads((source/'model.safetensors.index.json').read_text())['weight_map']
    payload = {}
    for expert in GLOBAL_IDS:
        for proj in ('gate_proj','up_proj','down_proj'):
            for field in ('trellis','suh','svh','mcg'):
                name = f'model.layers.{layer}.mlp.experts.{expert}.{proj}.{field}'
                with safe_open(source/index[name],framework='pt',device='cpu') as f:
                    payload[expert,proj,field] = f.get_tensor(name)
    return payload


def cpu_checks(payload):
    results = []
    assert set(OWNERS[0]).isdisjoint(OWNERS[1])
    assert set(OWNERS[0]) | set(OWNERS[1]) == set(GLOBAL_IDS)
    assert all(len(row)==TOP_K for case in route_cases() for row in case)
    for owned in OWNERS:
        placement = mapping(owned)
        forward,inverse = validate_placement(placement,len(owned))
        assert inverse == owned
        loader = make_global_loader(forward)
        stored = {}
        def accept(tensor,*,expert_id,shard_id):
            stored[expert_id,shard_id] = tensor
        param = SimpleNamespace(load_exl3_weight=accept)
        for global_id in GLOBAL_IDS:
            tensor = payload[global_id,'gate_proj','trellis']
            ok = loader(param,tensor,'unused','w1',global_id,return_success=True)
            assert ok == (global_id in owned)
        assert len(stored) == len(owned)
        for local,global_id in enumerate(owned):
            # Identity as well as equality: loading must not TP-slice or modify.
            assert stored[local,'w1'] is payload[global_id,'gate_proj','trellis']
            assert stored[local,'w1'].shape == (320,96,64)
        results.append({'local_to_global':list(inverse),'loaded_whole_experts':len(stored)})
    for expert in GLOBAL_IDS:
        for projection in ('gate_proj','up_proj','down_proj'):
            marker = int(payload[expert,projection,'mcg'].item())&0xffffffff
            assert marker == 0xcbac1fed
    # Full256->128 static ownership contract, independent of GPU kernels.
    for rank in (0,1):
        ids = tuple(range(rank,256,2))
        assert validate_placement(mapping(ids),128)[1] == ids
    return results


def gpu_checks(payload):
    from b12x.moe import fused_moe
    from b12x.preparation import PreparedCall,PreparationSession
    device = torch.device('cuda:0')
    torch.manual_seed(127)
    x = torch.randn((4,5120),dtype=torch.bfloat16,device=device)*.1
    ids = torch.empty((4,TOP_K),dtype=torch.int32,device=device)
    weights = torch.arange(1,TOP_K+1,dtype=torch.float32,device=device).repeat(4,1)
    weights.div_(weights.sum(dim=1,keepdim=True))
    localids = torch.empty_like(ids)
    contexts = []
    def prepare(owned,route_map):
        def stack(proj,field):
            return torch.stack([payload[e,proj,field] for e in owned]).to(device).contiguous()
        wp = fused_moe.plan_weights(source=fused_moe.Exl3TrellisSource(bits=4),
            activation=fused_moe.ActivationSpec(mode=fused_moe.ActivationMode.A16,
                nonlinearity='silu',io_dtype=torch.bfloat16),
            geometry=fused_moe.MoEGeometry(num_experts=len(owned),hidden_size=5120,intermediate_size=1536))
        ep = fused_moe.prepare_weights(plan=wp,weights=fused_moe.Exl3TrellisWeights(
            w13=torch.stack((stack('gate_proj','trellis'),stack('up_proj','trellis'))),
            w2=stack('down_proj','trellis'),gate_suh=stack('gate_proj','suh'),
            up_suh=stack('up_proj','suh'),down_svh=stack('down_proj','svh'),
            intermediate_rotations=torch.cat((stack('gate_proj','svh'),stack('up_proj','svh'),stack('down_proj','suh')),dim=1),mcg=0xcbac1fed))
        plan = route_plan(ep,4,TOP_K,256 if route_map is not None else 0)
        planids = ids if route_map is not None else localids
        output = torch.empty((4,5120),dtype=torch.float32,device=device)
        scratch = []
        kwargs = dict(a=x,experts=ep,topk_ids=planids,topk_weights=weights,output=output)
        if route_map is not None:
            kwargs['route_expert_map'] = route_map.to(device)
        def prime(state):
            scratch.extend(torch.empty(s.shape,dtype=s.dtype,device=s.device) for s in state.scratch.scratch_specs())
            binding = state.bind(scratch=tuple(scratch),**kwargs)
            return PreparedCall(run=binding.run,output=output,owners=(binding,))
        session = PreparationSession(device=device,autotune=False,compile_workers=2)
        session.prepare((plan.request(name='ep2-route-probe',prepare_call=prime),))
        binding = fused_moe.bind(plan,scratch=tuple(scratch),**kwargs)
        def run():
            return fused_moe.run(binding=binding)
        contexts.append((session,plan,binding))
        return run,output
    # Prime with valid initialized IDs; both sides include local and remote.
    remap = {g:i for i,g in enumerate(GLOBAL_IDS)}
    routes = route_cases()
    ids.copy_(torch.tensor(routes[0],device=device))
    localids.copy_(torch.tensor([[remap[g] for g in row] for row in routes[0]],device=device))
    reference,refout = prepare(GLOBAL_IDS,None)
    partials = [prepare(owned,mapping(owned)) for owned in OWNERS]
    graphs = []
    for run,out in [(reference,refout),*partials]:
        run(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run()
        graphs.append(g)
    tests = []
    for route in routes:
        x.copy_(torch.randn_like(x)*.1)
        weights.copy_(torch.rand_like(weights))
        weights.div_(weights.sum(dim=1,keepdim=True))
        ids.copy_(torch.tensor(route,device=device))
        localids.copy_(torch.tensor([[remap[g] for g in row] for row in route],device=device))
        for g in graphs:
            g.replay()
        torch.cuda.synchronize()
        expected = refout.clone()
        summed = partials[0][1]+partials[1][1]
        err = float(torch.linalg.vector_norm(summed-expected)/torch.linalg.vector_norm(expected))
        if not torch.isfinite(summed).all() or err>.005:
            raise AssertionError(('EP partial sum mismatch',err))
        for (owned,(run,out),g) in zip(OWNERS,partials,graphs[1:]):
            captured = out.clone()
            run()
            if not torch.equal(out,captured):
                raise AssertionError('EP eager/replay mismatch')
            if all(global_id not in owned for row in route for global_id in row):
                if torch.count_nonzero(out):
                    raise AssertionError('remote-only EP rank returned stale/nonzero partial')
        tests.append({'global_routes':route,'sum_relative_l2':err})
    return tests


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--layer',type=int,default=1)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    payload=checkpoint(args.source,args.layer)
    result={'schema':'dots3-ep2-route-probe-v2','layer':args.layer,
        'scope':'16 real whole experts; eight local per emulated rank; no collective or full model',
        'top_k':TOP_K,'global_experts':256,'fixture_global_ids':list(GLOBAL_IDS),
        'loader':cpu_checks(payload)}
    if args.gpu:
        result['gpu']=gpu_checks(payload)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
