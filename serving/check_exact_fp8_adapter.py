#!/usr/bin/env python3
"""Exercise the opt-in Q-B method with real TP2 source shards on an idle GPU."""
import argparse
import json
import hashlib
import statistics
from pathlib import Path
from types import SimpleNamespace
import tempfile

from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    destroy_model_parallel, destroy_distributed_environment,
)

import torch
from safetensors import safe_open
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.linear import WEIGHT_LOADER_V2_SUPPORTED
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from dots3_b12x_fp8 import Dots3ExactFp8Method


def load(source, name, index):
    with safe_open(source / index[name], framework='pt', device='cpu') as f:
        return f.get_tensor(name)


def metrics(a, b):
    x, y = a.float().flatten(), b.float().flatten()
    return {'relative_l2': float(torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(y)),
            'max_abs': float((x-y).abs().max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--owner-full', action='store_true', help='Full owner DSA/SWA QB and DSA output; no TP slicing')
    parser.add_argument('--timing', action='store_true', help='Time exact adapter versus native processed kernel: five samples of forty graph replays')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Real one-rank NCCL/model-parallel context. Checkpoint inputs below are
    # explicitly sliced to TP2 shapes; this does not test a distributed loader.
    with tempfile.TemporaryDirectory(prefix='fp8-dist-', dir=args.output.parent) as directory:
        rendezvous = (Path(directory) / 'store').resolve().as_uri()
        try:
            torch.cuda.set_device(0)
            init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                         distributed_init_method=rendezvous, backend='nccl')
            with set_current_vllm_config(VllmConfig()):
                initialize_model_parallel(tensor_model_parallel_size=1, backend='nccl')
                run_checks(args)
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def run_checks(args):
    torch.manual_seed(743)
    torch.set_default_dtype(torch.bfloat16)
    cfg = VllmConfig()
    cfg.model_config = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type='dots3_note'))
    index = json.loads((args.source/'model.safetensors.index.json').read_text())['weight_map']
    result = {'schema': 'dots3-exact-fp8-adapter-v1', 'cases': [],
              'distributed_world_size': 1, 'checkpoint_partition_size': 1 if args.owner_full else 2,
              'loader_scope': 'explicit source tensors; distributed loader not tested',
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'adapter_sha256':hashlib.sha256(Path(__file__).with_name('dots3_b12x_fp8.py').read_bytes()).hexdigest(),
              'gpu':str(torch.cuda.get_device_properties(0)), 'timing_enabled':args.timing}
    planned_rows = (1,4,8,16) if args.owner_full else (4,)
    fallback_rows = 2 if args.owner_full else 1
    projections = [(0,'q_b_proj'),(2,'q_b_proj')] + ([(0,'o_proj')] if args.owner_full else [])
    for layer_id, projection in projections:
        name = f'model.layers.{layer_id}.self_attn.{projection}'
        weight = load(args.source, name+'.weight', index)
        scale = load(args.source, name+'.weight_scale_inv', index)
        if not args.owner_full:
            weight, scale = weight.chunk(2,0)[0], scale.chunk(2,0)[0]
        weight, scale = weight.contiguous().cuda(), scale.contiguous().cuda()
        n, k = weight.shape
        original_w, original_s = weight.clone(), scale.clone()
        layer = torch.nn.Module()
        layer.tp_size = 1 if args.owner_full else 2  # Validate the real shard geometry in create_weights.
        with set_current_vllm_config(cfg):
            method = Dots3ExactFp8Method(Fp8Config(is_checkpoint_fp8_serialized=True,
                activation_scheme='dynamic', weight_block_size=[128, 128]), projection, planned_rows)
            assert method.input_dtype == method.out_dtype == torch.bfloat16
            assert type(method).__name__ in WEIGHT_LOADER_V2_SUPPORTED
            # Exercise inherited parameter creation and metadata, then install
            # actual rank-zero checkpoint slices through parameter copies.
            method.create_weights(layer, input_size_per_partition=k,
                output_partition_sizes=[n], input_size=k, output_size=n*layer.tp_size,
                params_dtype=torch.bfloat16)
            layer.cuda()
            layer.weight.data.copy_(weight)
            layer.weight_scale_inv.data.copy_(scale)
            method.process_weights_after_loading(layer)
            quantizer = QuantFP8(static=False, group_shape=GroupShape(1,128),
                num_token_padding=None, use_ue8m0=False)
        assert torch.equal(method.source_weight.view(torch.uint8), original_w.view(torch.uint8))
        assert torch.equal(method.source_scale, original_s)
        reference_w = original_w.float()*original_s.repeat_interleave(128,0).repeat_interleave(128,1)
        cases = []
        for rows in (*planned_rows, fallback_rows):
            x = torch.randn((rows,k), device='cuda', dtype=torch.bfloat16)*.2
            for _ in range(4):
                method.apply(layer,x)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = method.apply(layer,x)
            native_graph = None
            if args.timing:
                for _ in range(4):
                    method.fp8_linear.apply_weights(layer,x)
                torch.cuda.synchronize()
                native_graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(native_graph):
                    native_captured=method.fp8_linear.apply_weights(layer,x)
            previous = None
            checks = []
            for change in range(3):
                x.copy_(torch.randn_like(x)*(.1+change*.2))
                eager = method.apply(layer,x).clone()
                graph.replay()
                torch.cuda.synchronize()
                if not torch.isfinite(captured).all() or not captured.count_nonzero():
                    raise AssertionError('nonfinite or zero adapter output')
                if not torch.equal(captured,eager):
                    raise AssertionError(('graph/eager mismatch',metrics(captured,eager)))
                if previous is not None and torch.equal(previous,captured):
                    raise AssertionError('changed input produced stale graph output')
                previous = captured.clone()
                if rows in planned_rows:
                    av, asc = quantizer(x,None,None,use_triton=False)
                    expected = ((av.float()*asc.repeat_interleave(128,1))@reference_w.T).to(torch.bfloat16)
                    err = metrics(captured,expected)
                    if err['relative_l2'] > .005:
                        raise AssertionError(err)
                else:
                    expected = method.fp8_linear.apply_weights(layer,x)
                    if not torch.equal(captured,expected):
                        raise AssertionError('unplanned rows failed exact native fallback parity')
                    err = metrics(captured,expected)
                if native_graph is not None:
                    native_eager=method.fp8_linear.apply_weights(layer,x).clone()
                    native_graph.replay()
                    torch.cuda.synchronize()
                    if not torch.equal(native_captured,native_eager):
                        raise AssertionError('native timing graph/eager mismatch')
                checks.append(err)
            case={'rows':rows,'path':'exact' if rows in planned_rows else 'native',
                'changed_input_checks':checks,'graph_eager_equal':True}
            if args.timing:
                timings={'adapter':[],'native':[]}
                graphs={'adapter':graph,'native':native_graph}
                for sample in range(5):
                    order=('adapter','native') if sample%2==0 else ('native','adapter')
                    for label in order:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record()
                        for _ in range(40):graphs[label].replay()
                        end.record();end.synchronize()
                        timings[label].append(start.elapsed_time(end)*1000/40)
                medians={key:statistics.median(values) for key,values in timings.items()}
                case['timing']={'samples_us':timings,'median_us':medians,'samples':5,'replays_per_sample':40,
                    'native_over_adapter':medians['native']/medians['adapter'],
                    'scope':'Full apply path including activation quantization; CUDA graph replay device time; unplanned rows use native fallback'}
            cases.append(case)
        assert torch.equal(method.source_weight.view(torch.uint8),original_w.view(torch.uint8))
        assert torch.equal(method.source_scale,original_s)
        result['cases'].append({'weight':name,'shape':[n,k],
            'native_kernel':type(method.fp8_linear).__name__,'checks':cases,
            'source_immutable':True,'input_dtype':str(method.input_dtype),'out_dtype':str(method.out_dtype)})
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
