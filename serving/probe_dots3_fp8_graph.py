#!/usr/bin/env python3
"""Offline real-weight FP8 graph comparison; run only on an idle target GPU.

Example: --source <installed snapshot> --weight model.layers.0.self_attn.q_b_proj
--tp-axis 0 --tp-size 2 --rows 1 4 16 64 512 --output <project .cache>/fp8.json
This does not change a running server or select a serving implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from safetensors import safe_open
from b12x.gemm import blockscaled
from b12x.preparation import PreparedCall, PreparationSession
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.fp8 import create_fp8_quant_key, init_fp8_linear_kernel
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


def load(source, name, index):
    with safe_open(source / index[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def error(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    return dict(relative_l2=float(torch.linalg.vector_norm(a-b) / torch.linalg.vector_norm(b)),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
                max_abs=float((a-b).abs().max()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--weight', required=True, help='Module prefix, excluding .weight')
    p.add_argument('--tp-axis', type=int, choices=(0, 1))
    p.add_argument('--tp-size', type=int, default=2)
    p.add_argument('--tp-rank', type=int, default=0)
    p.add_argument('--rows', type=int, nargs='+', default=[1, 4, 16, 64, 512])
    p.add_argument('--iterations', type=int, default=30)
    p.add_argument('--replays', type=int, default=20)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not (args.tp_size > 0 and 0 <= args.tp_rank < args.tp_size):
        p.error('invalid TP partition')
    if min(args.rows + [args.iterations, args.replays]) <= 0:
        p.error('rows, iterations, replays must be positive')
    torch.manual_seed(1947)
    index = json.loads((args.source / 'model.safetensors.index.json').read_text())['weight_map']
    w = load(args.source, args.weight + '.weight', index)
    s = load(args.source, args.weight + '.weight_scale_inv', index)
    if w.dtype != torch.float8_e4m3fn or s.dtype != torch.float32:
        raise ValueError('requires original E4M3 weights and float32 block scales')
    if args.tp_axis is not None:
        axis = args.tp_axis
        if w.shape[axis] % (128 * args.tp_size):
            raise ValueError('TP split must align to source 128-element scale blocks')
        w = w.chunk(args.tp_size, dim=axis)[args.tp_rank].contiguous()
        s = s.chunk(args.tp_size, dim=axis)[args.tp_rank].contiguous()
    n, k = w.shape
    if n % 128 or k % 128 or tuple(s.shape) != (n // 128, k // 128):
        raise ValueError('exact blockscaled path requires aligned N/K; no implicit padding')
    weight, scale = w.cuda(), s.cuda()
    source_reference = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    config = VllmConfig()
    config.model_config = SimpleNamespace(dtype=torch.bfloat16, hf_text_config=SimpleNamespace(model_type='dots3_note'))
    with set_current_vllm_config(config):
        quantizer = QuantFP8(static=False, group_shape=GroupShape(1, 128), num_token_padding=None, use_ue8m0=False)
        native = init_fp8_linear_kernel(
            create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
            create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
            torch.bfloat16, torch.bfloat16, tuple(weight.shape))
    layer = torch.nn.Module()
    layer.register_parameter('weight', torch.nn.Parameter(weight.clone(), requires_grad=False))
    layer.register_parameter('weight_scale_inv', torch.nn.Parameter(scale.clone(), requires_grad=False))
    layer.weight_block_size, layer.input_scale = [128, 128], None
    native.process_weights_after_loading(layer)
    result = dict(schema='dots3-fp8-graph-probe-v1', weight=args.weight, shape=[n, k],
                  tp_axis=args.tp_axis, tp_size=args.tp_size, tp_rank=args.tp_rank,
                  gpu=str(torch.cuda.get_device_properties(0)), torch=torch.__version__,
                  native_kernel=type(native).__name__, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=[], ratio_definition='native median / exact B12x median; >1 favors B12x')
    for rows in args.rows:
        x = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16) * .2
        av, asc = quantizer(x, None, None, use_triton=False)
        plan = blockscaled.plan(blockscaled.query_from_call((av, asc), (weight, scale),
            ab_dtype='float8_e4m3fn', sf_dtype='float32', sf_vec_size=128, block_fp8=True, c_dtype='bfloat16'))
        def prepare(state):
            return PreparedCall(run=lambda: state.run_serialized(av, asc, weight, scale, None,
                ab_dtype='float8_e4m3fn', sf_dtype='float32', c_dtype='bfloat16', sf_vec_size=128, block_fp8=True, stream=None))
        with PreparationSession(device=x.device, autotune=False, compile_workers=2) as session:
            session.prepare((plan.request(name='dots3-fp8-graph-probe', prepare_call=prepare),))
            def exact():
                values, scales = quantizer(x, None, None, use_triton=False)
                return blockscaled.mm_block_fp8(values, scales, weight, scale, plan=plan)
            calls = {'exact': exact, 'native': lambda: native.apply_weights(layer, x)}
            graphs, outputs = {}, {}
            for name, call in calls.items():
                for _ in range(5):
                    call()
                torch.cuda.synchronize()
                graphs[name] = torch.cuda.CUDAGraph()
                with session.capture(), torch.cuda.graph(graphs[name]):
                    outputs[name] = call()
            checks = []
            for _ in range(2):
                x.copy_(torch.randn_like(x) * .2)
                av, asc = quantizer(x, None, None, use_triton=False)
                ref = ((av.float() * asc.repeat_interleave(128, 1)) @ source_reference.T).to(torch.bfloat16)
                original = (x.float() @ source_reference.T).to(torch.bfloat16)
                for g in graphs.values():
                    g.replay()
                torch.cuda.synchronize()
                e = error(outputs['exact'], ref)
                if not torch.isfinite(outputs['exact']).all() or e['relative_l2'] > .005:
                    raise AssertionError(e)
                checks.append(dict(exact_against_quantized_source=e,
                    exact_against_bf16_source=error(outputs['exact'], original),
                    native_against_bf16_source=error(outputs['native'], original)))
            timings = {name: [] for name in graphs}
            for i in range(args.iterations):
                for name in (['exact', 'native'] if i % 2 == 0 else ['native', 'exact']):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.replays):
                        graphs[name].replay()
                    end.record(); end.synchronize()
                    timings[name].append(start.elapsed_time(end) * 1000 / args.replays)
            medians = {name: statistics.median(v) for name, v in timings.items()}
            result['rows'].append(dict(m=rows, checks=checks, gpu_us=timings, median_us=medians,
                native_over_exact=medians['native']/medians['exact'], config=str(plan.selection.config)))
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(result['rows'][-1]), flush=True)


if __name__ == '__main__':
    main()
