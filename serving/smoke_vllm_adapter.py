#!/usr/bin/env python3
"""Load production EXL3 expert tensors through the Dots3 vLLM method."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file

from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.quantization.dots3_exl3_fp8 import (
    Dots3B12xExl3MoEMethod,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--b12x", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    payload = load_file(args.fixture, device="cpu")
    case = load_file(args.b12x, device="cpu")
    x = case["x"].to(device="cuda")
    ids = case["topk_ids"].to(device="cuda")
    weights = case["topk_weights"].to(device="cuda")
    hidden = x.shape[-1]
    intermediate = 768
    layer = SimpleNamespace(
        layer_name="model.layers.1.mlp.experts",
        local_num_experts=args.experts,
        hidden_size=hidden,
        exl3_hidden_size=hidden,
        exl3_intermediate_size_per_partition=intermediate,
        exl3_params_dtype=torch.bfloat16,
        exl3_tp_size=2,
        exl3_tp_rank=0,
        dots3_b12x_capacity=args.capacity or int(x.shape[0]),
        top_k=int(ids.shape[1]),
        activation=MoEActivation.SILU,
        expert_map=None,
        apply_router_weight_on_input=False,
        ckpt_gate_proj_name="gate_proj",
        ckpt_up_proj_name="up_proj",
        ckpt_down_proj_name="down_proj",
    )
    for prefix in ("w13", "w2"):
        for field in ("suh", "svh", "trellis", "mcg", "mul1"):
            setattr(layer, f"{prefix}_{field}", SimpleNamespace(
                exl3_tensors={}, device=torch.device("cuda:0")
            ))
    for expert in range(args.experts):
        for projection, shard in (
            ("gate_proj", "w1"),
            ("up_proj", "w3"),
            ("down_proj", "w2"),
        ):
            prefix = "w2" if shard == "w2" else "w13"
            for field in ("suh", "svh", "trellis", "mcg"):
                tensor = payload[f"expert{expert}.{projection}.{field}"]
                getattr(layer, f"{prefix}_{field}").exl3_tensors[(expert, shard)] = tensor

    quant_config = SimpleNamespace(
        rank_sliced_metadata=None,
        codebook_for_prefix=lambda _prefix: "mcg",
    )
    method = Dots3B12xExl3MoEMethod(quant_config, SimpleNamespace())
    loaded = []

    def record_weight(**kwargs):
        loaded.append((kwargs["shard_id"], kwargs["expert_id"], kwargs["loaded_weight"].shape))
        return True

    loader_layer = SimpleNamespace(
        layer_name="model.layers.1.mlp.experts",
        quant_method=method,
        get_expert_mapping=lambda include_fused: [
            ("experts.w13_", "experts.0.gate_proj.", 0, "w1")
        ],
        w13_trellis=SimpleNamespace(weight_loader=record_weight),
    )
    trellis = payload["expert0.gate_proj.trellis"]
    loaded_names = list(RoutedExperts.load_weights(
        loader_layer, [("0.gate_proj.trellis", trellis)]
    ))
    if loaded_names != ["w13_trellis"] or loaded != [("w1", 0, trellis.shape)]:
        raise AssertionError(f"EXL3 per-expert Trellis loader misread tensor: {loaded}")
    method.process_weights_after_loading(layer)
    output = method.apply(layer, x, weights, ids, None, None)
    expected = case["b12x_output"].to(device="cuda", dtype=torch.bfloat16)
    difference = (output - expected).abs()
    max_abs = float(difference.max().item())
    if max_abs > 0.02 or not torch.isfinite(output).all():
        raise AssertionError(f"Dots3 vLLM adapter differs from B12x: {max_abs}")
    one = method.apply(layer, x[:1], weights[:1], ids[:1], None, None)
    one_max_abs = float((one - expected[:1]).abs().max().item())
    if one_max_abs > 0.02:
        raise AssertionError(f"one-token B12x route differs: {one_max_abs}")
    graph_max_abs = None
    if args.graph:
        eager = output.clone()
        for _ in range(2):
            method.apply(layer, x, weights, ids, None, None)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = method.apply(layer, x, weights, ids, None, None)
        graph.replay()
        torch.cuda.synchronize()
        graph_max_abs = float((captured - eager).abs().max().item())
        if graph_max_abs > 0.02:
            raise AssertionError(f"Dots3 B12x CUDA graph differs from eager: {graph_max_abs}")
    print(json.dumps({
        "event": "dots3-vllm-adapter-smoke-complete",
        "shape": list(output.shape),
        "max_abs_vs_b12x_bf16": max_abs,
        "one_token_max_abs_vs_b12x_bf16": one_max_abs,
        "planned_capacity": layer.dots3_b12x_capacity,
        "graph_max_abs_vs_eager": graph_max_abs,
    }), flush=True)


if __name__ == "__main__":
    main()
