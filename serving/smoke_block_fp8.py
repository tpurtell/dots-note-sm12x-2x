#!/usr/bin/env python3
"""Bounded real-weight check of B12x on a Dots3 FP8 core projection."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm._shared.wo_mxfp8 import empty_dense_gemm_mnl_view
from b12x.preparation import PreparedCall, PreparationSession


def load(source: Path, name: str) -> torch.Tensor:
    index = json.loads((source / "model.safetensors.index.json").read_text())
    with safe_open(source / index["weight_map"][name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    name = "model.layers.0.self_attn.q_a_proj"
    weight = load(args.fp8_source, name + ".weight").to("cuda")
    scale = load(args.fp8_source, name + ".weight_scale_inv").to("cuda")
    if (weight.dtype, tuple(weight.shape), scale.dtype, tuple(scale.shape)) != (
        torch.float8_e4m3fn, (1024, 5120), torch.float32, (8, 40)
    ):
        raise ValueError("unexpected Dots3 layer-0 q_a_proj geometry")
    x = (torch.randn((1, 5120), device="cuda") * 0.2).to(torch.bfloat16)
    packed = bfl.pack_weight(weight, scale)
    plan = bfl.plan(bfl.Caps(
        device="cuda", max_tokens=1, in_features=5120,
        out_features=1024, source_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
    ))
    output = empty_dense_gemm_mnl_view(1, 1024, 1, device=x.device, dtype=torch.bfloat16)
    owned = {}

    def prepare(state):
        (spec,) = state.scratch.scratch_specs()
        owned["scratch"] = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            scratch=owned["scratch"], source=x, packed_weight=packed, output=output,
        )
        return PreparedCall(run=lambda: state.run_binding(binding), output=output)

    with PreparationSession(device=x.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="dots3-real-fp8-core", prepare_call=prepare),))
        binding = bfl.bind(
            plan, scratch=owned["scratch"], source=x,
            packed_weight=packed, output=output,
        )
        actual = bfl.run(binding=binding).clone()
        reference_weight = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
        reference = (x.float() @ reference_weight.T).to(torch.bfloat16)
        torch.cuda.synchronize()
        difference = (actual.float() - reference.float()).abs()
        relative_l2 = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float()))
        cosine = float(torch.nn.functional.cosine_similarity(actual.float(), reference.float()).item())
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured = bfl.run(binding=binding)
        graph.replay()
        torch.cuda.synchronize()
        graph_error = float((captured.float() - actual.float()).abs().max().item())
        result = {
            "schema": "dots3-real-block-fp8-component-v1",
            "weight": name,
            "weight_shape": list(weight.shape),
            "scale_shape": list(scale.shape),
            "cosine_against_dequantized_source": cosine,
            "relative_l2_against_dequantized_source": relative_l2,
            "max_abs_error_against_dequantized_source": float(difference.max().item()),
            "graph_max_abs_error": graph_error,
            "selected_config": str(plan.selection.config),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
