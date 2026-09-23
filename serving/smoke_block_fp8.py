#!/usr/bin/env python3
"""Bounded real-weight check of B12x on a Dots3 FP8 core projection."""

import argparse
import json
from pathlib import Path
import statistics

import torch
from safetensors import safe_open
from types import SimpleNamespace

from b12x.gemm import block_fp8_linear as bfl
from b12x.gemm import blockscaled
from b12x.gemm._shared.wo_mxfp8 import empty_dense_gemm_mnl_view
from b12x.preparation import PreparedCall, PreparationSession
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.fp8 import (
    create_fp8_quant_key,
    init_fp8_linear_kernel,
)


def load(source: Path, name: str) -> torch.Tensor:
    index = json.loads((source / "model.safetensors.index.json").read_text())
    with safe_open(source / index["weight_map"][name], framework="pt", device="cpu") as shard:
        return shard.get_tensor(name)


def timed(call, iterations: int = 30) -> list[float]:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


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
    with set_current_vllm_config(VllmConfig()):
        quantizer = QuantFP8(
            static=False, group_shape=GroupShape(1, 128),
            num_token_padding=None, use_ue8m0=False,
        )
        a_values, a_scale = quantizer(x, None, None, use_triton=False)
    exact_query = blockscaled.query_from_call(
        (a_values, a_scale), (weight, scale),
        ab_dtype="float8_e4m3fn", sf_dtype="float32", sf_vec_size=128,
        block_fp8=True, c_dtype="bfloat16",
    )
    exact_plan = blockscaled.plan(exact_query)

    def prepare_exact(state):
        return PreparedCall(run=lambda: state.run_serialized(
            a_values, a_scale, weight, scale, None,
            ab_dtype="float8_e4m3fn", sf_dtype="float32",
            c_dtype="bfloat16", sf_vec_size=128,
            block_fp8=True, stream=None,
        ))

    with PreparationSession(device=x.device, autotune=False, compile_workers=2) as session:
        session.prepare((exact_plan.request(name="dots3-exact-block-fp8", prepare_call=prepare_exact),))
        exact = blockscaled.mm_block_fp8(
            a_values, a_scale, weight, scale, plan=exact_plan,
        )
        exact_reference = (
            (a_values.float() * a_scale.repeat_interleave(128, 1))
            @ reference_weight.T
        ).to(torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            captured_exact = blockscaled.mm_block_fp8(
                a_values, a_scale, weight, scale, plan=exact_plan,
            )
        graph.replay()
        torch.cuda.synchronize()
        exact_difference = (exact.float() - exact_reference.float()).abs()
        result.update({
            "exact_block_fp8_cosine": float(torch.nn.functional.cosine_similarity(
                exact.float(), exact_reference.float()
            ).item()),
            "exact_block_fp8_relative_l2": float(
                torch.linalg.vector_norm(exact_difference)
                / torch.linalg.vector_norm(exact_reference.float())
            ),
            "exact_block_fp8_max_abs_error": float(exact_difference.max().item()),
            "exact_block_fp8_graph_max_abs_error": float(
                (captured_exact.float() - exact.float()).abs().max().item()
            ),
            "exact_block_fp8_config": str(exact_plan.selection.config),
        })
        def exact_from_bf16():
            values, scales = quantizer(x, None, None, use_triton=False)
            return blockscaled.mm_block_fp8(
                values, scales, weight, scale, plan=exact_plan,
            )

        full_graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(full_graph):
            full_captured = exact_from_bf16()
        full_graph.replay()
        torch.cuda.synchronize()
        result["exact_full_graph_max_abs_error"] = float(
            (full_captured.float() - exact.float()).abs().max().item()
        )
        exact_us = timed(exact_from_bf16)
    native_config = VllmConfig()
    native_config.model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        hf_text_config=SimpleNamespace(model_type="dots3_note"),
    )
    with set_current_vllm_config(native_config):
        native = init_fp8_linear_kernel(
            create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128)),
            create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128)),
            torch.bfloat16, torch.bfloat16, tuple(weight.shape),
        )
    layer = torch.nn.Module()
    layer.register_parameter("weight", torch.nn.Parameter(weight.clone(), requires_grad=False))
    layer.register_parameter("weight_scale_inv", torch.nn.Parameter(scale.clone(), requires_grad=False))
    layer.weight_block_size = [128, 128]
    layer.input_scale = None
    native.process_weights_after_loading(layer)
    native_output = native.apply_weights(layer, x)
    torch.cuda.synchronize()
    native_difference = (native_output.float() - exact_reference.float()).abs()
    original_reference = (x.float() @ reference_weight.T).to(torch.bfloat16)
    exact_original_difference = (exact.float() - original_reference.float()).abs()
    native_original_difference = (native_output.float() - original_reference.float()).abs()
    native_us = timed(lambda: native.apply_weights(layer, x))
    result.update({
        "native_kernel": type(native).__name__,
        "native_cosine_against_source_fp8_reference": float(
            torch.nn.functional.cosine_similarity(
                native_output.float(), exact_reference.float()
            ).item()
        ),
        "native_relative_l2_against_source_fp8_reference": float(
            torch.linalg.vector_norm(native_difference)
            / torch.linalg.vector_norm(exact_reference.float())
        ),
        "native_max_abs_error_against_source_fp8_reference": float(
            native_difference.max().item()
        ),
        "exact_relative_l2_against_original_source": float(
            torch.linalg.vector_norm(exact_original_difference)
            / torch.linalg.vector_norm(original_reference.float())
        ),
        "native_relative_l2_against_original_source": float(
            torch.linalg.vector_norm(native_original_difference)
            / torch.linalg.vector_norm(original_reference.float())
        ),
        "exact_median_gpu_us": statistics.median(exact_us),
        "native_median_gpu_us": statistics.median(native_us),
        "exact_gpu_us": exact_us,
        "native_gpu_us": native_us,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
