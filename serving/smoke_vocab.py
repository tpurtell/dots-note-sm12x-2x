#!/usr/bin/env python3
"""Check the Dots3 TP2 BF16 lm_head shard through B12x on SM12x."""

import argparse
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch
from safetensors import safe_open

from b12x.gemm import bf16_vocab_projection as vocab
from b12x.preparation import PreparedCall, PreparationSession
from vllm.model_executor.layers.quantization.dots3_exl3_fp8 import (
    Dots3B12xVocabMethod,
)


def timed(call, *, iterations: int = 30) -> list[float]:
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    elapsed = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end) * 1000)
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-source", required=True, type=Path)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-plan-reuse", action="store_true")
    args = parser.parse_args()
    if args.tp_rank not in (0, 1):
        raise ValueError("TP rank must be 0 or 1")
    index = json.loads((args.fp8_source / "model.safetensors.index.json").read_text())
    shard = args.fp8_source / index["weight_map"]["lm_head.weight"]
    with safe_open(shard, framework="pt", device="cpu") as source:
        weight = source.get_tensor("lm_head.weight")
    if weight.dtype != torch.bfloat16 or tuple(weight.shape) != (152064, 5120):
        raise ValueError(f"unexpected Dots3 lm_head: {weight.dtype} {tuple(weight.shape)}")
    weight = weight.narrow(0, args.tp_rank * 76032, 76032).to("cuda").contiguous()
    x = (torch.randn((1, 5120), device="cuda", dtype=torch.bfloat16) * 0.125).contiguous()
    plan = vocab.plan(vocab.Caps(
        device=x.device, max_tokens=1, in_features=5120, out_features=76032,
    ))

    def prepare(state):
        return PreparedCall(
            run=lambda: state.run(x, weight), owners=(x, weight),
        )

    with PreparationSession(device=x.device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="dots3-tp2-bf16-vocab", prepare_call=prepare),))
        if plan.selection is None or plan.selection.config.backend != "triton":
            raise ValueError(f"B12x did not select its native vocab backend: {plan.selection}")

        def b12x_call():
            return vocab.run(vocab.bind(plan, source=x, weight=weight))

        reference = torch.nn.functional.linear(x, weight)
        actual = b12x_call()
        torch.cuda.synchronize()
        difference = (actual.float() - reference.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(actual.float(), reference.float())
        top20_same = torch.equal(actual.topk(20).indices, reference.topk(20).indices)
        if not torch.isfinite(actual).all() or cosine.item() < 0.99999 or not top20_same:
            raise AssertionError("B12x vocabulary output failed native BF16 parity")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = b12x_call()
        graph.replay()
        torch.cuda.synchronize()
        graph_error = float((captured.float() - actual.float()).abs().max().item())
        if graph_error:
            raise AssertionError(f"B12x vocabulary graph replay differs: {graph_error}")
        b12x_us = timed(b12x_call)
        native_us = timed(lambda: torch.nn.functional.linear(x, weight))
        result = {
            "event": "dots3-vocab-smoke-complete",
            "host_gpu": torch.cuda.get_device_name(0),
            "tp_rank": args.tp_rank,
            "weight_shape": list(weight.shape),
            "b12x_config": str(plan.selection.config),
            "cosine": float(cosine.item()),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "top20_equal": top20_same,
            "graph_max_abs_error": graph_error,
            "b12x_gpu_us": b12x_us,
            "native_gpu_us": native_us,
            "b12x_median_gpu_us": statistics.median(b12x_us),
            "native_median_gpu_us": statistics.median(native_us),
        }
    adapter_layer = SimpleNamespace(weight=weight)
    adapter = Dots3B12xVocabMethod()
    adapter.process_weights_after_loading(adapter_layer)
    adapter_output = adapter.apply(adapter_layer, x)
    adapter_error = float((adapter_output.float() - actual.float()).abs().max().item())
    if adapter_error:
        raise AssertionError(f"vLLM vocabulary adapter differs from B12x: {adapter_error}")
    result["adapter_max_abs_error"] = adapter_error
    if args.check_plan_reuse:
        import gc
        import weakref

        other = SimpleNamespace(weight=-weight)
        adapter.process_weights_after_loading(other)
        if other.dots3_b12x_vocab_plan is not adapter_layer.dots3_b12x_vocab_plan:
            raise AssertionError("Equal-geometry heads did not share the prepared plan")
        # Keep both outputs live: sharing a plan must not share output storage.
        adapter.apply(other, x)
        reuse_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(reuse_graph):
            first_output = adapter.apply(adapter_layer, x)
            other_output = adapter.apply(other, x)
        if first_output.data_ptr() == other_output.data_ptr():
            raise AssertionError("Shared-plan vocabulary outputs alias")
        replay_errors = []
        for _ in range(2):
            x.mul_(0.5)
            other.weight.neg_()
            reuse_graph.replay()
            torch.cuda.synchronize()
            for layer, output in ((adapter_layer, first_output), (other, other_output)):
                eager = adapter.apply(layer, x)
                native = torch.nn.functional.linear(x, layer.weight)
                error = float((output.float() - eager.float()).abs().max().item())
                replay_errors.append(error)
                if error or torch.nn.functional.cosine_similarity(output.float(), native.float()).item() < 0.99999:
                    raise AssertionError("Shared-plan changed-weight/input graph failed")
        temporary = SimpleNamespace(weight=weight.clone())
        adapter.process_weights_after_loading(temporary)
        obsolete = weakref.ref(temporary.weight)
        allocated_before = torch.cuda.memory_allocated()
        temporary.weight = weight
        gc.collect()
        torch.cuda.synchronize()
        released = allocated_before - torch.cuda.memory_allocated()
        if obsolete() is not None or released < weight.numel() * weight.element_size():
            raise AssertionError("Obsolete draft vocabulary weights remain retained")
        result["plan_reuse"] = {
            "same_plan": True, "distinct_output_storage": True,
            "changed_input_weight_graph_errors": replay_errors,
            "obsolete_weight_released_bytes": released,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if not isinstance(value, list)}), flush=True)


if __name__ == "__main__":
    main()
