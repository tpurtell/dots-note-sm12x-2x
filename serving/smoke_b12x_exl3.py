#!/usr/bin/env python3
"""Exercise uniform EXL3 K4 through B12x's prepared fused-MoE path on a GPU."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from b12x.moe import fused_moe
from b12x.preparation import PreparedCall, PreparationSession


def fixture_weights(path: Path, *, experts: int, hidden: int, intermediate: int, device):
    payload = load_file(path, device="cpu")
    projections = {}
    for expert in range(experts):
        for name in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"expert{expert}.{name}."
            marker = int(payload[prefix + "mcg"].item()) & 0xFFFFFFFF
            if marker != 0xCBAC1FED:
                raise ValueError(f"invalid EXL3 MCG marker for {prefix}")
            projections[(expert, name)] = {
                field: payload[prefix + field] for field in ("trellis", "suh", "svh")
            }

    def stack(name, field, *, dim=None, length=None):
        rows = []
        for expert in range(experts):
            tensor = projections[(expert, name)][field]
            if dim is not None:
                tensor = tensor.narrow(dim, 0, length)
            rows.append(tensor)
        return torch.stack(rows).to(device=device).contiguous()

    gate_svh = stack("gate_proj", "svh", dim=0, length=intermediate)
    up_svh = stack("up_proj", "svh", dim=0, length=intermediate)
    down_suh = stack("down_proj", "suh", dim=0, length=intermediate)
    return fused_moe.Exl3TrellisWeights(
        w13=torch.stack((
            stack("gate_proj", "trellis", dim=1, length=intermediate // 16),
            stack("up_proj", "trellis", dim=1, length=intermediate // 16),
        )).contiguous(),
        w2=stack("down_proj", "trellis", dim=0, length=intermediate // 16),
        gate_suh=stack("gate_proj", "suh"),
        up_suh=stack("up_proj", "suh"),
        intermediate_rotations=torch.cat((gate_svh, up_svh, down_suh), dim=1).contiguous(),
        down_svh=stack("down_proj", "svh"),
        mcg=0xCBAC1FED,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-experts", type=int)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if args.top_k > args.experts:
        raise ValueError("top-k cannot exceed expert count")
    capacity = args.tokens if args.capacity is None else args.capacity
    if capacity < args.tokens:
        raise ValueError("capacity must cover live tokens")
    fixture_experts = args.experts if args.fixture_experts is None else args.fixture_experts
    if args.fixture is not None and args.experts % fixture_experts:
        raise ValueError("fixture expert count must divide target expert count")
    device = torch.device("cuda:0")
    source = fused_moe.Exl3TrellisSource(bits=4)
    activation = fused_moe.ActivationSpec(
        mode=fused_moe.ActivationMode.A16,
        nonlinearity="silu",
        io_dtype=torch.bfloat16,
    )
    geometry = fused_moe.MoEGeometry(
        num_experts=args.experts,
        hidden_size=args.hidden,
        intermediate_size=args.intermediate,
    )
    weight_plan = fused_moe.plan_weights(
        source=source, activation=activation, geometry=geometry
    )
    e, h, n, bits = args.experts, args.hidden, args.intermediate, source.bits
    weights = (
        fixture_weights(args.fixture, experts=fixture_experts, hidden=h, intermediate=n, device=device)
        if args.fixture is not None
        else fused_moe.Exl3TrellisWeights(
            w13=torch.zeros((2, e, h // 16, n // 16, 16 * bits), dtype=torch.int16, device=device),
            w2=torch.zeros((e, n // 16, h // 16, 16 * bits), dtype=torch.int16, device=device),
            gate_suh=torch.ones((e, h), dtype=torch.float16, device=device),
            up_suh=torch.ones((e, h), dtype=torch.float16, device=device),
            intermediate_rotations=torch.ones((e, 3 * n), dtype=torch.float16, device=device),
            down_svh=torch.ones((e, h), dtype=torch.float16, device=device),
            mcg=0xCBAC1FED,
        )
    )
    if args.fixture is not None and fixture_experts != e:
        copies = e // fixture_experts
        weights = fused_moe.Exl3TrellisWeights(
            w13=weights.w13.repeat(1, copies, 1, 1, 1).contiguous(),
            w2=weights.w2.repeat(copies, 1, 1, 1).contiguous(),
            gate_suh=weights.gate_suh.repeat(copies, 1).contiguous(),
            up_suh=weights.up_suh.repeat(copies, 1).contiguous(),
            intermediate_rotations=weights.intermediate_rotations.repeat(copies, 1).contiguous(),
            down_svh=weights.down_svh.repeat(copies, 1).contiguous(),
            mcg=weights.mcg,
        )
    experts = fused_moe.prepare_weights(plan=weight_plan, weights=weights)
    plan = fused_moe.plan_execution(
        experts=experts,
        capacity=fused_moe.ExecutionCapacity(max_tokens=capacity, top_k=args.top_k),
    )
    x = torch.randn((args.tokens, h), dtype=torch.bfloat16, device=device)
    ids = torch.arange(args.tokens * args.top_k, dtype=torch.int32, device=device)
    ids = ids.reshape(args.tokens, args.top_k).remainder_(e).contiguous()
    route_weights = torch.full((args.tokens, args.top_k), 1 / args.top_k, dtype=torch.float32, device=device)

    def prepare(state):
        dummy_x = torch.randn((capacity, h), dtype=torch.bfloat16, device=device)
        dummy_ids = torch.arange(capacity * args.top_k, dtype=torch.int32, device=device)
        dummy_ids = dummy_ids.reshape(capacity, args.top_k).remainder_(e).contiguous()
        dummy_weights = torch.full(
            (capacity, args.top_k), 1 / args.top_k,
            dtype=torch.float32, device=device,
        )
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in state.scratch.scratch_specs()
        )
        output = torch.empty_like(dummy_x, dtype=torch.float32)
        binding = state.bind(
            scratch=scratch,
            a=dummy_x,
            experts=experts,
            topk_weights=dummy_weights,
            topk_ids=dummy_ids,
            output=output,
        )
        return PreparedCall(
            run=binding.run, output=output,
            owners=(scratch, dummy_x, dummy_ids, dummy_weights, binding),
        )

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="dots3-uniform-exl3-k4", prepare_call=prepare),))
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=device)
            for spec in plan.scratch_specs()
        )
        binding = fused_moe.bind(
            plan,
            scratch=scratch,
            a=x,
            experts=experts,
            topk_weights=route_weights,
            topk_ids=ids,
        )
        result = fused_moe.run(binding=binding)
        torch.cuda.synchronize()
        finite_count = int(torch.isfinite(result).sum().item())
        if result.shape != x.shape or finite_count != result.numel():
            raise AssertionError(
                f"B12x EXL3 K4 output invalid: shape={tuple(result.shape)}, "
                f"finite={finite_count}/{result.numel()}, "
                f"dtype={result.dtype}"
            )
        if args.save is not None:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            save_file({
                "x": x.detach().cpu().contiguous(),
                "topk_ids": ids.detach().cpu().contiguous(),
                "topk_weights": route_weights.detach().cpu().contiguous(),
                "b12x_output": result.detach().cpu().contiguous(),
            }, args.save)
        print(json.dumps({
            "event": "b12x-exl3-k4-smoke-complete",
            "device": torch.cuda.get_device_name(0),
            "shape": list(result.shape),
            "capacity": capacity,
            "max_abs": float(result.abs().max().item()),
        }), flush=True)


if __name__ == "__main__":
    main()
