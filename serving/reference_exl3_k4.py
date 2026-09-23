#!/usr/bin/env python3
"""Compare B12x's uniform K4 fused MoE with the ExLlamaV3 projection kernel."""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from gptqmodel.exllamav3.modules.quant.exl3 import LinearEXL3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--b12x", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--intermediate", type=int, default=768)
    args = parser.parse_args()
    tensors = load_file(args.fixture, device="cpu")
    case = load_file(args.b12x, device="cpu")
    x = case["x"].to(device="cuda", dtype=torch.float16)
    topk_ids = case["topk_ids"].to(device="cuda", dtype=torch.long)
    topk_weights = case["topk_weights"].to(device="cuda", dtype=torch.float16)
    hidden_size = x.shape[-1]
    intermediate_size = args.intermediate
    output = torch.zeros_like(x, dtype=torch.float32)

    def projection(expert: int, name: str) -> LinearEXL3:
        prefix = f"expert{expert}.{name}."
        trellis = tensors[prefix + "trellis"]
        suh = tensors[prefix + "suh"]
        svh = tensors[prefix + "svh"]
        if name in ("gate_proj", "up_proj"):
            trellis = trellis.narrow(1, 0, intermediate_size // 16)
            svh = svh.narrow(0, 0, intermediate_size)
            in_features, out_features = hidden_size, intermediate_size
        else:
            trellis = trellis.narrow(0, 0, intermediate_size // 16)
            suh = suh.narrow(0, 0, intermediate_size)
            in_features, out_features = intermediate_size, hidden_size
        return LinearEXL3(
            None, in_features, out_features,
            suh=suh.to(device="cuda").contiguous(),
            svh=svh.to(device="cuda").contiguous(),
            trellis=trellis.to(device="cuda").contiguous(),
            mcg=tensors[prefix + "mcg"].to(device="cuda"),
            out_dtype=torch.float16,
        )

    for expert in range(args.experts):
        positions = (topk_ids == expert).nonzero(as_tuple=False)
        if positions.numel() == 0:
            continue
        token_ids = positions[:, 0]
        route_ids = positions[:, 1]
        inputs = x.index_select(0, token_ids)
        gate = projection(expert, "gate_proj").forward(inputs, {})
        up = projection(expert, "up_proj").forward(inputs, {})
        hidden = torch.nn.functional.silu(gate) * up
        down = projection(expert, "down_proj").forward(hidden, {})
        output.index_add_(
            0, token_ids,
            (down * topk_weights[token_ids, route_ids].unsqueeze(-1)).to(torch.float32),
        )

    reference = output.cpu()
    actual = case["b12x_output"].float()
    error = actual - reference
    cosine = torch.nn.functional.cosine_similarity(actual, reference, dim=-1)
    print(json.dumps({
        "event": "exl3-k4-parity",
        "shape": list(actual.shape),
        "cosine_min": float(cosine.min().item()),
        "cosine_mean": float(cosine.mean().item()),
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
        "relative_l2": float(error.norm().item() / reference.norm().item()),
        "reference_max_abs": float(reference.abs().max().item()),
    }), flush=True)


if __name__ == "__main__":
    main()
