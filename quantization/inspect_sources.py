#!/usr/bin/env python3
"""Audit the two Dots3 Note sources and enumerate the uniform K4 replacement."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import struct


EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
LAYERS = range(1, 46)
EXPERTS = range(256)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPECTED_EXPERTS = len(LAYERS) * len(EXPERTS) * len(PROJECTIONS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata(snapshot: Path) -> tuple[dict, dict[str, dict]]:
    config_path = snapshot / "config.json"
    index_path = snapshot / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())["weight_map"]
    tensors: dict[str, dict] = {}
    for shard in sorted(set(index.values())):
        path = snapshot / shard
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"implausible safetensors header: {path}")
            header = json.loads(stream.read(header_size))
        header.pop("__metadata__", None)
        if set(header) != {name for name, filename in index.items() if filename == shard}:
            raise ValueError(f"index/header disagreement: {path}")
        for name, value in header.items():
            tensors[name] = {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "bytes": value["data_offsets"][1] - value["data_offsets"][0],
            }
    if set(tensors) != set(index):
        raise ValueError(f"incomplete index: {snapshot}")
    return config, tensors


def inspect(bf16: Path, fp8: Path) -> dict:
    bf16_config, bf16_tensors = metadata(bf16)
    fp8_config, fp8_tensors = metadata(fp8)
    for label, config in (("BF16", bf16_config), ("FP8", fp8_config)):
        geometry = {
            "model_type": "dots3_note",
            "num_hidden_layers": 46,
            "first_k_dense_replace": 1,
            "n_routed_experts": 256,
            "num_experts_per_tok": 8,
        }
        for key, expected in geometry.items():
            if config.get(key) != expected:
                raise ValueError(f"{label} {key}: {config.get(key)!r} != {expected!r}")
    if bf16_config.get("quantization_config") is not None:
        raise ValueError("BF16 source declares quantization")
    if fp8_config.get("quantization_config") != {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }:
        raise ValueError("FP8 source has an unexpected quantization contract")

    expected_names = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in LAYERS
        for expert in EXPERTS
        for projection in PROJECTIONS
    }
    found_bf16 = {name for name in bf16_tensors if EXPERT.fullmatch(name)}
    found_fp8 = {name for name in fp8_tensors if EXPERT.fullmatch(name)}
    if found_bf16 != expected_names or found_fp8 != expected_names:
        raise ValueError("routed expert namespace is incomplete or unexpected")
    if len(expected_names) != EXPECTED_EXPERTS:
        raise AssertionError("incorrect routed expert count")
    if set(bf16_tensors) - set(fp8_tensors):
        raise ValueError("BF16 tensor missing from the FP8 checkpoint")
    if any(
        not name.endswith(".weight_scale_inv")
        for name in set(fp8_tensors) - set(bf16_tensors)
    ):
        raise ValueError("FP8 checkpoint has unexpected extra tensors")

    routed_bytes = 0
    for name in expected_names:
        source = bf16_tensors[name]
        old = fp8_tensors[name]
        if source["dtype"] != "BF16" or old["dtype"] != "F8_E4M3":
            raise ValueError(f"unexpected routed weight dtype: {name}")
        if source["shape"] != old["shape"]:
            raise ValueError(f"routed weight shape mismatch: {name}")
        scale = fp8_tensors.get(name.removesuffix(".weight") + ".weight_scale_inv")
        if scale is None or scale["dtype"] != "F32":
            raise ValueError(f"missing FP8 scale: {name}")
        routed_bytes += source["bytes"]

    preserved = set(fp8_tensors) - expected_names - {
        name.removesuffix(".weight") + ".weight_scale_inv" for name in expected_names
    }
    classes = Counter()
    for name in preserved:
        classes["vision" if name.startswith("vision_encoder.") else
                "audio" if name.startswith("audio_encoder.") else "core"] += 1

    return {
        "schema": "dots3-note-uniform-exl3-k4-source-audit-v1",
        "bf16": {
            "snapshot": str(bf16.resolve()),
            "config_sha256": sha256(bf16 / "config.json"),
            "index_sha256": sha256(bf16 / "model.safetensors.index.json"),
        },
        "fp8": {
            "snapshot": str(fp8.resolve()),
            "config_sha256": sha256(fp8 / "config.json"),
            "index_sha256": sha256(fp8 / "model.safetensors.index.json"),
        },
        "expert_layers": [LAYERS.start, LAYERS.stop - 1],
        "experts_per_layer": len(EXPERTS),
        "projections": list(PROJECTIONS),
        "replacement_tensors": len(expected_names),
        "source_bf16_routed_bytes": routed_bytes,
        "exl3_bits_per_weight": 4,
        "preserved_fp8_tensor_classes": dict(classes),
        "vision_expert_policy": "preserve FP8 checkpoint tensors",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = inspect(args.bf16, args.fp8)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
