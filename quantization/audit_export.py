#!/usr/bin/env python3
"""Audit the final hybrid checkpoint before Hub publication."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


ROUTED = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
FIELDS = ("trellis", "suh", "svh", "mcg")


def checkpoint(root: Path) -> tuple[dict, dict[str, tuple[Path, dict, int]]]:
    config = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    by_shard = {}
    for name, shard in index.items():
        by_shard.setdefault(shard, set()).add(name)
    tensors = {}
    for shard, names in by_shard.items():
        path = root / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            if header_size > 64 * 1024 * 1024:
                raise ValueError(f"oversized safetensors header: {path}")
            header = json.loads(stream.read(header_size))
        header.pop("__metadata__", None)
        if set(header) != names:
            raise ValueError(f"index and safetensors header differ: {path}")
        for name, metadata in header.items():
            tensors[name] = (path, metadata, 8 + header_size)
    if set(tensors) != set(index):
        raise ValueError("checkpoint index is incomplete")
    return config, tensors


def region_sha256(entry: tuple[Path, dict, int]) -> str:
    path, metadata, payload_offset = entry
    start, end = metadata["data_offsets"]
    remaining = end - start
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(payload_offset + start)
        while remaining:
            block = stream.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise EOFError(f"short tensor payload: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def routed_names() -> dict[str, tuple[str, str]]:
    expected = {}
    for layer in range(1, 46):
        for expert in range(256):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                prefix = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
                for field in FIELDS:
                    expected[f"{prefix}.{field}"] = (projection, field)
    return expected


def check_shape(name: str, metadata: dict, projection: str, field: str) -> int:
    is_down = projection == "down_proj"
    expected = {
        "trellis": ("I16", [96, 320, 64] if is_down else [320, 96, 64]),
        "suh": ("F16", [1536] if is_down else [5120]),
        "svh": ("F16", [5120] if is_down else [1536]),
        "mcg": ("I32", []),
    }[field]
    if (metadata["dtype"], metadata["shape"]) != expected:
        raise ValueError(f"invalid uniform K4 tensor {name}: {metadata}")
    start, end = metadata["data_offsets"]
    if end <= start:
        raise ValueError(f"empty tensor: {name}")
    return end - start


def audit(fp8_source: Path, output: Path, *, verify_core_bytes: bool) -> dict:
    source_config, source_tensors = checkpoint(fp8_source)
    output_config, output_tensors = checkpoint(output)
    quant = output_config.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "exl3":
        raise ValueError(f"export does not declare EXL3: {quant}")
    if quant.get("bits") != 4 or quant.get("codebook") != "mcg":
        raise ValueError(f"export is not uniform MCG K4: {quant}")
    if source_config.get("model_type") != output_config.get("model_type"):
        raise ValueError("model type changed during export")
    expected_routed = routed_names()
    old_weights = {
        name for name in source_tensors
        if (match := ROUTED.fullmatch(name))
        and 1 <= int(match.group(1)) <= 45
        and 0 <= int(match.group(2)) < 256
    }
    if len(old_weights) != 45 * 256 * 3:
        raise ValueError(f"FP8 source routed weight count: {len(old_weights)}")
    old_scales = {
        name.removesuffix(".weight") + ".weight_scale_inv"
        for name in old_weights
    }
    preserved = set(source_tensors) - old_weights - old_scales
    expected = preserved | set(expected_routed)
    if set(output_tensors) != expected:
        missing = sorted(expected - set(output_tensors))[:8]
        extra = sorted(set(output_tensors) - expected)[:8]
        raise ValueError(f"export tensor namespace differs: missing={missing}, extra={extra}")

    quant_bytes = 0
    for name, (projection, field) in expected_routed.items():
        quant_bytes += check_shape(name, output_tensors[name][1], projection, field)
    core_bytes = 0
    core_hash = hashlib.sha256()
    for name in sorted(preserved):
        original = source_tensors[name][1]
        exported = output_tensors[name][1]
        original_shape = (original["dtype"], original["shape"])
        exported_shape = (exported["dtype"], exported["shape"])
        original_bytes = original["data_offsets"][1] - original["data_offsets"][0]
        exported_bytes = exported["data_offsets"][1] - exported["data_offsets"][0]
        if original_shape != exported_shape or original_bytes != exported_bytes:
            raise ValueError(f"FP8 core tensor metadata changed: {name}")
        core_bytes += exported_bytes
        if verify_core_bytes:
            old_hash = region_sha256(source_tensors[name])
            new_hash = region_sha256(output_tensors[name])
            if old_hash != new_hash:
                raise ValueError(f"FP8 core tensor bytes changed: {name}")
            core_hash.update(name.encode() + b"\0" + bytes.fromhex(new_hash))
    return {
        "schema": "dots3-hybrid-uniform-k4-export-audit-v1",
        "source": str(fp8_source.resolve()),
        "output": str(output.resolve()),
        "fp8_core_tensors": len(preserved),
        "fp8_core_bytes": core_bytes,
        "fp8_core_byte_verified": verify_core_bytes,
        "fp8_core_tensor_digest": core_hash.hexdigest() if verify_core_bytes else None,
        "exl3_routed_projections": len(expected_routed) // len(FIELDS),
        "exl3_routed_tensors": len(expected_routed),
        "exl3_routed_bytes": quant_bytes,
        "quantization_config": quant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--skip-core-byte-check", action="store_true")
    args = parser.parse_args()
    result = audit(
        args.fp8_source, args.output,
        verify_core_bytes=not args.skip_core_byte_check,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
