#!/usr/bin/env python3
"""Build a zero-copy, indexed FP8-core/BF16-routed-expert source on a Spark.

The source shards are linked, never rewritten. GPTQModel's LazyTurtle uses the
index to select individual tensors from the appropriate checkpoint.
"""

import argparse
import json
import os
import re
from pathlib import Path


EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
FP8_REVISION = "7c14222e22423d6df6848eb0d1c5c3a88a00311a"
BF16_REVISION = "1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b"


def index(path: Path) -> dict[str, str]:
    with (path / "model.safetensors.index.json").open() as handle:
        data = json.load(handle)
    result = data["weight_map"]
    if not isinstance(result, dict) or not result:
        raise ValueError(f"invalid checkpoint index: {path}")
    return result


def build(fp8: Path, bf16: Path, output: Path) -> dict:
    fp8_weights = index(fp8)
    bf16_weights = index(bf16)
    routed = {
        name: shard for name, shard in bf16_weights.items()
        if (match := EXPERT.fullmatch(name)) and 1 <= int(match.group(1)) <= 45 and 0 <= int(match.group(2)) < 256
    }
    if len(routed) != 45 * 256 * 3:
        raise ValueError(f"expected 34,560 BF16 routed expert tensors, found {len(routed)}")
    missing = set(routed) - set(fp8_weights)
    if missing:
        raise ValueError(f"BF16 routed keys absent from FP8 source: {list(sorted(missing))[:5]}")
    fp8_core = {
        name: shard for name, shard in fp8_weights.items()
        if name not in routed and not (
            name.endswith(".weight_scale_inv") and name.removesuffix("_scale_inv") in routed
        )
    }
    if any(EXPERT.fullmatch(name) for name in fp8_core):
        raise AssertionError("FP8 routed tensor escaped replacement")
    output.mkdir(parents=True, exist_ok=True)
    weight_map: dict[str, str] = {}
    for prefix, root, names in (("fp8", fp8, fp8_core), ("bf16", bf16, routed)):
        for name, shard in names.items():
            target = root / shard
            if not target.is_file():
                raise FileNotFoundError(target)
            link = output / f"{prefix}-{shard}"
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f"existing link points elsewhere: {link}")
            elif link.exists():
                raise ValueError(f"expected symlink: {link}")
            else:
                link.symlink_to(target.resolve())
            weight_map[name] = link.name
    for filename in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
        source = fp8 / filename
        if source.is_file():
            link = output / filename
            if link.is_symlink() and link.resolve() == source.resolve():
                continue
            if link.exists() or link.is_symlink():
                raise ValueError(f"existing file differs: {link}")
            link.symlink_to(source.resolve())
    metadata = {
        "format": "dots3-hybrid-lazy-source-v1",
        "fp8_revision": FP8_REVISION,
        "bf16_revision": BF16_REVISION,
        "fp8_core_tensors": len(fp8_core),
        "bf16_routed_tensors": len(routed),
        "dropped_fp8_routed_scales": len(fp8_weights) - len(fp8_core) - len(routed),
    }
    index_tmp = output / "model.safetensors.index.json.tmp"
    index_tmp.write_text(json.dumps({"metadata": metadata, "weight_map": weight_map}, sort_keys=True) + "\n")
    os.replace(index_tmp, output / "model.safetensors.index.json")
    (output / "hybrid-source.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8", type=Path, required=True)
    parser.add_argument("--bf16", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.fp8, args.bf16, args.output), indent=2))


if __name__ == "__main__":
    main()
