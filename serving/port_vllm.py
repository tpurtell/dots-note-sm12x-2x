#!/usr/bin/env python3
"""Install the Dots3 EXL3/FP8 adapter into pinned vLLM v0.30.0."""

import hashlib
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])
TARGETS = {
    "model_executor/layers/quantization/__init__.py":
        "55f1d9d309d9caee6b0d47a95004dfa380e504c6cca061a6766c92c2654e11e4",
    "model_executor/layers/fused_moe/routed_experts.py":
        "68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d",
}


def replace(source: str, old: str, new: str, *, count: int = 1) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} matches for {old!r}, got {actual}")
    return source.replace(old, new)


for name, digest in TARGETS.items():
    path = ROOT / name
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise RuntimeError(f"vLLM v0.30.0 source identity changed: {name}")
    if name.endswith("quantization/__init__.py"):
        updated = replace(
            source,
            '    "deepseek_v4_fp8",\n',
            '    "deepseek_v4_fp8",\n    "exl3",\n',
        )
        updated = replace(
            updated,
            "    from .experts_int8 import ExpertsInt8Config\n",
            "    from .experts_int8 import ExpertsInt8Config\n"
            "    from .dots3_exl3_fp8 import Dots3HybridExl3Config\n",
        )
        updated = replace(
            updated,
            '        "deepseek_v4_fp8": deepseek_config,\n',
            '        "deepseek_v4_fp8": deepseek_config,\n'
            '        "exl3": Dots3HybridExl3Config,\n',
        )
    else:
        updated = replace(
            source,
            "            is_fused = loaded_weight.dim() == 3\n",
            "            # EXL3 per-expert Trellis tiles also have rank three.\n"
            "            is_fused = (\n"
            "                loaded_weight.dim() == 3\n"
            "                and not getattr(self.quant_method, \"exl3_per_expert_trellis\", False)\n"
            "            )\n",
        )
    compile(updated, str(path), "exec")
    path.write_text(updated)
print("Dots3 EXL3/FP8 vLLM v0.30.0 port applied")
