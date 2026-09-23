#!/usr/bin/env python3
"""Extract a small verified routed layer from durable quant checkpoints."""

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from gptqmodel.utils.exl3_projection_checkpoint import EXL3ProjectionCheckpointStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.boundary.read_text())
    layer = int(manifest["layer_index"])
    if not 1 <= layer <= 45 or not 1 <= args.experts <= 256:
        raise ValueError("invalid routed layer or fixture expert count")
    entries = {entry["module"]: entry for entry in manifest["projection_entries"]}
    store = EXL3ProjectionCheckpointStore(args.checkpoints)
    payload = {}
    for expert in range(args.experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            entry = entries[name]
            loaded = store.load_committed(entry["request_sha256"])
            if loaded is None:
                raise ValueError(f"missing projection checkpoint: {name}")
            request, tensors, _result = loaded
            if request["module"] != name or request["quantizer_contract"]["bits"] != 4:
                raise ValueError(f"projection contract mismatch: {name}")
            if set(tensors) != {"trellis", "suh", "svh", "mcg"}:
                raise ValueError(f"projection tensor set mismatch: {name}")
            for field, tensor in tensors.items():
                payload[f"expert{expert}.{projection}.{field}"] = tensor.contiguous()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, args.output)
    print(json.dumps({
        "event": "exl3-fixture-complete",
        "layer": layer,
        "experts": args.experts,
        "tensors": len(payload),
        "bytes": args.output.stat().st_size,
    }), flush=True)


if __name__ == "__main__":
    main()
