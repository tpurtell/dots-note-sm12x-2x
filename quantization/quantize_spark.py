#!/usr/bin/env python3
"""Run uniform K4 Dots3 routed-expert quantization on a Spark.

This is the local coordinator entry point. The source is an indexed FP8/BF16
hybrid built by build_hybrid_source.py; the source shards remain immutable.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


EXPERT_PATTERN = (
    r"^model\.layers\.(?:[1-9]|[1-3][0-9]|4[0-5])\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)$"
)


class Layer0Complete(Exception):
    """Intentional early exit for a bounded replay benchmark."""


def calibration_texts(path: Path, limit: int | None) -> list[str]:
    texts = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("calibration row has no prompt")
            texts.append(prompt)
            if limit and len(texts) >= limit:
                break
    if not texts:
        raise ValueError("empty calibration corpus")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--benchmark-layer0", action="store_true")
    parser.add_argument("--remote-config", type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    if args.source.is_symlink() or args.state.is_symlink() or args.output.is_symlink():
        raise ValueError("source/state/output root must not be symbolic links")
    args.state.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("GPTQMODEL_EXLLAMAV3_BUILD_ROOT", str(args.state / "jit" / "exllamav3"))
    os.environ.setdefault("GPTQMODEL_EXL3_ERROR_JOURNAL", str(args.state / "errors.jsonl"))
    os.environ.setdefault("GPTQMODEL_EXL3_CAPTURE_FRONTIER", str(args.state / "capture-frontiers"))

    import torch
    from gptqmodel import GPTQModel
    from gptqmodel.models.definitions.dots3_note import Dots3NoteQModel
    from gptqmodel.quantization import AutoModuleDecoderConfig, EXL3Config

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one local Spark GPU")
    provenance = None
    if args.remote_config is not None:
        with args.remote_config.open() as stream:
            provenance = json.load(stream)
        if not isinstance(provenance, dict) or not isinstance(provenance.get("family_join"), dict):
            raise ValueError("remote config has no immutable family identity")
    qcfg = EXL3Config(
        bits=4,
        codebook="mcg",
        out_scales="auto",
        module_include=[EXPERT_PATTERN],
        preprocessors=[AutoModuleDecoderConfig(target_dtype=torch.bfloat16)],
        fallback=None,
        offload_to_disk=True,
        offload_to_disk_path=str(args.state / "offload"),
        device="cuda:0",
        calibration_data_device="cpu",
        dense_vram_strategy_devices=["cuda:0"],
        moe_vram_strategy="balanced",
        moe_vram_strategy_devices=["cuda:0"],
        meta={"ds4rt_error_ledger": provenance} if provenance is not None else {},
    )
    print(json.dumps({"event": "model-load-start", "source": str(args.source)}), flush=True)
    model = GPTQModel.load(str(args.source), quantize_config=qcfg, trust_remote_code=False)
    if not isinstance(model, Dots3NoteQModel):
        raise RuntimeError(f"unexpected GPTQModel definition: {type(model).__name__}")
    turtle = getattr(model, "turtle_model", None)
    if turtle is None or len(getattr(turtle, "_weight_map", {})) != 38469:
        raise RuntimeError("hybrid LazyTurtle source did not materialize")
    print(json.dumps({"event": "model-load-complete", "source_tensors": len(turtle._weight_map)}), flush=True)
    if args.load_only:
        return
    if provenance is not None and not args.benchmark_layer0:
        from dots3_layer_boundary_store import (
            Dots3LayerBoundaryController,
            Dots3LayerBoundaryStore,
        )

        config = model.model.config
        checkpoint_root = provenance["run"]["projection_checkpoint"]["root"]
        identity = {
            "family_join": provenance["family_join"],
            "calibration": str(args.calibration),
            "source": str(args.source),
        }
        plan_sha256 = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        store = Dots3LayerBoundaryStore(
            args.state / "layer-boundaries",
            plan_sha256=plan_sha256,
            family_join=provenance["family_join"],
            projection_checkpoint_root=checkpoint_root,
            error_journal_path=args.state / "errors.jsonl",
            hidden_size=int(config.hidden_size),
            activation_rank=3,
            routed_experts=int(config.n_routed_experts),
            first_target_layer=1,
            last_target_layer=int(config.num_hidden_layers) - 1,
        )
        model.quantization_layer_boundary_checkpoint = Dots3LayerBoundaryController(
            store, defer_publication_materialization=True
        )
    texts = calibration_texts(args.calibration, args.limit)
    print(json.dumps({"event": "quantize-start", "prompts": len(texts)}), flush=True)
    if args.benchmark_layer0:
        def stop_before_routed_layer(_module, _name, layer_index, _config):
            if layer_index >= 1:
                raise Layer0Complete
            return True

        model.should_quantize_layer = stop_before_routed_layer
    started = time.monotonic()
    try:
        model.quantize(texts, batch_size=args.batch_size, calibration_sort=None)
    except Layer0Complete:
        print(json.dumps({"event": "layer0-benchmark-complete", "prompts": len(texts), "batch_size": args.batch_size, "seconds": time.monotonic() - started}), flush=True)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("output directory must be empty before publication")
    args.output.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output), max_shard_size="8GB")
    print(json.dumps({"event": "export-complete", "path": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
