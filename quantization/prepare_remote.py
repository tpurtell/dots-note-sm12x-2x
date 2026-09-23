#!/usr/bin/env python3
"""Create a Spark EXL3 coordinator/worker contract under the project cache."""

import argparse
import hashlib
import json
import os
import secrets
from pathlib import Path


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rhea-image", required=True)
    parser.add_argument("--moa-image", required=True)
    parser.add_argument("--rhea-gpu", required=True)
    parser.add_argument("--moa-gpu", required=True)
    parser.add_argument("--corpus-sha256", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.output.is_symlink() or any(not v.startswith("sha256:") for v in (args.rhea_image, args.moa_image)):
        raise ValueError("invalid output or image identity")
    source = {
        "fp8_revision": "7c14222e22423d6df6848eb0d1c5c3a88a00311a",
        "bf16_revision": "1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b",
    }
    family = {
        "recipe": "dots3-note-uniform-exl3-k4-v1",
        "model_type": "dots3_note",
        "source": source,
        "corpus_sha256": args.corpus_sha256,
        "bits": 4,
        "codebook": "mcg",
        "quantizer_seed": 787,
        "route_evidence_contract": "ds4rt.exl3-natural-route",
        "zero_route_recovery_contract": "ds4rt.exl3-zero-route-recovery",
        "zero_route_recovery_recipe": {
            "trigger": "natural-route-count-below-1024",
            "sample_source": "same-fixed-calibration-selection",
            "capture_method": "direct-expert-router-ranks-9-16-then-identity-residual",
            "selection_policy": "rank-ascending-then-fixed-replay-order-v1",
            "candidate_rank_min": 9,
            "candidate_rank_max": 16,
            "selection_cap": 1024,
            "target_sample_count": 1024,
            "identity_calibration_policy": "normalized-2i-residual-to-effective-count-1024-v2",
        },
    }
    rhea_preflight = digest({"role": "coordinator", "gpu": args.rhea_gpu, "image": args.rhea_image, "source": source})
    moa_preflight = digest({"role": "worker", "gpu": args.moa_gpu, "image": args.moa_image, "source": source})
    remote = {
        "contract": "ds4rt.exl3-remote-worker-v1",
        "scheduler": "dynamic-pipelined-slot-projection-v2",
        "endpoints": [{
            "name": "moa", "url": "http://moa:8765",
            "preflight_sha256": moa_preflight, "image_digest": args.moa_image,
        }],
        "coordinator_slots": [{
            "device": "cuda:0", "gpu_uuid": args.rhea_gpu,
            "preflight_sha256": rhea_preflight, "image_digest": args.rhea_image,
        }],
        "assignment_store": "/work/state/remote-assignments",
        "token_env": "DOTS3_EXL3_REMOTE_TOKEN",
        "timeout_seconds": 7200,
        "max_attempts": 2,
        "orchestration_workers": 4,
    }
    config = {
        "family_join": family,
        "run": {
            "projection_checkpoint": {
                "contract": "ds4rt.exl3-projection-checkpoint-v1",
                "root": "/work/state/projection-checkpoints",
            },
            "remote_workers": remote,
        },
    }
    path = args.output / "remote-config.json"
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    secret_path = args.output / "remote.env"
    if not secret_path.exists():
        descriptor = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"DOTS3_EXL3_REMOTE_TOKEN={secrets.token_urlsafe(48)}\n")
    print(json.dumps({"config": str(path), "secret": str(secret_path), "moa_preflight": moa_preflight, "rhea_preflight": rhea_preflight}, sort_keys=True))


if __name__ == "__main__":
    main()
