#!/usr/bin/env python3
"""Run one real K4 projection through rhea -> moa and verify checkpoint reuse."""

import json
import sys

import torch

from gptqmodel.utils.exl3_projection_checkpoint import build_projection_request
from gptqmodel.utils.exl3_remote import (
    EXL3_HESSIAN_CAPTURE_CONTRACT,
    EXL3_HESSIAN_NUMERICAL_CONTRACT,
    EXL3_HESSIAN_SYMMETRY_CONTRACT,
    EXL3RemoteClient,
    remote_client_from_provenance,
)


def main() -> None:
    with open(sys.argv[1], encoding="utf-8") as stream:
        provenance = json.load(stream)
    client = remote_client_from_provenance(provenance)
    endpoint = client.endpoints[0]
    client.qualify(endpoint)
    torch.manual_seed(787)
    weight = torch.randn(128, 128, dtype=torch.float32) * 0.02
    hessian = torch.eye(128, dtype=torch.float32) * 2048
    contract = {
        "bits": 4,
        "codebook": "mcg",
        "apply_out_scales": True,
        "sigma_reg": 0.025,
        "seed": 787,
        "hessian_capture": EXL3_HESSIAN_CAPTURE_CONTRACT,
        "hessian_numerical": EXL3_HESSIAN_NUMERICAL_CONTRACT,
        "hessian_symmetry": EXL3_HESSIAN_SYMMETRY_CONTRACT,
        "execution": EXL3RemoteClient.execution_contract(endpoint),
    }
    request = build_projection_request(
        module_full_name="smoke.projection",
        layer_index=1,
        input_weight=weight,
        hessian=hessian,
        sample_count=2048,
        quantizer_contract=contract,
        family_join=provenance["family_join"],
        route_evidence=None,
    )
    for attempt in range(2):
        packed, result, metadata = client.quantize(
            endpoint=endpoint,
            request_manifest=request,
            input_weight=weight,
            hessian=hessian,
        )
        assert set(packed) == {"trellis", "suh", "svh", "mcg"}
        assert bool(metadata["worker_checkpoint_hit"]) == bool(attempt)
        print(json.dumps({"attempt": attempt, "duration_seconds": result["duration_seconds"], "checkpoint_hit": metadata["worker_checkpoint_hit"]}), flush=True)


if __name__ == "__main__":
    main()
