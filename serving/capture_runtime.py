#!/usr/bin/env python3
"""Capture auditable vLLM startup, hardware and memory evidence."""

import argparse
import json
import platform
from pathlib import Path
import subprocess
import time


ENVIRONMENT = {
    "VLLM_HOST_IP", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME",
    "DOTS3_B12X_VOCAB", "CUTE_DSL_ARCH", "VLLM_EXL3_TRELLIS_MIN_M",
    "VLLM_EXL3_PREFILL_TRELLIS", "OMP_NUM_THREADS",
    "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_BREAKABLE_CUDAGRAPH",
    "DOTS3_B12X_EXACT_FP8", "DOTS3_B12X_EXACT_FP8_ROWS", "DOTS3_COMPACT_DSA_CACHE",
    "DOTS3_INDEXER_PREFILL_CONTEXTS", "VLLM_HYBRID_LAYER_PARTITION",
    "VLLM_HYBRID_PACKED_ROUTING", "VLLM_HYBRID_BALANCE_KV_GROUPS", "VLLM_HYBRID_ATTESTATION_PATH",
    "VLLM_HYBRID_MM_OWNERS", "VLLM_HYBRID_FUSED_PACK", "VLLM_HYBRID_OVERLAP_SHARED", "VLLM_HYBRID_BOUNDARY_OWNERS",
    "VLLM_ENABLE_PCIE_ALLREDUCE", "VLLM_PCIE_ALLREDUCE_BACKEND",
    "DOTS3_B12X_ROCE", "DOTS3_B12X_ROCE_EAGER", "DOTS3_B12X_ROCE_ROWS",
    "B12X_ROCE_HCA", "B12X_ROCE_GID_INDEX", "B12X_ROCE_SPIN_LIMIT",
    "B12X_COMPILE_CACHE_DIR", "B12X_ROCE_CACHE_DIR",
    "DOTS3_ALLOCATOR_FRACTION", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF",
}
LOG_MARKERS = (
    "Model loading took", "GPU KV cache size:", "Available KV cache memory:",
    "Graph capturing finished", "Captured CUDA graph", "Selected DeepGemm",
    "Selected DeepGEMM", "B12x", "prefix caching", "xgrammar",
    "Using V2 Model Runner", "Using V1 Model Runner",
)


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def capture_hybrid_attestation(container, info):
    env = dict(item.split("=", 1) for item in info["Config"]["Env"] if "=" in item)
    path = env.get("VLLM_HYBRID_ATTESTATION_PATH", "")
    if not env.get("VLLM_HYBRID_LAYER_PARTITION"):
        return {"status": "disabled"}
    if not path:
        return {"status": "not-configured"}
    if "--headless" in info.get("Args", []):
        return {"status": "headless-worker", "path": path,
                "scope": "Aggregate two-rank receipt is written by EngineCore head"}
    script = """import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.is_file():
 b=p.read_bytes()
 print(json.dumps({'status':'captured','path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'raw_json':b.decode()}))
else:
 print(json.dumps({'status':'pending','path':str(p)}))
"""
    try:
        return json.loads(command("docker", "exec", container, "python3", "-c", script, path))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {"status": "unavailable", "path": path, "error": str(exc)}


def capture_allocator_policy(container, info):
    env = dict(item.split('=', 1) for item in info['Config']['Env'] if '=' in item)
    if not env.get('DOTS3_ALLOCATOR_FRACTION'):
        return {'status': 'disabled'}
    rank = 1 if '--headless' in info.get('Args', []) else 0
    parent = Path(env.get('VLLM_HYBRID_ATTESTATION_PATH') or '/root/.cache/vllm-runtime/ownership.json').parent
    path = str(parent / f'allocator-policy-rank{rank}.json')
    script = """import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1])
if p.is_file():
 b=p.read_bytes()
 print(json.dumps({'status':'captured','path':str(p),'sha256':hashlib.sha256(b).hexdigest(),'raw_json':b.decode()}))
else:
 print(json.dumps({'status':'pending','path':str(p)}))
"""
    try:
        return json.loads(command('docker', 'exec', container, 'python3', '-c', script, path))
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return {'status': 'unavailable', 'path': path, 'error': str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    info = json.loads(command("docker", "inspect", args.container))[0]
    logs = command("docker", "logs", args.container)
    stats = command(
        "docker", "stats", "--no-stream", "--format", "{{json .}}", args.container,
    )
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"MemTotal", "MemFree", "MemAvailable", "SwapTotal", "SwapFree"}:
            meminfo[key] = value.strip()
    report = {
        "schema": "dots3-spark-vllm-runtime-v1",
        "captured_unix_seconds": time.time(),
        "host": platform.node(),
        "architecture": platform.machine(),
        "container": args.container,
        "image_id": info["Image"],
        "args": info["Args"],
        "hybrid_attestation": capture_hybrid_attestation(args.container, info),
        "allocator_policy": capture_allocator_policy(args.container, info),
        "started_at": info["State"]["StartedAt"],
        "status": info["State"]["Status"],
        "device_requests": info["HostConfig"]["DeviceRequests"],
        "selected_environment": [
            item for item in info["Config"]["Env"]
            if item.split("=", 1)[0] in ENVIRONMENT
        ],
        "selected_startup_lines": [
            line for line in logs.splitlines()
            if any(marker in line for marker in LOG_MARKERS)
        ],
        "docker_stats": json.loads(stats),
        "meminfo": meminfo,
        "gpu_inventory_csv": command(
            "nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader",
        ).splitlines(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"event": "runtime-captured", "host": report["host"], "container": args.container}), flush=True)


if __name__ == "__main__":
    main()
