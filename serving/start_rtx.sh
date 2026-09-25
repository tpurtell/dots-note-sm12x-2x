#!/usr/bin/env bash
set -euo pipefail

# Development images may predate the parser; qualified release/run.py pins dots3.
reasoning_args=()
case "${REASONING_PARSER:-}" in
  '') ;;
  dots3) reasoning_args=(--reasoning-parser dots3) ;;
  *) echo "REASONING_PARSER must be empty (development) or dots3" >&2; exit 2 ;;
esac
hybrid_args=()
if [[ -n "${VLLM_HYBRID_LAYER_PARTITION:-}" ]]; then
  # Resolve cache geometry before rank-local layer construction: a rank may
  # begin with SWA and otherwise choose 16 before encountering DSA's 64.
  hybrid_args=(--block-size 64)
fi

hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_revision="${MODEL_REVISION:-d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da}"
model_path="hub/models--wrldsuksgo2mars--dots3-note-prev-exl3-k4-v1/snapshots/$model_revision"
[[ -f "$hf_home/$model_path/config.json" ]] || { echo "Missing model: $hf_home/$model_path" >&2; exit 1; }
container="${CONTAINER_NAME:-dots3-vllm-rtx}"
if docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container $container already exists" >&2
  exit 1
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_cache="${RUNTIME_CACHE:-$project_root/.cache/serving/rtx/runtime}"
mkdir -p "$runtime_cache"

docker run -d --name "$container" --gpus all --ipc=host --network=host \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e DOTS3_COMPACT_DSA_CACHE="${DOTS3_COMPACT_DSA_CACHE:-0}" \
  -e DOTS3_INDEXER_PREFILL_CONTEXTS="${DOTS3_INDEXER_PREFILL_CONTEXTS:-40}" \
  -e VLLM_HYBRID_LAYER_PARTITION="${VLLM_HYBRID_LAYER_PARTITION:-}" \
  -e VLLM_HYBRID_PACKED_ROUTING="${VLLM_HYBRID_PACKED_ROUTING:-0}" \
  -e VLLM_HYBRID_ATTESTATION_PATH="${VLLM_HYBRID_ATTESTATION_PATH:-}" \
  -e DOTS3_B12X_EXACT_FP8="${DOTS3_B12X_EXACT_FP8:-}" \
  -e DOTS3_B12X_EXACT_FP8_ROWS="${DOTS3_B12X_EXACT_FP8_ROWS:-4,16,64,512}" \
  -e DOTS3_B12X_VOCAB="${DOTS3_B12X_VOCAB:-0}" \
  -e VLLM_ENABLE_PCIE_ALLREDUCE="${DOTS3_B12X_PCIE:-0}" \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x \
  -e B12X_COMPILE_CACHE_DIR=/root/.cache/vllm-runtime/b12x/compile \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm-runtime/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/vllm-runtime/triton \
  -e CUDA_CACHE_PATH=/root/.cache/vllm-runtime/cuda \
  -e NCCL_DEBUG=INFO -e OMP_NUM_THREADS="${CPU_THREADS:-8}" \
  -e VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}" \
  -v "$hf_home:/root/.cache/huggingface:ro" \
  -v "$runtime_cache:/root/.cache/vllm-runtime" \
  "${IMAGE:-dots3-vllm-rtx:dev}" \
  "/root/.cache/huggingface/$model_path" \
  --served-model-name dots3-note-exl3-k4 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --distributed-executor-backend mp \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-512}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" --enable-prefix-caching \
  --structured-outputs-config '{"backend":"xgrammar"}' \
  --limit-mm-per-prompt '{"image":1,"audio":1,"video":0}' \
  --enable-auto-tool-choice --tool-call-parser dots \
  --host 0.0.0.0 --port "${PORT:-8001}" "${reasoning_args[@]}" "${hybrid_args[@]}" "$@"
