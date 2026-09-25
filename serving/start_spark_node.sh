#!/usr/bin/env bash
set -euo pipefail

# Development images may predate the parser; qualified release/run.py pins dots3.
reasoning_args=()
case "${REASONING_PARSER:-}" in
  '') ;;
  dots3) reasoning_args=(--reasoning-parser dots3) ;;
  *) echo "REASONING_PARSER must be empty (development) or dots3" >&2; exit 2 ;;
esac

# Start rank 1 first, then rank 0. Known development hosts have defaults;
# other Spark pairs supply NODE_RANK, HOST_IP and MASTER_ADDR explicitly.
case "$(hostname -s)" in
  rhea) default_rank=0; default_ip=10.55.1.5; default_master=10.55.1.5 ;;
  moa) default_rank=1; default_ip=10.55.1.6; default_master=10.55.1.5 ;;
  *) default_rank=; default_ip=; default_master= ;;
esac
node_rank="${NODE_RANK:-$default_rank}"
host_ip="${HOST_IP:-$default_ip}"
master_addr="${MASTER_ADDR:-$default_master}"
case "$node_rank" in
  0) role=head ;;
  1) role=worker ;;
  *) echo "Set NODE_RANK=0 (head) or NODE_RANK=1 (worker)" >&2; exit 2 ;;
esac
[[ -n "$host_ip" && -n "$master_addr" ]] || {
  echo "Set HOST_IP to this node's address and MASTER_ADDR to the head's address" >&2
  exit 2
}
socket_ifname="${SOCKET_IFNAME:-enP2p1s0f0np0}"

hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_cache="models--wrldsuksgo2mars--dots3-note-prev-exl3-k4-v1"
model_mount=()
if [[ -n "${MODEL_DIR:-}" ]]; then
  # A byte-audited staging export can be qualified before Hub publication.
  model_snapshot="$(realpath -e "$MODEL_DIR")"
  model_in_container=/model
  model_mount=(-v "$model_snapshot:$model_in_container:ro")
else
  model_revision="${MODEL_REVISION:-d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da}"
  model_snapshot="$hf_home/hub/$model_cache/snapshots/$model_revision"
  model_in_container="/root/.cache/huggingface/hub/$model_cache/snapshots/$model_revision"
fi
if [[ ! -f "$model_snapshot/config.json" ]]; then
  echo "Missing accepted checkpoint at $model_snapshot" >&2
  exit 1
fi
if docker ps --format '{{.Names}}' | rg -q '^dots3-(quant-rhea|exl3-worker)$'; then
  echo "Stop the quantization process before starting the serving recipe" >&2
  exit 1
fi

container="dots3-vllm-${role}"
if docker ps -a --format '{{.Names}}' | rg -q "^${container}$"; then
  echo "Container $container already exists" >&2
  exit 1
fi

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_cache="${RUNTIME_CACHE:-$project_root/.cache/serving/spark/runtime}"
mkdir -p "$runtime_cache" "$hf_home"

args=(
  "$model_in_container"
  --served-model-name dots3-note-exl3-k4
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --distributed-executor-backend mp
  --nnodes 2 --node-rank "$node_rank"
  --master-addr "$master_addr" --master-port "${MASTER_PORT:-29501}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-512}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.80}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --enable-prefix-caching
  --structured-outputs-config '{"backend":"xgrammar"}'
  --limit-mm-per-prompt '{"image":1,"audio":1,"video":0}'
  --enable-auto-tool-choice
  --tool-call-parser dots
)
if [[ -n "${VLLM_HYBRID_LAYER_PARTITION:-}" ]]; then
  args+=(--block-size 64)
fi
if [[ "$role" == worker ]]; then
  args+=(--headless)
else
  args+=(--host 0.0.0.0 --port "${PORT:-8000}")
fi

docker run -d --name "$container" --gpus all --ipc=host --network=host \
  --cap-add=IPC_LOCK --ulimit memlock=-1 --device /dev/infiniband \
  -e VLLM_HOST_IP="$host_ip" \
  -e NCCL_SOCKET_IFNAME="=$socket_ifname" \
  -e GLOO_SOCKET_IFNAME="$socket_ifname" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e DOTS3_COMPACT_DSA_CACHE="${DOTS3_COMPACT_DSA_CACHE:-0}" \
  -e DOTS3_INDEXER_PREFILL_CONTEXTS="${DOTS3_INDEXER_PREFILL_CONTEXTS:-40}" \
  -e VLLM_HYBRID_LAYER_PARTITION="${VLLM_HYBRID_LAYER_PARTITION:-}" \
  -e VLLM_HYBRID_PACKED_ROUTING="${VLLM_HYBRID_PACKED_ROUTING:-0}" \
  -e VLLM_HYBRID_FUSED_PACK="${VLLM_HYBRID_FUSED_PACK:-0}" \
  -e VLLM_HYBRID_BOUNDARY_OWNERS="${VLLM_HYBRID_BOUNDARY_OWNERS:-}" \
  -e VLLM_HYBRID_MM_OWNERS="${VLLM_HYBRID_MM_OWNERS:-}" \
  -e VLLM_HYBRID_ATTESTATION_PATH="${VLLM_HYBRID_ATTESTATION_PATH:-}" \
  -e DOTS3_B12X_EXACT_FP8="${DOTS3_B12X_EXACT_FP8:-}" \
  -e DOTS3_B12X_EXACT_FP8_ROWS="${DOTS3_B12X_EXACT_FP8_ROWS:-4,16,64,512}" \
  -e DOTS3_B12X_VOCAB="${DOTS3_B12X_VOCAB:-1}" \
  -e DOTS3_B12X_ROCE="${DOTS3_B12X_ROCE:-0}" \
  -e DOTS3_B12X_ROCE_ROWS="${DOTS3_B12X_ROCE_ROWS:-1-64}" \
  -e DOTS3_B12X_ROCE_EAGER="${DOTS3_B12X_ROCE_EAGER:-0}" \
  -e B12X_ROCE_HCA="${B12X_ROCE_HCA:-}" \
  -e B12X_ROCE_GID_INDEX="${B12X_ROCE_GID_INDEX:-3}" \
  -e B12X_ROCE_SPIN_LIMIT="${B12X_ROCE_SPIN_LIMIT:-20000000}" \
  -e B12X_COMPILE_CACHE_DIR=/root/.cache/vllm-runtime/b12x/compile \
  -e VLLM_CACHE_ROOT=/root/.cache/vllm-runtime/vllm \
  -e TRITON_CACHE_DIR=/root/.cache/vllm-runtime/triton \
  -e CUDA_CACHE_PATH=/root/.cache/vllm-runtime/cuda \
  -e NCCL_DEBUG=INFO \
  -e OMP_NUM_THREADS="${CPU_THREADS:-8}" \
  -v "$hf_home:/root/.cache/huggingface:ro" \
  "${model_mount[@]}" \
  -v "$runtime_cache:/root/.cache/vllm-runtime" \
  "${IMAGE:-dots3-vllm-spark:dev}" "${args[@]}" "${reasoning_args[@]}" "$@"

# Spark shares physical RAM with the host. Keep the qualified host-headroom
# monitor outside the container so it can stop this workload under pressure.
if [[ "${MEMORY_GUARD:-1}" == 1 ]]; then
  nohup python3 "$project_root/serving/watch_spark_memory.py" \
    --container "$container" --min-available-gib "${MIN_HOST_AVAILABLE_GIB:-1}" \
    --ready-directory "$runtime_cache" \
    > "$runtime_cache/memory-watch.log" 2>&1 < /dev/null &
  guard_pid=$!
  echo "$guard_pid" > "$runtime_cache/memory-watch.pid"
  ready_file="$runtime_cache/memory-watch-ready-$guard_pid"
  guard_ready=0
  for _ in {1..100}; do
    if ! kill -0 "$guard_pid" 2>/dev/null; then break; fi
    if [[ -f "$ready_file" ]]; then guard_ready=1; rm -f "$ready_file"; break; fi
    sleep 0.1
  done
  if [[ "$guard_ready" != 1 ]]; then
    echo "Memory guard failed to become ready; stopping $container" >&2
    docker kill "$container" >&2
    exit 1
  fi
  echo "Host memory monitor PID $guard_pid; log: $runtime_cache/memory-watch.log"
fi
