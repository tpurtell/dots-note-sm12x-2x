#!/usr/bin/env bash
set -euo pipefail

# Run on rhea and moa after the accepted checkpoint is installed in each
# host's Hugging Face cache. Start moa first, then rhea.
case "$(hostname -s)" in
  rhea) node_rank=0; host_ip=10.55.1.5; role=head ;;
  moa)  node_rank=1; host_ip=10.55.1.6; role=worker ;;
  *) echo "This recipe is for rhea and moa only" >&2; exit 2 ;;
esac

model_revision="${MODEL_REVISION:?Set MODEL_REVISION to the accepted Hub commit}"
hf_home="${HF_HOME:-$HOME/.cache/huggingface}"
model_cache="models--wrldsuksgo2mars--dots3-note-prev-exl3-k4-v1"
model_snapshot="$hf_home/hub/$model_cache/snapshots/$model_revision"
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

model_in_container="/root/.cache/huggingface/hub/$model_cache/snapshots/$model_revision"
runtime_cache="${RUNTIME_CACHE:-$HOME/.cache/dots3-vllm}"
mkdir -p "$runtime_cache" "$hf_home"

args=(
  "$model_in_container"
  --served-model-name dots3-note-exl3-k4
  --tensor-parallel-size 2
  --pipeline-parallel-size 1
  --distributed-executor-backend mp
  --nnodes 2 --node-rank "$node_rank"
  --master-addr 10.55.1.5 --master-port "${MASTER_PORT:-29501}"
  --max-model-len "${MAX_MODEL_LEN:-32768}"
  --max-num-seqs "${MAX_NUM_SEQS:-16}"
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-2048}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
  --kv-cache-dtype fp8
  --enable-prefix-caching
  --structured-outputs-config '{"backend":"xgrammar"}'
)
if [[ "$role" == worker ]]; then
  args+=(--headless)
else
  args+=(--host 0.0.0.0 --port "${PORT:-8000}")
fi

docker run -d --name "$container" --gpus all --ipc=host --network=host \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e VLLM_HOST_IP="$host_ip" \
  -e NCCL_SOCKET_IFNAME='=enP2p1s0f0np0' \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e DOTS3_B12X_VOCAB=1 \
  -e OMP_NUM_THREADS="${CPU_THREADS:-8}" \
  -v "$hf_home:/root/.cache/huggingface:ro" \
  -v "$runtime_cache:/root/.cache/vllm-runtime" \
  "${IMAGE:-dots3-vllm-spark:dev}" "${args[@]}" "$@"
