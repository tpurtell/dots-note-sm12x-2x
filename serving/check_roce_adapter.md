# Two-Spark adapter qualification

Run only after serving and other GPU work have stopped on **both** Sparks. This
harness imports the adapter installed in `dots3-vllm-spark:roce-dev`, constructs
the real patched vLLM `CudaCommunicator`, and tests its actual backend dispatch.
It does not load a model or establish a whole-model performance result.

Coverage: BF16 width5120 at rows1/2/4/8/16/32/48/64; graph-only eager fallback;
unsupported row65, width5128 and float32 native fallback; noncontiguous input
rejection; consecutive distinct output addresses; changed-input graph replay;
and alternating graph sizes. Both graph outputs are checked exactly against the
same communicator's native PyNccl backend. Native setup uses the Gloo CPU group;
no extra Torch NCCL process group is created.

The script requires16GiB initial physical headroom and exits after three samples
below8GiB or a1200-second deadline. The adapter's own50ms health watchdog remains
active. Graphs are destroyed before communicator teardown; the outer watchdog
also covers cleanup. A final `complete` event on **both ranks** with no cleanup
errors is required. This is a healthy-transport test; intentional peer/proxy
failure qualification is separate.

Copy `check_roce_adapter.py` and `check_roce.py` into each remote project's
`serving/` directory. Choose a fresh receipt filename when repeating a run.
Start on **Moa** first:

```bash
cd /home/tj/dots-note-work/recipe
mkdir -p .cache/roce-adapter
NODE_RANK=1
docker run --rm --name dots3-roce-adapter-worker \
  --gpus all --ipc=host --network=host --device /dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_SOCKET_IFNAME==enP2p1s0f0np0 -e NCCL_DEBUG=INFO \
  -e OMP_NUM_THREADS=4 \
  -e B12X_COMPILE_CACHE_DIR=/probe/.cache/roce-adapter/b12x-compile \
  -e B12X_ROCE_CACHE_DIR=/probe/.cache/roce-adapter/proxy \
  -v "$PWD:/probe" -w /probe --entrypoint torchrun \
  dots3-vllm-spark:roce-dev \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr=10.55.1.5 --master-port=29639 \
  serving/check_roce_adapter.py \
  --rows 1 2 4 8 16 32 48 64 --hcas roceP2p1s0f0 --gid-index 3 \
  --output .cache/roce-adapter/adapter-rank1.jsonl
```

Then on **Rhea**:

```bash
cd /home/tj/dots-note-work/recipe
mkdir -p .cache/roce-adapter
NODE_RANK=0
docker run --rm --name dots3-roce-adapter-head \
  --gpus all --ipc=host --network=host --device /dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_SOCKET_IFNAME==enP2p1s0f0np0 -e NCCL_DEBUG=INFO \
  -e OMP_NUM_THREADS=4 \
  -e B12X_COMPILE_CACHE_DIR=/probe/.cache/roce-adapter/b12x-compile \
  -e B12X_ROCE_CACHE_DIR=/probe/.cache/roce-adapter/proxy \
  -v "$PWD:/probe" -w /probe --entrypoint torchrun \
  dots3-vllm-spark:roce-dev \
  --nnodes=2 --nproc-per-node=1 --node-rank="$NODE_RANK" \
  --master-addr=10.55.1.5 --master-port=29639 \
  serving/check_roce_adapter.py \
  --rows 1 2 4 8 16 32 48 64 --hcas roceP2p1s0f0 --gid-index 3 \
  --output .cache/roce-adapter/adapter-rank0.jsonl
```

The fresh proxy directory may compile the same native proxy already present in
the image; its files stay in the project cache. Record the image ID, B12x pin,
complete host logs and both receipts. The receipts additionally bind installed
adapter/communicator and harness source hashes. No SSH or GPU run was performed
while preparing this harness.
