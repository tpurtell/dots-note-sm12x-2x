# Isolated Spark RoCE collective probe

`check_roce.py` is a candidate qualification probe, not a serving integration.
It has not yet been executed on the GPUs. Run it only when the model engines
and other GPU benchmarks have stopped on **both** hosts.

## What it measures

- BF16 sums at 1/2/4/8/16/32 rows × hidden width5120.
- Exact parity against stable vLLM's native `PyNcclCommunicator`, including
  independently changed random inputs on each rank and rank-identical outputs.
- Eager and captured graph execution; changed-input replays, alternating small
  and large graphs, and ordered transitions between two eager streams.
- Five samples of100 iterations after10 warmups per arm; CUDA event timing,
  both ranks' raw samples and the slower rank's median. Arm order reverses at
  alternating shapes. This is an isolated collective result, not model speed.
- Explicit post-synchronization health checks, proxy statistics, memory usage,
  complete cleanup and append-only JSONL evidence on each rank.

The CPU rendezvous/exchange group uses Gloo. Native NCCL goes through vLLM's
PyNccl binding, matching the serving communication path without creating the
extra Torch NCCL process group that the B12x docs warn can cost several GiB on
Spark. All reduction output buffers and the prepared runtime remain alive
until their graphs are destroyed. No fallback occurs after poison or failure.

## Native two-host launch

Copy the current `serving/check_roce.py` to each remote recipe first. The paths
below match this project's Spark layout. Use the same native runtime image on
both hosts; `dots3-vllm-spark:dev` is a development candidate, not a release pin.
The runtime image must include B12x, vLLM, a C compiler and libibverbs headers.
The RDMA proxy is a small native C build on first use.

Verified read-only on both hosts: `rocep1s0f0` is active at GID3 with
10.55.0.5/6; `roceP2p1s0f0` is active at GID3 with10.55.1.5/6. These two HCA
names are deliberately in the same order on both ranks. NCCL receives the same
explicit HCA selection. Keep NCCL INFO output to verify its NET/IB path.

Start this command on **Moa** first:

```bash
cd /home/tj/dots-note-work/recipe
mkdir -p .cache/roce-probe
# Choose a new receipt filename for every run; the probe refuses to overwrite.
docker run --rm --name dots3-roce-probe-worker \
  --gpus all --ipc=host --network=host --device /dev/infiniband \
  --cap-add=IPC_LOCK --ulimit memlock=-1 \
  -e GLOO_SOCKET_IFNAME=enP2p1s0f0np0 \
  -e NCCL_SOCKET_IFNAME==enP2p1s0f0np0 -e NCCL_DEBUG=INFO \
  -e OMP_NUM_THREADS=4 \
  -e B12X_COMPILE_CACHE_DIR=/probe/.cache/roce-probe/b12x-compile \
  -e B12X_ROCE_CACHE_DIR=/probe/.cache/roce-probe/proxy \
  -v "$PWD:/probe" -w /probe --entrypoint torchrun \
  dots3-vllm-spark:dev \
  --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr=10.55.1.5 --master-port=29671 \
  serving/check_roce.py \
  --hcas rocep1s0f0,roceP2p1s0f0 --gid-index 3 \
  --peer-hosts 10.55.1.5 10.55.1.6 \
  --output .cache/roce-probe/moa-dual-001.jsonl
```

Then on **Rhea**, use the identical command with these replacements:

```text
--name dots3-roce-probe-head
--node-rank=0
--output .cache/roce-probe/rhea-dual-001.jsonl
```

For an isolated single-link control, use `--hcas roceP2p1s0f0` on **both** ranks
and fresh receipt names. The default six shapes use at most320KiB per reduction.
The CLI can change the matrix while bounding each payload to1MiB. Both ranks
must agree on all probe settings except local output path and HCA names.
`--help` works without importing Torch or initializing CUDA.

## Memory and failure expectations

At the default capacity, the pinned transport has six320KiB slots plus metadata
(about1.9MiB). Public preparation adds two320KiB alignment buffers, and this
probe bounds gather capacity to16bytes instead of inheriting the16MiB default.
Input/oracle/output tensors for the entire six-shape matrix total about1.85MiB.
CUDA contexts, NCCL, graph metadata, Python and compilation add further memory;
the probe does not claim an exact total process footprint.

Startup requires at least16GiB **physical MemAvailable**. PyTorch allocation is
capped at5% of device memory; pinned/native allocations are outside that cap.
Compilation uses the main process (`compile_workers=0`), avoiding a pool of
compiler processes. A watchdog samples physical headroom each second and exits
only the probe after three samples below8GiB. A20-minute process deadline also
covers stuck synchronization/cleanup. These controls reduce risk but do not
make concurrent model benchmarking appropriate or prevent every possible
system failure. The model containers are never stopped by the probe.

Gloo operations have a120-second timeout. RoCE uses its default20-million-spin
bound (the library documents roughly tens of seconds, not a precise deadline).
The preparation callback rendezvous occurs after compilation and before the
first GPU collective, so one rank compiling sooner does not consume the peer's
spin budget. On error, the process records the traceback and poison state,
destroys graphs, closes the preparation session and runtime, destroys NCCL,
then destroys the CPU group. A hard deadline exit cannot perform that normal
cleanup; the process/container must be allowed to terminate before retrying.

Success requires both ranks' final `complete` event, exact-parity records for
all shapes, alternating-graph and stream checks, and clean NCCL/IB logs. A
partial receipt or surviving peer is failure, not a usable timing result. A
successful microbenchmark still requires optional serving adapter work and
matched whole-model C1/C16 validation before enabling B12x RoCE in a recipe.
