# Disposable RoCE adapter fail-stop gate

Run only after stopping both model workers, in the same native Spark candidate
image as the normal adapter gate. This probe allocates small BF16 buffers, caps
PyTorch allocation to 5% of device memory, requires 16 GiB physical headroom,
aborts after three samples below 8 GiB, and limits total runtime to 20 minutes.
It never changes interfaces or host configuration.

Mount the recipe at `/workspace`. Start Moa first, then Rhea, using host
networking and the same RDMA device permissions as the ordinary adapter probe.
Run these commands **inside the disposable probe containers**, with the same
cache mount and installed candidate vLLM/B12x as the ordinary adapter gate:

```bash
# Moa, rank 1
cd /workspace
GLOO_SOCKET_IFNAME=enP2p1s0f0np0 torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank=1 \
  --master-addr=10.55.1.5 --master-port=29687 \
  serving/check_roce_failstop.py --hcas roceP2p1s0f0 --gid-index 3 \
  --output .cache/serving/spark/roce/moa-adapter-failstop-001.json

# Rhea, rank 0
cd /workspace
GLOO_SOCKET_IFNAME=enP2p1s0f0np0 torchrun \
  --nnodes=2 --nproc-per-node=1 --node-rank=0 \
  --master-addr=10.55.1.5 --master-port=29687 \
  serving/check_roce_failstop.py --hcas roceP2p1s0f0 --gid-index 3 \
  --output .cache/serving/spark/roce/rhea-adapter-failstop-001.json
```

Each torchrun worker is a CPU supervisor of one disposable GPU child. Children
construct the real patched vLLM communicator, warm it, capture three dependent
all-reduces, and verify a healthy replay numerically. Rank 1 then stops only its
own runtime proxy, following the fault mechanism in the vendored transport's
GPU tests. Both ranks replay the graph. Three dependent collectives ensure the
rank that initially receives its peer's payload also encounters a failed wait.

Success requires **both** receipts to report `passed: true`: child exit 70,
`fault_armed` after successful healthy verification, and the adapter watchdog's
specific fatal-poison message. An exception, NCCL failure, supervisor timeout,
ordinary child exit, or unarmed early crash fails the gate. Preserve the adjacent
`.worker.stdout` and `.worker.stderr` files. CPU supervisors exit zero only for
the expected adapter-induced termination.

After injection there is deliberately no GPU synchronization, Python cleanup,
or native fallback in the child: independent host watchdog termination is the
behavior under test. Process exit releases its registrations and CUDA resources.
Discard the containers after the gate. This proves the adapter watchdog path;
the separate ordinary adapter harness proves healthy cleanup and fallback. It
does not simulate a NIC reset, cable removal, or arbitrary driver hang.
