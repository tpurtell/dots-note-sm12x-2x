# Layer ownership with tensor-parallel experts

This development feature assigns non-expert decoder layers to owners while
retaining tensor parallelism across **every routed expert**. The global vLLM
configuration remains TP2 / PP1. Both workers execute each expert layer; there
is no expert placement map or expert parallelism.

## Execution

`VLLM_HYBRID_LAYER_PARTITION=23,23` assigns decoder layers 0–22 to rank 0 and
23–45 to rank 1. The owner constructs the full attention projections, norms,
router and shared experts and holds that layer's KV cache. Each worker holds
half of the intermediate dimension of all 256 routed experts in layers 1–45.

For each routed layer:

1. The owner runs attention, normalization and routing.
2. Both workers receive the activation, expert IDs and routing weights.
3. Both execute their tensor partitions of the selected experts.
4. Expert partial outputs reduce to the owner, which adds the shared expert
   contribution once and continues the residual stream.

`VLLM_HYBRID_PACKED_ROUTING=1` combines the three input broadcasts into one
aligned byte payload. BF16 activations, int32 IDs and FP32 weights retain their
native representations. The expert output reduction is still required.
Buffers are prepared before CUDA graph capture, and only active rows are sent.
Hidden states and residuals move at the decoder ownership boundary. The MTP
decoder belongs to the last owner. Input embeddings and vocabulary projection
currently retain their existing TP interfaces.

DSA layers are 0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41 and 45. With the
initial 23/23 split, ranks own seven and six DSA layers respectively. The SWA
layers have their own QKV/cache; they do not directly read an earlier DSA
layer's sparse indices or KV. A split need not coincide with a DSA boundary.

## Multimodal ownership and memory balance

The separate opt-in `VLLM_HYBRID_MM_OWNERS=0,1` assigns vision to rank 0 and
audio to rank 1. The peer skips tower construction and checkpoint loading.
Native encoder outputs, including actual per-item lengths, are broadcast to
both workers' multimodal caches. This requires the existing `weights` encoder
TP mode and eager encoder execution; decoder CUDA graphs remain supported.

The initial loaded RTX receipt measured **7.459 GiB vision** and **1.651 GiB
audio** on *each* worker. These include registered buffers and deduplicate
storage aliases. Vision's runtime FP8 conversion makes this smaller than its
12.799 GiB checkpoint tensor total. Owner-only placement is expected to remove
one copy of each tower across the pair; usable KV gains require a new startup
profile and an uneven layer split.

Choose that split using actual owner weight allocations, per-rank KV block
costs and measured profiling headroom. Equal layer counts or equal free bytes
alone do not maximize the shared token capacity. Admission and long-context
requests must validate any estimated improvement.

## Interfaces and qualification

[hybrid_parallel.py](../serving/hybrid_parallel.py) contains the model-independent
owner plan, explicit dense/expert contexts and transport interface. The
[Dots adapter](../serving/dots3_hybrid_parallel.py) supplies model execution and
loading behavior. Distributed global TP state is not temporarily rewritten.
[Loaded ownership receipts](../serving/hybrid_attestation.py) check actual
parameter geometry, expert partitions, cache ownership and storage allocations.

The initial unpacked hybrid and native TP2 passed the same 12-request 524K
coding screen. See the [matched comparison](../benchmarks/development/rtx-native524-hybrid-comparison).
Packed communication has passed eager and changed-input CUDA graph component
checks on RTX; whole-model packed and owner-only multimodal qualification is
still in progress. These development features do not change the published
RTX v1 profile until release-image qualification is complete on each platform.
