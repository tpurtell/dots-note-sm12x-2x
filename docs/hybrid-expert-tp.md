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
12.799 GiB checkpoint tensor total. Loaded ownership receipts now verify that
the peer skips each tower, removing one copy of each across the pair. Usable KV
gains depend on the new startup profile and layer split.

Choose that split using actual owner weight allocations, per-rank KV block
costs and measured profiling headroom. Equal layer counts or equal free bytes
alone do not maximize the shared token capacity. Admission and long-context
requests must validate any estimated improvement.

### Capacity calculation

For this checkpoint and FP8 cache, each DSA layer retains **576 bytes of MLA
state plus 132 bytes of indexer state per token**. Each SWA layer retains
**1,088 bytes per token** within its 513-token window. There are 33 target SWA
layers plus the MTP SWA layer. Their logical window payload totals about
**18.10 MiB per request** across both GPUs; scheduler in-flight reservations
and physical arena padding are additional.

The current allocator uses a shared block-ID pool with different physical
strides on the two owners. For measured per-rank budgets `K0`, `K1` and
physical block strides `S0`, `S1`, available blocks are:

```
B = min(floor(K0 / S0), floor(K1 / S1))
R(L) = ceil(L / 64) + G_swa * A_swa(L, in_flight_tokens)
accounted_context_tokens = floor(B / R(L) * L)
```

`G_swa` is the global number of SWA groups after refinement;
`A_swa` comes from vLLM's actual SWA admission rule. At batch 512 with the
current asynchronous runner, 1,024 tokens may be in flight and the SWA rule
reserves 25 blocks per group per request. These values must be recomputed when
the batch cap changes. The formula reports equivalent capacity at the selected
length `L`; it does not prove concurrent requests at that length have passed.

`VLLM_HYBRID_BALANCE_KV_GROUPS=1` splits oversized SWA groups before worker
projection so bounded SWA pages do not unnecessarily widen every history
block. It preserves a common global group list and layer membership, appending
overflow groups while retaining the original IDs. This reduces padding within
the existing allocator; it does not introduce independent per-type arenas.
The [placement planner](../serving/plan_hybrid_placement.py) invokes the actual
vLLM grouping/allocation code and reports useful history, bounded SWA payload,
padding and the limiting rank separately. A fresh loaded receipt is required
to replace its estimated budgets after changing the cut.

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
checks on both platforms. Whole-model owner-only multimodal checks passed on
both; the optimized RTX 18/28 screen also passed all 12 coding requests and 40
API cases. See the [current qualification ledger](serving-progress.md) for
measured profiles and remaining gates. These development features do not change the published
RTX v1 profile until release-image qualification is complete on each platform.
