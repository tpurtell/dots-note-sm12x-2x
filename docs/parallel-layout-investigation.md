# Dots3 parallel layout investigation

Source inspection: 2026-09-25. This records implementation constraints and proposed experiments, **not measured performance**. Current serving layout remains TP2, DCP1 with native inter-host collectives on Spark. Paths below refer to the checked-out source; `.cache/vllm-v0.30.0` is the pristine stable reference.

## Decisions

| Candidate | Source support | Practical disposition |
|---|---|---|
| TP2, DCP1 | Current implemented path | Control for all comparisons |
| TP2 + EP2, replicated tokens | B12x has a rank-local EP primitive; EXL3 loader and Dots adapter currently reject EP | Plausible bounded loader/adapter project; does not remove final all-reduce |
| TP2 + DCP2 | Dots SWA prefill explicitly rejects DCP; custom sparse backend has no owner exchange | Requires attention/cache work across both attention types; not a launch-flag experiment |
| Spark B12x RoCE all-reduce | Native integrated-GPU RDMA runtime and prepared collective APIs exist | Most isolated communication candidate; preserve native fallback and qualify on both hosts |
| RTX B12x RoCE | Runtime requires integrated GPU with active RDMA hardware | Inapplicable to discrete RTX; separate PCIe candidate already exists |

## TP2 versus EP2

The current adapter prepares all local experts with each expert's TP-sharded intermediate width. It refuses `expert_map` at execution. The inherited EXL3 method also raises for EP during weight creation, so removing the adapter check alone cannot enable EP.

Source anchors:

- [EXL3 EP rejection and TP metadata](../serving/vendor/exl3.py#L1285), [intermediate tensor slicing](../serving/vendor/exl3.py#L1452).
- [Current slab geometry and weight preparation](../serving/dots3_exl3_fp8.py#L115), [runtime rejection](../serving/dots3_exl3_fp8.py#L238).
- [B12x replicated-input EP contract](../third_party/sparkinfer-glmrt/b12x/moe/ep_moe/__init__.py#L1), [immutable map validation](../third_party/sparkinfer-glmrt/b12x/moe/ep_moe/_impl.py#L89).

For this 256-expert model, balanced static EP2 would hold 128 whole experts of intermediate width1536 per rank; TP2 holds 256 experts of width768. This is the same leading-order expert weight volume. EP may change tile utilization, rotation work and load balance; these are hypotheses to measure, not capacity or speed claims. A skewed routed batch can concentrate work on one EP rank.

B12x EP consumes the replicated hidden states and global top-k IDs, ignores experts owned by the other rank, and returns a partial output. Its global-to-local map must contain `-1` for remote experts and exactly one occurrence of each local slot. It **does not perform token dispatch or output reduction**. The existing vLLM runner already reduces un-reduced EP outputs when appropriate ([final reduction](../.cache/vllm-v0.30.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L493)). For DP1 without sequence parallelism, naive dispatch is not selected merely because EP is enabled ([dispatch predicate](../.cache/vllm-v0.30.0/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L793)). Therefore replicated-input EP2 retains an output all-reduce of the hidden-state tensor; it cannot be advertised as eliminating the main TP communication.

### Bounded implementation route

1. Use vLLM's canonical static placement map and preserve local-to-global metadata identity. Load only locally owned whole expert tensors; disable expert intermediate TP slicing in EP mode. Codebook checks currently form checkpoint names from local IDs and must use global IDs for EP ([validation](../serving/vendor/exl3.py#L1414)). Reject EPLB until migration/repacking exists.
2. Retain uniform MCG K4 without modifying checkpoint weights. **First investigate the existing fused-MoE route-map path**: its W4A16 binder accepts a contiguous int32 `route_expert_map` of `plan.route_E` entries ([binding validation](../third_party/sparkinfer-glmrt/b12x/moe/fused_moe/_impl.py#L3040)), and planning explicitly supports a global router namespace larger than local packed weights ([route capacity](../third_party/sparkinfer-glmrt/b12x/moe/fused_moe/_impl.py#L3340)). This suggests retaining the current prepared EXL3 payload with128 experts, declaring `route_num_experts=256`, and binding the canonical global-to-local map rather than introducing another weight format. Confirm remote `-1` masking and partial output zeroing in numerical tests. Brandon also uses the unified route-map binding for its projection-mixed EP path ([reference binding](../../brandon-glm-5.3-flash/recipe/patches/port-exl3-projection-mixed-glm53.py#L650)).

   The separate legacy EP API is another route, but carries a different preparation contract. Its `Caps` currently takes `weight_plan`, and its Trellis `full_rotation` branch requires float16 plan I/O rather than the ordinary BF16 branch ([capacity contract](../third_party/sparkinfer-glmrt/b12x/moe/ep_moe/_impl.py#L157), [rotation detection](../third_party/sparkinfer-glmrt/b12x/moe/ep_moe/_impl.py#L217)). Do not assume the current fused-MoE BF16 preparation object is interchangeable.
3. Prepare the immutable 256→128 map and serially shared scratch/output capacity before graph capture. Return only the local partial; ensure vLLM performs exactly one final reduction and preserves shared-expert/scaling semantics.
4. Validate each rank's partial against the reference sum over its owned experts, then validate the summed output against TP2. Include all-local/all-remote, uneven ownership, repeated routes, zero local routes, changed-input graph replay, and arbitrary supported static placement. Inspect actual loaded global IDs and memory before benchmarks.
5. Compare matched TP2/EP2 profiles at C1 and C16, short decode and512-token prefill, including per-rank imbalance, memory, collective timing and whole-model contracts.

Brandon is a concrete integration reference: [EP placement and loader port](../../brandon-glm-5.3-flash/recipe/patches/port-exl3-ep-glm53.py#L203), [prepared-map publication](../../brandon-glm-5.3-flash/recipe/patches/port-exl3-ep-glm53.py#L330), [binding and partial-output contract](../../brandon-glm-5.3-flash/recipe/patches/port-exl3-ep-glm53.py#L488). Its runtime and source version differ; copying the patch wholesale is not justified.

## DCP2: both attention families need correct ownership

The custom B12x sparse backend explicitly permits DCP1 only ([backend guard](../serving/dots3_b12x_attention.py#L40)). Its binding supplies rank-local block tables and global logical selections without a DCP ownership exchange ([binding](../serving/dots3_b12x_attention.py#L107)). Simply relaxing the guard would send remote logical selections through the wrong local cache mapping.

Stable vLLM's Dots SWA metadata builder independently raises `NotImplementedError` for DCP prefill ([SWA metadata](../.cache/vllm-v0.30.0/vllm/models/dots3_note/nvidia/attention.py#L380)). Although the model exposes query-replication plumbing ([SWA projections](../.cache/vllm-v0.30.0/vllm/models/dots3_note/nvidia/model.py#L382)), that does not establish end-to-end DCP support.

A correct DCP implementation requires global indexer top-k ownership, rank-local cache index translation, query exchange or replication, and numerically correct attention output/LSE merging. SWA prefill additionally needs remote history gathering with the causal sliding-window bounds. KV allocation, prefix cache reuse, short contexts, uneven page ownership and speculative multi-token verification must all agree on the same DCP mapping. Large recycled physical page IDs must retain64-bit arithmetic.

Brandon's [documented DCP2 owner exchange and PCIe A2A](../../brandon-glm-5.3-flash/recipe/README.md#L149) illustrates the missing work; its GLM cache layout does not supply a Dots1088-byte SWA/DSA implementation. DCP2 might increase cache capacity by reducing replication, but the actual heterogeneous pool and communication cost must be measured after implementation. It is not presently a supported recipe option.

## Spark B12x RoCE

The vendored library has a real RDMA one-shot all-reduce/all-gather implementation for integrated GPUs. Support requires `is_integrated` and an active HCA ([support check](../third_party/sparkinfer-glmrt/b12x/comm/roce/roce_oneshot.py#L119)); construction exchanges ranks, allocates pinned memory, connects the proxy and fixes capacity ([constructor](../third_party/sparkinfer-glmrt/b12x/comm/roce/roce_oneshot.py#L191)). Supported dtypes include BF16 and world sizes include2 ([constants](../third_party/sparkinfer-glmrt/b12x/comm/roce/roce_oneshot.py#L42)).

The current Spark launcher already exposes `/dev/infiniband` and permits locked memory ([launcher](../serving/start_spark_node.sh#L83)). That is prerequisite access, not evidence that the B12x collective is installed or faster than NCCL. Qwen's Spark reference qualified independent **TP1** servers ([reference scope](../../rtx6k-exl3-qwen3.8-flash-next/docs/spark-work.md#L60)); its vocabulary/MTP/memory tuning does not qualify two-host transport.

A bounded optional integration should replace only eligible TP all-reduces at first:

1. Construct one caller-owned runtime using a CPU exchange group on both ranks, explicit matching HCA/GID configuration and conservative payload capacity. Account pinned buffers in Spark host headroom.
2. Obtain the public `query_from_runtime`/`plan` and prepare actual collectives before graph capture ([prepared API](../third_party/sparkinfer-glmrt/b12x/comm/roce/_preparation.py#L91)). Both ranks must take identical preparation/selection branches. Keep unsupported sizes/dtypes on native collectives.
3. Use appropriate stream/channel ownership and stable output buffers. Before enabling graph replay, test alternating native and B12x calls, changed contents, graph sizes, multistream transitions and repeated requests on the real pair.
4. Call `check_health` after replay and fail the worker on poison. The runtime explicitly says timed-out data is untrustworthy and later launches do nothing; a poisoned runtime cannot silently fall back ([health contract](../third_party/sparkinfer-glmrt/b12x/comm/roce/roce_oneshot.py#L649)). Close the runtime before destroying its groups.
5. Measure end-to-end BF16 hidden-width5120 all-reduces at deployment decode/verification/prefill row counts against current NCCL/IB, then compare matched whole-model C1/C16 and correctness. Do not infer improvement from bandwidth or link count.

No EP/DCP/RoCE implementation or performance claim is introduced by this investigation. The next smallest communication experiment is the isolated Spark collective qualification; EP needs a coordinated loader/runtime patch, and DCP needs a separate attention implementation.
