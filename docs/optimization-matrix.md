# Dots3 optimization coverage

This ledger tracks development candidates. Final defaults require whole-model
measurements on the named platform and qualification on its release image.
References: [Brandon RTX recipe](../../brandon-glm-5.3-flash/recipe/README.md)
and [Qwen Spark recipe](../../rtx6k-exl3-qwen3.8-flash-next/README.md).

| Area | Current implementation / evidence | Remaining decision |
| --- | --- | --- |
| Uniform K4 routed experts | B12x planned fused MoE; TP2 loads the full checkpoint. Shared preparation/runtime scratch recovered about 7.8 GiB per RTX GPU at 512-token capacity. | Profile routing/packing, explore EP2 and larger capacities. |
| FP8 core projections | Native block-FP8 methods preserve source scales; isolated B12x `q_a_proj` numerical check passed on GB10. | Compare useful B12x dense shapes in full-model execution on each platform. |
| Sparse MLA | B12x TP2 strided 1088-byte cache integration; exact geometry, interleaved pages, >2 GiB offsets, and changed-input graph checks passed. RTX full-model checks pass. | Spark full-model qualification; prefill/decode profiling and DCP investigation. |
| Sliding-window attention | Native Dots3 execution with an added 1088-wide gather/dequantizer; component checks passed on both architectures. RTX 262K boundary passed. | Profile allocations and evaluate useful B12x alternatives. |
| Vocabulary projection | Native RTX default; optional B12x single-token BF16 projection. GB10 component parity and graph checks passed. | Matched full-model A/B on both platforms. |
| RTX TP communication | Native custom all-reduce control; optional Brandon-derived B12x adapter updated to current planned API. Exact parity and changed-input graph replay pass at 1/2/4/8/16/32 rows. | Full-model C1 difference was +0.35%; larger-message components were slower. Native remains default; B12x is an evaluation option. |
| Spark communication | Earlier startup logs confirm NCCL NET/IB on both fast links. | User confirmed recovery; corrected target-only runtime is loading with memory monitoring. Evaluate applicable B12x RoCE paths. |
| Parallel layouts | Current adapter supports TP2/DCP1. | EP2 requires route-map integration; DCP needs exact owner/index/cache semantics before benchmarking. |
| KV capacity | FP8 KV, prefix caching; RTX 32K and 262K exact boundaries passed. 262K candidate reports 310,070 equivalent cache tokens at utilization 0.94. | Native model limit is 524,288; maximize useful capacity and qualify retrieval/concurrency under the final allocation. |
| Scheduler / prefill | 16 slots, initial 512-token chunks; actual C1/2/4/8/16 overlap verified on RTX. | Sweep chunks and slot counts with memory accounting and latency/throughput tradeoffs. |
| CUDA graphs / warmup | Live multimodal input contract repaired. RTX generation, tools, modalities and concurrent graph paths pass. | Package compiled platform kernels; qualify release startup warmup and post-ready JIT behavior. |
| Speculation | Native MTP weights and vLLM support verified. Hybrid FP8 config conversion probe passes. | MTP1 at 128K passed 16/21 content contracts plus tools, prefix/JSON and multimodal checks (136.92 weighted tokens/s). MTP2 measured 170.53 with 18/21 contracts; native control passed 19/21. Final quality/capacity/tuning remain. |
| XGrammar / tools | Named/required JSON parsing repaired while auto retains Dots XML parsing. Eight RTX API cases pass. | Spark and final-release qualification; repeat with speculation if enabled. |

Raw development evidence currently lives under `.cache/serving/{rtx,spark}`.
The final benchmark report must preserve accepted evidence with exact source,
model, image, hardware and launch settings. Current results are not GHCR release
qualification.
