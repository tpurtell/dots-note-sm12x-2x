# Dots3 optimization coverage

This ledger tracks development candidates. Final defaults require whole-model
measurements on the named platform and qualification on its release image.
References: [Brandon RTX recipe](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx)
and [Qwen Spark recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next).

## Qualified hybrid release profiles

The detailed ordinary-TP experiments below are historical controls. Current
candidates use owner-local non-expert work and KV with **routed-expert TP2**,
a **17/29** decoder cut, vision/audio owners **0/1**, native boundary tables,
MTP3, batch **512**, and a native **524,288-token** context limit. RTX published-image
qualification is complete on both platforms; inherited Spark stages and each continuation are explicit in the reports.

- **RTX:** retain narrow B12x SWA Q-B rows **4/8/16**, native vocabulary and
  collectives, packed routing, fused packing, shared-expert overlap, and balanced
  KV groups. The complete native build and public wrapper reproduce **2,004,801
  accounted cache tokens**. Larger batch and boundary-owner decisions are
  documented below. [Final RTX v2 qualification](../benchmarks/releases/rtx-20260925-v2/report.json)
  completed with C1/C2/C4 rates **163.02/129.26/98.80 tokens/s**, 36/36 natural
  coding completions and static checks, and the full 524K boundary. The report
  explicitly preserves 14 stages from before a hardware interruption and three
  continued stages; it does not claim uninterrupted operation.
- **Spark:** retain batch **512**, utilization **0.80**, and exactly **1 GiB**
  physical host reserve. Batch 2048/4096 exceeded the allowed 200,000-token KV
  loss. Batch 1024 passed the accounting limit but failed actual 128K prefill:
  the memory guard terminated the head container after three low-headroom
  samples. Its roughly 10% short-prefill improvement does not qualify it.
  [Batch evidence](../benchmarks/development/spark-prefill-batch-gates/manifest.json).
  Final Spark selects native FP8 projections and B12x RoCE, with post-warmup
  allocator fraction/GC threshold 0.90. Actual wrapper capacity is **2,777,333**.
  [Spark report](../benchmarks/releases/spark-20260925-v1/report.json) records
  36/36 natural/static coding passes, 6/6 retrieval and **150/176** tool points.
  Public lifecycle checks also pass; fresh/restart KV capacities 2,722,237 and
  2,762,204 disclose startup variability without replacing the benchmark figure.
  The exact boundary and selected functional/context checks are inherited with
  source/profile differences; final-digest measurements are separately identified.
- **Both:** reject forced sparse MQA. Long-prefill MQA was already selected on
  SM12; the flag changes the short dense shortcut, and does not remove the
  hypothesized long-prefill expansion. See the explanation below and the
  [qualification ledger](serving-progress.md).

## FP8 kernel choices

The checkpoint remains uniform EXL3 K4 for routed experts with its original
FP8 core. Two separate kernel choices are under evaluation:

- **Projection kernels:** `DOTS3_B12X_EXACT_FP8=swa_q_b_proj` selects B12x
  execution for sliding-window attention's query projection. The current
  hybrid screen restricts it to rows `4,8,16`, retaining native row-1 execution.
  This differs from the older ordinary-TP screen below. Its copied weights
  consume memory, so both coding throughput and KV capacity are measured.
  The [hybrid screen](../benchmarks/development/rtx-owner-swa-qb-screen/manifest.json)
  provisionally retains it: C1/C2/C4 rates 167.41/139.41/97.04 tokens/s and
  2,004,801 accounted cache tokens. The accepted RTX v2 release retains this
  choice; final measurements above are separate from the candidate screen.
- **Sparse-attention prefill:** vLLM's
  `--attention-config '{"sparse_mla_force_mqa":true}'` disables its dense-MHA
  shortcut for prefixes no longer than the sparse top-k (2,048 tokens).
  Long-context prefills already use our compact-cache MQA adapter on SM12.
  In vLLM 0.30, `_is_masked_mha_available` requires capability family 100;
  the masked-MHA long-prefill branch is unavailable on SM120/121. Our earlier
  hypothesis that this flag would avoid large long-context K/V expansion on
  these GPUs was incorrect. It does not alter SWA execution or expert
  quantization. The RTX screen found unchanged KV capacity and peak memory,
  with slightly slower prefill, so the flag is rejected for RTX. Component
  FP8-query reference checks alone do not establish whole-model equivalence.

Neither option changes the published checkpoint. Results in the table below
include earlier ordinary-TP controls; later hybrid sections identify the
current layer-ownership experiments.

| Area | Current implementation / evidence | Remaining decision |
| --- | --- | --- |
| Uniform K4 routed experts | B12x planned fused MoE; TP2 loads the full checkpoint. Shared preparation/runtime scratch recovered about 7.8 GiB per RTX GPU at 512-token capacity. | Shared primer storage also removes duplicate preparation buffers. EP2 top-8 placement and graph probes pass, but full-model EP2 loses 262K capacity and C1 performance; native TP2 remains preferred. |
| FP8 core projections | Native block-FP8 methods preserve source scales; isolated B12x `q_a_proj` numerical check passed on GB10. | RTX graph probes: Q-B improves at rows 4/16/64/512 (1.64/1.25/1.20/1.10×), slightly loses at row 1; output projection loses at several sizes. Exact blockscaled scales differ from native UE8M0 rounding. Optional Q-B full-model MTP3 screen: 187.29 weighted tokens/s versus 183.02 native, but 16/21 versus 17/21 contracts and C16 679.93 versus 723.82. Extra source copies cost ~0.46 GiB/rank and reduce KV capacity. Native remains default. |
| Sparse MLA | B12x TP2 strided 1088-byte cache integration; exact geometry, interleaved pages, >2 GiB offsets, and changed-input graph checks passed. RTX full-model checks pass. | Spark full-model qualification; prefill/decode profiling and DCP investigation. |
| Sliding-window attention | Native Dots3 execution with an added 1088-wide gather/dequantizer; component checks passed on both architectures. RTX 262K boundary passed. | Profile allocations and evaluate useful B12x alternatives. |
| Vocabulary projection | Native RTX default; optional B12x single-token BF16 projection. GB10 parity/graphs passed. RTX shared-plan component checks pass on both TP shards: distinct outputs, changed-input/weight graph parity, and obsolete draft-head release of 778,567,680 bytes/rank. B12x medians 490.50/489.36 µs versus native 507.28/507.78 µs (native/B12x 1.034/1.038×). | Shared-plan fix restores 5.03 GiB KV / 288,320 tokens at 262K / 0.95. One-run whole-model screen: B12x +1.58% C1, −3.61% C4 per-request decode rate; retain native vocabulary for balanced RTX default. Different answer lengths and 7/8 versus 8/8 natural completions preclude causal quality claims. [Component](../benchmarks/component/vocab/manifest.json) and [whole-model evidence](../benchmarks/development/rtx-vocab/manifest.json). |
| RTX TP communication | Native custom all-reduce control; optional Brandon-derived B12x adapter updated to current planned API. Exact parity and changed-input graph replay pass at 1/2/4/8/16/32 rows. | Full-model C1 difference was +0.35%; larger-message components were slower. Native remains default; B12x is an evaluation option. |
| Spark communication | B12x RoCE passes BF16 eager/changed-input graph checks at rows 1–64, real vLLM communicator dispatch/fallback and distinct-output checks, and disposable proxy-failure watchdog checks on both Sparks. | Matched MTP3 coding C1/C4 three-run comparison at 32K/0.80: paired median decode gains +3.34%/+5.26%. C1 natural stops 12/12 for both; C4 RoCE 10/12 versus native 12/12, with two async answers reaching 8192 tokens. Completion latency also reflects different output lengths. This historical screen informed the final RoCE selection; the completed Spark report now records C1/C2/C4 with MTP3 and native-context retrieval. Native remains the historical control. [Raw evidence](../benchmarks/development/spark-roce/manifest.json). |
| Parallel layouts | Current adapter supports TP2/DCP1. | Experimental EP2 now loads arbitrary whole-expert placement and reduces correctly. RTX 128K screen: 165.83 weighted tokens/s, 18/21 contracts; 262K rejected for insufficient KV (3.09 GiB). TP2 retains more KV. DCP remains 1: upstream Dots3 SWA prefill explicitly rejects DCP, and its decode gather does not consume distributed sequence lengths. Our sparse adapter also lacks owner localization, query exchange and output/LSE reduction. The 513-token SWA window is already bounded; DCP2 does not simply halve that workspace. Implementing both attention families is substantial and has no established C1–C4 latency benefit. |
| KV capacity | FP8 KV, prefix caching; RTX MTP2 262K exact boundary passed at utilization 0.95 with 5.10 GiB KV / 292,303 equivalent cache tokens. The same profile at 0.94 rejected startup cleanly for insufficient KV. | Native model limit is 524,288; maximize useful capacity and qualify retrieval/concurrency under the final allocation. |
| Scheduler / prefill | 16 slots, initial 512-token chunks; actual C1/2/4/8/16 overlap verified on RTX. | A 1024-token RTX chunk improved 32K prefill to 5,024 tokens/s versus 4,371 at 512, but reduced decode throughput and KV capacity; 512 remains preferred. Final measurements focus on agentic coding/reasoning C1–C4 per user preference. |
| CUDA graphs / warmup | Live multimodal input contract repaired. RTX generation, tools, modalities and concurrent graph paths pass. | Package compiled platform kernels; qualify release startup warmup and post-ready JIT behavior. |
| Speculation | Native MTP weights and vLLM support verified. Hybrid FP8 config conversion probe passes. | MTP1 at 128K passed 16/21 content contracts plus tools, prefix/JSON and multimodal checks (136.92 weighted tokens/s). MTP2 measured 170.53 with 18/21 contracts; MTP3 183.02 with 17/21; MTP4 179.83 with 19/21; native control passed 19/21. MTP2 led the current one-run C16 screen (776.21 versus MTP3 723.82/MTP4 640.74). Matched 262K MTP2/MTP3 prose tests favored MTP2 at C1 (138.99 vs 130.58) and C16 (774.39 vs 726.89), but the user now prioritizes coding/reasoning at C1–C4: run dedicated matched tests before final selection. Spark MTP1/MTP2 seven-workload results: 33.79/37.80 tokens/s (17/21 and 18/21 contracts); MTP3 is being qualified. |
| Dynamic speculative depth | Both platforms use the V2 model runner. Its batch-size schedule can reduce target verification rows while retaining full graphs, but draft generation still runs the configured maximum number of steps. | Keep fixed depth unless the coding curves cross clearly; a dynamic candidate needs direct measurement of changing batch sizes and grammar boundaries. It cannot be assumed to match fixed K2 at its K2 tier. |
| XGrammar / tools | Named/required JSON parsing repaired while auto retains Dots XML parsing. Eight RTX API cases pass. | Spark native MTP1/2/3 prefix, JSON, eight tool modes and both modalities pass. Final release-image qualification remains. |

Selected RTX development receipts are preserved in [the evidence manifest](../benchmarks/development/rtx-optimization/manifest.json), with lossless gzip payloads and hashes. Additional in-progress evidence lives under `.cache/serving/{rtx,spark}`.

The subsequent matched reasoning/coding comparison favors RTX MTP3 over MTP2
at the user's priority concurrency levels: median per-request decode rates are
181.33 versus 166.76 tokens/s at C1, 141.38 versus 130.56 at C2, and 102.53
versus 96.43 at C4. Each candidate completed all 36 measured requests naturally;
each passed 35 static response checks. Those checks do not execute generated
code. Output lengths differ, so completed-answer latency is reported alongside
output token counts instead of being treated as an isolated speed ranking.
The [matched coding evidence](../benchmarks/development/rtx-coding/manifest.json)
preserves the payloads, responses and runtime identities. The completed MTP4
extension measured 185.29 / 140.07 / 103.07 tokens/s at C1/C2/C4 versus MTP3's
181.33 / 141.38 / 102.53. MTP4 finished 34/36 requests naturally; two async-pool
responses exhausted 8192 tokens during reasoning. MTP3 finished 36/36.
**RTX selects fixed MTP3 for the requested C1–C4 balance.** The small rate
crossovers do not establish a dynamic-depth benefit. These limited samples do
not show that MTP4 causes lower answer quality; completion and output length
are reported separately from decode speed. A later native-vocabulary MTP3 screen
also had an async-pool reasoning response reach the same 8192-token limit. Spark selection remains in progress;
all of these are development measurements.

The final benchmark report must preserve accepted evidence with exact source,
model, image, hardware and launch settings. Current results are not GHCR release
qualification.

## RTX SWA Q-B finite screen: retain native FP8

The opt-in `swa_q_b_proj` selector at rows 1/4/16 completed an eight-request
matched C1/C4 screen. Native versus candidate median decode rates were
180.436/186.543 tokens/s at C1 (+3.38%) and 104.648/105.269 at C4 (+0.59%).
Native completed 8/8 naturally; the candidate completed 7/8, with one request
reaching the benchmark's 8,192-token output budget including reasoning.
Different sampled outputs prevent a causal quality-regression conclusion.
Retain native FP8: this finite screen does not establish a clear balanced
benefit, the copied weights add about 264 MiB per GPU, and native 524K-context
capacity is now the priority. [Exact raw evidence and provenance](../benchmarks/development/rtx-swa-qb-priority/manifest.json)
are archived; no benchmark extension is planned.

## Hybrid ownership: prefill batch selection

The ordinary TP results above remain historical controls. The current hybrid
candidates keep routed experts at TP2 while assigning attention, other decoder
work and KV storage to layer owners. Their larger KV budgets require a fresh
prefill comparison; the earlier 512-token choice is not a final hybrid default.

After selecting a layer cut, compare `max_num_batched_tokens=512`, `2048` and
`4096` separately on RTX and Spark. Keep the image, layer cut, kernel options,
MTP depth, context limit, sequence limit and memory utilization identical.

- Capture each startup's actual KV budget, block count, group stride and
  scheduler admission bounds. Larger chunks can consume both activation
  workspace and additional bounded SWA blocks.
- Reject increases costing more than **200,000 accounted cache tokens** relative
  to the same candidate at 512, before running expensive performance screens.
- Measure repeated cold-prefill throughput and TTFT with matched prompt lengths
  and no reusable prefix. Preserve actual prompt-token counts and individual
  samples; prefix-cache hits must not masquerade as faster prefill.
- Promote a larger batch only for a repeatable, meaningful prefill improvement
  while preserving the requested C1–C4 reasoning/coding performance. Otherwise
  retain 512. The two platforms may select different values.

These screens do not establish maximum-context concurrency. Final release
qualification must still exercise the selected profile at its context boundary.

## Rejected indexer profiling lifetime change

Holding the dummy logits allocation through one profile forward did not remove rank1 fragmentation and worsened rank0 capacity. Same-cut18 startup: rank0 inactive splits1.547→5.047GiB and KV6.654→2.654GiB; rank1 inactive splits stayed3.025GiB while peak increased512MiB. `VLLM_INDEXER_PROFILE_HOLD_LOGITS` is not adopted; `port_indexer_profile_lifetime.py` is a rejected experiment retained for reproducibility, not part of native image/launcher integration. [Raw comparison and provenance](../benchmarks/development/rtx-rejected-indexer-lifetime/manifest.json). No performance traffic or accepted fix is claimed.

## Batch-token cap: retain512 after warm return control

Batch1024 improved cold prefill by about9–10% but lost132,471 KV tokens. After matched prefill load, warm512 coding measured164.10/129.17/98.83 tokens/s atC1/C2/C4 versus1024 at148.70/121.83/82.05 (about−9.4/−5.7/−17.0%). Warm512 completed12/12 naturally/static;1024 completed11/12 with one async8192-budget truncation. Retain512. This does not establish causal quality or thermal effects. [Raw prefill/coding, return-control telemetry and exact hashes](../benchmarks/development/rtx-prefill-batch-capacity/manifest.json).

## Boundary0/1 ownership rejected

Same narrow SWA-QB profile with boundary0/1 measured149.82/122.24/93.71 coding tokens/s versus167.41/139.41/97.04 atC1/C2/C4 with native TP2 boundary tables. Capacity declined2,004,801→1,899,086. Despite12/12 natural/static completions and functional passes, retain native TP2 embedding/head. No causal quality claim. [Raw ownership/alias, functional and timing evidence](../benchmarks/development/rtx-rejected-boundary-ownership/manifest.json).
