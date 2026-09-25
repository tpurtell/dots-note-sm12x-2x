# Dots3 Note Preview: EXL3 K4 on two Sparks or two RTX GPUs

This repository provides one Dots3 Note Preview checkpoint and is developing two-GPU vLLM recipes for DGX Spark and RTX PRO 6000. The completed checkpoint retains the [FP8 source](https://huggingface.co/dots-studio/dots3-note-prev-fp8) outside the routed language-model experts. Every routed expert `gate_proj`, `up_proj`, and `down_proj` weight in layers 1–45 comes from the [BF16 source](https://huggingface.co/dots-studio/dots3-note-prev) and uses uniform EXL3 K4. Vision experts remain as supplied by the FP8 checkpoint. The published checkpoint is [`wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1`](https://huggingface.co/wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1).

**Status (2026-09-25):** quantization, tensor audit, publication and local cache installation are complete. The checkpoint is pinned to `d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da`; its 36 artifact files total 163,552,088,967 bytes. **RTX v2 published-image qualification is complete at the native 524,288-token context.** Its public container fast path has passed pull, fresh-cache startup, health, logs, stop and restart checks. Spark memory qualification and its final container remain in progress; unmeasured Spark table entries stay blank. See the [RTX v2 report](benchmarks/releases/rtx-20260925-v2/report.json) and [qualification ledger](docs/serving-progress.md).

## Container fast path

The public RTX `20260925-v2` image is:

```text
ghcr.io/tpurtell/dots3-note-exl3-k4-rtx@sha256:d350ceb8c9be1dce3851ab20fba4c586f1530bef0a65a7094305b4ee8d2df16e
```

The [deployment evidence](benchmarks/releases/rtx-20260925-v2/fastpath/manifest.json) records these checks with zero benchmark requests and no model download.

Follow the [container run instructions](serving/release/README.md#container-fast-path) and [pinned release settings](serving/release/settings.json). Mount the entire existing `HF_HOME`; no model download is needed. RTX uses one amd64 container with two GPUs. Spark uses a separate arm64 image on both hosts, worker first; its final public fast path is pending.

The qualified RTX profile uses **17/29 owner-local decoder layers**, **TP2 for every routed expert**, vision on rank 0 and audio on rank 1, FP8 KV, **524,288 context**, **0.95 utilization**, 16 slots, batch 512 and **MTP3**. It enables packed/fused routing, shared-expert overlap, compact cache storage and balanced KV groups. Optional B12x SWA Q-B projection runs only at rows 4/8/16; vocabulary, boundary tables and other dense projections retain the selected native paths. Accounted KV capacity is **2,004,801 tokens**, not a promise that every request mix can use that total. See [execution and cache accounting](docs/hybrid-expert-tp.md).

The report explicitly records a hardware interruption and restart: **14 completed stages were preserved**, and **three unfinished stages** were completed with the same image, model, launch profile and workload sources. This was not an uninterrupted run. Original receipts, prior-artifact hashes, driver failure evidence and both runtime identities are retained. No temperature, power or cooling settings were changed for the continuation.

The earlier ordinary-TP2, 262,144-context RTX v1 remains available as [historical release evidence](benchmarks/releases/rtx-20260925-v1/report.json); the tables below describe hybrid v2.

## Source and calibration

| Input | Resolved revision | Use |
| --- | --- | --- |
| `dots-studio/dots3-note-prev-fp8` | `7c14222e22423d6df6848eb0d1c5c3a88a00311a` | Preserved core and multimodal tensors |
| `dots-studio/dots3-note-prev` | `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` | BF16 routed expert weights |

The [source audit](quantization/inspect_sources.py) checks the 46-layer, 256-expert geometry, all 34,560 routed weights and their FP8 scales, and the exact BF16/FP8 source dtypes. The initial local inventory measured **517.27 GiB** of BF16 routed weights, **264.15 GiB** of FP8 routed weights, and **13.94 GiB** of non-expert FP8 checkpoint weights. These are source tensor bytes, not an EXL3 artifact size or serving memory measurement.

The [calibration builder](quantization/build_calibration_corpus.py) adapts GLMRT's source-disjoint selection to Dots3 Note's non-thinking chat template. Its renderer was checked against the source `chat_template.jinja`. The first generated selection contains 1,437 calibration prompts / 1,081,453 prompt tokens, 89 held-out prompts / 67,466 tokens, and 146 screening prompts / 110,279 tokens. The manifest binds the tokenizer hash, builder revision, source groups, and split hashes. Calibration and held-out source groups must remain disjoint.

Both Sparks have 121 GiB unified memory and local NVMe. The source checkpoint exceeds one Spark's memory, so GPTQModel streams layers through an indexed hybrid source. The BF16 checkpoint interleaves layers across all 131 model shards; file-level layer partitioning does not reduce transfer volume. The completed quantization run replayed calibration prompts on rhea and distributed K4 expert projection work across rhea and moa. The cross-host K4 kernel and checkpoint-reuse path passed a live projection test. Layer 1's rolling activation boundary passed a full shard-hash audit, and a live restart resumed directly at layer 2. See the [quantization runbook](quantization/README.md) for the prompt-splitting tradeoff.

## Serving target

Start from [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), the latest stable release rechecked on 2026-09-25. Native Dots3 Note support entered through [#51255](https://github.com/vllm-project/vllm/pull/51255), and the merged [Dots3 runtime optimization #53517](https://github.com/vllm-project/vllm/pull/53517) is present in the pinned source. The open [video-audio cache repair #57655](https://github.com/vllm-project/vllm/pull/57655) remains relevant if video with audio is qualified. The [two-node launch script](serving/start_spark_node.sh) targets two GB10 GPUs at tensor parallel size 2 with **80% GPU memory utilization** in current candidates, based on measured unified-memory headroom and a 1 GiB host-memory guard. Final limits require request qualification.

Qualification must prove prefix-cache hits on repeated long prompts, including Dots3's sliding-window and sparse-MLA cache groups. Record cached-token counters and warm versus cold TTFT; review vLLM's [prefix caching contract](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and sparse-MLA alignment rules. Exercise xgrammar JSON and tool calls with and without speculation. The earlier Qwen recipe carried a termination fix related to [#52805](https://github.com/vllm-project/vllm/pull/52805), with a [reported follow-up](https://github.com/vllm-project/vllm/issues/53181); carry a patch only if the pinned release still needs it.

Use B12x where its exact Dots3 geometry and semantics qualify: EXL3 MoE, dense projections, MLA/DSA or SWA attention, and vocabulary projection. Record correctness and graph replay before performance comparisons. Dots3's padded DSA cache and sliding-window MLA require separate checks from existing GLM sparse-MLA paths.

The [Brandon-derived GLM RTX recipe](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx) guides the RTX B12x investigation, and the [Qwen Spark recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next) guides the Spark integration and reporting. See the [optimization decisions](docs/optimization-matrix.md) for Dots3-specific results.

## Headline measurements

| Metric | 2× Spark | 2× RTX |
|---|---|---|
| C1 reasoning/coding aggregate output tokens/s | — | 161.30 |
| C2 reasoning/coding aggregate output tokens/s | — | 229.07 |
| C4 reasoning/coding aggregate output tokens/s | — | 327.47 |
| C1 seven-workload weighted decode tokens/s | — | 165.60 |
| Seven content contracts | Pending | 18/21 |
| Exact context boundary, input + output | — | 524,032 + 256 = 524,288 |
| C1 greedy merge_intervals median tokens/s | — | 202.57 |
| Sampled async code, depth 0, burst-excluded tokens/s | — | 187.95 |
| C16 sampled clients aggregate tokens/s | — | 665.05 |

## Reasoning-enabled coding: C1–C4

| Platform | C | Aggregate output tokens/s | Completed latency s | Natural | Static checks | Truncated | Output tokens median (range) |
|---|---|---|---|---|---|---|---|
| spark | 1 | — | — | —/— | —/— | — | — (—–—) |
| spark | 2 | — | — | —/— | —/— | — | — (—–—) |
| spark | 4 | — | — | —/— | —/— | — | — (—–—) |
| rtx | 1 | 161.30 | 23.28 | 12/12 | 12/12 | 0 | 3,783.50 (2336–7282) |
| rtx | 2 | 229.07 | 34.76 | 12/12 | 12/12 | 0 | 4,343.00 (2754–6266) |
| rtx | 4 | 327.47 | 39.02 | 12/12 | 12/12 | 0 | 3,844.00 (2557–5994) |

Coding aggregate throughput is total streamed output tokens divided by summed concurrent-wave wall time, including reasoning and prefill. It is measured directly, not concurrency times a per-stream median. Completed latency excludes truncated answers. The benchmark output budget 8192 includes reasoning and is not the server output limit; clients may request more within context. Static checks are not execution-based code correctness. Variable output lengths affect latency.

## Seven content workloads: C1

| Case | Spark median tokens/s | RTX median tokens/s |
|---|---|---|
| code | — | 202.57 |
| math | — | 204.42 |
| fable | — | 126.22 |
| hello | — | 168.35 |
| topic | — | 136.71 |
| structured-json | — | 209.66 |
| multilingual | — | 143.32 |

RTX content contract misses: fable run1: fable has 113 words, outside 140..170; fable run2: fable has 100 words, outside 140..170, response does not end with a moral about sharing credit; topic run2: response omits paging.

## Sampled prose clients

| Platform | Clients | Aggregate tokens/s | Minimum overlap |
|---|---|---|---|
| rtx | 1 | 112.94 | 1 |
| rtx | 2 | 185.71 | 2 |
| rtx | 4 | 294.05 | 4 |
| rtx | 8 | 442.00 | 8 |
| rtx | 16 | 665.05 | 16 |

## Sampled async code and context scaling

| Platform | Probe | Input depth | TTFT s | Prompt tokens / TTFT | Decode tokens/s |
|---|---|---|---|---|---|
| rtx | code-agent | 0 | 0.06 | 2,372.99 | 187.95 |
| rtx | code-agent | 8192 | 0.08 | 98,452.59 | 178.41 |
| rtx | code-agent | 24000 | 0.09 | 268,774.51 | 178.81 |
| rtx | context-2048 | 2048 | 0.61 | 3,332.80 | 205.47 |
| rtx | context-8192 | 8192 | 2.29 | 3,583.20 | 204.58 |
| rtx | context-32768 | 32768 | 9.29 | 3,528.71 | 201.83 |
| rtx | context-65536 | 65536 | 19.14 | 3,423.60 | 201.65 |
| rtx | context-131072 | 131072 | 40.75 | 3,216.11 | 202.17 |
| rtx | context-262144 | 262144 | 92.13 | 2,845.40 | 202.29 |
| rtx | context-524032 | 524032 | 271.38 | 1,931.01 | 182.53 |

Decode excludes the entire first SSE token burst. Prompt tokens / TTFT includes tokenization and first-token handoff. For unique cold context probes this estimates effective prefill throughput. Sampled code-agent history probes can reuse prefixes; their ratios are not prefill-speed measurements. Fixed 256-output probes do not demonstrate natural completion.

## Tool-use quality: hard mode, Basic / Hard / Total

| Platform | Suite | Points | Score % | Pass/partial/fail |
|---|---|---|---|---|
| spark | Basic | —/— | — | —/—/— |
| spark | Hard | —/— | — | —/—/— |
| spark | Total | —/— | — | —/—/— |
| rtx | Basic | 116/138 | 84.06 | 51/14/4 |
| rtx | Hard | 31/38 | 81.58 | 14/3/2 |
| rtx | Total | 147/176 | 83.52 | 65/17/6 |

Thinking is enabled. The run uses temperature 0, eight parallel scenarios, one trial, at most eight turns and a 4,096-token benchmark response budget. This budget is not a server default. The source model documents thinking on/off, with no named reasoning-effort levels.

Pinned public suite: 69 Basic + 19 Hard = 88, partial credit 0/1/2. Missing entries are unmeasured, not zero. Total weights scenario counts. Parser/API compatibility checks below are a separate measure. [Reproduce the hard-mode suite](serving/benchmarks/tool_quality.md).

The [failure audit](benchmarks/development/rtx-tool-quality-audit/README.md) separates unfinished tasks, ineffective retries, overclarification and prompt-injection failures from brittle grader matches. The official **147/176** score is unchanged; no failures are excluded or manually rescored.

## Functional checks and sampled memory

| Platform | Reasoning/tools/JSON | Prefix hits/queries | Cold TTFT s | Warm TTFT s | Passed retrieval probes | MM examples |
|---|---|---|---|---|---|---|
| rtx | 80/80 | 3520.0/3612.0 | 9.75 | 0.06 | 6 | 2 |

Prefix counters establish reuse. Cold/warm latency includes first-use overhead and is not an isolated cache-speedup measurement.

RTX: minimum sampled host MemAvailable 163.82GiB; GPU0 peak 94.92GiB, GPU1 peak 94.14GiB.

Sampled peaks can miss brief excursions; Spark memory is shared with the host. Quality misses/truncations and raw traces remain in the reports.

## Release evidence

- RTX: [report](benchmarks/releases/rtx-20260925-v2/report.json), [lossless evidence manifest](benchmarks/releases/rtx-20260925-v2/archive-manifest.json); SHA256 `bd29e1e31b8d0adf2a0b04654429118795bcfcd72815ace1d74516c4f8f6de50`.

## Third-party sources

The quantization and kernel sources are pinned as Git submodules:

| Path | Purpose | Branch |
| --- | --- | --- |
| `third_party/GPTQModel` | Quantization | `main` |
| `third_party/sparkinfer-glmrt` | GPU kernels and serving integration | `dots3-tp2-strided-mla` |

Clone with `git clone --recurse-submodules https://github.com/tpurtell/dots-note-sm12x-2x.git`, or run `git submodule update --init --recursive` after an ordinary clone. The parent repository records exact commits. Make library changes inside its submodule and push them before committing the updated pointer here. To advance to tracked branches, run `git submodule update --remote --merge`, review and test, then commit the pointer updates.
