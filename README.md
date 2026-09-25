# Dots3 Note Preview: EXL3 K4 on two Sparks or two RTX GPUs

This repository provides one Dots3 Note Preview checkpoint and is developing two-GPU vLLM recipes for DGX Spark and RTX PRO 6000. The completed checkpoint retains the [FP8 source](https://huggingface.co/dots-studio/dots3-note-prev-fp8) outside the routed language-model experts. Every routed expert `gate_proj`, `up_proj`, and `down_proj` weight in layers 1–45 comes from the [BF16 source](https://huggingface.co/dots-studio/dots3-note-prev) and uses uniform EXL3 K4. Vision experts remain as supplied by the FP8 checkpoint. The published checkpoint is [`wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1`](https://huggingface.co/wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1).

**Status (2026-09-25):** quantization, tensor audit, publication and local cache installation are complete. The checkpoint is pinned to `d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da`; its 36 artifact files total 163,552,088,967 bytes. Both serving tracks pass development checks for text, image/audio, prefix caching, constrained JSON and tool calls. **RTX selects MTP3 with native vocabulary projection for reasoning-enabled coding at C1–C4; Spark selects its settings independently.** RTX uses 0.95 GPU memory utilization; current Spark candidates use 0.80 with the host-memory guard. The Dots-aware `dots3` reasoning parser is required by both final release profiles. RTX published-image qualification is complete and its benchmark profile is accepted; the public container fast path has passed fresh-start and restart checks. Spark qualification and native GHCR publication remain in progress. See the [RTX release report](benchmarks/releases/rtx-20260925-v1/report.json) and [qualification ledger](docs/serving-progress.md).

## Container fast path

The public **amd64 RTX** container is available; the separate **arm64 Spark** container is under qualification. Each contains its platform kernels and runtime caches. Use the [container run instructions](serving/release/README.md#container-fast-path): pull the platform image, mount the entire existing `HF_HOME`, and start the pinned profile. Spark runs one container on each host, worker first. Model downloads are unnecessary when the published checkpoint is already installed.

The [release settings](serving/release/settings.json) qualify each platform independently. RTX is qualified and its public fast path is verified. Spark remains pending. The runner refuses incomplete platform settings. Development launchers have different defaults; see [serving development and qualification](serving/README.md).

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

The tables use the workload families from the [Qwen3.8 Flash Next recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next). A dash reserves an entry for final release-image qualification. Development measurements and their contract failures are recorded in the [qualification ledger](docs/serving-progress.md).

The following GB10 results are component checks, separate from the developing whole-model measurements:

| Component check | Measured result |
| --- | --- |
| Real K4 top-8 expert fixture versus GPTQModel ExLlamaV3 | Minimum cosine 0.99999940; relative L2 0.000905 |
| Real TP=2 BF16 vocabulary shard, one-token GPU median | B12x 3.06 ms on each shard; PyTorch 3.18 / 3.14 ms |
| Real FP8 core `q_a_proj`, exact-scale B12x versus dequantized-FP8 reference | BF16 output relative L2 0.0000114; CUDA graph replay difference 0 |

The [serving component probes](serving/README.md) describe their input sizes and limits; their raw vocabulary and FP8 samples are in [benchmarks/component](benchmarks/component). No entry above represents an end-to-end Dots3 serving rate.

| Measurement | 2× DGX Spark | 2× RTX PRO 6000 |
| --- | ---: | ---: |
| C1 seven-workload weighted decode, tokens/s | — | 181.48 (17/21 contracts) |
| C1 greedy `merge_intervals` median, tokens/s | — | 233.52 |
| C1 sampled async code, first-burst-excluded median tokens/s | — | 203.21 |
| C16 sampled-prose aggregate median, tokens/s | — | 693.71 |
| Full-context boundary, input + output tokens | — | 261,888 + 256 = 262,144 |
| API tool constraints / retrieval probes | — | 80/80 reasoning/tool/JSON cases; 6/6 retrieval |

RTX uses the published MTP3/dots3-parser image with native vocabulary and collectives, FP8 KV, 0.95 memory utilization, 262,144 context, 16 slots and 512 batched tokens. Measurements use three runs; coding medians pool 12 requests per concurrency. The seven-workload headline is weighted by timed decode duration. Quality checks pass 17/21 seven-workload contracts and 34/36 coding responses. The sampled async-code headline uses the depth-0 prompt and excludes the entire first speculative SSE token burst from decode tokens/time; the older `(output tokens − 1)` convention is retained separately in raw/report evidence. Prefill rates use actual prompt tokens divided by client TTFT, including first-token handoff.

See the [RTX release report](benchmarks/releases/rtx-20260925-v1/report.json), [lossless evidence hashes](benchmarks/releases/rtx-20260925-v1/archive-manifest.json), and [public pull/start/restart checks](benchmarks/releases/rtx-20260925-v1/fastpath/report.json) for exact image/source identities, output lengths, memory observations and deployment evidence.

### Reasoning-enabled coding: C1–C4

Four debugging tasks use natural stopping and an 8,192-token output budget, including reasoning. Report completed-answer latency alongside token rates, output lengths, truncation and static response checks. Static checks do not establish behavioral code correctness. Identical request payloads are required for MTP comparisons; select defaults separately on each platform.

| Clients | Spark per-request decode tokens/s | Spark completed-answer latency, s | RTX per-request decode tokens/s | RTX completed-answer latency, s |
| ---: | ---: | ---: | ---: | ---: |
| 1 | — | — | 180.44 | 19.72 |
| 2 | — | — | 140.87 | 23.74 |
| 4 | — | — | 102.81 | 35.78 |

RTX natural completions and static checks: **34/36**. One C2 and one C4 request hit this benchmark’s 8,192-token per-request output budget, including reasoning; completed-answer latency excludes those two requests. Decode rates include reasoning and all measured requests. This budget is a benchmark setting, not a server output limit; clients can request larger outputs within the available context. Output lengths include reasoning:

| Clients | RTX output tokens, median (min–max) | Natural completions |
| ---: | ---: | ---: |
| 1 | 3,602.0 (2,434–7,311) | 12/12 |
| 2 | 3,317.5 (1,968–8,192) | 11/12 |
| 4 | 3,710.0 (2,230–8,192) | 11/12 |

### Seven content workloads: C1

| Workload | Spark decode tokens/s | Contract | RTX decode tokens/s | Contract |
| --- | ---: | ---: | ---: | ---: |
| Code | — | — | 233.52 | 3/3 |
| Math | — | — | 225.36 | 3/3 |
| Fable | — | — | 129.50 | 1/3 |
| Hello | — | — | 189.80 | 3/3 |
| Topic | — | — | 157.25 | 1/3 |
| Structured JSON | — | — | 234.54 | 3/3 |
| Multilingual | — | — | 161.51 | 3/3 |

RTX seven-workload contracts pass **17/21**: two fables miss the word-count range and two topic responses omit paging.

### Sampled prose: independent clients

| Clients | Spark aggregate tokens/s | Minimum overlap | RTX aggregate tokens/s | Minimum overlap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | — | — | 124.30 | 1 |
| 2 | — | — | 195.64 | 2 |
| 4 | — | — | 311.94 | 4 |
| 8 | — | — | 455.96 | 8 |
| 16 | — | — | 693.71 | 16 |

### Prefill and context scaling: C1

| Prompt tokens | Spark prefill tokens/s | Spark TTFT, s | Spark decode tokens/s | RTX prefill tokens/s | RTX TTFT, s | RTX decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | — | — | — | 4219.33 | 0.485 | 225.23 |
| 8,192 | — | — | — | 4096.49 | 2.000 | 226.14 |
| 32,768 | — | — | — | 4021.17 | 8.149 | 223.47 |
| 65,536 | — | — | — | 3851.92 | 17.014 | 223.41 |
| 131,072 | — | — | — | 3523.21 | 37.202 | 222.77 |
| Maximum qualified context | — | — | — | 3014.80 | 86.867 | 221.69 |

### Functional checks

| Check | 2× DGX Spark | 2× RTX PRO 6000 |
| --- | --- | --- |
| Repeated-prefix cached tokens and TTFT | — | 3,520 hits / 3,612 queried; cold 9.713 s → warm 0.058 s |
| xgrammar JSON and tool constraints | — | 80/80 thinking/nonthinking, stream/nonstream API cases |
| Text, image, and audio requests | — | Passed text, source-example image and audio contracts |
| CUDA graphs | — | Full/piecewise capture and C1–C16 request checks passed; [component replay evidence](docs/optimization-matrix.md) |
| Sampled peak GPU memory / minimum host available | — | GPU 0: 93.17 GiB; GPU 1: 93.15 GiB; host available minimum 163.62 GiB |

The RTX maximum-context row uses 261,888 prompt tokens plus 256 output tokens. Cold/warm TTFT includes first-use runtime overhead; the cache counters establish prefix reuse, while the timing ratio is not an isolated cache speedup. GPU memory entries are sampled peaks and can miss brief higher allocations. RTX GPUs were limited to 400 W each.

Raw responses, timing samples, exact image and source revisions, GPU mode, memory snapshots, and quantization error evidence accompany each accepted table entry.

### Tool-use quality: Basic / Hard / Total

Final RTX and Spark recipes will run the pinned `tool-eval-bench` public suite with Hard Mode enabled: **69 Basic + 19 Hard = 88 scenarios**. Scores include partial credit (0/1/2 points per scenario); they are distinct from parser/API compatibility checks. Infrastructure exclusions prevent qualification and remain visible in raw evidence.

| Platform | Basic (69) | Hard (19) | Total (88) |
|---|---:|---:|---:|
| 2× RTX | Pending final-profile run | Pending | Pending |
| 2× Spark | Pending final-profile run | Pending | Pending |

[Reproduction and raw evidence format](serving/benchmarks/tool_quality.md). No tool-quality score is inferred from the existing coding or tool-parser checks.

## Third-party sources

The quantization and kernel sources are pinned as Git submodules:

| Path | Purpose | Branch |
| --- | --- | --- |
| `third_party/GPTQModel` | Quantization | `main` |
| `third_party/sparkinfer-glmrt` | GPU kernels and serving integration | `dots3-tp2-strided-mla` |

Clone with `git clone --recurse-submodules https://github.com/tpurtell/dots-note-sm12x-2x.git`, or run `git submodule update --init --recursive` after an ordinary clone. The parent repository records exact commits. Make library changes inside its submodule and push them before committing the updated pointer here. To advance to tracked branches, run `git submodule update --remote --merge`, review and test, then commit the pointer updates.
