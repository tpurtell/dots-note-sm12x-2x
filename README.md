# Dots3 Note Preview: EXL3 K4 on two DGX Sparks

This repository is building one Dots3 Note Preview checkpoint and a two-Spark vLLM recipe. The checkpoint will retain the [FP8 source](https://huggingface.co/dots-studio/dots3-note-prev-fp8) outside the routed language-model experts. Every routed expert `gate_proj`, `up_proj`, and `down_proj` weight in layers 1–45 will come from the [BF16 source](https://huggingface.co/dots-studio/dots3-note-prev) and use uniform EXL3 K4. Vision experts remain as supplied by the FP8 checkpoint. The target publication is [`wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1`](https://huggingface.co/wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1).

**Status (2026-09-23):** the distributed quantization run is active on rhea and moa. The quantized checkpoint and serving measurements are pending; no throughput values below are inferred from another model. The RTX recipe begins after the Spark recipe is accepted.

## Source and calibration

| Input | Resolved revision | Use |
| --- | --- | --- |
| `dots-studio/dots3-note-prev-fp8` | `7c14222e22423d6df6848eb0d1c5c3a88a00311a` | Preserved core and multimodal tensors |
| `dots-studio/dots3-note-prev` | `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b` | BF16 routed expert weights |

The [source audit](quantization/inspect_sources.py) checks the 46-layer, 256-expert geometry, all 34,560 routed weights and their FP8 scales, and the exact BF16/FP8 source dtypes. The initial local inventory measured **517.27 GiB** of BF16 routed weights, **264.15 GiB** of FP8 routed weights, and **13.94 GiB** of non-expert FP8 checkpoint weights. These are source tensor bytes, not an EXL3 artifact size or serving memory measurement.

The [calibration builder](quantization/build_calibration_corpus.py) adapts GLMRT's source-disjoint selection to Dots3 Note's non-thinking chat template. Its renderer was checked against the source `chat_template.jinja`. The first generated selection contains 1,437 calibration prompts / 1,081,453 prompt tokens, 89 held-out prompts / 67,466 tokens, and 146 screening prompts / 110,279 tokens. The manifest binds the tokenizer hash, builder revision, source groups, and split hashes. Calibration and held-out source groups must remain disjoint.

Both Sparks have 121 GiB unified memory and local NVMe. The source checkpoint exceeds one Spark's memory, so GPTQModel streams layers through an indexed hybrid source. The BF16 checkpoint interleaves layers across all 131 model shards; file-level layer partitioning does not reduce transfer volume. The current run replays calibration prompts on rhea and distributes K4 expert projection work across rhea and moa. The cross-host K4 kernel and checkpoint-reuse path passed a live projection test. Layer 1's rolling activation boundary passed a full shard-hash audit, and a live restart resumed directly at layer 2. See the [quantization runbook](quantization/README.md) for the prompt-splitting tradeoff.

## Serving target

Start from [vLLM v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), the latest stable release checked on 2026-09-23. Native Dots3 Note support entered through [#51255](https://github.com/vllm-project/vllm/pull/51255), and the merged [Dots3 runtime optimization #53517](https://github.com/vllm-project/vllm/pull/53517) is present in the pinned source. The open [video-audio cache repair #57655](https://github.com/vllm-project/vllm/pull/57655) remains relevant if video with audio is qualified. The [two-node launch script](serving/start_spark_node.sh) targets two GB10 GPUs at tensor parallel size 2 and approximately **85% GPU memory utilization**, subject to a measured unified-memory budget and successful startup.

Qualification must prove prefix-cache hits on repeated long prompts, including Dots3's sliding-window and sparse-MLA cache groups. Record cached-token counters and warm versus cold TTFT; review vLLM's [prefix caching contract](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/) and sparse-MLA alignment rules. Exercise xgrammar JSON and tool calls with and without speculation. The earlier Qwen recipe carried a termination fix related to [#52805](https://github.com/vllm-project/vllm/pull/52805), with a [reported follow-up](https://github.com/vllm-project/vllm/issues/53181); carry a patch only if the pinned release still needs it.

Use B12x where its exact Dots3 geometry and semantics qualify: EXL3 MoE, dense projections, MLA/DSA or SWA attention, and vocabulary projection. Record correctness and graph replay before performance comparisons. Dots3's padded DSA cache and sliding-window MLA require separate checks from existing GLM sparse-MLA paths.

## Headline measurements

The tables use the workload families from the [Qwen3.8 Flash Next recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next). A dash means the workload has not been measured on this checkpoint.

The available GB10 results are component checks, not whole-model throughput:

| Component check | Measured result |
| --- | --- |
| Real K4 top-8 expert fixture versus GPTQModel ExLlamaV3 | Minimum cosine 0.99999940; relative L2 0.000905 |
| Real TP=2 BF16 vocabulary shard, one-token GPU median | B12x 3.06 ms on each shard; PyTorch 3.18 / 3.14 ms |
| Real FP8 core `q_a_proj`, exact-scale B12x versus dequantized-FP8 reference | BF16 output relative L2 0.0000114; CUDA graph replay difference 0 |

The [serving component probes](serving/README.md) describe their input sizes and limits. No entry above represents an end-to-end Dots3 serving rate.

| Measurement | 2× DGX Spark | 2× RTX PRO 6000 |
| --- | ---: | ---: |
| C1 seven-workload weighted decode, tokens/s | — | — |
| C1 greedy `merge_intervals` median, tokens/s | — | — |
| C1 sampled async coding task median, tokens/s | — | — |
| C16 sampled-prose aggregate median, tokens/s | — | — |
| Full-context boundary, input + output tokens | — | — |
| API tool constraints / retrieval probes | — | — |

### Seven content workloads: C1

| Workload | Spark decode tokens/s | Contract | RTX decode tokens/s | Contract |
| --- | ---: | ---: | ---: | ---: |
| Code | — | — | — | — |
| Math | — | — | — | — |
| Fable | — | — | — | — |
| Hello | — | — | — | — |
| Topic | — | — | — | — |
| Structured JSON | — | — | — | — |
| Multilingual | — | — | — | — |

### Sampled prose: independent clients

| Clients | Spark aggregate tokens/s | Minimum overlap | RTX aggregate tokens/s | Minimum overlap |
| ---: | ---: | ---: | ---: | ---: |
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 4 | — | — | — | — |
| 8 | — | — | — | — |
| 16 | — | — | — | — |

### Prefill and context scaling: C1

| Prompt tokens | Spark prefill tokens/s | Spark TTFT, s | Spark decode tokens/s | RTX prefill tokens/s | RTX TTFT, s | RTX decode tokens/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | — | — | — | — | — | — |
| 8,192 | — | — | — | — | — | — |
| 32,768 | — | — | — | — | — | — |
| 65,536 | — | — | — | — | — | — |
| 131,072 | — | — | — | — | — | — |
| Maximum qualified context | — | — | — | — | — | — |

### Functional checks

| Check | 2× DGX Spark | 2× RTX PRO 6000 |
| --- | --- | --- |
| Repeated-prefix cached tokens and TTFT | — | — |
| xgrammar JSON and tool constraints | — | — |
| Text, image, and audio requests | — | — |
| CUDA graph replay and stable workspace | — | — |

Raw responses, timing samples, exact image and source revisions, GPU mode, memory snapshots, and quantization error evidence belong beside each accepted table entry.

## Third-party sources

The quantization and kernel sources are pinned as Git submodules:

| Path | Purpose | Branch |
| --- | --- | --- |
| `third_party/GPTQModel` | Quantization | `main` |
| `third_party/sparkinfer-glmrt` | GPU kernels and serving integration | `master` |

Clone with `git clone --recurse-submodules https://github.com/tpurtell/dots-note-sm12x-2x.git`, or run `git submodule update --init --recursive` after an ordinary clone. The parent repository records exact commits. Make library changes inside its submodule and push them before committing the updated pointer here. To advance to tracked branches, run `git submodule update --remote --merge`, review and test, then commit the pointer updates.
