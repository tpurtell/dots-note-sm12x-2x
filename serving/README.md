# RTX and two-Spark serving development

For published containers, use the [release fast path](release/README.md#public-fast-path-enabled-after-final-qualification).
It pins separate native amd64/RTX and arm64/Spark image digests and each
platform's qualified settings. Publication and final qualification are still in
progress; pending settings intentionally refuse launch.

## Current platform decisions

RTX selects **MTP3**, native vocabulary projection, native collectives, and
0.95 GPU memory utilization for the C1–C4 reasoning/coding balance. Spark
selection is independent; current Spark candidates use **0.80** utilization
with the host-memory guard enabled and a 1 GiB minimum physical headroom.
Both final profiles require `REASONING_PARSER=dots3`, prefix caching, xgrammar,
and the separate `dots` tool-call parser. See the [qualification ledger](../docs/serving-progress.md)
and [optimization decisions](../docs/optimization-matrix.md) for measured
tradeoffs and remaining gates.

The [Brandon-derived GLM RTX recipe](https://github.com/tpurtell/glm-5.3-flash-ext3-4-bit-2x-rtx)
informs RTX B12x optimization; the [Qwen Spark recipe](https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next)
informs Spark integration and measurement layout. Each B12x path must qualify
on Dots3 before becoming a platform default.

## Native development builds

Both Dockerfiles pin vLLM v0.30.0, include the hybrid FP8-core/EXL3-expert
adapter and pinned B12x source, and install the Dots-aware reasoning parser.
For a reproducible checkout, select the parent recipe commit recorded in the
accepted report, then restore its submodule commits rather than advancing their
branches:

```bash
git checkout RECIPE_COMMIT_FROM_REPORT
git submodule update --init --recursive
git submodule status
```

The current pins are GPTQModel `b903382057e4903b5629fcd49838ffc3ccce5f14`
for quantization and B12x `c963d8f7c98792a026eaf81a96a72d01b4aa0047` for new
native builds on both serving platforms. Published RTX `20260925-v1` retains
its original B12x `c5e23d830c3d1e76be56a5df290d13e30bc66702`; use that image
digest and recorded source revision to reproduce v1. GPTQModel is not copied into the serving image. The model is
already quantized; building a serving image does not repeat quantization.
B12x is copied from the checked-out submodule into `/opt/b12x`. The source
checkout must be clean for those commit IDs to describe the copied code.

Both Dockerfiles default to the same immutable multi-architecture vLLM base:
`vllm/vllm-openai@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
Use the recorded default rather than substituting a mutable vLLM tag. The RTX
build targets `sm_120a`; Spark targets `sm_121a`. Build on the matching native
CPU/GPU platform:

```bash
# On the RTX host:
docker build --platform linux/amd64 -f serving/Dockerfile.rtx -t dots3-vllm-rtx:dev .
# On each Spark:
docker build --platform linux/arm64 -f serving/Dockerfile.spark -t dots3-vllm-spark:dev .
```

Each Dockerfile copies the repository's hybrid quantization, attention,
multimodal/cache, optional dense/EP adapters and `dots3` reasoning parser, then
applies `port_vllm.py`, `port_reasoning.py`, and `port_compact_cache.py`.
The compact DSA cache integration is opt-in: native builds default to
`DOTS3_COMPACT_DSA_CACHE=0`. The separate indexer workspace candidate defaults
to `DOTS3_INDEXER_PREFILL_CONTEXTS=0`, preserving upstream workspace sizing.
Its compact-cache candidate reduces DSA records to 576 bytes and
bounds the sliding-window cache pool; enabling it requires its own native
context, prefix, retrieval and memory qualification. The published v1 image
does not acquire these changes from a source checkout update. RTX additionally applies
`port_pcie.py`; Spark applies `port_roce.py` and builds its native verbs proxy.
Optional integrations are enabled only by the selected runtime profile.
Both builds install hash-pinned SoundFile 0.13.1 with dependencies left intact.
The shared EXL3 format adapter is vendored at `serving/vendor/exl3.py`.

These parent builds contain native source and integration patches. Producing
the distributed release container additionally requires warming the selected
profile, exporting its actual compiled artifacts, and building the
[release wrapper](release/README.md#build-sequence). Its manifest records exact
source/dependency fingerprints and parent image ID. A new build is not assumed
to reproduce an already published image byte for byte; use its registry digest
when reproducing that exact image.

The accepted checkpoint is already published and installed. Mount the entire
Hugging Face cache with `HF_HOME`; leave `MODEL_DIR` unset. `MODEL_DIR` remains
a development-only option for auditing a staged export. Start
`start_spark_node.sh` on moa before rhea, using the same native image on both.
The verified development addresses are `10.55.1.5/6` on the second RoCE link;
the launch uses vLLM's multi-node multiprocessing executor with TP=2. RTX uses
`start_rtx.sh` for TP=2 on one host.

Raw development launchers deliberately preserve older defaults: RTX utilization
0.90, Spark 0.82, 32,768-token context, 512 batched tokens, 16 sequences,
parser off, and no MTP unless explicitly requested. They are **not** the chosen
release profiles. For current candidate qualification, explicitly set
`GPU_MEMORY_UTILIZATION=0.95` (RTX) or `0.80` (Spark) and
`REASONING_PARSER=dots3`, with the exact context, MTP and B12x options under
comparison. The release runner supplies all stored profile values itself.
Do not copy the RTX MTP choice onto Spark without its comparison results.

This is an integration image, not a qualified serving release. The Dots3
adapter packs uniform per-expert EXL3 K4 tensors into B12x's native fused-MoE
owner after vLLM's tensor-parallel slicing. It prepares a capacity-bound B12x
plan before CUDA graph capture and shares reusable scratch across decoder
layers. The generic vendored EXL3 adapter remains only as a loader and format
validator for this model.

A four-expert fixture extracted from the layer-1 production checkpoints was
run on a GB10 through the image's B12x K4 path. For four BF16 tokens with two
routes each, comparison against GPTQModel's ExLlamaV3 projection kernel gave
minimum output cosine 0.99999899 and relative L2 error 0.000946. A second
fixture used eight experts and the model's real top-8 routing width: minimum
output cosine 0.99999940, maximum absolute error 0.00352, and relative L2
error 0.000905. This is a small fused-MoE
parity test, not end-to-end model parity or a throughput claim. The exact
hardware commands are `smoke_b12x_exl3.py` and `reference_exl3_k4.py`.
The Dots3 vLLM adapter loaded the same per-expert tensors, returned the same
BF16 result as the direct B12x test, and replayed a four-token CUDA graph with
zero observed difference from eager execution for both route widths. The real
rank-3 Trellis tensor also passed the patched vLLM per-expert loader path.
The top-8 adapter additionally reused one eight-token plan for one-token and
four-token live calls; both matched the direct BF16 output exactly.
Before full-model startup, the same preparation path also completed with the
launch capacity of 2,048 tokens and all 256 routed experts on GB10. That
bounded geometry smoke replicated eight real layer-1 K4 experts to fill the
256 slots, ran four live top-8 tokens, and returned finite outputs. Its
[completion receipt](../benchmarks/component/capacity2048-full-geometry.log)
is retained with the component evidence.

The real TP=2 BF16 `lm_head` shards were checked with `smoke_vocab.py` on
GB10. Both shards selected the B12x native single-token kernel, kept the
same top-20 token order as PyTorch, and replayed CUDA graphs with zero
observed difference. The vLLM vocabulary method matched direct B12x output
exactly. Isolated median GPU times were 3.06 ms for B12x on each shard,
versus 3.18 and 3.14 ms for PyTorch. When enabled, this method applies only to
single-token decode; other shapes use vLLM's ordinary BF16 projection. These
component timings are not full-model decode measurements.

After the full service starts, run `qualify_prefix_xgrammar.py` on an otherwise
idle endpoint. It records cold and repeated-prompt TTFT, prefix-cache query
and hit counter deltas, a constrained JSON response from xgrammar, and a
forced Dots-format function call with integer arguments. Its
result is accepted only when the repeated request records cache hits and the
JSON and tool responses conform to their requested schemas.

The `benchmarks/` scripts adapt the same seven content contracts, independent
client timing, exact-length prefill, and context scaling used in the adjacent
Qwen recipe. The legacy content and prose chat workloads set
`enable_thinking=False` for direct-answer measurements. The separate
`coding_clients.py` workload enables thinking and drives the current C1–C4
selection; use `--output-tokens 8192` to reproduce that comparison budget.
Run them from a client while the server is otherwise idle and
retain the JSON/JSONL files under this project's `.cache/bench/` until the
measurement is accepted. `context.py` and `prefill.py` start at 2K and 8K;
pass longer depths only after the matching service context limit is qualified.

```bash
python3 serving/benchmarks/coding_clients.py --base-url http://rhea:8000/v1 --output-tokens 8192 --output .cache/bench/spark-coding.jsonl
python3 serving/benchmarks/reasoning_api.py --base-url http://rhea:8000 --repeats 2 --require-mtp --require-boundary-chunk --output .cache/bench/spark-reasoning.jsonl
python3 serving/benchmarks/workloads.py --suite seven --base-url http://rhea:8000 --output .cache/bench/spark-seven.jsonl
python3 serving/benchmarks/clients.py --base-url http://rhea:8000/v1 --output .cache/bench/spark-clients.json
python3 serving/benchmarks/prefill.py --base-url http://rhea:8000/v1 --output .cache/bench/spark-prefill.json
python3 serving/benchmarks/context.py --base-url http://rhea:8000 --output .cache/bench/spark-context.jsonl
python3 serving/benchmarks/retrieval.py --base-url http://rhea:8000 --output .cache/bench/spark-retrieval.jsonl
python3 serving/benchmarks/tools.py --base-url http://rhea:8000 --output .cache/bench/spark-tools.jsonl
python3 serving/benchmarks/multimodal.py --base-url http://rhea:8000 --output .cache/bench/spark-multimodal.json
```

The scripts keep per-token SSE timestamps and server usage for their reported
decode rates. The seven-workload summary divides all timed decode tokens by
all timed decode seconds. The client script requires independently overlapping
requests for C1, C2, C4, C8, and C16. No model performance number is accepted
from the standalone component smokes.

Run `capture_runtime.py dots3-vllm-head --output /path/to/rhea-runtime.json` on
rhea and the same command with `dots3-vllm-worker` on moa. It keeps image,
launch flags, selected startup lines, Docker memory use, system memory, and
GPU inventory alongside the benchmark results. Use both receipts to check the
0.80 Spark candidate allocation against actual unified-memory availability.
Keep the host guard active throughout qualification. Use the same capture
script with `dots3-vllm-rtx` on the RTX host.

Full-model development loading, text/image/audio, prefix-cache hits, xgrammar
and tool checks have passed on both platforms. Final container qualification
must repeat them with the selected parser, MTP, context and memory settings.
Component checks below preserve earlier investigation results and do not
replace those release gates. Exact installed dependencies and source hashes
are recorded by the release cache export rather than inferred from package
metadata requirements.

`smoke_block_fp8.py` exercised two B12x FP8 paths on the real layer-0
`q_a_proj` weight and scale. The weight-only MXFP8 path re-quantizes arbitrary
FP32 checkpoint scales into UE8M0 at runtime and showed about 3.3% relative L2
difference from a dequantized-source reference for one BF16 token. The exact
serialized block-FP8 path kept the checkpoint weight and scales unchanged;
its output matched an explicit dequantized-FP8 reference in BF16 and the
complete activation-quantization-plus-GEMM CUDA graph matched eager output.
With vLLM's dynamic FP8 activation quantizer included in both timed paths,
the exact B12x route took 82.7 microseconds median versus 64.4 microseconds
for vLLM's selected DeepGEMM path on this one-token projection. Its relative
L2 to the original BF16-input/FP8-weight reference was 2.58% versus 3.95%
for DeepGEMM, which re-quantizes the checkpoint scales. These are one-shape
component results. The initial launch keeps native FP8 for dense projections
because the tested exact route is slower. vLLM v0.30.0 also contains a B12x
block-FP8 wrapper, but it calls an older plan-free API and needs adaptation
before it can use this pinned B12x version.
