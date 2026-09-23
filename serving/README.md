# Two-Spark vLLM integration (in progress)

`Dockerfile.spark` pins the multi-architecture vLLM v0.30.0 image, vendors the
Apache-2.0 EXL3 adapter source, installs the Dots3 FP8-core/EXL3-expert
configuration, and includes the pinned B12x submodule. It builds on ARM64 and
imports the native Dots3 model and hybrid quantization config on moa.

```bash
docker build -f serving/Dockerfile.spark -t dots3-vllm-spark:dev .
```

After the strict export audit, copy the export to moa with `rdmasync` and set
`MODEL_DIR` to the audited export on both hosts for prepublication model
qualification. After publishing and installing the accepted checkpoint into
each Spark's Hugging Face cache, set `MODEL_REVISION` to the Hub commit and
unset `MODEL_DIR`. Build the same image on both hosts; start
`start_spark_node.sh` on moa, then on rhea. The script
uses the two verified 100 Gb/s RoCE interfaces at `10.55.1.5/6`, vLLM's
multi-node multiprocessing executor, TP=2, explicit prefix caching, xgrammar,
the native `dots` tool-call parser, and a starting GPU memory utilization
target of 0.85. It refuses to start
while the quantization containers are active. The initial 32K context and
2048 batched-token settings are qualification settings; increase them after
measured memory and prefix-cache checks.

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
versus 3.18 and 3.14 ms for PyTorch. The launch enables this method only for
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
Qwen recipe. Run them from a client while the server is otherwise idle and
retain the JSON/JSONL files under this project's `.cache/bench/` until the
measurement is accepted. `context.py` and `prefill.py` start at 2K and 8K;
pass longer depths only after the matching service context limit is qualified.

```bash
python3 serving/benchmarks/workloads.py --suite seven --base-url http://rhea:8000 --output .cache/bench/spark-seven.jsonl
python3 serving/benchmarks/clients.py --base-url http://rhea:8000/v1 --output .cache/bench/spark-clients.json
python3 serving/benchmarks/prefill.py --base-url http://rhea:8000/v1 --output .cache/bench/spark-prefill.json
python3 serving/benchmarks/context.py --base-url http://rhea:8000 --output .cache/bench/spark-context.jsonl
python3 serving/benchmarks/retrieval.py --base-url http://rhea:8000 --output .cache/bench/spark-retrieval.jsonl
python3 serving/benchmarks/tools.py --base-url http://rhea:8000 --output .cache/bench/spark-tools.jsonl
```

The scripts keep per-token SSE timestamps and server usage for their reported
decode rates. The seven-workload summary divides all timed decode tokens by
all timed decode seconds. The client script requires independently overlapping
requests for C1, C2, C4, C8, and C16. No model performance number is accepted
from the standalone component smokes.

Full-model loading, block-FP8 core parity, padded DSA attention, prefix-cache
hits, xgrammar requests, full-model CUDA graph replay, and the 85% memory target still
require the completed checkpoint.
The vLLM base currently ships Torch 2.13 and CuTe DSL 4.7.1 while the pinned
B12x package declares CuTe DSL 4.6.2. The small GB10 parity and graph tests
above pass with this combination; full-model qualification is pending.

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
