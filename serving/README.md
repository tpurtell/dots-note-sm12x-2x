# Two-Spark vLLM integration (in progress)

`Dockerfile.spark` pins the multi-architecture vLLM v0.30.0 image, vendors the
Apache-2.0 EXL3 adapter source, installs the Dots3 FP8-core/EXL3-expert
configuration, and includes the pinned B12x submodule. It builds on ARM64 and
imports the native Dots3 model and hybrid quantization config on moa.

```bash
docker build -f serving/Dockerfile.spark -t dots3-vllm-spark:dev .
```

After publishing and installing the accepted checkpoint into each Spark's
Hugging Face cache, build the same image on both hosts. Set `MODEL_REVISION` to
the Hub commit, start `start_spark_node.sh` on moa, then on rhea. The script
uses the two verified 100 Gb/s RoCE interfaces at `10.55.1.5/6`, vLLM's
multi-node multiprocessing executor, TP=2, explicit prefix caching, xgrammar,
and a starting GPU memory utilization target of 0.85. It refuses to start
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

After the full service starts, run `qualify_prefix_xgrammar.py` on an otherwise
idle endpoint. It records cold and repeated-prompt TTFT, prefix-cache query
and hit counter deltas, and a constrained JSON response from xgrammar. Its
result is accepted only when the repeated request records cache hits and the
JSON response conforms to the requested schema.

Full-model loading, block-FP8 core parity, padded DSA attention, prefix-cache
hits, xgrammar requests, full-model CUDA graph replay, and the 85% memory target still
require the completed checkpoint.
The vLLM base currently ships Torch 2.13 and CuTe DSL 4.7.1 while the pinned
B12x package declares CuTe DSL 4.6.2. The small GB10 parity and graph tests
above pass with this combination; full-model qualification is pending.
