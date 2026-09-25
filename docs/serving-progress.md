# Serving qualification ledger

## Objective

Qualify and publish independent 2× DGX Spark (ARM64/SM121) and 2× RTX PRO
6000 (AMD64/SM120) recipes in parallel. Both consume the existing uniform-K4
checkpoint at `d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da`.

References: `../brandon-glm-5.3-flash/recipe/README.md` for RTX B12x integration;
`../rtx6k-exl3-qwen3.8-flash-next/README.md` for Spark choices and measurement
conventions. Evaluate all applicable B12x features against Dots3 geometry.

## Baseline revalidation: 2026-09-25

- GitHub releases API still reports vLLM v0.30.0 as latest stable, published
  2026-09-22. Docker registry manifest matches the existing pinned digest.
- Base manifest: `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`.
- ARM64: `sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56`.
- AMD64: `sha256:5f5e535216848d0c52159c8c13a0af04be5f6fe1a84e79914300610796f76d40`.
- Both RTX GPUs idle; both Sparks have no running GPU processes or containers.
- Moa has the prepared Spark image; transfer to Rhea uses 10.55.1.5.
- Native RTX image build started from the same pinned multiarchitecture base.
- Dots3 PR review: #57655 remains open (video/audio processor-cache miss
  handling), #56304 remains open (video/audio decode bounds). Review these
  against tested multimodal scope. #53517 runtime optimization is in baseline.

## Required acceptance evidence

Each item requires separate Spark and RTX evidence; a component check does
not qualify whole-model serving.

- [ ] Full model startup, correct generation, startup memory and KV capacity.
- [ ] B12x optimization decisions: MoE, routing, vocabulary, FP8 dense,
      sparse MLA/indexing, SWA, graph buffers, and applicable collectives.
- [ ] Parallelism, batching, cache/context, graph and speculation tuning.
- [ ] Text/image/audio, prefix-cache counters and cold/warm TTFT, xgrammar JSON,
      stream/nonstream tools, retrieval, context boundary and stability.
- [ ] Seven workloads with contracts; greedy and sampled coding; C1/2/4/8/16;
      prefill/TTFT/decode depth curves; memory and maximum-context receipts.
- [ ] Separate GHCR release images with compiled platform kernels, versioned
      tags, immutable digests, fresh pulls and release-image qualification.
- [ ] Tested container fast paths and source builds, startup/readiness/warmup,
      stop/restart/logs/API instructions, complete README comparison tables.

Local temporary work stays in `.cache/`. `/mnt/scratch` is read-only. Existing
HF files are reused; no model download or quantization is required.

## Initial startup findings

The first full startup attempts revealed two serving configuration compatibility
issues before weights were loaded. The pinned Transformers release rejects the
nested RoPE configuration when used through vLLM's DeepSeek-derived Dots3
configuration; vLLM also rejects generic heterogeneous MLA metadata. The serving
port now translates nested DSA/SWA RoPE and checks that HF per-layer overrides
exactly match Dots3's native `swa_*` dimensions before consuming the native
representation. Checkpoint files remain unchanged. `serving/check_config.py`
checks the actual model configuration, roundtrip, routing and invalid inputs.

Native image builds and startup retries are in progress. There are no full-model
performance results yet. Local build logs and failed startup traces are under
`.cache/serving/rtx`; Spark logs are under `/home/tj/dots-note-work/serving`.

Both native images pass the real-config and GPTQModel tensor metadata probes.
The adapter now reads the existing `quantize_config.json` and checks all 34,560
K4 records. Initial RTX worker construction selects FlashAttnMLASparse, which
rejects FP8 KV. A BF16-cache baseline retry isolates model loading while the
FP8 sparse-attention integration remains an explicit optimization task. This
retry does not qualify the final cache format or maximum context.

## Padded sparse attention: TP2 component qualification

The explicit BF16 retry also fails: SM12x selection offers only the FP8
FlashInfer sparse backend, while Dots3 then replaces it with a Hopper-only
padded FlashAttention implementation. This needs an attention integration,
not another cache flag. Spark logs independently confirm both RoCE interfaces
are used by NCCL (`NET/IB/0` and `NET/IB/1`).

B12x already provides `attention.sparse_mla.strided` for Dots3's exact 1088-byte
physical rows, 576 QK dimensions and 512 value dimensions, but restricted its
caps to TP8. The implementation now accepts TP2 (64 local query heads) as well.
The existing five numerical/graph/addressing cases run at both TP sizes:

- RTX SM120: **11 passed**, 8.82 seconds.
- Moa SM121: **11 passed**, 9.72 seconds.
- Cases cover padding isolation, masked selections, request-relative remapping,
  layer-interleaved pages, graph replay with changing inputs and no allocation,
  and physical offsets beyond signed 32-bit addressing.
- Receipts: `.cache/serving/{rtx,spark}/strided-tp2.xml`; raw logs in local RTX
  directory and Moa's `/home/tj/dots-note-work/serving/strided-tp2.log`.

This qualifies the TP2 component extension only. Connecting it to vLLM's Dots3
prefill/decode lifecycle, quantization scales and cache allocation is next.
The services are currently stopped after failed baseline initialization.

## vLLM sparse adapter and audio configuration

`serving/dots3_b12x_attention.py` connects the padded ordinary-E4M3 cache to
B12x's TP2 strided path. It consumes native causal top-k indices for prefill
and decode and shares capacity-planned scratch between sparse layers. The
adapter's numerical oracle and changed-query CUDA graph replay passed on
both RTX and Moa (`serving/check_sparse_adapter.py`). Full-service graph
warmup, latency, quality and capacity remain unqualified.

With this backend, RTX startup passed attention construction and reached
checkpoint loading. The first remaining shape error was in the audio path:
vLLM's legacy audio config ignored the newer HF adapter dimensions and built
2048-wide output instead of 5120. The port now maps the explicit HF audio
encoder and adapter dimensions into its native Whisper-shaped config. Both
platform images have been rebuilt for the next full-model loading attempt.

## Exact multimodal source restoration

The first audio dimension translation exposed a second mismatch (10240 vs
5120 FC1 width). Reading the original cached FP8 source configuration showed
that normalization also omitted SwiGLU, vision `patch_merger`, and
`pre_pixel_shuffle=True`. The partial audio translation has been replaced with
exact source audio/vision sections in `serving/source_multimodal_config.json`.
The processor accepts these sections only when the canonical export-config
SHA256 matches the pinned checkpoint; model files are unchanged.

The configuration probe now checks audio dimensions and SwiGLU plus vision
adapter type, shuffle semantics, and output width. RTX has completed all 19
shard loads and post-load preparation: rank 0 reports 78.81 GiB model memory,
rank 1 reports 78.87 GiB, about 62.5 seconds. Encoder profiling is running;
these numbers are startup observations, not final serving memory or throughput.

RTX's initial full-model profile (2048 prefill tokens, default multimodal limits)
ran out of memory after model preparation. A retry uses 512 prefill tokens and
explicit image=1/audio=1/video=0 limits. Video is outside the requested
text/image/audio qualification scope. The previous processor warmup error was
traced to a combined native-video + separate-image/audio dummy request, which
Dots3 explicitly rejects. Spark's original loading attempt is still live and
must be polled before any restart.

## SWA gather integration

RTX's 512-token retry passed memory profiling far enough to exercise real
sliding-window attention. Its native cache-gather CUDA op supports widths
320/512/576 only and rejects Dots3's 1088. `dots3_cache_gather.py` now provides
that supporting gather/dequantization operation with 64-bit page arithmetic.
`check_cache_gather.py` passed exact-reference and graph-replay checks on RTX
and Moa, including ragged boundaries, interleaved layers and offsets >2 GiB.
RTX has restarted with this integration; Spark's replacement image is building
while its prior profile run remains live.

Both Sparks completed full model preparation (Moa about 258 seconds; Rhea
about 387 seconds). Memory pressure and swap during loading require further
work before publishing startup expectations. No serving endpoint has yet been
qualified and no end-to-end performance result is available.

## RTX profiling and first live request

With 256 prefill tokens and utilization 0.95, RTX reached HTTP health 200.
The unshared-preparation baseline reported 3.66 GiB KV memory, 199,308
request-equivalent tokens, and 6.08x concurrency at the configured 32,768
context. These are development observations, not qualified release defaults.
Receipt: `.cache/serving/rtx/profile-256-unshared.log`.

The first live request failed a breakable-CUDA-graph argument-address check:
capture retained `input_ids` alongside `inputs_embeds`, whereas live execution
passes only embeddings. The port now matches the live input contract during
dummy input preparation, preserving raw tokens for models that require them.

Preparation also retained a separate MoE scratch allocation per routed layer.
It now uses the same capacity/spec-keyed scratch arena as serial live layer
execution. At the same 512-token/0.90 settings, available KV memory improved
from -5.08 GiB to +2.74 GiB per GPU. Startup reported 80.44 GiB consumed,
2.29 GiB peak activation headroom, and a 144,715-token equivalent KV pool.
The first live graph-backed chat completed with the correct answer to 2+2.
Receipts: `.cache/serving/rtx/profile-512-shared.log` and
`.cache/serving/rtx/first-chat-shared.json`. Broader qualification is pending.

Both Sparks currently time out before the SSH banner on RDMA and LAN addresses.
This does not establish that their running profiling jobs have exited. No
replacement Spark job has been launched; host-console status was requested.

The user will reboot the Sparks and signal readiness; until then, work stays
on RTX. The corrected shared adapter must be deployed to both Sparks before
any restart, with a reduced initial prefill capacity and measured headroom.

RTX's first image/audio probes identified two cats and correctly transcribed
the nursery-rhyme sample. The image answer reached its 128-token cap, so a
completed-response qualification remains necessary. A unique repeated prompt
recorded 2,304 cached-token hits and TTFT 0.478s cold / 0.095s warm. Constrained
JSON returned exactly `{"answer":42}`. These initial probes are not final
benchmark measurements; raw receipts are in `.cache/serving/rtx/`.

Named tool choice exposed a v0.30.0 parser mismatch: xgrammar emits the JSON
arguments, while Dots' `supports_required_and_named=False` sends them to its
XML extractor. The port enables the standard named/required JSON parsing;
automatic tool calls retain Dots XML parsing. Verification is pending.

The parser fix passed all eight required/named/auto/none × streaming/nonstreaming
cases in `tools-json-parser.jsonl`. The combined prefix/JSON/forced-tool probe
also passed: 3,584 cached-token hits, cold/warm TTFT 0.751s/0.042s, exact
JSON `{"answer":42}`, and `add_numbers(a=2,b=3)`. Text, image and audio completed
with correct content in the follow-up multimodal contract probe; image output
now uses a concise request and both modalities finish normally. All receipts
remain development evidence in `.cache/serving/rtx/`, pending final-image and
broader workload qualification. A one-pass seven-workload quality screen is
running on the same image. No release throughput claim is made yet.

The seven content contracts all passed. Initial client runs completed C1/2/4/8/16
without engine errors (one measured run per point, saved in
`clients-initial.json`). Six retrieval probes passed at approximately 8K and
24K prompt tokens, with needles at 10%, 50%, and 90%. The 32K configured
boundary passed with exactly 32,512 input + 256 output tokens; the model's
native 524,288-token context remains a
separate capacity and qualification target. Final repeated measurements and
optimization comparisons remain outstanding.

The initial client screen verified actual overlapping stream counts of
1/2/4/8/16, matching each requested concurrency. RTX remains running at
TP2, FP8 KV, 512 prefill tokens, 32K context, 16 slots and memory utilization
0.90. The working baseline is ready for larger-context and B12x tuning.
