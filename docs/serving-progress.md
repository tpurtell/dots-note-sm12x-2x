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

After saving the baseline and source commit `65ef447`, RTX was deliberately
restarted for a 262,144-token capacity candidate at utilization 0.94, retaining
512 prefill tokens and 16 slots. Container `dots3-vllm-rtx` is running startup
(image includes shared scratch and the graph/tool fixes). This larger-context
candidate is not yet qualified. Do not restart it solely on an observation
timeout. Sparks still await the user's explicit ready signal after reboot.

## Larger-context RTX and PCIe comparison

The 262K candidate passed exactly 261,888 input + 256 output tokens. Startup
reported 5.41 GiB KV memory and 310,070 request-equivalent tokens. The single
boundary run measured 80.57s TTFT and 77.05 decode tokens/s. Temporary CUDA
allocation retries occurred during prefill, but the request completed. Receipts:
`context-262k-boundary.jsonl`, `startup-262k.log`, and
`runtime-262k-boundary.json` in `.cache/serving/rtx/`.

A three-run seven-workload control at this capacity passed all contracts and
measured 86.30 weighted decode tokens/s (`seven-262k-native-ar.jsonl`). This is
the control for the optional B12x PCIe adapter adapted from the Brandon recipe
to the current planned B12x API. It prepares BF16 5120-channel row counts 1–32
before graph capture and retains native fallbacks outside that admission set.

The isolated PCIe test passed exact changed-input parity against NCCL for
1/2/4/8/16/32 rows. It also compares native vLLM graphs: B12x was similar at
small rows and slower at 16/32, so no default change is justified yet. Raw
receipt: `pcie-check.log`. The full model is starting with the optional B12x
adapter for the matched end-to-end comparison; that candidate is unqualified.

The matched PCIe model run completed all content contracts: 86.60 weighted
tokens/s versus 86.30 native (about +0.35%). Given the small C1 difference and
slower 16/32-row components, native remains the default while other candidates
are investigated. The B12x path stays an explicit evaluation option, not a
release optimization claim. Receipts: `seven-262k-b12x-ar.jsonl` and
`runtime-pcie-candidate.json`.

The checkpoint does include native MTP: layer 46 contains FP8 projection and
dense MLP weights plus the predictor norms, and `model.mtp.embed_tokens.weight`
is present. vLLM supplies `num_nextn_predict_layers=1` and selects sliding
attention for the prediction layer. The hybrid adapter previously recognized
only `dots3_note`; it now also retains the FP8 core for `dots3_note_mtp`.
The configuration probe covers this conversion. An MTP1 candidate is being
prepared with native communication; actual loading and execution are pending.

MTP configuration checks passed. RTX container `dots3-vllm-rtx` is now
starting image `dots3-vllm-rtx:mtp-dev` with MTP1, 131,072 context, memory
utilization 0.94, 512 prefill tokens, and native communication. Poll this
container before making further lifecycle changes. Sparks await user readiness.

## Spark recovery and initial RTX MTP1 evidence

The user confirmed both Sparks are ready after reboot. Both were responsive
with about 115 GiB available and zero swap use. Fresh native images were built
on both hosts, and the adapter/gather hashes match the local source. Their
configuration probes pass, including MTP FP8 handling. Both now run the corrected
TP2 target-only startup at 0.85 utilization, 512 prefill tokens and 32K context.
Host-side memory monitors sample once a second and stop this recipe's container
after three samples below 8 GiB available RAM. Monitor logs live in each remote
recipe's `.cache/serving/spark/memory-watch.log`; verify the process and log
before trusting the guard. These monitors do not replace memory qualification.

RTX MTP1 loaded successfully (79.6 GiB model memory/rank), reports 5.0 GiB KV
memory and 283,447 equivalent tokens, and returned a correct first chat response.
Three seven-workload runs passed all content contracts at 136.92 weighted
tokens/s. Eight tool checks and prefix/JSON/forced-tool checks pass with MTP1;
the latter recorded 3,520 cached-token hits. Native no-spec control used 262K
configured context while this MTP candidate uses 128K, so the final tuning
comparison must align settings. MTP1 concurrency testing is running.

MTP1 concurrency completed at C1/2/4/8/16 with actual overlap matching all five
levels; its single-run C16 aggregate was 739.23 tokens/s. Image/audio content
and normal completion contracts passed. The candidate's runtime and cumulative
speculation counters were saved (`runtime-mtp1-qualified-initial.json`,
`mtp1-metrics-after-checks.txt`). MTP2 is now starting on RTX with the same 128K,
0.94 utilization, 512 prefill and native-communication settings. Spark loading
continues with both memory monitors live and substantial available RAM.
