# Serving qualification ledger

## Objective

Qualify and publish independent 2× DGX Spark (ARM64/SM121) and 2× RTX PRO
6000 (AMD64/SM120) recipes in parallel. Both consume the existing uniform-K4
checkpoint at `d8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da`.

References: `../brandon-glm-5.3-flash/recipe/README.md` for RTX B12x integration;
`../rtx6k-exl3-qwen3.8-flash-next/README.md` for Spark choices and measurement
conventions. Evaluate all applicable B12x features against Dots3 geometry.

## Current development: hybrid layer owners and expert TP2

The next release targets native 524,288 context with compact DSA cache storage,
four indexer prefill contexts, and owner-local non-expert layers. Every routed
expert remains TP2 across both GPUs. The initial 23/23 split is provisional;
owner-only vision/audio placement and measured KV budgets will guide an uneven
split independently for RTX and Spark. See [execution and memory details](hybrid-expert-tp.md).

The matched RTX 524K screen measured native TP2 at 184.68 / 135.08 / 101.63
per-request decode tokens/s versus the initial unpacked hybrid at 154.96 /
130.54 / 97.97 for C1/C2/C4. Both completed all 12 requests naturally and passed
the static response checks. These are small development screens; the published
RTX v1 results remain the release baseline. [Archived comparison](../benchmarks/development/rtx-native524-hybrid-comparison).

The packed RTX candidate combines routing inputs into one broadcast and has
passed loaded ownership checks. Whole-model measurements are in progress.
Spark's initial hybrid candidate passed 40 API cases and prefix/JSON/tool
checks at 524K, utilization 0.80 and a 1 GiB host reserve; its coding screen is
in progress. Neither hybrid candidate is yet a qualified release image.

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

A three-run seven-workload control at this capacity passed 19/21 contracts and
measured 86.30 weighted decode tokens/s (`seven-262k-native-ar.jsonl`). This is
the control for the optional B12x PCIe adapter adapted from the Brandon recipe
to the current planned B12x API. It prepares BF16 5120-channel row counts 1–32
before graph capture and retains native fallbacks outside that admission set.

The isolated PCIe test passed exact changed-input parity against NCCL for
1/2/4/8/16/32 rows. It also compares native vLLM graphs: B12x was similar at
small rows and slower at 16/32, so no default change is justified yet. Raw
receipt: `pcie-check.log`. The full model is starting with the optional B12x
adapter for the matched end-to-end comparison; that candidate is unqualified.

The matched PCIe model run passed 19/21 content contracts: 86.60 weighted
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
Three seven-workload runs passed 16/21 content contracts at 136.92 weighted
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

## Content-contract audit correction

The first single-pass screen passed 7/7. Later three-run summaries were
overstated from selected live log rows: native and B12x PCIe each passed
19/21, MTP1 16/21, and MTP2 18/21. Misses concern fable length/moral and the
paging-topic requirement. Raw responses are unchanged; the derived
`content-contract-audit.json` binds counts/issues to source-file SHA256 hashes.
The benchmark now prints aggregate pass counts and every failed contract in
its final summary. Performance results must always be accompanied by these
quality counts. MTP2 measured 170.53 weighted tokens/s; no MTP release default
is selected. Tools, prefix/JSON and multimodal checks are independent receipts.

## Current candidates

MTP2 passed all eight tool cases, prefix/JSON/forced-tool checks, image/audio
contracts, and C1/2/4/8/16 client execution. Its preliminary C16 aggregate is
776.21 tokens/s; content quality remains 18/21 on the three-run seven-workload
screen. Runtime and counters are saved in `runtime-mtp2-initial.json` and
`mtp2-metrics.txt`. RTX is now starting MTP3 at the same 128K/0.94/512 settings.

The corrected Spark pair reached HTTP health 200 at utilization 0.85. Its
reported KV allocation was 15.24 GiB, but physical available RAM on Rhea was
only about 8.3 GiB before real-request workspace allocation. Both containers
were deliberately stopped after saving `target-85-startup.log`, container
inspection and `memory-watch-85.log` on each host. No first request was sent
at that tight budget. The pair is now restarting at 0.82 utilization with the
same shared-scratch fix and memory monitors. The Spark launch default follows
this provisional 0.82 budget; final tuning still targets the user's approximate
85% goal subject to real unified-memory headroom.

The Spark launcher now accepts NODE_RANK, HOST_IP, MASTER_ADDR and SOCKET_IFNAME
for other two-Spark deployments while retaining the verified Rhea/Moa defaults.

## Parallel qualification after Spark recovery

The Spark 0.82 candidate reached health and answered its first real request
correctly. Physical available memory was approximately 9.7 GiB on Rhea and
10.9 GiB on Moa with both host memory guards active and zero swap. Functional
qualification is running; these are startup observations, not a final memory
capacity claim.

RTX MTP3 completed the repeated seven-workload screen at 183.02 weighted
decode tokens/s with 17/21 content contracts. Eight tool cases, prefix caching
(3,520 cached tokens), constrained JSON, forced tool arguments, and both
image/audio contracts passed. Cold/warm prefix TTFT was 0.766/0.057 seconds.
The one-run C1/2/4/8/16 screening rates were 136.68/212.21/317.06/497.56/723.82
aggregate tokens/s. This did not improve on MTP2 concurrency consistently.
Evidence: `seven-128k-mtp3.jsonl`, `tools-mtp3.jsonl`, `prefix-mtp3.json`,
`multimodal-mtp3.json`, `clients-mtp3.json`, and `runtime-mtp3-initial.json`.
MTP4 is now starting with the same 128K/0.94/512 settings; no final speculation
default has been selected.

MTP4 completed the seven-workload screen at 179.83 weighted tokens/s with
19/21 content contracts. C1/2/4/8/16 screening rates were
129.38/192.08/277.83/425.83/640.74 tokens/s. The reference sampled async coding
task at task-only depth measured a 240.23 tokens/s median. Its fixed 256-token
output measures speed, not generated-code correctness. The Dots3 coding-depth
harness now uses vLLM's supported `/tokenize` chat interface.

Spark target-only at 0.82 passed prefix/xgrammar, all eight tools, image/audio,
and completed the repeated seven screen at 24.38 weighted tokens/s with 16/21
content contracts. Rhea's available-memory minimum was about 8.5 GiB.
Concurrency was deliberately deferred at this allocation. The next candidate
uses 0.80 with MTP1 and native vocabulary projection to preserve headroom.

Actual RTX graph FP8 probes preserve original FP32 weight scales. Q-B improved
at rows 4/16/64/512 but lost slightly at row 1; output projection has no broadly
useful gain. These component results do not establish a whole-model benefit.
See `fp8-qb-graph.json` and `fp8-o-graph.json`. The optional exact path also
differs from native DeepGemm scale rounding and must pass model qualification.

## Matched control and exact FP8 candidate

The 128K native no-speculation control completed at 86.25 weighted decode
tokens/s with 19/21 contracts, confirming the earlier 262K control's rate.
Its sampled coding-task median was 85.67 tokens/s. The audio-dependency image
passed image/audio checks. Receipts: `seven-128k-native-control.jsonl`,
`code-agent-native-task.jsonl`, `multimodal-audio-native.json`,
`runtime-native-128k.json`.

The optional exact-FP8 Q-B adapter is disabled by default. Real DSA/SWA TP2
shards passed source immutability, changed-input graph/eager equality, exact
M4 reference and native M1 fallback checks. The probe uses a real single-rank
NCCL context with explicitly sliced TP2 weights; distributed loader validation
comes from the pending whole-model launch. Source scales remain FP32, whereas
the native path rounds them for DeepGemm. This may change model output and
requires quality qualification. Whole-model MTP3 plus this candidate is now
starting at 128K/0.94/512; it is not a selected release default.

Both launchers now persist B12x compile objects inside the project runtime
cache. Earlier containers only persisted vLLM/Triton caches, so removing them
lost B12x compile artifacts and added repeated startup work. Spark's current
MTP1 container predates this launcher change; its artifacts will be copied
before the next removal.

## Exact Q-B whole-model result and scheduler tuning

The opt-in Q-B/MTP3 image loaded successfully and completed both workload and
concurrency screens. Weighted decode was 187.29 tokens/s with 16/21 contracts;
C1/2/4/8/16 rates were 137.24/202.26/311.81/500.23/679.93. Native Q-B/MTP3 was
183.02 with 17/21 contracts and 723.82 at C16. Extra originals raised model
allocation to 80.02–80.06 GiB/rank and reduced available KV to 4.62 GiB. The
small C1 aggregate difference does not justify the quality/capacity/concurrency
tradeoff; native FP8 remains the default. This opt-in experiment is retained
for reproducibility, with receipts `seven-128k-mtp3-exact-qb.jsonl`,
`clients-mtp3-exact-qb.json`, and `runtime-mtp3-exact-qb.json`.

RTX next tests MTP2/native FP8 at 1,024-token prefill chunks, 128K context and
0.94 utilization to measure prefill/latency and capacity against 512-token
chunks. Spark MTP1 at 0.80 has passed prefix/xgrammar, eight tools and image/audio
with about 11–13 GiB physical memory remaining after modalities; repeated
workloads are running.

## Prefill tradeoff, shared preparation and 262K MTP2

RTX MTP2 at 1,024-token chunks achieved about 5,024 input tokens/s median
at 32K, but C1/2/4/8/16 decode was only
120.98/199.70/294.47/447.97/654.88 aggregate tokens/s. KV capacity fell to
2.61 GiB (145,902 equivalent tokens) at 128K/0.94. Larger chunks are not the
balanced default. Receipts: `prefill-mtp2-batch1024.json`,
`clients-mtp2-batch1024.json`, `runtime-mtp2-batch1024.json`.

The MoE preparation code now shares dummy inputs/routes/outputs by static
geometry, in addition to shared scratch. This removes duplicate retained
primers; whole-model validation is running. With 512-token chunks and MTP2,
the 262K/0.94 startup explicitly rejected insufficient KV (4.15 GiB available
versus 4.57 required); the container exited cleanly. The 0.95 RTX candidate
started with 5.10 GiB and 292,303 equivalent tokens. This RTX allocation does
not apply to unified-memory Sparks. 32K prefill median was 4,371 tokens/s;
the seven screen was 169.92 weighted decode tokens/s with 18/21 contracts and
eight tool cases passed. Boundary, prefix, modalities and concurrency checks
are running. The different configured context limits prevent attributing
these KV differences solely to primer sharing.

Spark MTP1 at 0.80 completed C1/2/4/8/16 at
31.52/46.59/74.02/107.18/161.83 aggregate tokens/s; actual overlap matched
each level. Minimum available physical memory was 11.196 GiB on Rhea and
12.343 GiB on Moa. Its seven screen was 33.79 with 17/21 contracts. Both
containers stopped cleanly after receipts and compile-cache preservation.
MTP2 is now loading with the same allocation/context/chunks/native vocabulary.

The public Spark launcher now starts a host memory monitor automatically.
The monitor binds to the original container ID, tolerates inspection timeouts
and retries failed stop attempts. It remains a best-effort monitor, not a
hard memory limit. Current development jobs continue with their already
verified manual monitors; final fast-path qualification must verify the
automatically launched monitor.

RTX shared-primer MTP2/262K/0.95 passed the exact261,888+256 boundary
(81.79s TTFT,188.11 decode tokens/s), prefix3,520 hits with cold/warm
0.803/0.057s, constrained JSON, image/audio and all concurrency levels.
C1/2/4/8/16 single-screen aggregate was
143.62/213.11/319.16/510.48/715.77. Native allocator retry warnings occurred
during long prefill, but the request completed and subsequent clients passed.
Receipts: `context-mtp2-shared-boundary.jsonl`, `prefix-mtp2-shared.json`,
`multimodal-mtp2-shared.json`, `clients-mtp2-shared.json`,
`runtime-mtp2-shared-qualified-initial.json`.

An EP2 component probe passed with four real whole experts, arbitrary local
placement, remote-only and repeated routes, and changed-input graphs. Summed
rank-local partials matched the unmapped oracle with relativeL2 <=1.91e-8.
This emulates ranks on one GPU and does not prove distributed loading/reduction.
The optional whole-model EP2 adapter is now building for that qualification;
it remains disabled unless `--enable-expert-parallel` is supplied.

The clients benchmark now retains request payloads and raw SSE events in
addition to timing samples; its timing convention is unchanged. Existing
development client files predate this extra response evidence. Final release
qualification uses the enhanced recorder.

## EP2 and Spark RoCE investigations

EP2 loaded all language experts and completed the 128K MTP2 workload/client
screens:165.83 weighted tokens/s,18/21 content contracts and
129.98/209.41/346.46/479.69/746.03 aggregate tokens/s at C1/2/4/8/16. Its
262K/0.95 startup rejected only3.09GiB of available KV versus4.57GiB required;
the same TP2 profile had5.10GiB. EP2's128K KV was3.79GiB. The capacity loss
and lower C1 rate do not justify selecting it from these preliminary screens.
The2.01GiB profile difference remains unisolated; neither extra communicator
memory nor scratch growth has been established as its cause. Receipts:
`ep2-startup-attempt1.log`, `seven-128k-mtp2-ep2.jsonl`,
`clients-mtp2-ep2.json`, `runtime-mtp2-ep2.json`.

The strengthened top8 component probe used16 real whole experts with arbitrary
placement and global IDs up to255. Mixed, remote-only, repeated and skewed
routes, changed inputs/router weights and graph replay passed; maximum partial
sum relativeL2 was6.84e-8. See `ep2-top8-gpu-check.json`. This remains an
experimental option rather than the release profile.

Spark MTP2 completed37.80 weighted tokens/s with18/21 contracts, all functional
checks and C1/2/4/8/16 at30.76/45.65/69.23/110.30/150.74 aggregate tokens/s.
The measured minimum host availability was10.682/11.771GiB. After saving
receipts both serving containers stopped for isolated collective testing.

The first Spark RoCE probe found a prepared-API dtype conversion bug before
collective execution. B12x now converts its normalized torch dtype back to the
launcher's required name; regression tests cover three dtypes. The fix is
pushed as `c5e23d830c3d1e76be56a5df290d13e30bc66702`. Both ranks then passed
BF16 exact eager/graph results, changed-input replay, six shapes, alternating
graphs and ordered streams. Graph microtimings favored B12x, but the unusually
large native graph costs do not establish whole-model gains. Receipts and
overlay provenance are under `.cache/serving/spark/roce/`. A serving adapter
is being prepared for a matched comparison; native NCCL remains current.

Spark now builds the current native source (audio dependency, shared primers,
optional disabled EP/exact FP8 and the RoCE library fix) for native MTP3
qualification. RTX is comparing MTP2/MTP3 with identical per-request nonce
seeds and three C1/C16 samples at262K, reducing input variation in the final
speculation decision.

## RTX speculation selection

The matched262K/0.95/512 comparison used identical payloads across all51
measured requests, verified directly from preserved raw receipts. MTP2 median
C1/C16 sampled-prose rates were138.99/774.39 tokens/s versus
130.58/726.89 for MTP3. MTP3's separate seven workload mix was181.37 with
17/21 contracts versus the MTP2 shared-primer169.92 with18/21. The balanced
RTX release candidate therefore uses MTP2, retaining native FP8, native
collectives and TP2. MTP3 remains a documented workload-dependent option.
Evidence: `clients-mtp2-matched.json`, `clients-mtp3-matched.json`,
`seven-262k-mtp3-shared.jsonl`, `runtime-mtp3-matched.json`.

The Spark RoCE library fix and regression tests are now pinned in the parent.
Current Spark candidate-v2 includes the audio dependency and shared primers,
and is testing MTP3 with automatically launched host monitors. The new RoCE
serving adapter is still being prepared; no whole-model communication gain
has been claimed or selected.

## User preference: agentic coding and reasoning at C1–C4

The final default must prioritize coding/reasoning latency and useful throughput at C1, C2 and C4. Prose C16 and the seven-workload aggregate remain reporting data, not the main selection criterion. The preliminary RTX MTP2 choice is therefore provisional pending matched concurrent coding tests with actual thinking enabled. Existing reference coding-depth tests explicitly disable thinking and cannot establish this new preference.

## Reasoning-enabled coding baseline and parser preparation

RTX MTP2 completed the new natural-stop coding/reasoning workload at 262K / 0.95 / 512-token chunks, using the release wrapper with native collectives. Three measured runs per C1/C2/C4 each contain four debugging tasks; the output budget is 8,192 tokens and thinking is enabled. All 36 measured requests finished naturally with no HTTP or token-accounting errors. Median per-request decode rates: 166.76 / 130.56 / 96.43 tokens/s; completed-answer latency: 23.61 / 35.59 / 40.29 seconds. These rates include reasoning. Median output lengths differ (3,836 / 4,870.5 / 3,850 tokens), so latency alone cannot select a candidate. The source-identical MTP3 comparison is running; Spark's native MTP3 control is also running.

Current static checks pass 35/36 RTX requests. One C4 retry answer omitted the requested final JSON. A separate validator ambiguity was repaired without changing prompts: `successful_reservations` may be a count of one or a list containing one request ID. The comparison report preserves the originally reported checks and the recomputed checks, with validator hashes. These checks do not execute generated code or establish behavioral correctness.

The original 4,096-token screening budget truncated the async task inside reasoning; its partial receipts are retained. Under 8,192 tokens one RTX warmup was truncated, but all measured requests completed. Raw baseline: `.cache/serving/rtx/coding-mtp2-8192.jsonl`; audited report: `.cache/serving/rtx/coding-comparison/mtp2-report.json`. Final release measurements remain pending.

The RTX wrapper started from a new empty runtime cache and passed prefix caching, eight tool modes, image/audio, and the repeated seven-workload suite (170.63 weighted tokens/s, 18/21 contracts). It has not been published. A newer parent image now contains the optional Dots3 reasoning parser. Its CPU evidence covers nonthinking, prior turns, tool returns, grammar start after thinking, speculative validation rollback and xgrammar reset. The actual API gate must still pass on both platforms before this parser is enabled in release profiles.

The latest stable vLLM was rechecked through the official releases API on 2026-09-25: v0.30.0, published 2026-09-22. The pinned baseline therefore remains current.

The 2026-09-25 Dots3 PR search is preserved in `benchmarks/development/dots3-pr-review-20260925.jsonl`. The video/audio processor-cache repair #57655 is still open; video remains outside this recipe’s qualified modalities. Newer hits include processor naming cleanup and experimental media preprocessing, which do not justify replacing the stable runtime during qualification. No additional PR was applied solely from its search title.

## Matched RTX coding result: MTP3 leads MTP2 at C1–C4

Both complete candidates use the same container image, 262K context, 0.95 utilization, 512-token chunks, 16 slots, native collectives, and exactly matched measured request payloads. Three runs of four reasoning-enabled debugging tasks at each concurrency finished naturally: 36/36 per candidate, zero HTTP or token-accounting errors. Recomputed static checks pass 35/36 for each. Evidence and validator audit are preserved in [the comparison](../benchmarks/development/rtx-coding/mtp2-vs-mtp3.json), with [raw receipt hashes](../benchmarks/development/rtx-coding/manifest.json).

| Clients | MTP2 median per-request decode tokens/s | MTP3 median per-request decode tokens/s | MTP2 / MTP3 completed-answer latency, s |
| --- | ---: | ---: | ---: |
| 1 | 166.76 | 181.33 | 23.61 / 24.00 |
| 2 | 130.56 | 141.38 | 35.59 / 26.98 |
| 4 | 96.43 | 102.53 | 40.29 / 43.40 |

Responses have different lengths; compare token rate alongside latency and output lengths, not latency alone. All measured requests visibly finished reasoning. Concurrent waves reach their requested overlap, then naturally drain as tasks finish. MTP3 is the current RTX preference for the requested C1–C4 coding balance; a matched one-run MTP4 screen is running before final choice. The earlier MTP2 preference from sampled prose/C16 no longer drives the default. Dynamic depth is not needed to reconcile these fixed-K curves because MTP3 leads at all three coding levels.


## Completed RTX MTP4 extension: retain MTP3 for C1–C4

The completed one-run MTP4 screen was extended with runs 1 and 2, preserving its
14 warmup/measured wave objects exactly. The screen's raw hash matches the
combined receipt's resume-source hash. Lossless raw streams, screen, runtime and
container logs are archived in the [coding evidence manifest](../benchmarks/development/rtx-coding/manifest.json);
the [three-candidate comparison](../benchmarks/development/rtx-coding/mtp2-vs-mtp3-vs-mtp4.json)
verifies matched payloads and includes validator hashes and original/recomputed
static checks.

| Clients | MTP3 median decode tokens/s | MTP4 median decode tokens/s | MTP3 / MTP4 natural completions |
| --- | ---: | ---: | ---: |
| 1 | 181.33 | 185.29 | 12/12 / 12/12 |
| 2 | 141.38 | 140.07 | 12/12 / 11/12 |
| 4 | 102.53 | 103.07 | 12/12 / 11/12 |

Select **RTX MTP3** for balanced reasoning/coding at C1–C4. MTP4 provides a small
C1 rate gain and nearly equal C2/C4 rates, with 34/36 natural completions versus
MTP3's 36/36. Its two `async_pool` requests at C2/run1 and C4/run1 reached the
8192-token limit before finishing reasoning. This is a limited, three-run sample;
it does not establish that speculative depth caused an answer-quality change.
Answer lengths differ, and completed-only latency excludes those truncations,
so neither latency nor static checks alone ranks overall coding quality.

Final release-image qualification and README performance tables remain pending.
The next RTX checks use the selected MTP3 with the Dots-aware reasoning parser
and evaluate the optional vocabulary optimization against its native control.


## RTX vocabulary shared-plan component qualification

The optional vocabulary path initially retained the temporary MTP head through
its preparation session. Sharing the weight-independent prepared plan avoids a
second persistent head. On both RTX shards, GPU checks now verify distinct live
output storage, exact eager/graph equality after changing inputs and weights,
and release of **778,567,680 bytes per rank** when the obsolete head is replaced.
BF16-native cosine is at least 0.99999994 and top20 indices agree. B12x median
latencies are 490.50/489.36 µs versus native 507.28/507.78 µs—a small component
benefit, not a full-model speed claim. [Raw receipts and command/source/image provenance](../benchmarks/component/vocab/manifest.json)
are preserved. Recovery of 262K model startup capacity and matched whole-model
vocabulary A/B remain pending.


## RTX vocabulary screen: retain native projection; capacity recovered

The original vocabulary-enabled image (`7055b0…`) rejected 262K startup at 0.95:
it exposed 4.30 GiB KV versus the required 4.57 GiB. The shared-plan fix in image
`5ecf76…` restores **5.03 GiB KV / 288,320 cache tokens**, matching the native
control without raising utilization. The failed startup log, accepted runtime
receipts, raw streams, API checks and comparison are preserved losslessly with
hashes in [the vocabulary evidence manifest](../benchmarks/development/rtx-vocab/manifest.json).

Both accepted candidates use MTP3 and passed all 80 reasoning API checks. The
matched one-run coding screen has four measured tasks each at C1 and C4:

| Clients | Native vocabulary decode tokens/s | B12x vocabulary decode tokens/s | B12x relative rate change |
| --- | ---: | ---: | ---: |
| 1 | 180.405 | 183.263 | +1.58% |
| 4 | 110.825 | 106.821 | −3.61% |

Retain **native vocabulary (`DOTS3_B12X_VOCAB=0`) with MTP3** for the final RTX
recipe: this screen shows no clear balanced whole-model speed benefit. It is a
limited sample and the images differ by the vocabulary lifetime fix. Native
completed 7/8 measured responses naturally, versus B12x 8/8; native's remaining
`async_pool` response exhausted 8192 tokens during reasoning. Output lengths
differ, so these completion and latency observations do not establish causal
quality or speed changes. The MTP3 truncation also confirms that this output
budget can truncate other samples; the earlier MTP4 observations do not identify
an intrinsic MTP4 defect. Final release measurements remain pending.

## RTX wrapper publication and initial registry verification

The native RTX wrapper `ghcr.io/tpurtell/dots3-note-exl3-k4-rtx:20260925-v1`
was pushed at digest
`sha256:2aff95d9896b3f3d3f8e3f4cfbaed90c77344d41d58fe7322eff9f4c73ec63b6`.
Authenticated digest pull passed. At the initial publication check, its package
was private and anonymous pull returned `unauthorized`; public access needs
its own successful verification after visibility is changed.

The prepublication wrapper started with an empty runtime cache and passed
prefix/xgrammar, image/audio, and 80/80 reasoning API cases. Its single-run
seven-workload screen passed 6/7 contracts (fable word count failed). Lossless
raw receipts, registry logs and exact image/source/cache provenance are in the
[wrapper archive](../benchmarks/development/rtx-release-wrapper/manifest.json).
Full qualification is now running separately against the published digest
with another fresh runtime cache. Release settings and final performance
tables remain pending.

## Completed RTX SWA Q-B priority screen

The finite C1/C4 comparison completed eight matched measured requests per
candidate with unchanged MTP3/262K/0.95 settings. Native median decode rates
were 180.436 and 104.648 tokens/s; the SWA Q-B exact-FP8 candidate measured
186.543 and 105.269 (+3.38% and +0.59%). Native finished 8/8 naturally versus
7/8 for the candidate; one candidate response reached the benchmark's
8,192-token output budget. This sampled completion difference does not establish
causal quality loss. The default remains native FP8: no clear balanced benefit
is established by this limited screen, extra weight copies cost about 264 MiB
per GPU, and preserving memory for native 524K context is the next priority.
[Lossless outputs, matched baseline subset, build/source/runtime provenance and
hashes](../benchmarks/development/rtx-swa-qb-priority/manifest.json) are archived.
The screen is complete and will not be extended.
