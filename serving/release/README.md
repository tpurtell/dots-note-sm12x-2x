# Native release packaging

These tools prepare separate `linux/amd64` RTX and `linux/arm64` Spark release
images from the exact runtime images qualified on their native hosts. They do
not download model weights, publish images, or choose serving defaults.

## Spark host memory guard

The release profile pins `memory_guard: true` and at least 1 GiB host memory
headroom. Release start/restart ignore inherited disabling settings and refuse
when current physical memory is below that threshold. The launcher starts
`watch_spark_memory.py` on each host and requires a successful initial memory
sample before reporting success. The monitor stops its container after three
one-second samples below the threshold. Logs and PID are under
`RUNTIME_CACHE/memory-watch.{log,pid}`.

Release restart rearms a monitor immediately after Docker restart. If readiness
cannot be established within ten seconds, the container is killed. Start and
restart both require a host Python interpreter and working Docker CLI. The
monitor runs outside the container; it does not survive a host reboot. Start
the recipe again after reboot. GPU utilization is separate from available host
RAM on Spark.

## Reasoning parser contract

Both final profiles require explicit `reasoning_parser: "dots3"`. The release
runner sets `REASONING_PARSER=dots3`, rejects parser override flags, and requires
the image's verified parser label before start/restart. The wrapper build checks
the installed registry and exact recipe parser source hash using CPU-only AST
inspection. Live reasoning, tools, JSON and speculative boundary gates must
still pass on each final platform image before marking settings qualified.

Development launchers leave `REASONING_PARSER` unset by default to support older
parser-off images. Set `REASONING_PARSER=dots3` when qualifying the new parent and
wrapper. Building an older development wrapper requires the explicit
`--development-without-reasoning-parser` option; its parser label is `none`, and
the final release runner rejects it. This option does not establish support for
reasoning in the old image.

## Compact-cache candidate versus published v1

The current native Dockerfiles include the compact-cache port and B12x
`bf5677c69197499433314d61aad70f79e89c47c9`, with
`DOTS3_COMPACT_DSA_CACHE=0` and `DOTS3_INDEXER_PREFILL_CONTEXTS=0` by default.
Both leave the optional memory optimizations disabled. RTX `20260925-v1` remains immutable and
uses B12x `c5e23d830c3d1e76be56a5df290d13e30bc66702` and its qualified padded
cache layout. Compact-cache experiments need a separately qualified parent,
new warmed cache export and release wrapper; do not reuse v1's qualification
report or assume its compiled artifacts match the changed runtime sources.

## Build sequence

1. Build the native runtime image from the recorded recipe/submodule commits
   using the [native build instructions](../README.md#native-development-builds).
   Both Dockerfiles include the SoundFile dependency from `audio-requirements.txt`.
   The pinned wheels include
   libsndfile; `--no-deps` preserves the base NumPy/CFFI stack. Validate import and
   actual audio decoding on each platform.
2. Start and qualify the final recipe, including all intended modalities,
   concurrency levels, context lengths and speculative settings. Keep cache
   paths as defined by the existing launchers. Stop benchmark traffic before
   exporting so no compilation is changing the exported files.
3. Export on each native Docker host:

   ```bash
   python3 serving/release/cache_bundle.py export \
     --container dots3-vllm-rtx --platform rtx \
     --recipe-revision "$(git rev-parse HEAD)" \
     --evidence benchmarks/RELEASE_REPORT.json \
     --output .cache/release/rtx-seed
   python3 serving/release/build.py \
     --bundle .cache/release/rtx-seed --image QUALIFIED_RUNTIME_IMAGE \
     --tag ghcr.io/tpurtell/dots3-note-exl3-k4-rtx:VERSION
   ```

   For Spark use `--platform spark`, its native image, and a separate tag. Each
   Spark rank has different physical device cache identities. Export both ranks;
   the current exporter makes one bundle per container. A single Spark image
   seeded from one rank is valid but the other rank may compile B12x kernels.
   Merge both rank executable caches before building the Spark image:

   ```bash
   python3 serving/release/cache_bundle.py merge \
     --bundle .cache/release/rhea-seed \
     --add-b12x-from .cache/release/moa-seed \
     --output .cache/release/spark-seed
   ```

   This verifies matching runtime source/dependency identities and preserves
   rank provenance. B12x same-path collisions must have identical hashes. The
   primary bundle's vLLM/Triton cache is retained; secondary rank UUID-specific
   B12x executables are added. Copy Moa's exported bundle to Rhea beforehand.
4. Qualify the wrapper with a **new empty runtime cache** using the existing
   launcher and `IMAGE=... RUNTIME_CACHE=... REASONING_PARSER=dots3`. Do this for both Spark ranks.
   Inspect startup, run functional and performance gates, then publish approved
   versioned tags with `docker push`. Record the registry digest.
5. Pull each image by digest and repeat the release gates. Public access must be
   checked using an unauthenticated Docker configuration. Record that digest in `settings.json` together with the final profile and
   report hash; the public `run.sh` fast path reads this file.

The existing launchers mount the **entire** Hugging Face cache read-only; this is
necessary for this checkpoint's shared blob links. Set `HF_HOME` to that cache.
The image has no weights and offline mode remains enabled by the launchers.

## What the image includes

The payload includes actual B12x CuTe `.o` executables and metadata, Triton
cubins and compilation metadata, and vLLM compilation caches. Each file has a
SHA256 digest. The manifest records the exact parent Docker image ID, native
architecture, CUTE target, recipe revision, qualification evidence reference,
model revision, installed dependency versions, Python source tree hashes and
B12x physical device identities. Model data, CUDA driver caches and B12x tuning
selection caches are excluded.

At startup the entrypoint checks architecture, target, paths, dependency
versions, source hashes and all payload hashes. It atomically copies missing
files into the writable mounted runtime cache, preserving existing entries.
Triton group metadata contains absolute paths, so the container cache paths
must stay `/root/.cache/vllm-runtime/{triton,vllm}`. The B12x executable cache is
seeded under that same mount via `B12X_COMPILE_CACHE_DIR`. An existing stale
runtime cache is still governed by each library's own cache identity checks;
use a new directory for release qualification.

## Interpreting post-ready Triton warnings

vLLM 0.30 logs Triton's process-first `jit_post_compile_hook` as “JIT
compilation” even when Triton's compiler returns an existing disk-cached kernel.
The warning alone therefore does not establish a new compilation. In the RTX
wrapper check, all 72 cached files for the four warning kernel families
(including `_bmm_outer_product_kernel`) matched the parent seed hashes. The only
new cache paths were CUDA driver-cache files, which the seed intentionally
excludes. [CPU audit and exact image/source hashes](../../benchmarks/component/cache/rtx-wrapper-post-ready-audit.json)
preserve that evidence. First-use loading still has overhead; this observation
does not promise zero compilation across other shapes, libraries or hardware.
No monitor suppression or additional wrapper rebuild was needed for these warnings.

## Portability and remaining startup work

B12x intentionally keys its persistent executables by **physical GPU UUID**,
source fingerprint, toolchain, compile options and explicit kernel spec. Its
private offline compiler API still requires the target UUID; it is a way to
compile in a child without CUDA initialization, not a portable artifact import
API. This wrapper never rewrites those identities. A new user's same-model GPU
can therefore compile B12x kernels on its first run. New shapes/configurations
can also compile. CUDA graphs and live device allocations must be recreated.

The images include compiled native kernels but do **not** promise zero JIT on
arbitrary hardware. Architecture-wide B12x artifact reuse needs a supported,
validated library feature with compatibility checks and new-device numerical
qualification. Caches from a later source or dependency build must be exported
again; the release wrapper rejects dependency or source drift.

## Registry access: qualified RTX v2

RTX `20260925-v2` is **public**, with authenticated and anonymous digest pulls
verified. The active RTX settings pin this immutable native amd64 image:

```bash
docker pull --platform linux/amd64 \
  ghcr.io/tpurtell/dots3-note-exl3-k4-rtx@sha256:d350ceb8c9be1dce3851ab20fba4c586f1530bef0a65a7094305b4ee8d2df16e
```

The [completed v2 report](../../benchmarks/releases/rtx-20260925-v2/report.json)
records functional, performance and hard-mode tool-quality results, including
quality misses and the documented restart continuation. The public runner's
pull, fresh-cache start, health, status, logs, stop and restart all passed;
see the [lifecycle receipt](../../benchmarks/releases/rtx-20260925-v2/fastpath/receipt.json.gz)
and [artifact hashes](../../benchmarks/releases/rtx-20260925-v2/fastpath/manifest.json).
The lifecycle check reused local model weights and sent no additional benchmark
requests. [Publication evidence](../../benchmarks/development/rtx-native-v2-wrapper/README.md)
preserves the registry checks and prepublication wrapper gates.

RTX v1 remains an immutable historical 262,144-context recipe; its
[report](../../benchmarks/releases/rtx-20260925-v1/report.json) and
[deployment evidence](../../benchmarks/releases/rtx-20260925-v1/fastpath/report.json)
remain available. The current commands select v2 through qualified settings,
not a mutable registry tag.

The separate native ARM64 Spark wrapper has been published as
`ghcr.io/tpurtell/dots3-note-exl3-k4-spark:20260925-v1`, digest
`sha256:fbe12925a19f529a35ea036a1f0ed9db0ba6de77f7455a6401816ea54508dbad`.
The package is **public**: authenticated pulls on both hosts and an anonymous
exact-digest pull passed. See the [publication archive](../../benchmarks/development/spark-native-v1-wrapper/README.md).
Spark qualification is complete and its settings are active; see the [final report](../../benchmarks/releases/spark-20260925-v1/report.json).
Both Spark hosts must use this same ARM64 digest. Source-bound inherited stages
and the timeout-only retrieval continuation remain explicit in the report.
Public pull, fresh-cache start, health, status, logs, stop and coordinated restart
passed on both hosts. Both starts verified the actual allocator policy and host
guards; old guards exited without duplicates. See the [lifecycle receipt](../../benchmarks/releases/spark-20260925-v1/fastpath/receipt.json.gz)
and [artifact hashes](../../benchmarks/releases/spark-20260925-v1/fastpath/manifest.json).
No generation requests, model downloads or thermal changes were made by this check.

## Container fast path

RTX settings contain the v2 published digest, qualified profile and completed
report hash. Spark v1 settings likewise contain its public ARM64 digest, completed
report hash and independently selected profile. `run.sh` refuses incomplete
platform settings.
No development image tag is a release fallback. Keep this recipe checkout,
including its qualification reports and `serving/start_*.sh` launchers.

RTX v2 selects 524,288 context, 512 batched tokens, 16 sequence slots, MTP3,
FP8 KV and 0.95 GPU memory utilization. Its 17/29 layer ownership keeps routed
experts at TP2; vision/audio towers use owners 0/1. It enables compact DSA,
owner-aware KV grouping, packed/fused routing, bounded shared-expert overlap,
and exact SWA Q-B for rows 4/8/16. Vocabulary projection and boundary tables
retain their native implementations. The report and settings bind every choice.
Spark selects 0.80 utilization, a 1 GiB physical-memory guard, MTP3, batch 512,
17/29 ownership, B12x RoCE and native FP8 projections. Its post-warmup allocator
fraction and reclamation threshold are both 0.90; actual receipts bind both ranks.
Final wrapper KV capacity is 2,777,333 tokens, distinct from parent/diagnostic
allocations. Fresh lifecycle startup accounted for 2,722,237 tokens and restart
for 2,762,204, both above the 2,550,605 acceptance floor; exact capacity can vary
between startups without a profile change.
Bare development launcher defaults do not reproduce these profiles.

Run from the same recipe checkout revision on both Spark hosts, or from the
RTX checkout root. Each host needs Docker with NVIDIA GPU support, Bash,
Python 3 and `rg` on the host. RTX requires native x86-64; Spark requires native
ARM64. Do not substitute the RTX image on Spark or use architecture emulation.

For RTX:

```bash
export HF_HOME="$HOME/.cache/huggingface"
# Optional: persistent kernel/graph cache on a native Linux filesystem.
export RUNTIME_CACHE="$PWD/.cache/release/rtx-runtime"
bash serving/release/run.sh rtx pull
bash serving/release/run.sh rtx start
bash serving/release/run.sh rtx health
bash serving/release/run.sh rtx logs
```

For Spark, run `pull` on each native ARM64 host, then start the **worker first**.
If its published package requires authentication, log Docker in on both hosts
with an account that has access before pulling.
Set addresses and the network interface for your own hosts. The interface must
reach the other Spark; the existing launcher also exposes `/dev/infiniband`.
The tested hosts used `SOCKET_IFNAME=enP2p1s0f0np0`,
`B12X_ROCE_HCA=roceP2p1s0f0` and GID index `3`. Set the matching interface,
HCA and RoCE GID for your hosts explicitly; an empty HCA was not the tested
configuration. Preserve these exports for restart as well.

```bash
# On both hosts:
export HF_HOME="$HOME/.cache/huggingface"
export RUNTIME_CACHE="$PWD/.cache/release/spark-runtime"
export MASTER_ADDR=HEAD_NETWORK_IP
export SOCKET_IFNAME=RDMA_NETWORK_INTERFACE
export B12X_ROCE_HCA=RDMA_HCA
export B12X_ROCE_GID_INDEX=3
bash serving/release/run.sh spark pull

# On the worker:
export NODE_RANK=1 HOST_IP=WORKER_NETWORK_IP
bash serving/release/run.sh spark start

# Then on the head:
export NODE_RANK=0 HOST_IP=HEAD_NETWORK_IP
bash serving/release/run.sh spark start
bash serving/release/run.sh spark health
```

Use `status`, `logs`, `stop`, `restart` or `remove` instead of `start` for local
container management. `logs` follows output; Ctrl-C stops following. `stop`
retains the container; `restart` uses its original settings and verifies its
image, checkpoint, parser, memory/context limits, fixed MTP and active B12x
settings against the qualified profile. A same-image container with different
settings is rejected. To change settings, stop and remove the container, then start
again. On Spark, stop the head before the worker; restart the worker before the
head. `health` runs on the head only; a worker has no HTTP endpoint. HTTP health
alone does not establish functional request readiness. Startup is asynchronous:
inspect logs while weights load, then run `health` again once the API is ready.
A failed early health request does not stop startup. Keep the same per-host
network, `PORT`, `HF_HOME` and `RUNTIME_CACHE` exports for subsequent commands.

For a coordinated Spark restart, stop both ranks first:

```bash
# Head host (NODE_RANK=0):
bash serving/release/run.sh spark stop
# Worker host (NODE_RANK=1), after head stops:
bash serving/release/run.sh spark stop
bash serving/release/run.sh spark restart
# Head host, after worker restart:
bash serving/release/run.sh spark restart
bash serving/release/run.sh spark health
```

`remove` removes only the stopped local container, not the mounted model or
runtime cache. `start` requires that the old container has been removed;
`restart` reuses an existing container.

`HF_HOME` must contain the already installed checkpoint and is mounted in full,
read-only, with offline mode enabled. Set it to the directory containing `hub/`,
not to `hub/` itself or a model snapshot. Each Spark needs its own complete local
cache, including all blobs targeted by snapshot symlinks. A snapshot containing
only small symlinks is not sufficient unless their blob targets exist inside
the mounted cache. No command downloads model weights.
`RUNTIME_CACHE` must be writable; do not use this project's `/mnt/scratch` disk.
`PORT` can customize the HTTP port. Extra arguments after `--` are passed to
vLLM, for example `... rtx start -- --api-key YOUR_LOCAL_KEY`. Reasoning parser
override flags are rejected. Other extra flags may change qualified behavior; published performance applies to the stored profile.
Avoid putting secrets in shared shell history.

### API example

Run this on the RTX host, or use port `8000` on the qualified Spark head. The model name is the same on both platforms.

```bash
curl --fail-with-body --silent --show-error http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @- <<'JSON'
{"model":"dots3-note-exl3-k4","messages":[{"role":"user","content":"Identify and fix the bug in this Python function: def square(x): return x + x"}],"temperature":0,"max_tokens":2048,"chat_template_kwargs":{"enable_thinking":true}}
JSON
```

The response separates `choices[0].message.reasoning` from the final answer in
`choices[0].message.content`. Set `enable_thinking` to `false` for direct answers;
add `"stream":true` to use SSE. Output budgets include reasoning tokens. The
benchmark's 8,192-token budget is not a server output limit; larger requests
must still fit the model's available total context.

### Final settings contract

The JSON schema version is `1`. Each platform has its own `status`; set that
platform to `qualified` only after its published digest and complete report are
verified. RTX can qualify while Spark remains pending, or vice versa. The
legacy top-level status is used only when a platform status is absent. Each
platform stores its own:

- Native architecture and immutable `ghcr.io/...@sha256:...` image reference.
- Repository-relative qualification report path and SHA256 of its exact bytes.
- GPU memory utilization, context limit, sequence limit and prefill chunk size.
- FP8 KV format and platform-specific MTP token count (`0` explicitly disables MTP).
- `compact_dsa_cache`: opt-in compact sparse-attention cache layout. RTX v2
  uses `true` with its measured report; historical v1 uses `false`. Start/restart bind this setting to the captured
  `DOTS3_COMPACT_DSA_CACHE` environment. Older profiles default to `false`.
- Required `reasoning_parser: "dots3"`, verified against the image label.
- `hybrid_layer_partition`: empty for ordinary TP2, or two positive decoder
  layer counts totaling 46 for hybrid non-expert ownership with expert TP2.
  This is bound to the qualification report and explicitly set on launch;
  an inherited shell setting cannot change a qualified profile.
- `hybrid_packed_routing`: whether the hybrid owner broadcasts activation and
  routing metadata as one byte-preserving payload. Defaults to `false` and
  requires hybrid ownership; qualification must use the same transport setting.
- `indexer_prefill_contexts`: indexer gather workspace budget in full contexts
  (1–40). Older profiles use 40. A smaller budget requires a supporting image
  and qualification with the same captured `DOTS3_INDEXER_PREFILL_CONTEXTS` value.
- For Spark, `memory_guard: true` and `min_host_available_gib` of at least 1.
- Boolean B12x vocab and RTX PCIe settings.
- Spark RoCE enable/eager booleans and explicit admitted row counts (disabled by default).
- Explicit exact-FP8 projection selector string (empty disables) and row counts.

The runner checks the report hash and binds its completed platform, image,
checkpoint and measured profile to these settings, without an additional
JSON-schema dependency.
It uses the existing platform launcher, overriding profile environment variables
with the qualified values. Optional networking, host cache paths and CPU thread
settings use the existing launcher conventions. Separate settings files can be
selected with `--settings PATH`; qualification report paths remain relative to
the recipe repository. Custom profiles require their own qualification evidence.

## New final qualification and source coverage

New RTX/Spark qualification runners use schema v2 and require the pinned
69 Basic +19 Hard tool-quality stage. Final reports validate all88 scenarios,
partial-credit totals, raw Markdown/SQLite hashes and runtime identity; endpoint
exclusions prevent completion. [Tool-quality reproduction](../benchmarks/tool_quality.md)
uses the same pinned suite for both platforms. Historical RTX v1 reports retain
their original schema and do not imply these newly required measurements.

Native Dockerfiles install all hybrid model/cache/MM/boundary ports before the
CPU quantization import check. Optional flags remain off until selected in a
qualified profile. The cache bundle fingerprints the complete installed vLLM
and B12x source trees plus `/opt/dots3/hybrid_attestation.py`, so changed fused
pack kernels, cache-group hooks, boundary factories or profiling helpers require
a new warmed-parent export. Their Triton artifacts use the existing exported
Triton cache. The exact parent image ID remains bound through wrapper build and
qualification; development thin-image tags and source checkout updates do not
modify the published v1 container.
