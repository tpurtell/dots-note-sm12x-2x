# Native release packaging

These tools prepare separate `linux/amd64` RTX and `linux/arm64` Spark release
images from the exact runtime images qualified on their native hosts. They do
not download model weights, publish images, or choose serving defaults.

## Spark host memory guard

The release profile pins `memory_guard: true` and at least 8 GiB host memory
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

## Build sequence

1. Build the native runtime image, including the optional soundfile dependency
   from `audio-requirements.txt` before qualifying it. The pinned wheels include
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

## Public fast path (enabled after final qualification)

`settings.json` intentionally contains null digest/profile fields and
`pending-qualification` status. `run.sh` refuses to launch until the final
published image digests, qualified profiles and report hashes are recorded.
No development image tag is a release fallback. Keep this recipe checkout,
including its qualification reports and `serving/start_*.sh` launchers.

RTX currently selects MTP3, native vocabulary and 0.95 GPU memory utilization.
Spark's selection is independent; its current candidates use 0.80 utilization
with the host guard. These decisions must be reflected in the final settings
after the corresponding release images pass qualification. Bare development
launcher defaults do not reproduce these profiles.

Run the following commands from the recipe checkout root. For RTX, after the
final release settings are committed:

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
Set addresses and the network interface for your own hosts. The interface must
reach the other Spark; the existing launcher also exposes `/dev/infiniband`.

```bash
# On both hosts:
export HF_HOME="$HOME/.cache/huggingface"
export RUNTIME_CACHE="$PWD/.cache/release/spark-runtime"
export MASTER_ADDR=HEAD_NETWORK_IP
export SOCKET_IFNAME=RDMA_NETWORK_INTERFACE
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
retains the container; `restart` uses its original settings and verifies it is
the pinned image. To change settings, stop and remove the container, then start
again. On Spark, stop the head before the worker; restart the worker before the
head. `health` runs on the head only; a worker has no HTTP endpoint. HTTP health
alone does not establish functional request readiness.

`HF_HOME` must contain the already installed checkpoint and is mounted in full,
read-only, with offline mode enabled. No command downloads model weights.
`RUNTIME_CACHE` must be writable; do not use this project's `/mnt/scratch` disk.
`PORT` can customize the HTTP port. Extra arguments after `--` are passed to
vLLM, for example `... rtx start -- --api-key YOUR_LOCAL_KEY`. Reasoning parser
override flags are rejected. Other extra flags may change qualified behavior; published performance applies to the stored profile.
Avoid putting secrets in shared shell history.

### Final settings contract

The JSON schema version is `1`; `status` must become `qualified` only after the
published digest has been pulled and verified. Each platform stores its own:

- Native architecture and immutable `ghcr.io/...@sha256:...` image reference.
- Repository-relative qualification report path and SHA256 of its exact bytes.
- GPU memory utilization, context limit, sequence limit and prefill chunk size.
- FP8 KV format and platform-specific MTP token count (`0` explicitly disables MTP).
- Required `reasoning_parser: "dots3"`, verified against the image label.
- For Spark, `memory_guard: true` and `min_host_available_gib` of at least 8.
- Boolean B12x vocab and RTX PCIe settings.
- Spark RoCE enable/eager booleans and explicit admitted row counts (disabled by default).
- Explicit exact-FP8 projection selector string (empty disables) and row counts.

The runner validates these fields without an additional JSON-schema dependency.
It uses the existing platform launcher, overriding profile environment variables
with the qualified values. Optional networking, host cache paths and CPU thread
settings use the existing launcher conventions. Separate settings files can be
selected with `--settings PATH`; qualification report paths remain relative to
the recipe repository. Custom profiles require their own qualification evidence.
