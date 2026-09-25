# Native release packaging

These tools prepare separate `linux/amd64` RTX and `linux/arm64` Spark release
images from the exact runtime images qualified on their native hosts. They do
not download model weights, publish images, or choose serving defaults.

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
   launcher and `IMAGE=... RUNTIME_CACHE=...`. Do this for both Spark ranks.
   Inspect startup, run functional and performance gates, then publish approved
   versioned tags with `docker push`. Record the registry digest.
5. Pull each image by digest and repeat the release gates. Public access must be
   checked using an unauthenticated Docker configuration. Use the digest in the
   documented `IMAGE=ghcr.io/...@sha256:...` fast path.

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
