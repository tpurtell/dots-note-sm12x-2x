# Final native parent and wrapper sequence

Prepared commands only. No builds or lifecycle actions were run for this checklist. Keep published RTX `20260925-v1` immutable. New version names and final optional flags remain pending the current screens.

## 1. Freeze source and runtime profiles

Record a clean recipe commit and submodule pins. Copy that exact checkout to each native build host. Do not rebuild the two Sparks independently for final qualification: build one ARM parent, transfer it to the other host, and verify identical image IDs.

Required source coverage in both native Dockerfiles:

- B12x `bf5677c6` full-owner sparse MLA and compact record support.
- `dots3_b12x_fp8.py` including `f264e11`: actual sole MTP layer46 prefix is SWA, with or without `mtp_block`.
- `hybrid_pack_kernel.py` mixed int32/int64 route-ID and FP32/BF16/FP16 weight support (`eab9696`).
- Shared overlap restricted to1–16 rows (`a8d8744`).
- Owner-aware cache-group module/port (`9cf7958`), attestation/profile telemetry (`0c121f4`), and lazy boundary factories (`f7556d5`).
- Standard quantization/reasoning/compact and native communicator ports first; hybrid model/cache-group/MM/boundary ports afterward. The final quantization import smoke must pass.
- Rejected `port_indexer_profile_lifetime.py` is not applied.

Capture actual bytes as well as commit names:

```bash
mkdir -p .cache/release/final-provenance
git rev-parse HEAD > .cache/release/final-provenance/recipe-revision.txt
git submodule status > .cache/release/final-provenance/submodules.txt
git status --porcelain > .cache/release/final-provenance/worktree-status.txt
sha256sum serving/Dockerfile.* serving/dots3_b12x_fp8.py serving/hybrid*.py serving/dots3_hybrid*.py serving/port*.py > .cache/release/final-provenance/runtime-source.sha256
```

Provisional RTX: cut17/29,batch512,.95,MTP3,MM0/1,packed/fused/overlap/groups,narrow SWA-QB rows4/8/16, native boundary tables. Spark: cut17/29,batch512,.80,1GiB physical reserve and guard, common hybrid options. Native524288 context on both. Force-MQA remains a separate decision; record its exact optional JSON flag only if selected. Do not treat this paragraph as qualified settings.

## 2. Build native parents

Set unique candidate tags before running:

```bash
: "${RTX_PARENT_TAG:?unique RTX parent tag}"
docker build --platform linux/amd64 -f serving/Dockerfile.rtx -t "$RTX_PARENT_TAG" .
docker image inspect "$RTX_PARENT_TAG" > .cache/release/final-provenance/rtx-parent-image.json
```

On Rhea, from the frozen source:

```bash
: "${SPARK_PARENT_TAG:?unique Spark parent tag}"
docker build --platform linux/arm64 -f serving/Dockerfile.spark -t "$SPARK_PARENT_TAG" .
mkdir -p .cache/release/final-provenance
docker image inspect "$SPARK_PARENT_TAG" > .cache/release/final-provenance/spark-parent-image.json
docker save "$SPARK_PARENT_TAG" | ssh moa docker load
```

Inspect the same parent tag on both Sparks and require identical `.Id`. Image transport does not fetch any model files; use the already-installed complete HF cache.

## 3. Warm and export platform caches

Launch using existing native launchers with the frozen profile, whole `HF_HOME`, explicit `REASONING_PARSER=dots3`, and fresh platform runtime caches. Spark worker first, then head; preserve the physical-memory guard. Qualify functional paths and warm every selected kernel before export; pause request traffic during export.

For each host/container:

```bash
: "${CONTAINER:?warmed container}" "${PLATFORM:?rtx or spark}" "${SEED:?new project cache bundle path}" "${EVIDENCE:?accepted parent evidence path}"
python3 serving/release/cache_bundle.py export --container "$CONTAINER" --platform "$PLATFORM" \
  --recipe-revision "$(git rev-parse HEAD)" --evidence "$EVIDENCE" --output "$SEED"
```

Export both Spark ranks. Transfer Moa's bundle to Rhea and merge UUID-specific B12x artifacts with `cache_bundle.py merge --bundle RHEA_SEED --add-b12x-from MOA_SEED --output SPARK_SEED`. The wrapper remains bound to exact native sources/dependencies and parent identity; new GPU UUIDs/shapes may still compile.

## 4. Build, publish and pull wrappers

```bash
: "${PARENT_IMAGE:?exact warmed parent ID or inspected tag}" "${RELEASE_TAG:?new platform GHCR tag}"
python3 serving/release/build.py --bundle "$SEED" --image "$PARENT_IMAGE" --tag "$RELEASE_TAG"
```

Verify the wrapper using an empty writable runtime cache before publishing. Then `docker push "$RELEASE_TAG"`, record the real registry digest, and pull by that digest. Use separate RTX/ARM tags. Pull the **same final ARM digest on both Sparks**. Record authenticated and anonymous access results separately; do not assert public access until anonymous pull succeeds.

## 5. Final published-image qualification

Freeze validator/benchmark source for the run; it is hash-bound. The pinned tool-eval checkout must be clean at `cf54b4bfe705f12f71e8866f10730572497c8105` with frozen dependencies. The runners preflight this locally before contacting the model. Use fresh output paths.

```bash
python3 serving/release/qualify_rtx.py --container "$CONTAINER" \
  --expected-image-id "$PUBLISHED_LOCAL_IMAGE_ID" --expected-mtp 3 \
  --max-model-len 524288 --output-dir .cache/release/rtx-final-qualification --execute
```

```bash
python3 serving/release/qualify_spark.py --head-host rhea --worker-host moa \
  --head-container dots3-vllm-head --worker-container dots3-vllm-worker \
  --remote-project "$REMOTE_RECIPE" --expected-image-id "$PUBLISHED_LOCAL_IMAGE_ID" \
  --expected-mtp "$SPARK_MTP" --max-model-len 524288 \
  --gpu-memory-utilization .80 --min-host-available-gib 1 \
  --output-dir .cache/release/spark-final-qualification --execute
```

The v2 plan includes native-context boundary/retrieval, all performance/functional stages and69 Basic+19 Hard tool scenarios. Root/head aggregate ownership proof must match the selected partition/MM/context; the Spark worker references that receipt. A tool scenario count is not an HTTP request count. Preserve quality misses and infrastructure errors exactly.

## 6. Export reports and activate settings

After terminal `complete.json` exists, run `report.py --input QUALIFICATION --output benchmarks/releases/PLATFORM-VERSION --platform PLATFORM --image GHCR_DIGEST --cache-manifest SEED/manifest.json` plus `--registry-receipt REGISTRY.json` for RTX (and Spark when available). Use actual paths/digests, never placeholder values in committed settings.

Bind each platform's immutable image, exact profile and report SHA independently. New524K/hybrid profiles require completed v2 Basic/Hard/Total evidence. Then exercise public `run.sh` pull/start/health/logs/stop/restart paths with those settings and a fresh runtime cache. Publish README measurements from the accepted reports only. Existing v1 settings remain usable until replacements pass these gates.
