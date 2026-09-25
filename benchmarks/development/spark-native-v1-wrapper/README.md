# Spark v1 native wrapper publication provenance

**Published publicly; final qualification pending.** No final serving
performance claim is made by this archive.

Tag `ghcr.io/tpurtell/dots3-note-exl3-k4-spark:20260925-v1` resolves to
`sha256:fbe12925a19f529a35ea036a1f0ed9db0ba6de77f7455a6401816ea54508dbad`.
Push and authenticated digest pulls on both hosts passed. Anonymous exact-digest
pull on Rhea also passed using a newly created empty Docker config with
`DOCKER_AUTH_CONFIG` unset; directory listing and pull log are archived. Both
hosts inspect the same wrapper image. The redundant direct transfer to Moa was
cancelled after its successful registry pull, not because the image failed.
Final published-image startup/qualification remains excluded from this snapshot.

The native runtime parent is frozen recipe `8f0a580cb360e19ee757f1435d03e7bf7c6a52e4`.
Wrapper packaging used recipe/tool snapshot `2a84f3a8395fe296224a3a09633e9a1ccfb7f583`,
including the optional allocator helper fingerprint introduced by `e0d6988`.
These are different source scopes, not interchangeable runtime revisions.
Local Docker image IDs and registry digests are recorded separately in
`registry-receipt.json` and original image inspections.

Both rank cache manifests and the merged manifest are preserved; merge output
is at the start of `wrapper-build.log.gz`. A fresh seed smoke copied **4,014**
files with **0 existing**. This checks cache installation, not model requests.
Exact installed `hybrid_attestation.py` and `allocator_policy.py` fingerprints
are recorded in the seed and publication receipt. GPU-specific artifacts can
still need recompilation on another device.

`manifest.json` binds lossless raw build/transfer/export/merge/push/pull/seed
logs, frozen source hashes, wrapper tool hashes, native per-host image/runtime
and allocator policy receipts. This evidence supports final qualification and
explicit inheritance; it does not activate Spark release settings.
