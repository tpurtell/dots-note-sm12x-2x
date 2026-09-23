# Dots3 Note source and calibration gates

This directory contains the source audit, corpus, and Spark runner for uniform EXL3 K4 quantization of the routed language-model experts. The full run started on 2026-09-23; the finished checkpoint and acceptance evidence are pending. Do not publish an artifact from preparation scripts alone.

## Inputs

Use these exact Hugging Face snapshots from `HF_HOME=/mnt/scratch/hf_cache`:

- FP8: `models--dots-studio--dots3-note-prev-fp8/snapshots/7c14222e22423d6df6848eb0d1c5c3a88a00311a`
- BF16: `models--dots-studio--dots3-note-prev/snapshots/1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`

Before reading weights, run the source audit. It checks the complete routed namespace, source dtypes, scale presence, and config geometry without loading tensor payloads:

```bash
hf_hub=/mnt/scratch/hf_cache/hub
fp8="$hf_hub/models--dots-studio--dots3-note-prev-fp8/snapshots/7c14222e22423d6df6848eb0d1c5c3a88a00311a"
bf16="$hf_hub/models--dots-studio--dots3-note-prev/snapshots/1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b"
python3 quantization/inspect_sources.py --bf16 "$bf16" --fp8 "$fp8" \
  --output /path/to/state/source-audit.json
```

Build the source-disjoint calibration, screening, and held-out selections with the Dots3 tokenizer. The builder needs the `tokenizers` package and committed inputs in both this repository and `../training-data`:

```bash
python3 quantization/build_calibration_corpus.py \
  --training-data ../training-data \
  --tokenizer "$fp8/tokenizer.json" \
  --output /path/to/state/calibration
```

The builder retains GLMRT's fixed source categories and selection algorithm, with Dots3 Note's exact single-user non-thinking chat rendering. Its output manifest records prompt and token identities, tokenizer hash, source revisions, and disjoint split groups. Preserve it as run evidence.

## Two-Spark execution contract

The runner must use rhea and moa, each with one GB10 and local NVMe. It must stream model layers because a complete source checkpoint cannot reside in a Spark's unified memory. It must use the FP8 checkpoint for all non-routed tensors and BF16 source weights for every one of the 34,560 routed projections. It must never apply a mixed K3/K4, projection-tier, or visual-expert replacement policy.

The 131 BF16 model shards interleave routed weights from many layers. Transfer either the intact source snapshot to both hosts or repack by tensor with a content-verified index. Do not infer a layer ownership split from source shard filenames. Local checks found the intact FP8 snapshot to be about 279 GiB and the BF16 snapshot about 538 GiB on disk. A complete copy of both sources fits each Spark's available local NVMe, while per-layer streaming keeps runtime memory bounded.

The current distributed schedule runs sequential activation replay on rhea and sends routed projection quantization to moa through GPTQModel's authenticated, checkpointed remote EXL3 API. The coordinator also retains one local GPU slot. A real 128×128 K4 projection passed the cross-host request, result, and checkpoint-reuse path before the full run started. Source shards and the complete calibration selection are staged on both hosts, so the schedule can be revised using measured layer timings.

Splitting prompts between hosts would require a barrier at each subset of each layer: each Spark would replay half the prompts, reduce every routed expert's additive Hessian and route evidence, distribute identical packed weights, and then proceed with the next subset or layer. Gate and up share one 5,120×5,120 FP32 Hessian per expert; down has one 1,536×1,536 Hessian. That is about 27.2 GiB of unique Hessian exchange per routed layer before protocol overhead. A naive two-host prompt split without reduction would change the calibration result. The active single-coordinator run remains the correctness baseline while this protocol is developed and measured.

A bounded [layer-0 replay test](activation-batch-preflight.json) on moa used the first 128 prompts in native order. Excluding model load, batch size 1 took 18.08 seconds and batch size 4 took 23.07 seconds, so the production run retains batch size 1. This is a scheduling check, not a whole-model throughput claim.

A second bounded [input-placement test](activation-device-preflight.json) kept
the same first 128 prompts on the Spark GPU. It reduced layer-0 time from
18.08 to 11.66 seconds. A routed-layer capture comparison was attempted but
stopped before completion because loading its 256 BF16 experts would have
overlapped production work on moa. The production run still uses CPU-held
calibration inputs; the layer-0 result alone does not establish a routed-layer
speedup.

`build_hybrid_source.py` makes a zero-copy checkpoint index with 3,909 native FP8 tensors and exactly 34,560 BF16 routed projection tensors. It excludes the 34,560 stale FP8 routed scale tensors. Its shard links must resolve inside the container, so mount `/home/tj/dots-note-source` at the same absolute path. `quantize_spark.py` streams source layers through GPTQModel's LazyTurtle and writes state and output only to Spark NVMe. `prepare_remote.py` creates the authenticated coordinator/worker configuration and a private token file. `remote_exl3_worker.py` implements the worker protocol. The model adapter is in the pinned GPTQModel submodule.

Projection checkpoints are enabled on the coordinator and worker. The coordinator also writes a rolling decoder-layer activation boundary after a complete routed layer, so a restart can resume at the next layer with the authenticated projection index. GPTQModel's subset Hessian frontier is enabled where its execution plan permits; gate/up captures that also produce forward outputs are not independently reusable. Do not remove run state or source checkpoints while the quant is active.

After export, run `audit_export.py` on rhea against its local FP8 source and
the exported directory before upload. It checks every routed projection's
uniform K4 Trellis, sign rotations and MCG marker, rejects leftover FP8 routed
weights or scales, and compares the bytes of all preserved FP8 core tensors.
Keep its JSON report with the publication receipts.

Run `summarize_errors.py --journal /work/state/errors.jsonl --complete` after
the last layer. It rejects duplicate or missing routed projections,
nonuniform formats, incomplete Hessian metrics, and unexpected devices; its
report records the rhea/moa projection split, recovery counts, and Hessian
weighted errors. Omit `--complete` only for progress reports during the run.

The layer-1 boundary was committed with 1,437 BF16 activation shards (10,838,640,640 tensor bytes) and 768 indexed K4 projections. An independent pass verified every activation shard's size and xxh3 digest and the manifest SHA-256. A controlled restart on rhea then restored the boundary and resumed layer 2 without replaying the completed prefix. This is recovery evidence for the first routed boundary, not final artifact acceptance.

Layer 2 is an explicit progress milestone: notify the user when its quantization begins. Do not treat source transfer or calibration selection as reaching that milestone.

## Acceptance before publication

Require all routed projections present at K4, no FP8 routed weight or scale retained in the exported language-model expert namespace, and byte-identified FP8 non-routed tensors. Validate reconstruction and held-out model behavior, native B12x EXL3 parity on GB10, model loading, text and multimodal requests, and the final Hugging Face artifact inventory. Publish only the accepted artifact to `wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1`. Install that exact Hub revision into the local and both Spark Hugging Face caches, then remove temporary raw exports and run-state only after the published revision has been revalidated.
