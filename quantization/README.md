# Dots3 Note source and calibration gates

This directory contains preparation for a uniform EXL3 K4 quantization of the routed language-model experts. The quantization runner, finished checkpoint, and acceptance evidence have not yet been produced. Do not publish an artifact from the preparation scripts alone.

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

The main throughput decision is how to share calibration work. A disjoint prompt split can halve forward work, but EXL3 requires the additive Hessian evidence for each expert to be reduced across hosts before quantization. An expert split avoids Hessian exchange but duplicates the calibration forward pass. Compare measured end-to-end time, peak unified memory, RDMA traffic, and restart behavior on a small number of layers before selecting the full-run schedule. In either design, write content-bound per-layer checkpoints and resume only from validated boundaries. A completed layer must record route coverage, quantization error, exact K4 packed tensor inventory, and the next replay frontier.

Layer 2 is an explicit progress milestone: notify the user when its quantization begins. Do not treat source transfer or calibration selection as reaching that milestone.

## Acceptance before publication

Require all routed projections present at K4, no FP8 routed weight or scale retained in the exported language-model expert namespace, and byte-identified FP8 non-routed tensors. Validate reconstruction and held-out model behavior, native B12x EXL3 parity on GB10, model loading, text and multimodal requests, and the final Hugging Face artifact inventory. Publish only the accepted artifact to `wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1`. Install that exact Hub revision into the local and both Spark Hugging Face caches, then remove temporary raw exports and run-state only after the published revision has been revalidated.
