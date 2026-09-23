---
license: apache-2.0
base_model:
  - dots-studio/dots3-note-prev-fp8
  - dots-studio/dots3-note-prev
library_name: vllm
tags:
  - dots3-note
  - exl3
  - fp8
  - multimodal
---

# Dots3 Note Preview, FP8 core with uniform EXL3 K4 routed experts

This checkpoint preserves the non-routed tensors from
[`dots-studio/dots3-note-prev-fp8`](https://huggingface.co/dots-studio/dots3-note-prev-fp8)
at revision `7c14222e22423d6df6848eb0d1c5c3a88a00311a`. It replaces every
language-model routed expert weight in layers 1–45 with an EXL3 MCG K4
quantization of the corresponding BF16 tensor from
[`dots-studio/dots3-note-prev`](https://huggingface.co/dots-studio/dots3-note-prev)
at revision `1e1e7b0cd37a3a48a6c8d7fa55d5f9d14377006b`. All 256 experts and
all gate, up, and down projections use the same K4 format. The vision experts,
shared experts, attention, embeddings, vocabulary head, and multimodal
components remain as provided by the FP8 source.

Quantization used GPTQModel with 1,437 source-disjoint calibration prompts and
two NVIDIA DGX Sparks. The [recipe and audit
tools](https://github.com/tpurtell/dots-note-sm12x-2x) record source revisions,
calibration selection, uniform-format checks, and measured serving results.
This hybrid EXL3/FP8 checkpoint requires the Dots3 vLLM integration in that
recipe; generic FP8-only loaders do not understand its routed experts.

The upstream Dots3 Note Preview models are released under Apache-2.0. Consult
their model cards for model capabilities, evaluation details, and usage terms.
