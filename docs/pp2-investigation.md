# Layer ownership with expert tensor parallelism

The requested candidate assigns complete non-expert decoder modules to layer owners while retaining tensor partitions of **every routed expert on both ranks**. It uses no expert parallelism. Standard vLLM PP2 would omit whole layers from each rank and cannot implement this ownership pattern; PP2×TP2 ordinarily requires four workers.

## Candidate contract

`VLLM_HYBRID_LAYER_PARTITION=23,23` assigns the46 target decoder layers in contiguous ranges. The runtime remains TP2/PP1. The MTP decoder belongs to the last owner. The generic `hybrid_parallel.py` defines the ownership plan, explicit dense-local versus expert-TP contexts, capability validation, and graph-compatible broadcast/reduce primitives. No global TP group is replaced or monkeypatched.

For each decoder layer:

1. Its owner holds full attention projections, indexer, KV, normalization, router, and shared expert/dense MLP parameters.
2. Owner attention produces complete output locally; no attention all-reduce occurs.
3. The owner routes once. Activation, global expert IDs, and routing weights are broadcast.
4. Both ranks execute their existing EXL3 tensor partition for every routed expert selected. The adapter calls the native router and existing B12x `RoutedExperts.forward_modular`; expert tensor slicing and K4 weights remain unchanged.
5. The two partial FFN outputs are reduced **to the owner**, which adds the complete shared expert output once. The next owner's input layer normalization consumes an already reduced result.
6. Hidden state and residual transfer only at ownership boundaries. MTP's complete dense decoder is owner-local and avoids its former TP all-reduce.

Boundary embedding and vocabulary interfaces initially retain ordinary vLLM TP2 behavior; final normalized hidden state is broadcast to preserve the sampler's contract. Multimodal encoders retain their existing ownership. This first integration targets decoder ownership and does not claim all non-decoder weights are owner-exclusive.

## Concrete source dependencies

- Dots decoder construction: `vllm/models/dots3_note/nvidia/model.py:504–570`; shared expert forward adds to routed partial output at142–162.
- Existing decoder forward defers the FFN all-reduce to the following input normalization and reduces attention before post-attention normalization: `vllm/models/deepseek_v32/nvidia/model.py:122–169`. Both fused reductions must be bypassed for complete owner outputs.
- Native linear constructors accept explicit `disable_tp` and persist per-module TP rank/size, including loader metadata: `model_executor/layers/linear.py:268–318,473–509`.
- Existing router and modular expert split: `layers/fused_moe/runner/moe_runner.py:625–639`, `routed_experts.py:1172–1207`.
- V2 shared KV planner can project disjoint owner specs while preserving group IDs; the actual-source CPU checks are in `check_hybrid_cache.py`.
- PyNccl exposes stream-aware broadcast and root-directed reduce: `distributed/device_communicators/pynccl.py:424,538`. Both ranks must execute the same ordered collectives during warmup, capture, and replay.

## Prerequisite evidence and remaining qualification

CPU protocol checks cover both owner roots, changed activations, single router/shared contribution, and identical collective order. Source checks verify explicit full projection contexts in DSA/SWA constructors. Independent GPU checks qualify root-directed transport and128-head sparse attention. These are prerequisites, not full-model correctness or performance evidence.

The first full-model candidate must verify actual per-rank parameter and KV ownership, loading all routed shards, MTP correctness, cache reuse, structured output, multimodal compatibility, and graphs before any C1–C4 performance claim. Attention is computed on one GPU per layer, so reduced communication does not guarantee improved latency. Compare the same saved coding request payloads against compact native TP2; preserve natural-stop/output-length differences.
