# B12x switches under hybrid owner attention

CPU source audit, 2026-09-25. This is a candidate-selection note, not a GPU performance result. Native TP2 measurements do not establish the result with owner-local attention or multimodal towers. Keep partition, multimodal ownership, context, MTP depth, output budget, and memory utilization matched within each comparison.

## Exact FP8 projection selectors: audit and correction

The checkpoint has DSA 128 heads with QK width192 and V width128; SWA has64 heads with QK width256 and V width128. Both Q low-rank dimensions are1024. Owner construction disables dense TP splitting.

| Projection | Native TP2 weight shape | Owner weight shape |
|---|---|---|
| DSA Q-B | 12288 ×1024 | 24576 ×1024 |
| SWA Q-B | 8192 ×1024 | 16384 ×1024 |
| DSA output | 5120 ×8192 | 5120 ×16384 |
| SWA output | 5120 ×4096 | 5120 ×8192 |

The pre-correction [exact FP8 method](../serving/dots3_b12x_fp8.py) filters by loaded shape, not attention type:

- `DOTS3_B12X_EXACT_FP8=q_b_proj` selects all Q-B projections then rejects owner shapes outside its TP2 allowlist. Do not launch this flag on hybrid.
- `swa_q_b_proj` only permits8192×1024, so owner SWA Q-B falls back to native preparation/apply. This is not an active hybrid candidate.
- `dsa_o_proj` permits5120×8192. Under hybrid that selects **SWA output**, while actual owner DSA output falls back. The name no longer describes the selected layers. Do not present that flag as owner DSA coverage or combine it with the no-op SWA Q-B flag.

A useful later exact-FP8 screen requires explicit attention-type-aware selectors and qualified full-owner shapes first. Required gates: checkpoint scale/weight preservation, numeric parity with reference, changed-input graph replay for MTP/C1–C4 row counts, and actual dispatch evidence. No kernel changes are proposed by this audit. Retained source weights and scale copies consume memory; newly freed MM memory is not evidence that these projections are faster. Every such candidate needs a new startup memory/KV receipt.

## Narrow existing candidate: vocabulary projection

With the **current TP2 boundary embedding/head contract**, `DOTS3_B12X_VOCAB=1` remains shape-valid on both platforms: the method requires a contiguous BF1676032×5120 shard. It only dispatches B12x for exactly one input row and falls back natively otherwise. See [vocabulary adapter](../serving/dots3_exl3_fp8.py) and [hybrid output boundary](../serving/dots3_hybrid_parallel.py).

Recommended next optional kernel screen, after choosing packed/MM/placement settings:

1. Control `DOTS3_B12X_VOCAB=0` versus candidate `DOTS3_B12X_VOCAB=1`.
2. Keep `DOTS3_B12X_EXACT_FP8=` and both custom allreduce switches disabled; change only vocabulary.
3. Use matched C1/C2/C4 coding tasks and seeds, identical MTP3/context/placement, and inspect completed-only latency, output lengths, truncations, and per-request decode. Confirm one-row MTP/head dispatch instead of attributing multirow native fallback to B12x.
4. Preserve live reasoning/JSON/tool boundary and graph checks. Previous native TP2 parity/shared-plan evidence supports reuse of the current shape; whole-model hybrid performance remains unmeasured.

If boundary embedding/head ownership is enabled later, inspect the actual head layout first. A full152064×5120 head violates the current strict TP2 geometry check, so this flag is not valid for that configuration without adapter work and new correctness gates.

## Collective switches do not target the hybrid hot path

[PyNcclOwnerTransport](../serving/hybrid_parallel.py) directly invokes the PyNccl communicator's `broadcast` and root `reduce`. Packed routing combines activation/route metadata into one broadcast, then reduces routed-expert partials to the owner. Partition crossings and final hidden-state distribution also use broadcasts.

Existing [PCIe adapter](../serving/b12x_pcie_all_reduce.py) and [RoCE adapter](../serving/b12x_roce_all_reduce.py) replace TP `all_reduce`, not these calls. Enabling them does not accelerate the repeated routed-expert transport. Remaining native TP boundary operations may invoke allreduce, but that is a much narrower opportunity than the previous native TP2 screens. Avoid spending a full C1–C4 screen on these flags until dispatch counts establish material coverage. RoCE also adds session/proxy/output-buffer costs and fail-stop requirements; MM memory savings do not remove them.

Packed routing itself is an exact, active candidate on both platforms (`VLLM_HYBRID_PACKED_ROUTING=1`) and should be qualified before layering vocabulary. RTX component latency improvements are archived [here](../benchmarks/component/hybrid-packed-routing/README.md); they do not imply the same whole-model gain or a Spark result.

## Memory and existing B12x paths

Hybrid retains TP2 routed experts on both ranks, so the existing EXL3 fused-MoE path remains applicable. Full-head sparse MLA is already enabled and qualified separately; compact DSA records and bounded indexer workspace should remain fixed in matched screens. Owner-local MM frees asymmetric per-rank memory; use actual attested tower/dense storage, per-worker KV budget, and bounded SWA/full-history DSA placement to choose a split. Do not raise Spark memory utilization to spend the savings: preserve the qualified host guard and .80 setting while establishing useful native-context capacity.

## Selector correction prepared after this audit

The adapter now resolves actual DSA/SWA layer identity from the vLLM HF text config, recognizes the single Dots MTP SWA block, and accepts owner Q-B24576/16384×1024 and DSA-output5120×16384 alongside TP2 geometries. Wrong-type layers remain native; unresolved identities fail clearly. The CPU selector check covers the SWA-owner/DSA-TP2 shape collision. `check_exact_fp8_adapter.py --owner-full` prepares rows1/4/8/16 and tests changed-input graphs, exact source preservation, numeric reference, and unplanned-row native fallback for both full Q-B matrices and full DSA output. GPU qualification and whole-model memory/performance evidence are still required before using the new owner shapes.
