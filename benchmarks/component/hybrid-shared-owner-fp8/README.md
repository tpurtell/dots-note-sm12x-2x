# Shared overlap and full-owner FP8 component gates

Idle RTX probes only. Shared-overlap GPU0 v2 passed changed-input graphs using real shared FP8 MLP and eight TP2-sliced routed experts; this does not cover full256 routing or network transport. The initial fixture failed on missing `model_config.is_moe`, before the kernel checks. Runtime overlap is now limited to1–16 rows because512 was slower.

Owner FP8 GPU1 passed full DSA/SWA Q-B and full DSA output at rows1/4/8/16, changed-input graph/eager checks, source immutability and reference accuracy; row2 checked native fallback. This is explicit checkpoint tensor loading, not a distributed loader or whole-model test.

Raw logs/results and exact hashes are archived. Median component timings and maximum reference errors appear in manifest.json.
