# Fused pack dtype correction and owner FP8 timing

The original optimized-group18 startup rejected native router metadata dtype. Its failed startup/runtime/container evidence is preserved. Fix `eab9696` passed28 GPU conversion/layout cases including int64 route IDs and BF16/FP16 weights. This component gate does not establish whole-model success.

Owner FP8 timings compare full adapter apply (activation quantization included) against the native processed FP8 kernel using five samples of40 CUDA graph replays, with alternating order. Prior numeric/source/changed-input checks remain active.

| Weight | Rows | Native / adapter |
|---|---:|---:|
| model.layers.0.self_attn.q_b_proj | 1 | 0.701 |
| model.layers.0.self_attn.q_b_proj | 4 | 1.551 |
| model.layers.0.self_attn.q_b_proj | 8 | 1.202 |
| model.layers.0.self_attn.q_b_proj | 16 | 1.003 |
| model.layers.0.self_attn.q_b_proj | 2 | 0.999 |
| model.layers.2.self_attn.q_b_proj | 1 | 0.801 |
| model.layers.2.self_attn.q_b_proj | 4 | 1.249 |
| model.layers.2.self_attn.q_b_proj | 8 | 1.643 |
| model.layers.2.self_attn.q_b_proj | 16 | 1.452 |
| model.layers.2.self_attn.q_b_proj | 2 | 1.000 |
| model.layers.0.self_attn.o_proj | 1 | 0.873 |
| model.layers.0.self_attn.o_proj | 4 | 0.880 |
| model.layers.0.self_attn.o_proj | 8 | 0.586 |
| model.layers.0.self_attn.o_proj | 16 | 0.627 |
| model.layers.0.self_attn.o_proj | 2 | 0.999 |

Ratios above1 favor the adapter. Row2 is a native-fallback control, not an exact FP8 candidate. Recommended next screen: `DOTS3_B12X_EXACT_FP8=q_b_proj DOTS3_B12X_EXACT_FP8_ROWS=4,8,16`. Both owner Q-B shapes lose at row1; DSA output loses throughout. Keep those paths native. Whole-model C1–C4 and memory/functional qualification remain required before selecting a default.
