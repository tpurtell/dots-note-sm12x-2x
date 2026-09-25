# Spark optimized native prefill baseline

Completed development baseline for a separate prospective force-MQA comparison. Force-MQA is not selected. Hybrid17/29, packed/fused routing, bounded shared overlap, batch512; exact settings are archived.

The observed native image IDs differ between hosts (rhea174bee76, moa b2e57dd8). Both are recorded exactly; this is not shared published-digest final qualification.

- Native524288 context admitted; reported KV capacity2,750,605 tokens.
- Reasoning API40/40 passed.
- Nine cold-prefill requests completed: three per depth.

| Prompt tokens | Median effective prompt tokens/s | Median TTFT s |
|---:|---:|---:|
| 8192 | 839.350 | 9.760 |
| 32768 | 824.455 | 39.745 |
| 131072 | 541.280 | 242.152 |

The .jsonl prefill file contains one whole JSON object. Unique prompts avoid prefix reuse; client TTFT includes tokenization and first-token handoff. Both host logs/memory monitors, ownership receipt and128K-stage memory/GPU/smaps snapshots are losslessly archived with hashes. No final-release or force-MQA performance claim.

Dispatch correction: SM121 already uses B12x MQA for long prefill because masked long-MHA is restricted to capability family100. The force-MQA flag only removes the dense-MHA prefix at at most2048 tokens. This baseline must not be interpreted as long masked-MHA.
