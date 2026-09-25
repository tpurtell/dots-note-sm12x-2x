# RTX force-MQA rejection

The narrow SWA-QB candidate with `sparse_mla_force_mqa=true` passed40/40 API and prefix gates, but showed no capacity or sampled peak-NVML benefit. Capacity remains2,004,801 tokens. Retain the defaultfalse.

| Prompt tokens | Median effective prompt tokens/s |
|---:|---:|
| 8192 | 3976.519 |
| 32768 | 3746.947 |
| 131072 | 3210.669 |

The installed attention backend restricts masked long-MHA to family100; SM120/121 already use the B12x MQA path for long prefill. The flag changes the dense-MHA prefix of at most2048 tokens, not the long-context path. The measured rates do not justify adoption.

Three cold runs per depth, full API/prefix outputs, timing intervals, memory telemetry, loaded ownership, actual image/arguments and final logs are preserved. Development rejection only; no final release or causal thermal claim.
