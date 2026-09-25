# Spark MTP4 finite comparison

Both candidates use RoCE, 32K context, 0.80 allocation, batch 512 and native vocabulary projection. Each has one warmup and one timed run at C1/C4, four tasks per run, an 8192-token output budget, and matched request payloads. Compressed raw receipts and hashes are in the manifest.

| Metric | MTP3 | MTP4 |
|---|---:|---:|
| C1 median decode tokens/s | 41.17 | 42.07 |
| C4 median decode tokens/s | 24.92 | 25.43 |
| C1 natural stops / requests | 4/4 | 3/4 |
| C4 natural stops / requests | 3/4 | 3/4 |

Paired median decode ratios favor MTP4 by 2.19% at C1 and 1.90% at C4. Completed-request latency ratios are 0.852 and 1.160, with output-length ratios 0.858 and 1.160: latency differences largely follow output length. No API errors occurred. Static checks are not executed-code correctness, and stochastic truncation is not proof of a draft-depth defect.

MTP3 remains the established candidate with three-run evidence; this small MTP4 screen does not establish a robust overall improvement. Native 524288-context qualification takes priority over extending this screen.
