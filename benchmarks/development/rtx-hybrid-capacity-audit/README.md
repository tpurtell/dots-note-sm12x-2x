# Matched hybrid capacity audit: cut18 versus cut17

Both sequential capacity runs used exactly image `0d2c3c4d`; only the attestation helper differs from execution parent `202de164`. Both passed the prefix gate. Earlier cut17 multimodal and40/40 reasoning API evidence is included separately.

| Cut | Rank0 available KV GiB | Rank1 available KV GiB |
|---:|---:|---:|
| 17 | 6.911460 | 11.147844 |
| 18 | 6.675132 | 9.091203 |

The measured difference is dominated by `total_consumed`, not a large change in peak activation memory. Root-cause analysis remains open. The reported graph estimate is already included in peak activation and must not be added again. These startup capacities are not a new maximum-context qualification or throughput result.

Raw placement estimates from the earlier cut17 investigation are retained as historical diagnostics; use current CUDA-only storage accounting for decisions. Exact raw hashes, runtime/container identities and profiling byte fields are in manifest.json and lossless archives.
