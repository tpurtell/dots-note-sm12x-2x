# Packed hybrid routing component gate

Two RTX workers passed CPU layout checks and changed-input CUDA graph replay checks. This measures the routing communication component, not whole-model throughput.

| Rows | Unpacked median µs | Packed median µs | Reduction |
|---:|---:|---:|---:|
| 1 | 24.587 | 16.366 | 33.44% |
| 2 | 24.530 | 17.315 | 29.41% |
| 4 | 26.633 | 18.450 | 30.73% |
| 8 | 28.920 | 22.522 | 22.12% |
| 16 | 36.874 | 28.812 | 21.86% |
| 512 | 291.193 | 283.674 | 2.58% |

Medians pool both owner positions, both ranks and five samples per record. Raw samples and hash provenance are archived.
