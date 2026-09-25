# Native TP2 versus first hybrid screen

Matched four coding tasks, run 0, same image `f4c227`, context 524288, MTP3, .95 memory, batch512, compact cache and indexer cap4. Script, task hashes and request settings match. Hybrid uses partition23,23/block64; native launch details are archived.

| Concurrency | Native decode tokens/s | Hybrid decode tokens/s | Hybrid change |
|---:|---:|---:|---:|
| 1 | 184.683 | 154.955 | -16.10% |
| 2 | 135.084 | 130.538 | -3.37% |
| 4 | 101.626 | 97.973 | -3.59% |

Both completed 12/12 requests naturally with static checks passing. Decode excludes the initial SSE burst and includes reasoning tokens. Output budget8192 is benchmark-specific, not the server output limit.

This small development screen finds unpacked hybrid slower at all three concurrency levels; it does not qualify a release or establish broader quality differences. Output lengths vary with stochastic generation. Packed routing is a separate candidate requiring its own measurement.

Raw native receipts are lossless gzip files with SHA-256 hashes in manifest.json; hybrid raw receipts are in ../rtx-hybrid-first-screen. Historical container state does not assert current liveness.
