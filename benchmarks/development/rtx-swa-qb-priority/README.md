# RTX SWA Q-B finite priority screen

Eight measured requests per candidate compare identical payloads at C1/C4,
using MTP3, 262,144 context, 0.95 utilization, 16 slots, 512-token prefill chunks,
native vocabulary and native collectives. The candidate enables
`DOTS3_B12X_EXACT_FP8=swa_q_b_proj` at rows `1,4,16`.

| Clients | Native median decode tokens/s | SWA Q-B median decode tokens/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 180.436 | 186.543 | +3.38% |
| 4 | 104.648 | 105.269 | +0.59% |

Changes are ratios of these medians. Native completed 8/8 naturally; the
candidate completed 7/8, with one response reaching this benchmark's 8,192-token
output budget including reasoning. Output lengths and stochastic generation
differ, so this is not a causal quality-regression claim.

**Decision:** retain native FP8. This limited screen does not establish a clear
balanced benefit, and the candidate needs about 264 MiB of additional copied
weights per GPU. Preserving capacity for the requested native 524K-context
recipe now takes priority. No further benchmark extension is planned.

The [manifest](manifest.json) records exact image/source/profile/environment,
matched-payload evidence, per-concurrency results and original/compressed SHA256
hashes. Raw baseline subset, candidate outputs, comparison, source/build files,
runtime snapshots, prefix gate and logs are lossless gzip archives. Runtime
caches are excluded. The original build manifest's pre-run execution field is
preserved verbatim; the later completed run receipts establish execution.
