# RTX prefill versus batch-token admission

Completed prefill: batch512 baseline and batch1024, plus rejected2048/4096 admissions. Batch1024 coding and the monitored warm512 return control are complete; retain batch512 for this candidate. All use the exact profiling image `e54f202f`, cut17, native524288 context and recorded launch settings.

| Batch cap | KV capacity tokens | Disposition |
|---:|---:|---|
|512|2,049,116|Baseline; nine prefill requests completed|
|2048|707,424|Rejected: loses1,341,692 tokens; no timing run|
|4096|Not admitted|1.11GiB available versus1.88GiB required; no timing run|

Batch512 cold C1 prefill medians, three runs each:

| Prompt tokens | Effective prompt tokens/s |
|---:|---:|
| 8192 | 4108.742 |
| 32768 | 3927.787 |
| 131072 | 3305.417 |

Unique exact-length prompts avoid prefix reuse. Rate includes tokenization and first-token handoff. Startup warnings are recovered allocation pressure during weight postprocessing: checkpoint reading ended09:42:24, warnings began09:42:33, model loading completed09:42:39/43, and profiling followed. Captured allocator counters show rank0 retries0/OOMs0 and rank1 retries1/OOMs0. These were not thrown OOMs or prefill errors. The first1,006,632,960-byte request matches a TP2 EXL3 slab geometry, but its exact caller is unproven. This is not final-release qualification. Exact raw samples, memory profiles, arguments and hashes are archived.

## Completed batch1024 prefill

Capacity1,916,645 tokens, down132,471 from512. Three runs per depth:

| Prompt tokens | Batch512 tokens/s | Batch1024 tokens/s | Change |
|---:|---:|---:|---:|
| 8192 | 4108.742 | 4483.648 | +9.12% |
| 32768 | 3927.787 | 4319.068 | +9.96% |
| 131072 | 3305.417 | 3631.782 | +9.87% |

## Completed coding return control and decision

After equivalent prefill load, warm512 returned12/12 natural completions/static passes; batch1024 achieved11/12 with one async request reaching the8192 benchmark output budget.

| C | Warm512 decode tokens/s | Batch1024 | Change |
|---:|---:|---:|---:|
| 1 | 164.104 | 148.703 | -9.39% |
| 2 | 129.172 | 121.833 | -5.68% |
| 4 | 98.834 | 82.055 | -16.98% |

Retain batch512: the roughly10% prefill gain at1024 does not offset the measured decode cost and132,471-token capacity loss for this recipe. Temperature/clock monitoring, timing intervals and full raw outputs are preserved. No causal thermal or quality claim is made; stochastic output lengths affect completion latency. This remains development evidence, not final release qualification.
