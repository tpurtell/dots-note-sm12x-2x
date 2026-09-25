# RTX prefill versus batch-token admission

Completed prefill: batch512 baseline and batch1024, plus rejected2048/4096 admissions. Batch1024 coding is still active and is not included; no final profile decision is claimed. All use the exact profiling image `e54f202f`, cut17, native524288 context and recorded launch settings.

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
