# RTX prefill versus batch-token admission

Completed portion only: batch512 baseline and rejected2048/4096. Batch1024 is still measuring and has no result in this archive. All use the exact profiling image `e54f202f`, cut17, native524288 context and recorded launch settings.

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

Unique exact-length prompts avoid prefix reuse. Rate includes tokenization and first-token handoff. Raw startup logs retain allocator OOM diagnostics;512 subsequently became ready and completed all requests. This does not claim error-free startup or final-release qualification. Exact raw samples, memory profiles, arguments and hashes are archived.
