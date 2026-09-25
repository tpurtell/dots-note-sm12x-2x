# Archive a completed qualification

`report.py` reads local files only. It requires the runner's `manifest.json`,
a terminal `attempt-*/complete.json`, every accepted stage receipt, matching
artifact hashes, and the exact source checkout recorded by the runner. It
re-runs the CPU stage validators and refuses incomplete runs before creating
output. It never runs requests or changes release settings.

Once the full RTX runner finishes:

```bash
python3 serving/release/report.py \
  --input .cache/release/rtx-published-qualification \
  --output benchmarks/releases/rtx-20260925-v1 \
  --platform rtx \
  --image ghcr.io/tpurtell/dots3-note-exl3-k4-rtx@sha256:2aff95d9896b3f3d3f8e3f4cfbaed90c77344d41d58fe7322eff9f4c73ec63b6 \
  --cache-manifest .cache/release/rtx-final-seed/manifest.json \
  --registry-receipt benchmarks/development/rtx-release-wrapper/manifest.json
```

For Spark, use its completed runner directory, `--platform spark`, actual
published digest, and merged cache manifest. Its runner binds repository
digests on both hosts; an extra registry receipt is optional. No Spark digest
is supplied here before publication.

Output is a new directory containing `report.json`, `archive-manifest.json`,
and lossless deterministic gzip archives of accepted raw artifacts, command
records, logs, runtime/memory snapshots, cache provenance and bound scripts.
The archive manifest hashes both original and compressed bytes and the report.
Source files are archived so later reviews can reconstruct the exact validator
checkout. The exporter refuses source drift rather than using changed checks.

The report includes seven workloads, fixed-length code-agent timing, sampled
prose concurrency, reasoning-enabled coding, context/prefill curves, retrieval,
prefix hits, reasoning/grammar/tool gates and multimodal outcomes. Quality
misses and truncations remain explicit. Fixed-length throughput probes do not
prove natural completion. Client TTFT includes first-token handoff.

The report means the runner completed its declared plan. Publishing a public
package, verifying anonymous access, approving final settings and populating
README tables remain separate steps. An older private registry receipt stays
labeled with its original observation; add fresh public-access evidence when
available. Do not reinterpret authenticated pull as anonymous success.

## Timing and README data mapping

| Promised measurement | Report source and interpretation |
| --- | --- |
| C1 seven-workload weighted decode | `stage_results.seven.weighted_decode_tps`; measured decode tokens divided by measured decode seconds across the seven cases and three runs |
| C1 greedy `merge_intervals` | `stage_results.seven.median_tps_by_case.code`; temperature0, thinking disabled, first SSE burst excluded; report its contract outcomes |
| C1 sampled async coding baseline | `stage_results.code-agent.points` entry with `depth=0`; temperature0.2, thinking disabled, forced256-token completion |
| Sampled async coding timing convention | `decode_tokens_per_second_median` excludes the entire initial SSE burst. `reference_n_minus_one_tokens_per_second.median` instead uses `(output_tokens-1)/elapsed`, as in the older reference recipe. Name the convention explicitly; speculative first bursts can contain multiple tokens. |
| C16 aggregate sampled prose | `stage_results.clients.points` entry with `concurrency=16`; also report `minimum_overlap` |
| Reasoning-enabled coding C1/C2/C4 | `stage_results.coding.by_concurrency`; use completed-only latency with completion/truncation counts and the additional distributions below |
| Context/prefill | Each `stage_results.context-N.points` row; actual prompt count / TTFT includes first-token handoff. The final boundary is actual prompt+completion tokens, not a nominal model configuration alone. |

The sampled async coding headline refers to the reference task at baseline depth,
not the separate reasoning-enabled `async_pool` task. Neither fixed-length coding
probe establishes completed-code correctness. The report preserves both timing
metrics so older reference numbers are not silently mixed with burst-aware rates.

Reasoning coding adds `distributions_by_concurrency` with sample count, minimum,
median and maximum for output tokens and latency. Natural completions and length
truncations are separate groups; `all_terminal` retains all requests, including
any errors. Output tokens include reasoning and require valid stream/usage token
accounting. A completed-only latency excludes truncated answers and must not be
presented as latency for all attempted requests.

## Memory evidence

`memory_observations` summarizes the actual monitoring files from all runner
attempts, including resumed/failed attempts. It reports minimum physical host
`MemAvailable`, maximum system-wide swap use, and on RTX the observed per-GPU
peak memory use plus memory/utilization/power distributions. NVIDIA memory units
are MiB; host memory values are bytes. These are sampled observations, not
instantaneous peaks or memory attributable solely to the serving process. Monitor
errors and unavailable samples remain visible. Spark's current monitor provides
physical unified-memory samples; no separate GPU-allocation series is inferred.

`hardware.*.startup_memory_evidence` retains source log lines and parsed model
loading GiB, available KV GiB, KV-cache token capacity, and graph-capture GiB where
present. Per-rank and repeated graph phases remain separate; summing them would
misrepresent simultaneous allocation. Original snapshots and monitoring logs are
still archived losslessly.

## Final tool-use qualification

New RTX/Spark runners emit qualification schema v2 and include the mandatory pinned 88-scenario tool-quality stage. Reports require its completed Basic/Hard/Total scoring, all raw trace/SQLite hashes, and stage identity matching the run image/profile. The archive includes nested tool artifacts. Infrastructure exclusions reject completion; model quality misses and partial credit remain measurements. Historical v1 reports remain accepted without retroactively inventing tool-quality results; reconstructing them still requires their recorded source checkout. Plan counts treat this stage as88 scenarios, not88 HTTP requests.

RTX monitor samples now append GPU temperature, SM/memory clocks and software/hardware thermal slowdown flags. Unsupported thermal-field queries fall back to temperature/clocks, preserving the query warning in raw evidence. Reports accept historical six-column and new nine/eleven-column records, skip unavailable optional values, and publish observed ranges plus thermal-active sample counts. These samples can reveal thermal/order differences but do not establish their causal effect on performance.

## Explicit restart continuation

For an interrupted RTX run, `continue_rtx.py` can preserve completed work only with an explicit interruption reason/evidence and matching immutable image, model, profile, and unchanged workload/validator sources. The original manifest/receipts remain untouched. The exporter verifies the immutable restart ledger and all preserved file hashes, attributes each stage to its actual pre/post-restart identity, archives both attempts, and discloses the interruption in `restart_continuation`. This is not an uninterrupted-run claim. A changed profile or workload requires separate qualification.
