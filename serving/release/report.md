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
