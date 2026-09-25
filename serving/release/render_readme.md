# Render a release README draft

After each immutable published-image qualification has a completed `report.json`:

```bash
python3 serving/release/render_readme.py \
  --rtx benchmarks/releases/RTX_VERSION/report.json \
  --spark benchmarks/releases/SPARK_VERSION/report.json \
  --output .cache/release/README.dual-platform.draft.md
```

Omit a platform until its report exists. It stays pending; no measurements or
image digests are copied from the other platform. Historical v1 reports remain
readable and explicitly labeled; they cannot supply unmeasured hard-mode scores.
The tool refuses to overwrite the root README directly.

Review the draft against both report hashes, activate the matching settings only
after qualification, verify the launch commands, then merge the final content
into README. This renderer does not activate settings or verify registry access.
Pull-by-digest requires GHCR login while a package is private. The final Spark
instructions must use the same ARM image on worker and head, worker first.

The draft includes workload contracts and truncations, coding lengths/completed
latency, sampled-client overlap, burst-excluded code-agent timing, context curves,
Basic/Hard/Total tool quality, functional checks, and sampled memory. Raw reports
remain authoritative. These tables distinguish synthetic throughput and static
content checks from execution-based coding correctness.
