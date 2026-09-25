# Spark long-prefill allocator growth

This bounded diagnostic reproduces the memory growth from the failed native
524K boundary. It is not a completed performance or maximum-context test.
The diagnostic image adds worker telemetry to the native parent; it does not
change kernels or allocator policy. Static utilization remains 0.80, batch 512,
decoder cut 17/29, routed-expert TP2, and the physical-memory guard remains 1 GiB.

The controller submitted one 524,032-input / 256-output request and deliberately
stopped both containers when either host crossed 2 GiB available, before the
guard fired. It recorded progressing KV occupancy. No output was completed.
The 2 GiB threshold is a diagnostic early stop, not a new serving reserve.

During the approximately 213-second external monitoring window:

| Counter | Rhea | Moa |
|---|---:|---:|
| CUDA reserved, first → last aligned sample | 90.05 → 100.09 GiB | 92.40 → 102.13 GiB |
| Live allocated, min → max | 88.51 → 88.78 GiB | 91.19 → 91.47 GiB |
| Inactive split memory, first → last | 1.35 → 1.33 GiB | 0.96 → 0.87 GiB |

Worker CPU resident memory stayed approximately stable. Growing reservations
without growing live allocations or inactive splits identify accumulation of
freed, unsplit CUDA allocator segments. This supports testing proactive cache
reclamation rather than reducing the static model/KV utilization or declaring a
live-tensor leak. It does not yet identify every individual allocating call.

Native vLLM indexer prefill creates changing-size logits arrays, bounded by its
512 MiB logits budget; this is a candidate source of allocation churn. Shared
gather workspace is reused. Actual live bounds alone do not bound allocator
reservations on the unified-memory hosts.

[Summary](summary.json) uses only worker samples inside the external controller's
time window. [Manifest](manifest.json) binds lossless compressed telemetry,
controller, diagnostic extension, request output and failure log. Slightly
different endpoints from the full worker log reflect sample alignment.

Proactive reclamation is being investigated. PyTorch's native allocator
`garbage_collection_threshold:0.8` is not sufficient alone: its implementation
also requires an explicitly configured per-process memory maximum. An environment
setting that parses successfully does not prove reclamation is enabled. No such
policy was active in this diagnostic. Acceptance of a supported allocator policy
requires the unfinished actual 524K boundary to fit physical memory; the
allocation diagnosis alone does not qualify the candidate or release.
