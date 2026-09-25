# Final two-Spark qualification

Run this controller from the local recipe project. It invokes the same benchmark
CLIs and imports the common plan/validators from `qualify_rtx.py`; it never edits
the RTX runner, starts containers, changes settings, or runs GPU component tests.
Reserve the Spark API for qualification so timing, prefix hits and MTP counters
are attributable to this workload.

Inspect the plan first (no SSH, Docker or API calls):

```bash
python3 serving/release/qualify_spark.py \
  --expected-image-id sha256:REPLACE_WITH_FINAL_SPARK_IMAGE_ID \
  --expected-mtp 3 \
  --max-model-len 262144 \
  --output-dir .cache/release/spark-final
```

Use the selected final MTP/context values and append `--execute` to run. The
expected image ID must match **both** hosts. The default API URL is
`http://10.55.1.5:8000/v1`; root and `/v1` forms normalize consistently, with the
specific API-base convention required by `clients.py` handled by the shared plan.
SSH uses `BatchMode=yes` and inherits the calling shell's `SSH_AUTH_SOCK` and SSH
configuration. No agent socket or private key is hardcoded.

Defaults:

| Setting | Value |
| --- | --- |
| Head / worker SSH aliases | `rhea` / `moa` |
| Containers | `dots3-vllm-head` / `dots3-vllm-worker` |
| Remote recipe | `/home/tj/dots-note-work/recipe` |
| Expected vLLM memory utilization | `0.80` |
| Minimum physical host `MemAvailable` | `8 GiB` |

Explicit flags can override those host/container/project names, API URL,
`--gpu-memory-utilization`, and `--min-host-available-gib`. The guard threshold
cannot be below8 GiB. Both containers must already use TP2, the selected fixed
MTP depth/context, Dots reasoning/tool parsers, xgrammar, automatic tool choice
and prefix caching. The host guard must be alive, started after the current
container, bound by its command line to that container name, and reporting fresh
samples. Existing launcher guard readiness markers are consumed on startup;
absence is recorded as expected, not treated as failure. A present marker must
match the current container ID. No assertion is made that a consumed marker can
be reconstructed from the filesystem.

## Same benchmark definitions on both platforms

At262144 context, the entire benchmark plan equals the RTX plan: **359 generation
requests**, including warmups. It covers prefix/JSON, the80-case reasoning/tool
API suite, image/audio, three-run seven workloads, sampled reference coding at
baseline/8K/24K, independent prose clients C1/2/4/8/16, reasoning coding C1/2/4,
context curves, and early/middle/late retrieval. Sampling, seeds, prompts, output
budgets and timing definitions are inherited unchanged. Tokenization and metrics
add non-generation requests.

Context curves share2048/8192/32768/65536/131072 prompt lengths. The last point is
`max_model_len - 256`, generating256 tokens to verify the exact total boundary.
For524288, an additional262144 prompt point yields **363 generation requests**.
The final prompt is524032, and the actual prompt+completion total must equal
524288. Retrieval uses8192 and `max_model_len - 2144` filler tokens, each at
character fractions0.05/0.5/0.95. Actual server-counted prompt+output totals are
checked against the selected context limit; filler length is not total chat
prompt length.

Context rows also provide prefill throughput as prompt tokens divided by TTFT,
including the first-token handoff. This avoids a redundant long-context prefill
sweep. Seven/coding static-content misses and natural-stop versus truncation
remain measurements; they are not blanket execution failures or claims that
generated code is behaviorally correct. Explicit API/MM/retrieval contracts and
incomplete token evidence remain gates.

## Runtime evidence and resuming

Before and after the run, both hosts supply the existing `capture_runtime.py`
report, full container/guard logs, image IDs/repository digests, server arguments,
selected NCCL/Gloo/network environment, NIC addresses/link properties, actual
runtime-cache mount, and guard PID/start time/command line/readiness-marker state.
The remote capture helper writes its raw receipt under the remote project's
`.cache/qualification-runtime`; the controller also preserves it locally.
Physical `MemAvailable`, swap totals/free bytes, guard state, and container
identity are sampled every15 seconds via CPU-only SSH inspection. The existing
host guards remain responsible for stopping serving on memory exhaustion. If
inspection detects a failed guard, low headroom or host error, this controller
cancels its benchmark client and stops qualification; it does not kill containers.

Stage layout matches the RTX runner: raw output, exact command, log, and a
hash-bound validated receipt. Identical reruns skip only stages with unchanged
artifacts and the same two container/guard identities, configuration, image
references, source hashes and plan. Failed attempts are retained separately.
No receipts are reused across a container restart or guard replacement.

A successful run writes `attempt-*/complete.json` only after ending snapshots
and monitor checks succeed. It references context receipts for report generation.
A complete marker means the defined gates completed; seven/coding quality misses
must still be reported. It is not a universal zero-JIT or all-prompts-correct claim.
