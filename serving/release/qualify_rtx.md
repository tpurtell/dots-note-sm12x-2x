# Final RTX release qualification

`qualify_rtx.py` runs the existing benchmark CLIs sequentially against an already
running final container. It never launches or changes that container. Reserve
both GPUs and the API for this run; concurrent traffic invalidates timing and
prefix/MTP counter attribution.

First inspect the plan (no Docker, GPU, or API access):

```bash
python3 serving/release/qualify_rtx.py \
  --container dots3-rtx \
  --expected-image-id sha256:REPLACE_WITH_IMMUTABLE_LOCAL_IMAGE_ID \
  --expected-mtp 3 \
  --base-url http://127.0.0.1:8001 \
  --output-dir .cache/release/rtx-final
```

Set the selected final MTP value and actual image ID, then append `--execute`.
The script requires explicit TP2, `--max-model-len 262144`, and the expected MTP
in the running container arguments. The selected recipe must have its reasoning
parser, tools, xgrammar, multimodal support, and prefix caching enabled.

## Coverage

| Stage | Generation requests, including warmups |
| --- | ---: |
| Prefix reuse, JSON schema, forced tool | 4 |
| Reasoning API: thinking on/off, stream/nonstream, two turns, JSON, four tool choices and tool-result round trips; two repetitions | 80 |
| Image and audio examples | 2 |
| Seven workloads: one warmup plus three measured runs | 28 |
| Reference code-agent task at baseline/8K/24K: one warmup plus three runs | 12 |
| Independent prose clients C1/2/4/8/16: two warmup waves plus three measured waves, 128 output tokens | 155 |
| Reasoning coding C1/2/4: four tasks, one warmup plus three runs, natural stop within 8192 tokens | 48 |
| Context at 2048/8192/32768/65536/131072/261888: one warmup plus three runs, 256 output tokens | 24 |
| Retrieval at 8192 and 260000 filler tokens, early/middle/late | 6 |
| **Total** | **359** |

Tokenization, metrics, and runtime inspection add non-generation requests.
Multimodal uses the existing script's public example URLs. Retrieval positions
are character fractions, and its actual chat prompt lengths are recorded; 260000
is filler length, not a claim of exactly 260000 total prompt tokens.

The 261888-prompt + 256-output context point checks exactly **262144 total tokens**.
It is an acceptance test at that limit, not a rejection test beyond it.
Prefill throughput is derived as exact prompt length divided by TTFT from the
same unique-prefix context requests. This includes the first-token handoff;
it is not a kernel-only throughput measure. Avoiding a second prefill sweep
saves repeated long-context work. Reasoning API already covers all tool modes,
so the runner does not repeat `tools.py`.

## Evidence and resuming

Every stage stores the exact command, stdout/stderr, raw output, and a receipt
with its artifact hash and validated completion. Receipts are written only after
successful exit, structural validation, and unchanged container identity.
A repeated identical command skips only stages with valid receipts and unchanged
artifact hashes. Interrupted or failed attempts remain intact; their stage is
retried into a new attempt directory. There is no reuse of results from another
image, restarted container, changed source, or changed plan.

The root manifest binds the image ID, container ID/start time/restart count,
server arguments, environment hash, all benchmark source hashes, and plan.
Each invocation records runtime snapshots, full container logs and metrics before
and after, plus GPU memory/utilization/power and host memory every five seconds.
A final `attempt-*/complete.json` is written only after the ending snapshot
succeeds. It contains derived context curves and links to raw receipts.

Execution, token-evidence, retrieval, prefix, multimodal, and reasoning API gate
failures stop the run. Seven-suite and coding static-content misses are retained
as quality results; completing this runner does **not** mean all generated code
is correct or every content contract passed. No generated code is executed.
