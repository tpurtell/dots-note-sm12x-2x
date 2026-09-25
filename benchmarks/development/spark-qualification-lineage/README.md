# Spark qualification evidence lineage

This manifest implements the user's instruction to preserve completed tests. It does not assert that prior images or profiles are identical to the published wrapper. The runner archives each original compressed file and decoded raw bytes, checks both hashes, records both hosts' source and target profiles, and exposes their differences in the report.

| Inherited stage | Actual coverage | Source profile limitation |
| --- | --- | --- |
| Prefix | Four-request prefix/JSON/tool check | Prior 17/29 image, fused pack and shared overlap disabled |
| Reasoning API | 40/40 cases | Prior 17/29 image before post-warm allocator policy |
| Multimodal | One image and one audio check | Prior 23/23 image; later partition, grouping and execution changes disclosed |
| 8K / 32K / 128K prefill | Three samples per depth, one output token each | Prior 17/29 image; TTFT/prefill only, no decode measurement |
| Exact 524K boundary | One sample: 524032 input + 256 output | Diagnostic image with GC 0.9 / fraction 0.9; later metadata/sampler and image differences disclosed |

All source artifacts are committed archives. The boundary source did not report cached-token usage; the report must not turn this into a zero-cache claim. Component receipts are supporting evidence, not substitutes for a repeated final-image functional test.

Pass this manifest to the existing runner, alongside the actual final image ID and selected profile:

```bash
python3 serving/release/qualify_spark.py \
  --expected-image-id "$SPARK_IMAGE_ID" --expected-mtp 3 \
  --max-model-len 524288 --gpu-memory-utilization 0.80 \
  --inherit-evidence benchmarks/development/spark-qualification-lineage/manifest.json \
  --output-dir .cache/release/spark-published-qualification
```

The default is plan-only; add `--execute` only when the final containers are ready. The inherited stages send no requests. The remaining plan contains seven workloads, sampled code depths, concurrent clients, reasoning coding, Basic/Hard tools, context depths 2048/65536/262144, and early/middle/late retrieval at 8192/522144 filler tokens. The plan counts 349 new request/task entries including warmups (tool tasks can make multiple model calls).

CPU verification: six rejection/lineage tests plus materialization, hash validation, and report metric extraction against all seven real archived stage sources. No endpoint, SSH, Docker, or GPU traffic was used to create this manifest.
