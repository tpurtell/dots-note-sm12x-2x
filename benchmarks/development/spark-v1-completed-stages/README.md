# Spark v1 completed stages — qualification still running

This is a partial evidence archive, **not an accepted release qualification**. It preserves the already validated seven-workload, sampled-code, concurrent-client, reasoning-coding, tool-quality, context 2048/65536/262144 and retrieval 8192 stages on the published Spark image, plus the explicitly inherited prefix/API/multimodal receipts. Active near-maximum retrieval and unfinished stages were not read or archived.

Both hosts used image `sha256:0492bba788ce02b011c903fa2bf55b5a4a7678b4b1e48250b791b6f327c225b7`. The original qualification manifest and both before-runtime snapshots retain the full image digest, model revision, arguments, environment, allocator policy and container identities. Each accepted stage receipt binds its artifact hash and runtime identity. Inherited stages retain separate source/target identities and never claim execution on this image.

| Completed measurement | Result |
| --- | --- |
| Seven-workload weighted decode | 30.62 tokens/s; 17/21 static contracts |
| Sampled coding depth 0 / 8192 / 24000 | 34.97 / 33.10 / 36.23 tokens/s |
| Concurrent clients C1 / C2 / C4 / C8 / C16 | 21.53 / 37.14 / 56.91 / 88.13 / 136.63 aggregate tokens/s |

These are three-run measurements. Sampled coding uses temperature 0.2, thinking disabled, fixed 256-token outputs; its primary decode timing excludes the first SSE burst. Client throughput is aggregate over the global first-to-last SSE window. Neither fixed-length benchmark establishes natural completion or code correctness. All four seven-workload contract misses remain in the raw evidence and summary.

Inherited functional coverage is prefix/JSON/tool validation, 40 API cases, and one image plus one audio example from the declared prior profiles. See each stage's `evidence_provenance` in [summary.json](summary.json) for differences and limitations.

[manifest.json](manifest.json) lists SHA-256 hashes for all 105 original and losslessly compressed artifacts. [summary.json](summary.json) contains the extracted metrics and exact identities. The archive does not activate the Spark release profile; unfinished qualification must still complete.

## Additional completed stages

The expanded summary includes all 36 measured reasoning-coding responses, Basic/Hard tool results and misses, three measured samples at each newly completed context depth (2048/65536/262144), and all three early/middle/late retrieval checks at 8192 filler tokens. Exact metrics, quality outcomes and timing conventions remain in [summary.json](summary.json) and the lossless raw receipts. Original archived files and their hashes were preserved; this update adds only terminal, validated stages. Near-maximum retrieval remains excluded while active.
