# RTX v2 qualification

[Measured report](report.json) for the immutable public RTX image:

`ghcr.io/tpurtell/dots3-note-exl3-k4-rtx@sha256:d350ceb8c9be1dce3851ab20fba4c586f1530bef0a65a7094305b4ee8d2df16e`

The run was interrupted by GPU bus loss and explicitly continued after a user-coordinated host reboot. The user requested that finished tests not be repeated. **Fourteen completed stages were preserved unchanged; only the three unfinished context/retrieval stages ran afterward.** The report contains the old/new lifecycle identities, immutable restart ledger, prior-file hashes, failed attempt, and continuation records. Image, model, runtime profile, and benchmark/validator sources remained unchanged. No temperature, power, or cooling configuration changes were made.

- Native524K context: three exact524032+256 measurements completed.
- Retrieval: early/middle/late positions passed at both8K and nearmax522K,6/6.
- Coding/reasoning:36/36 natural completions,36/36 static response checks; generated code was not executed.
- General workload contracts18/21; all misses remain in the report.
- Official tool score147/176,83.52%; Basic84.06%, Hard81.58%. [Diagnostic grader caveats](../../development/rtx-tool-quality-audit/README.md) do not change the official score.

[Interruption evidence and audited recovery](../../development/rtx-native-v2-qualification-interrupted/README.md). The raw gzip artifacts are lossless and individually hashed in [archive-manifest.json](archive-manifest.json). Public launcher lifecycle checks are recorded separately; they do not repeat the benchmark suite.

## Public launcher lifecycle

[Lifecycle evidence](fastpath/manifest.json): public pull, start with an initially empty runtime cache, health, status, logs, stop, and restart all passed. Zero benchmark requests were sent. The verified container `dots3-vllm-rtx-v2-fastpath` remains running on port8001; use `CONTAINER_NAME=dots3-vllm-rtx-v2-fastpath ./serving/release/run.sh rtx status` to inspect it.
