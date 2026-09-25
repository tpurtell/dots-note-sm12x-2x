# Full-head sparse MLA component gates

B12x `bf5677c69197499433314d61aad70f79e89c47c9` permits TP1/128 heads while
preserving TP2/64 and TP8/16. The Dots3 adapter derives its local geometry from
its actual head count and keeps distinct shared workspaces for 64/128 heads.
This supports owner attention without changing expert placement.

Eight CPU metadata tests passed. Each physical RTX GPU passed 22 full-head
kernel tests across compact/padded records, interleaved/actual/stress layouts,
changed-input graphs and addresses beyond 2 GiB. Each also passed the actual
128-head vLLM adapter gate.

The initial combined commands exited 2 **after all kernel tests passed** because
the following adapter invocation used `--heads128`. Separate corrected
`--heads 128` invocations passed on both GPUs. Both the failed command logs and
corrected logs are archived; no failed exit is relabeled as a successful command.

The [manifest](manifest.json) binds image, source overlays, exact commands,
per-device outcomes and lossless hashes. These are component results; they do
not establish full-model hybrid correctness or performance.
