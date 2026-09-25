# Native TP2: 524,288-context development gate

The compact-cache candidate with `DOTS3_COMPACT_DSA_CACHE=1` and
`DOTS3_INDEXER_PREFILL_CONTEXTS=4` started successfully, passed repeated-prefix
and structured-output checks, and completed an exact **524,032 input + 256
output = 524,288 total-token** request. Prefix reuse recorded 3,520 hits from
3,612 queried tokens.

The single boundary sample measured 212.550 seconds TTFT and 228.723 decode
tokens/s. This is one development sample, not a final performance distribution.
Attention remained native TP2; hybrid owner attention was not active.

The [manifest](manifest.json) preserves image/container identity, full launch
arguments, selected environment and lossless hashes for prefix, boundary,
startup and container-inspection receipts. The container snapshot was captured
while running; root subsequently stopped the model safely. Final native-context
retrieval, full release qualification, and hybrid comparisons remain separate.
