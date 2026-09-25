# GB10 dense projection and vocabulary probes

These component results use actual accepted hybrid-checkpoint weights, TP2
shards, changed-input graph checks and interleaved GPU timings. They do not
establish whole-model gains. Ratios below are native time / B12x time; greater
than one favors B12x. Raw measurements and hashes are in `manifest.json`.

| Projection | TP shard | M1 | M2 | M4 | M8 | M16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DSA Q-B | Rhea rank0, 12288×1024 | 0.568 | 0.393 | 0.296 | 0.287 | 0.892 |
| SWA Q-B | Moa rank1, 8192×1024 | 1.054 | 1.699 | 1.405 | 1.167 | 1.025 |
| DSA output | Rhea rank0 | 1.192 | — | 1.203 | — | 1.174 |
| SWA output | Moa rank1 | 0.638 | — | 0.786 | — | 1.214 |

The source-scale B12x activation quantization differs from native UE8M0
rounding. Both source-reference errors are reported in the raw receipts.
Geometry-specific optional integration needs its own model quality/performance
measurement; blanket Q-B replacement would include the slow DSA path.

Single-token BF16 vocabulary medians were 3055.97/3159.36 µs B12x versus
3172.88/3225.84 µs native on rank0/rank1. Top-20 selections matched; adapter
graphs with changed input and weight matched exactly. Shared plans retained
distinct output storage and released 778,567,680 bytes of obsolete draft-head
weight per rank. This path only handles one token; larger batches use native
projection.
