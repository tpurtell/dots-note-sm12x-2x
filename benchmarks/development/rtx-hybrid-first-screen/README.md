# First RTX hybrid owner-attention screen

Development evidence only; matched native TP2 comparison and final qualification remain separate.

Image `f4c227423bff15dee498e1ba504b357f34c9013b175584b013d7a2ebdd8bd549`, hybrid source `347a470`, B12x `bf5677c6`. Partition `23,23`, explicit block size 64, native 524288 context, memory utilization .95, batch tokens 512, MTP3, compact cache enabled, indexer prefill cap 4.

- Reasoning/tools/JSON API: **40/40 passed**.
- Coding: **12/12 natural completions and static checks passed**, no truncations.
- Median per-request decode tokens/s: C1 **154.955**, C2 **130.538**, C4 **97.973**.
- Prefix receipt and full raw runtime/startup/container evidence are preserved.

The coding output budget is a per-request benchmark constraint, not a server output limit. These small stochastic task samples do not establish a causal quality improvement or a speed comparison with another profile.

Earlier startup diagnostics remain included: v1 failed the parameter-free nonowner MTP completeness check; v2 required explicit block size 64 on both ranks. Neither attempt is counted as a successful functional run.

All raw artifacts are losslessly gzip-compressed. `manifest.json` records original and compressed SHA-256 hashes, exact launch arguments and selected environment. Container state is the historical capture, not current liveness.
