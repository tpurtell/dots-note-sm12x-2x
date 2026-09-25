# Spark post-warm allocator reclamation boundary

One completed diagnostic-image boundary: **524,032 input + 256 output = 524,288**,
TTFT **1,071.070 s**, decode **39.190 tokens/s**. Both hosts used exact image
`sha256:8ff5bd07a3bf10af89f7cbab47b2fe54f066a69df7bcb870e5641809e4389c03`.
This is development evidence, **not final published-digest qualification**.

The native allocator was configured after compile/graph warmup with fraction
0.90 and garbage collection threshold 0.90. Actual before/after settings,
per-process ceilings and worker diagnostics are retained. The earlier startup
ownership receipt precedes this hook; actual policy proof here comes from the
post-warm log JSON, not a fabricated field in that earlier receipt.

External request-window minimum physical headroom was 3.955 GiB on Rhea and
5.334 GiB on Moa. `summary.json` also gives minima from the complete one-second
guard logs, including startup, and their exact sample counts. Sampled minima
are not continuous guarantees. Boundary cached-token accounting was omitted
(`null`); unique nonce/hash retained. The forced 256-token continuation is not
natural-completion or retrieval-quality evidence.

Actual capacity was 3,586,490 tokens, versus approximately 2.79M in an older
configuration. This confound prevents interpreting the result as an isolated
GC-only performance/capacity comparison. Production adds an atomic per-rank
policy receipt and removes the diagnostic sampling thread; final-image identity
and inheritance must be documented separately, not silently substituted here.

`manifest.json` records raw and gzip SHA256 hashes for boundary output, complete
container/startup/guard logs, runtime and Docker inspection, actual ownership and
policy, rank allocator telemetry, launch/monitor scripts and diagnostic build
sources/logs. No executable caches are included.
