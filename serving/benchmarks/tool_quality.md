# Pinned tool-use quality gate

This wrapper reproduces the Qwen reference recipe's public suite at commit `cf54b4bfe705f12f71e8866f10730572497c8105`: 69 Basic, 19 Hard, 88 Total. It uses Hard Mode, one trial, temperature0, reference date2026-09-07, eight turns, 900-second request timeout, thinking enabled, and default concurrency8. Use the same concurrency/reference date for both final platforms. No held-out packs are used.

The existing sibling checkout can be a different revision. Keep it unchanged:

```bash
git -C ../tool-eval-bench fetch https://github.com/SeraphimSerapis/tool-eval-bench.git cf54b4bfe705f12f71e8866f10730572497c8105
git -C ../tool-eval-bench worktree add --detach "$PWD/.cache/tool-eval-bench-cf54b4b" cf54b4bfe705f12f71e8866f10730572497c8105
UV_CACHE_DIR="$PWD/.cache/uv" uv sync --project .cache/tool-eval-bench-cf54b4b --frozen --no-dev
python3 serving/benchmarks/tool_quality.py --output .cache/qualification/rtx/tools --plan
```

After the final image/profile is healthy and the platform owner has reserved its endpoint:

```bash
python3 serving/benchmarks/tool_quality.py \
  --base-url http://127.0.0.1:8001/v1 \
  --model dots3-note-exl3-k4 \
  --output .cache/qualification/rtx/tools
```

For Spark use its qualified endpoint and a distinct output directory. The wrapper never starts/stops a model or downloads checkpoint weights. Output directories must be fresh. Setup and plan commands do not send endpoint traffic; running without `--plan` does.

## Evidence and qualification integration

The run directory owns its SQLite database (`data/benchmarks.sqlite`) and upstream Markdown traces. It preserves full `tools.json`, `tools.md`, CLI log, exit status, exact command/pin/lock/wrapper hashes, and Basic/Hard/Total summaries. `receipt.json` hashes every raw artifact. The exporter validates exactly TC-01 through TC-88, upstream point totals, partial credit and exclusions. It refuses a complete qualification receipt when upstream excluded infrastructure failures; all raw results remain available. Model quality failures are reported honestly and do not masquerade as harness errors.

Both final qualification runners now invoke this as a mandatory v2 stage after binding image/profile identity, use `tool_quality.validate(json.loads(tools.json))` to validate stage output, require `complete == True`, and archive the entire stage directory (including SQLite and Markdown). Match before/after image/config/runtime identities using the existing platform qualification controls. Do not add this stage to a currently running hash-bound qualification manifest. `--export-only --output EXISTING` can reconstruct split summaries from an already completed run without endpoint requests.

Scores are earned points divided by graded maximum points, with pass/partial/fail counts separately preserved. Basic and Hard percentages cannot be averaged equally: Total is weighted by the 69/19 scenario counts. The upstream rounded overall score and safety warnings remain in the JSON summary. Report endpoint failures/exclusions explicitly rather than claiming a smaller suite is the full88.
