# Interrupted RTX v2 qualification

This is **not a qualified release report**. The immutable public image was undergoing its complete qualification when GPU1 left the PCIe bus during the first 524K context warmup at **2026-09-25 11:30:41 UTC**.

Kernel evidence records **Xid 79, GPU has fallen off the bus**, at PCI `0000:e1:00.0`. Both GPUs then recorded Xid154 with **Node Reboot Required**. CUDA surfaced an unspecified launch failure in the native SWA FlashAttention call; the asynchronous error location does not establish the cause of the bus loss. The final pre-failure sample recorded GPU1 at91°C and406W, without asserted thermal slowdown. These observations do not establish a thermal, power, kernel, or allocator cause.

The poisoned container was stopped. No GPU reset or host reboot was attempted by the agent. Public RTX settings remain on the prior qualified v1 profile. The published v2 image is not yet activated.

## Preserved measurements

All14 completed stage receipts, raw outputs, full tool traces/database, runtime snapshots, continuous monitoring, failed-stage output, runner logs, and kernel failure window are archived losslessly under `raw/`; every input and gzip has a SHA256 in `manifest.json`.

- Reasoning/API80/80; prefix and image/audio gates passed.
- Seven workload contracts18/21; misses preserved.
- Coding36/36 natural completions and static response checks; C1/C2/C4 median per-request decode163.02/129.26/98.80tokens/s. Generated code was not executed.
- Tool score147/176=83.52%: Basic116/138, Hard31/38, with no excluded scenarios. [Diagnostic grader caveats](../rtx-tool-quality-audit/README.md) do not alter the official score.
- Context stages through262144 completed. The524032 warmup failed before any completed measurement. Both retrieval stages remained unexecuted.

The earlier native-parent single524K request passed, but it does not replace the interrupted published-image qualification.

## Recovery after the user reboots the host

Do not run these commands before host recovery. First verify both original GPUs enumerate normally and inspect new boot kernel logs for continuing faults. Do not change the selected runtime profile silently.

The original stopped container retains the exact image digest and command:

```bash
nvidia-smi
journalctl -k -b --no-pager | rg 'NVRM|Xid'
docker start dots3-vllm-rtx-v2-published
```

Wait for `/health` to succeed, inspect startup logs/ownership, and capture a new runtime receipt. The current launcher cache path is `.cache/release/rtx-native-v2/published/runtime`; only compiled caches survive the restart, while model/KV state starts fresh.

```bash
curl --fail http://127.0.0.1:8001/health
python3 serving/capture_runtime.py dots3-vllm-rtx-v2-published \
  --output .cache/release/rtx-native-v2/published/runtime-after-reboot.json
python3 serving/release/qualify_rtx.py \
  --container dots3-vllm-rtx-v2-published \
  --expected-image-id sha256:d350ceb8c9be1dce3851ab20fba4c586f1530bef0a65a7094305b4ee8d2df16e \
  --expected-mtp 3 --max-model-len 524288 \
  --base-url http://127.0.0.1:8001 \
  --output-dir .cache/release/rtx-published-v2-qualification-after-reboot \
  --execute
```

**Run the complete suite again in this new output directory.** The runner correctly binds container start/restart identity; do not bypass that gate or rewrite the interrupted manifest. Do not combine prior completed stages with a restarted engine as a single uninterrupted qualification.

After a terminal completion receipt, export the report using the existing registry receipt and cache manifest:

```bash
python3 serving/release/report.py \
  --input .cache/release/rtx-published-v2-qualification-after-reboot \
  --output benchmarks/releases/rtx-20260925-v2 \
  --platform rtx \
  --image ghcr.io/tpurtell/dots3-note-exl3-k4-rtx@sha256:d350ceb8c9be1dce3851ab20fba4c586f1530bef0a65a7094305b4ee8d2df16e \
  --cache-manifest .cache/release/rtx-native-v2/seed/manifest.json \
  --registry-receipt .cache/release/rtx-native-v2/registry-receipt.json
```

RTX-only settings activation and the fresh-cache public `run.sh` lifecycle gate remain pending. Preserve the Spark settings independently.
