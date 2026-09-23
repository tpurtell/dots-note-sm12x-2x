#!/usr/bin/env bash
set -euo pipefail

# Run on rhea after dots3-quant-rhea has exited successfully. All paths are
# local ext4 paths; the alternate /mnt/scratch Hugging Face cache is read-only.
if [[ "$(hostname -s)" != rhea ]]; then
  echo "Export acceptance must run on rhea" >&2
  exit 2
fi
work_root="${WORK_ROOT:-/home/tj/dots-note-work}"
fp8_source="${FP8_SOURCE:-/home/tj/dots-note-source/fp8}"
export_root="$work_root/export"
report_root="$work_root/state/acceptance"
recipe_root="$work_root/recipe/quantization"

container_state="$(docker inspect dots3-quant-rhea --format '{{.State.Status}} {{.State.ExitCode}}')"
if [[ "$container_state" != 'exited 0' ]]; then
  echo "Quant container has not completed successfully: $container_state" >&2
  exit 1
fi
if [[ ! -f "$export_root/model.safetensors.index.json" ]]; then
  echo "Quantized export has no safetensors index" >&2
  exit 1
fi
mkdir -p "$report_root"
python3 "$recipe_root/summarize_errors.py" \
  --journal "$work_root/state/errors.jsonl" \
  --output "$report_root/error-summary.json" --complete
python3 "$recipe_root/finalize_export.py" \
  --fp8-source "$fp8_source" --output "$export_root" \
  --report "$report_root/assets.json"
python3 "$recipe_root/audit_export.py" \
  --fp8-source "$fp8_source" --output "$export_root" \
  --report "$report_root/tensors.json"
cp "$recipe_root/model_card.md" "$export_root/README.md"
python3 "$recipe_root/hash_artifact.py" \
  --artifact "$export_root" --report "$report_root/artifact-files.json"
echo "Accepted local export: $export_root"
