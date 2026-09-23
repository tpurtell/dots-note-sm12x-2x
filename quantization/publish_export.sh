#!/usr/bin/env bash
set -euo pipefail

# Run on rhea only after accept_export.sh completes successfully.
if [[ "$(hostname -s)" != rhea ]]; then
  echo "Hub publication must run on rhea" >&2
  exit 2
fi
work_root="${WORK_ROOT:-/home/tj/dots-note-work}"
repo_id=wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1
export_root="$work_root/export"
report_root="$work_root/state/acceptance"
manifest="$report_root/artifact-files.json"
recipe_root="$work_root/recipe/quantization"
cache_dir="${HF_CACHE_DIR:-$HOME/.cache/huggingface/hub}"

if [[ ! -s "$manifest" || ! -s "$report_root/tensors.json" || ! -s "$report_root/error-summary.json" ]]; then
  echo "Complete export acceptance reports are required before publication" >&2
  exit 1
fi
python3 "$recipe_root/verify_artifact_files.py" \
  --artifact "$export_root" --manifest "$manifest" \
  --report "$report_root/local-before-upload.json"

# The hf CLI bundles its own Python environment and Hugging Face Hub package.
hf_python="$(dirname "$(readlink -f "$(command -v hf)")")/python3"
if [[ ! -x "$hf_python" ]]; then
  echo "Cannot find the Hugging Face CLI Python environment" >&2
  exit 1
fi
hf repos create "$repo_id" --public --exist-ok
"$hf_python" -c 'from huggingface_hub import HfApi; import sys; assert HfApi().model_info(sys.argv[1]).private is False, "target Hub repo must be public"' "$repo_id"
hf upload "$repo_id" "$export_root" . \
  --commit-message 'Uniform EXL3 MCG K4 routed experts with FP8 core'
revision="$($hf_python -c 'from huggingface_hub import HfApi; import sys; print(HfApi().model_info(sys.argv[1]).sha)' "$repo_id")"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Hub did not return an immutable model revision: $revision" >&2
  exit 1
fi
printf '%s\n' "$revision" > "$report_root/hub-revision.txt"
snapshot="$(hf download "$repo_id" --revision "$revision" --cache-dir "$cache_dir" --format quiet)"
python3 "$recipe_root/verify_artifact_files.py" \
  --artifact "$snapshot" --manifest "$manifest" \
  --report "$report_root/rhea-hub-cache.json" --allow-extra .gitattributes
echo "Published and verified $repo_id@$revision on rhea"
