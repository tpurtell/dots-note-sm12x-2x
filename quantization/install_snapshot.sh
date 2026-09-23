#!/usr/bin/env bash
set -euo pipefail

# Install and byte-verify the immutable published model revision on any host.
revision="${1:?Usage: install_snapshot.sh HUB_REVISION ACCEPTED_HASH_MANIFEST CACHE_DIR [REPORT]}"
manifest="${2:?Usage: install_snapshot.sh HUB_REVISION ACCEPTED_HASH_MANIFEST CACHE_DIR [REPORT]}"
cache_dir="${3:?Usage: install_snapshot.sh HUB_REVISION ACCEPTED_HASH_MANIFEST CACHE_DIR [REPORT]}"
report="${4:-}"
if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
  echo "An immutable 40-character Hub revision is required" >&2
  exit 2
fi
if [[ ! -s "$manifest" ]]; then
  echo "Accepted artifact hash manifest is missing: $manifest" >&2
  exit 2
fi
case "$(realpath -m "$cache_dir")" in
  /mnt/scratch|/mnt/scratch/*)
    echo "Refusing to write to the read-only NTFS scratch disk" >&2
    exit 2 ;;
esac
mkdir -p "$cache_dir"
repo_id=wrldsuksgo2mars/dots3-note-prev-exl3-k4-v1
snapshot="$(hf download "$repo_id" --revision "$revision" \
  --cache-dir "$cache_dir" --format quiet)"
args=(--artifact "$snapshot" --manifest "$manifest" --allow-extra .gitattributes)
if [[ -n "$report" ]]; then
  args+=(--report "$report")
fi
python3 "$(dirname "$0")/verify_artifact_files.py" "${args[@]}"
echo "Installed $repo_id@$revision at $snapshot"
