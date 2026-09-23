#!/usr/bin/env bash
set -euo pipefail

# A low-overhead watcher for a long Spark run. Start with nohup on rhea.
if [[ "$(hostname -s)" != rhea ]]; then
  echo "Quantization watcher must run on rhea" >&2
  exit 2
fi
recipe_root=/home/tj/dots-note-work/recipe/quantization
while true; do
  state="$(docker inspect dots3-quant-rhea --format '{{.State.Status}} {{.State.ExitCode}}')"
  case "$state" in
    'exited 0') break ;;
    running\ *|created\ *|restarting\ *) sleep 60 ;;
    *) echo "Quant container ended without success: $state" >&2; exit 1 ;;
  esac
done
bash "$recipe_root/accept_export.sh"
bash "$recipe_root/publish_export.sh"
