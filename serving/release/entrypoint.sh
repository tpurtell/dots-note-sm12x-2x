#!/usr/bin/env bash
set -euo pipefail
python3 /opt/dots3/cache_bundle.py seed
exec vllm serve "$@"
