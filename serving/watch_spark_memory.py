#!/usr/bin/env python3
"""Stop this recipe's container if Spark host memory headroom is exhausted."""
import argparse
import json
import subprocess
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--container', choices=['dots3-vllm-head', 'dots3-vllm-worker'], required=True)
parser.add_argument('--min-available-gib', type=float, default=8)
args = parser.parse_args()
threshold = int(args.min_available_gib * 1024**3)
low_samples = 0
while True:
    state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', args.container],
                           capture_output=True, text=True, timeout=5)
    if state.returncode or state.stdout.strip() != 'running':
        print(json.dumps({'event': 'container-stopped', 'state': state.stdout.strip()}), flush=True)
        break
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    available = values['MemAvailable']
    low_samples = low_samples + 1 if available < threshold else 0
    print(json.dumps({'time': time.time(), 'available_bytes': available,
                      'swap_used_bytes': values['SwapTotal'] - values['SwapFree'],
                      'low_samples': low_samples}), flush=True)
    if low_samples >= 3:
        print(json.dumps({'event': 'memory-headroom-exhausted', 'container': args.container,
                          'threshold_bytes': threshold}), flush=True)
        result = subprocess.run(['docker', 'kill', args.container], capture_output=True, text=True, timeout=15)
        print(json.dumps({'event': 'container-kill', 'returncode': result.returncode,
                          'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
        raise SystemExit(1)
    time.sleep(1)
