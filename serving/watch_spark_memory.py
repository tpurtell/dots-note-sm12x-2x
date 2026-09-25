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
if args.min_available_gib <= 0:
    parser.error('min-available-gib must be positive')
# Bind this monitor to the launched instance; never follow a replacement that
# later reuses the same container name.
container_id = subprocess.check_output(
    ['docker', 'inspect', '--format', '{{.Id}}', args.container], text=True, timeout=5
).strip()
low_samples = 0
while True:
    try:
        state = subprocess.run(['docker', 'inspect', '--format', '{{.State.Status}}', container_id],
                               capture_output=True, text=True, timeout=5)
    except subprocess.TimeoutExpired:
        state = None
        print(json.dumps({'event': 'docker-inspect-timeout', 'container_id': container_id}), flush=True)
    if state is not None and (state.returncode or state.stdout.strip() != 'running'):
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
        try:
            result = subprocess.run(['docker', 'kill', container_id], capture_output=True, text=True, timeout=15)
            print(json.dumps({'event': 'container-kill', 'returncode': result.returncode,
                              'stdout': result.stdout, 'stderr': result.stderr}), flush=True)
            if result.returncode == 0:
                raise SystemExit(1)
        except subprocess.TimeoutExpired:
            print(json.dumps({'event': 'docker-kill-timeout', 'container_id': container_id}), flush=True)
    time.sleep(1)
