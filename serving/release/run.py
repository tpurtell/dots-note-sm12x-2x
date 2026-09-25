#!/usr/bin/env python3
"""Run the existing launchers using qualified immutable release settings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import urllib.request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
MODEL_REVISION = 'd8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da'


def fail(message):
    raise SystemExit(message)


def settings(path, selected):
    doc = json.loads(path.read_text())
    if doc.get('schema') != 1 or doc.get('status') != 'qualified':
        fail('Release is not finalized. GHCR digests and qualified settings must be filled in after release-image verification.')
    if doc.get('model_revision') != MODEL_REVISION:
        fail('Unexpected checkpoint revision')
    config = doc['platforms'][selected]
    if selected == 'rtx':
        # Older RTX profiles predate the Spark-only transport fields.
        config = dict(config)
        config.setdefault('b12x_roce', False)
        config.setdefault('b12x_roce_eager', False)
        config.setdefault('b12x_roce_rows', list(range(1, 65)))
    expected = {'rtx': 'amd64', 'spark': 'arm64'}[selected]
    if config.get('architecture') != expected:
        fail('Release platform architecture mismatch')
    image = config.get('image')
    if not isinstance(image, str) or not re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}', image):
        fail('Release requires an immutable GHCR image digest, not a mutable tag')
    report = config.get('qualification_report')
    report_hash = config.get('qualification_report_sha256')
    if not isinstance(report, str) or not report or not isinstance(report_hash, str) or not re.fullmatch('[a-f0-9]{64}', report_hash):
        fail('Release requires a repository qualification report and SHA256')
    report_path = (PROJECT / report).resolve()
    if not report_path.is_relative_to(PROJECT) or not report_path.is_file() or hashlib.sha256(report_path.read_bytes()).hexdigest() != report_hash:
        fail('Qualification report missing or hash mismatch')
    memory = config.get('gpu_memory_utilization')
    if type(memory) not in (int, float) or not 0 < memory < 1:
        fail('Invalid qualified GPU memory utilization')
    for name in ['max_model_len', 'max_num_seqs', 'max_num_batched_tokens', 'port']:
        if type(config.get(name)) is not int or config[name] <= 0:
            fail(f'Invalid qualified {name}')
    if config['port'] > 65535 or config.get('kv_cache_dtype') != 'fp8':
        fail('Invalid release port or FP8 KV setting')
    if type(config.get('mtp_tokens')) is not int or config['mtp_tokens'] < 0:
        fail('mtp_tokens must be 0 (disabled) or a positive integer')
    for name in ['b12x_vocab', 'b12x_pcie', 'b12x_roce', 'b12x_roce_eager']:
        if type(config.get(name)) is not bool:
            fail(f'{name} must be a qualified boolean')
    if selected == 'spark' and config['b12x_pcie']:
        fail('The RTX PCIe collective is not supported by the Spark launcher')
    if selected == 'rtx' and config['b12x_roce']:
        fail('B12x RoCE is supported only by the Spark image')
    roce_rows = config.get('b12x_roce_rows')
    if not isinstance(roce_rows, list) or not roce_rows or any(type(n) is not int or not 1 <= n <= 64 for n in roce_rows):
        fail('b12x_roce_rows must explicitly select positive row counts within 1..64')
    if len(set(roce_rows)) != len(roce_rows):
        fail('b12x_roce_rows must not contain duplicate row counts')
    if not isinstance(config.get('b12x_exact_fp8'), str):
        fail('b12x_exact_fp8 must be an explicit string; empty disables it')
    rows = config.get('b12x_exact_fp8_rows')
    if not isinstance(rows, list) or not rows or any(type(n) is not int or n <= 0 for n in rows):
        fail('b12x_exact_fp8_rows must be explicit positive row counts')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('platform', choices=['rtx', 'spark'])
    parser.add_argument('action', choices=['pull', 'start', 'status', 'health', 'logs', 'stop', 'restart', 'remove'])
    parser.add_argument('--settings', type=Path, default=HERE / 'settings.json')
    argv = sys.argv[1:]
    split = argv.index('--') if '--' in argv else len(argv)
    args = parser.parse_args(argv[:split])
    extra = argv[split + 1:] if split < len(argv) else []
    if extra and args.action != 'start':
        fail('Extra vLLM arguments are supported only with start')
    config = settings(args.settings, args.platform)
    native = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
    if native != config['architecture']:
        fail(f'This {args.platform} image requires native {config["architecture"]}')
    env = os.environ.copy()
    if env.get('MODEL_DIR'):
        fail('The release fast path uses the qualified model in HF_HOME; unset MODEL_DIR')
    if args.platform == 'spark' and args.action != 'pull':
        if env.get('NODE_RANK') not in ['0', '1'] or not env.get('HOST_IP') or not env.get('MASTER_ADDR') or not env.get('SOCKET_IFNAME'):
            fail('Set NODE_RANK=0/1, HOST_IP, MASTER_ADDR and SOCKET_IFNAME for this Spark')
        container = 'dots3-vllm-head' if env['NODE_RANK'] == '0' else 'dots3-vllm-worker'
    else:
        container = env.get('CONTAINER_NAME', 'dots3-vllm-rtx')
    port = int(env.get('PORT', config['port']))
    env.update(IMAGE=config['image'], MODEL_REVISION=MODEL_REVISION, PORT=str(port),
        GPU_MEMORY_UTILIZATION=str(config['gpu_memory_utilization']),
        MAX_MODEL_LEN=str(config['max_model_len']), MAX_NUM_SEQS=str(config['max_num_seqs']),
        MAX_BATCHED_TOKENS=str(config['max_num_batched_tokens']), KV_CACHE_DTYPE=config['kv_cache_dtype'],
        DOTS3_B12X_VOCAB=str(int(config['b12x_vocab'])), DOTS3_B12X_PCIE=str(int(config['b12x_pcie'])),
        DOTS3_B12X_ROCE=str(int(config['b12x_roce'])),
        DOTS3_B12X_ROCE_EAGER=str(int(config['b12x_roce_eager'])),
        DOTS3_B12X_ROCE_ROWS=','.join(map(str, config['b12x_roce_rows'])),
        DOTS3_B12X_EXACT_FP8=config['b12x_exact_fp8'],
        DOTS3_B12X_EXACT_FP8_ROWS=','.join(map(str, config['b12x_exact_fp8_rows'])))
    if args.action == 'start':
        if extra:
            print('Additional vLLM flags can change the qualified behavior and performance.', file=sys.stderr)
        # Entire HF_HOME and the runtime cache are mounted by the existing launcher.
        command = ['bash', str(HERE.parent / ('start_rtx.sh' if args.platform == 'rtx' else 'start_spark_node.sh'))]
        if config['mtp_tokens']:
            command += ['--speculative-config', json.dumps({'method': 'mtp', 'num_speculative_tokens': config['mtp_tokens']})]
        command += extra
    elif args.action == 'pull':
        command = ['docker', 'pull', '--platform', f'linux/{config["architecture"]}', config['image']]
    elif args.action == 'status':
        command = ['docker', 'inspect', '--format', '{{json .State}}', container]
    elif args.action == 'health':
        if args.platform == 'spark' and env['NODE_RANK'] == '1':
            fail('Worker has no HTTP endpoint. Use status here and health on the head.')
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=10) as response:
            print(f'HTTP {response.status}; run functional qualification for request readiness.')
        return
    else:
        if args.action == 'restart':
            actual = subprocess.check_output(['docker', 'inspect', '--format', '{{.Config.Image}}', container], text=True).strip()
            if actual != config['image']:
                fail('Existing container uses a different image. Stop/remove it and start the pinned release.')
        command = {'logs': ['docker', 'logs', '--tail', '100', '-f', container],
                   'stop': ['docker', 'stop', '--time', '60', container],
                   'restart': ['docker', 'restart', '--time', '60', container],
                   'remove': ['docker', 'rm', container]}[args.action]
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
