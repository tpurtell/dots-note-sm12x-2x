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
import time
import urllib.request

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent
MODEL_REVISION = 'd8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da'


def fail(message):
    raise SystemExit(message)


def settings(path, selected):
    doc = json.loads(path.read_text())
    if doc.get('schema') != 1:
        fail('Unsupported release settings schema')
    if doc.get('platforms', {}).get(selected, {}).get('status', doc.get('status')) != 'qualified':
        fail(f'{selected} release is not finalized. Record its completed evidence and qualified settings after release-image verification.')
    if doc.get('model_revision') != MODEL_REVISION:
        fail('Unexpected checkpoint revision')
    config = doc['platforms'][selected]
    if selected == 'rtx':
        # Older RTX profiles predate the Spark-only transport fields.
        config = dict(config)
        config.setdefault('b12x_roce', False)
        config.setdefault('b12x_roce_eager', False)
        config.setdefault('b12x_roce_rows', list(range(1, 65)))
    if config.get('reasoning_parser') != 'dots3':
        fail('Qualified release requires explicit reasoning_parser=dots3; legacy parser-off profiles are development-only')
    if selected == 'spark' and (config.get('memory_guard') is not True or
            type(config.get('min_host_available_gib')) not in (int, float) or
            not 8 <= config['min_host_available_gib'] <= 64):
        fail('Qualified Spark release requires memory_guard=true and at least 8 GiB host headroom')
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
    validate_report(json.loads(report_path.read_text()), config, selected)
    return config


def parse_rows(text):
    try:
        rows = set()
        for item in text.split(','):
            if '-' in item:
                low, high = map(int, item.split('-'))
                if not 0 < low <= high <= 1048576:
                    raise ValueError('invalid row range')
                rows.update(range(low, high+1))
            else:
                rows.add(int(item))
        if not rows or min(rows) <= 0:
            raise ValueError('invalid rows')
        return rows
    except ValueError:
        fail('Invalid active row selection in qualification report')


def validate_report(read_report, config, selected):
    report = read_report
    if (report.get('schema') != 'dots3-release-report-v1' or
            report.get('platform') != selected or report.get('image') != config['image'] or
            report.get('model_revision') != MODEL_REVISION or
            report.get('completion', {}).get('completed') is not True):
        fail('Qualification report does not bind this completed platform/image/checkpoint')
    profiles = report.get('profile', {})
    if len(profiles) != (1 if selected == 'rtx' else 2):
        fail('Qualification report must cover every platform host')
    for host, profile in profiles.items():
        for key, setting in [('max_model_len', 'max_model_len'), ('max_num_seqs', 'max_num_seqs'),
                ('max_num_batched_tokens', 'max_num_batched_tokens'),
                ('kv_cache_dtype', 'kv_cache_dtype'), ('reasoning_parser', 'reasoning_parser')]:
            if str(profile.get(key)) != str(config[setting]):
                fail(f'Qualification profile mismatch for {host}: {setting}')
        if float(profile['gpu_memory_utilization']) != config['gpu_memory_utilization']:
            fail(f'Qualification memory allocation mismatch for {host}')
        if str(profile.get('tensor_parallel_size')) != '2' or profile.get('tool_call_parser') != 'dots':
            fail(f'Qualification TP2/tool parser mismatch for {host}')
        spec = json.loads(profile['speculative_config'])
        if spec.get('method') != 'mtp' or spec.get('num_speculative_tokens') != config['mtp_tokens']:
            fail(f'Qualification MTP mismatch for {host}')
        runtime_env = profile.get('selected_environment')
        if runtime_env is None:
            runtime_env = report.get('hardware', {}).get(host, {}).get('runtime', {}).get('selected_environment', [])
        env = dict(item.split('=', 1) for item in runtime_env)
        for variable, setting in [('DOTS3_B12X_VOCAB', 'b12x_vocab'),
                ('VLLM_ENABLE_PCIE_ALLREDUCE', 'b12x_pcie'), ('DOTS3_B12X_ROCE', 'b12x_roce')]:
            if env.get(variable, '0') != str(int(config[setting])):
                fail(f'Qualification B12x mismatch for {host}: {setting}')
        if env.get('DOTS3_B12X_EXACT_FP8', '') != config['b12x_exact_fp8']:
            fail(f'Qualification exact-FP8 selection mismatch for {host}')
        if config['b12x_pcie'] and env.get('VLLM_PCIE_ALLREDUCE_BACKEND') != 'b12x':
            fail(f'Qualification PCIe backend mismatch for {host}')
        if config['b12x_exact_fp8'] and parse_rows(env.get('DOTS3_B12X_EXACT_FP8_ROWS', '')) != set(config['b12x_exact_fp8_rows']):
            fail(f'Qualification exact-FP8 rows mismatch for {host}')
        if config['b12x_roce']:
            if env.get('DOTS3_B12X_ROCE_EAGER', '0') != str(int(config['b12x_roce_eager'])):
                fail(f'Qualification RoCE eager setting mismatch for {host}')
            if parse_rows(env.get('DOTS3_B12X_ROCE_ROWS', '')) != set(config['b12x_roce_rows']):
                fail(f'Qualification RoCE rows mismatch for {host}')


def check_spark_headroom(minimum_gib):
    available = next(int(line.split()[1]) * 1024 for line in
        Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available < minimum_gib * 2**30:
        fail(f'Spark has less than {minimum_gib} GiB physical memory available; refusing start/restart')


def rearm_spark_guard(container, env):
    cache = Path(env.get('RUNTIME_CACHE', PROJECT / '.cache/serving/spark/runtime'))
    try:
        cache.mkdir(parents=True, exist_ok=True)
        with (cache / 'memory-watch.log').open('a') as log:
            guard = subprocess.Popen([sys.executable, str(HERE.parent / 'watch_spark_memory.py'),
                '--container', container, '--min-available-gib', env['MIN_HOST_AVAILABLE_GIB'],
                '--ready-directory', str(cache)],
                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                env=env, start_new_session=True)
        (cache / 'memory-watch.pid').write_text(str(guard.pid) + '\n')
        # Require a successful initial memory sample, not just a process PID.
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if guard.poll() is not None:
                raise RuntimeError('Spark memory guard exited before readiness')
            marker = cache / f'memory-watch-ready-{guard.pid}'
            if marker.is_file():
                marker.unlink()
                print(f'Host memory monitor PID {guard.pid}; log: {cache / "memory-watch.log"}')
                return
            time.sleep(.1)
        raise RuntimeError('Spark memory guard did not become ready within 10 seconds')
    except BaseException:
        subprocess.run(['docker', 'kill', container], check=False, timeout=20)
        raise


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
    if any(flag.split('=', 1)[0] in ('--reasoning-parser', '--reasoning-parser-plugin') for flag in extra):
        fail('Release reasoning parser is pinned to dots3; use the development launcher for parser experiments')
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
    env.update(REASONING_PARSER=config['reasoning_parser'], IMAGE=config['image'], MODEL_REVISION=MODEL_REVISION, PORT=str(port),
        GPU_MEMORY_UTILIZATION=str(config['gpu_memory_utilization']),
        MAX_MODEL_LEN=str(config['max_model_len']), MAX_NUM_SEQS=str(config['max_num_seqs']),
        MAX_BATCHED_TOKENS=str(config['max_num_batched_tokens']), KV_CACHE_DTYPE=config['kv_cache_dtype'],
        DOTS3_B12X_VOCAB=str(int(config['b12x_vocab'])), DOTS3_B12X_PCIE=str(int(config['b12x_pcie'])),
        DOTS3_B12X_ROCE=str(int(config['b12x_roce'])),
        DOTS3_B12X_ROCE_EAGER=str(int(config['b12x_roce_eager'])),
        DOTS3_B12X_ROCE_ROWS=','.join(map(str, config['b12x_roce_rows'])),
        DOTS3_B12X_EXACT_FP8=config['b12x_exact_fp8'],
        DOTS3_B12X_EXACT_FP8_ROWS=','.join(map(str, config['b12x_exact_fp8_rows'])))
    if args.platform == 'spark':
        env.update(MEMORY_GUARD='1', MIN_HOST_AVAILABLE_GIB=str(config['min_host_available_gib']))
        if args.action in ('start', 'restart'):
            check_spark_headroom(config['min_host_available_gib'])
    if args.action in ('start', 'restart'):
        label = subprocess.check_output(['docker', 'image', 'inspect', '--format',
            '{{index .Config.Labels "io.tpurtell.dots3.reasoning-parser"}}', config['image']], text=True).strip()
        if label != config['reasoning_parser']:
            fail('Release image lacks the verified dots3 reasoning-parser label; legacy development wrappers cannot serve this profile')
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
    if args.platform == 'spark' and args.action == 'restart':
        try:
            subprocess.run(command, env=env, check=True, timeout=90)
        except BaseException:
            subprocess.run(['docker', 'kill', container], check=False, timeout=20)
            raise
        rearm_spark_guard(container, env)
        return
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
