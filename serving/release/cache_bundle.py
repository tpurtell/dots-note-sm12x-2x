#!/usr/bin/env python3
"""Export warmed native caches and seed them without replacing runtime files.

Run export on the Docker host after qualification; run seed in the release image.
This packages executable artifacts: only bundle caches from trusted builds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

ROOT = Path('/root/.cache/vllm-runtime')
PROBE = r"""
import hashlib, importlib.metadata as metadata, json
from pathlib import Path
result = {'packages': {d.metadata['Name']: d.version for d in metadata.distributions()}, 'sources': {}}
for root in ['/opt/b12x/b12x', '/usr/local/lib/python3.12/dist-packages/vllm']:
    digest = hashlib.sha256()
    paths = sorted(path for path in Path(root).rglob('*') if path.is_file() and path.suffix in {'.py', '.c', '.h', '.cu', '.cuh', '.cpp'})
    if not paths:
        raise RuntimeError('Missing runtime source tree: ' + root)
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    result['sources'][root] = digest.hexdigest()
print(json.dumps(result, sort_keys=True))
"""
CACHES = {
    'b12x/compile': '/root/.cache/b12x/compile',
    'triton': str(ROOT / 'triton'),
    'vllm': str(ROOT / 'vllm'),
}


def run(*args):
    return subprocess.check_output(args, text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def export(args):
    dest = Path(args.output).resolve()
    if dest.exists():
        raise SystemExit(f'Refusing existing output: {dest}')
    container = json.loads(run('docker', 'inspect', args.container))[0]
    image = json.loads(run('docker', 'image', 'inspect', container['Image']))[0]
    expected_arch = {'rtx': 'amd64', 'spark': 'arm64'}[args.platform]
    if image['Architecture'] != expected_arch:
        raise SystemExit('Container architecture does not match requested platform')
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix='.cache-bundle-') as tmp:
        tmp = Path(tmp)
        payload = tmp / 'payload'
        payload.mkdir()
        env = dict(item.split('=', 1) for item in container['Config']['Env'])
        expected_cute = {'rtx': 'sm_120a', 'spark': 'sm_121a'}[args.platform]
        if env.get('CUTE_DSL_ARCH') != expected_cute:
            raise SystemExit('CUTE_DSL_ARCH does not match requested platform')
        if env.get('TRITON_CACHE_DIR') != str(ROOT / 'triton'):
            raise SystemExit('Triton cache must use the release absolute path')
        sources = dict(CACHES)
        sources['b12x/compile'] = env.get('B12X_COMPILE_CACHE_DIR', sources['b12x/compile'])
        proxy_source = env.get('B12X_ROCE_CACHE_DIR', str(ROOT / 'b12x/roce'))
        if subprocess.run(['docker', 'exec', args.container, 'test', '-d', proxy_source],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
            sources['b12x/roce'] = proxy_source
        elif env.get('DOTS3_B12X_ROCE', '0').lower() not in {'', '0', 'false', 'no', 'off'}:
            raise SystemExit('Enabled RoCE runtime has no compiled proxy cache to export')
        for relative, source in sources.items():
            target = payload / relative
            target.mkdir(parents=True)
            subprocess.run(['docker', 'cp', f'{args.container}:{source}/.', str(target)], check=True)
        files = {}
        for path in sorted(payload.rglob('*')):
            if path.is_symlink():
                raise SystemExit(f'Unexpected cache symlink: {path}')
            if not path.is_file():
                continue
            if path.name.endswith(('.lock', '.tmp')):
                path.unlink()
                continue
            files[str(path.relative_to(payload))] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        if not any(p.startswith('b12x/compile/') and p.endswith('.o') for p in files):
            raise SystemExit('No compiled B12x objects found')
        if not any(p.startswith('triton/') and p.endswith('.cubin') for p in files):
            raise SystemExit('No compiled Triton cubins found')
        if 'b12x/roce' in sources and not any(p.startswith('b12x/roce/roce_proxy-') and p.endswith('.so') for p in files):
            raise SystemExit('RoCE proxy cache exists but contains no native proxy executable')
        uuids = set()
        for path in (payload / 'b12x/compile').rglob('*.json'):
            doc = json.loads(path.read_text())
            identity = doc.get('semantic_payload', {}).get('device_uuid')
            if identity:
                uuids.add(json.dumps(identity))
        # Record installed versions without importing torch or touching the GPU.
        runtime = json.loads(run('docker', 'exec', args.container, 'python3', '-c', PROBE))
        provenance = {
            'schema': 1, 'platform': args.platform, 'architecture': expected_arch,
            'cute_arch': expected_cute, 'source_image_id': container['Image'],
            'source_image_labels': image['Config'].get('Labels', {}),
            'recipe_revision': args.recipe_revision,
            'qualification_evidence': args.evidence,
            'model_revision': 'd8e3b9a48d3b5b8e23d9c6b3f6cc645f48b2f9da',
            'b12x_device_identities': sorted(uuids),
            'runtime': runtime, 'files': files, 'roce_proxy_bundled': 'b12x/roce' in sources,
            'limitations': ['B12x executable cache identities contain physical GPU UUIDs.',
                'Unseen GPUs, shapes, toolchains and configurations may compile at runtime.',
                'Triton and vLLM caches retain their original absolute paths.',
                'CUDA driver cache and device-specific tuning decisions are not bundled.'],
        }
        (tmp / 'manifest.json').write_text(json.dumps(provenance, indent=2, sort_keys=True) + '\n')
        os.rename(tmp, dest)
    print(json.dumps({'output': str(dest), 'files': len(files), 'bytes': sum(v['bytes'] for v in files.values())}))


def merge(args):
    """Add other ranks' B12x objects, preserving the primary graph/Triton cache."""
    primary = Path(args.bundle)
    output = Path(args.output).resolve()
    if output.exists():
        raise SystemExit(f'Refusing existing output: {output}')
    manifest = json.loads((primary / 'manifest.json').read_text())
    additions = []
    for name in args.add_b12x_from:
        other = Path(name)
        doc = json.loads((other / 'manifest.json').read_text())
        for key in ['schema', 'platform', 'architecture', 'cute_arch', 'runtime', 'recipe_revision', 'model_revision']:
            if doc[key] != manifest[key]:
                raise SystemExit(f'Rank bundle compatibility mismatch: {key}')
        additions.append((other, doc))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.cache-merge-') as tmp:
        tmp = Path(tmp)
        shutil.copytree(primary, tmp, dirs_exist_ok=True)
        for other, doc in additions:
            for name, entry in doc['files'].items():
                relative = Path(name)
                if relative.is_absolute() or '..' in relative.parts:
                    raise SystemExit(f'Invalid rank cache path: {name}')
                if not name.startswith('b12x/compile/'):
                    continue
                source = other / 'payload' / relative
                if source.is_symlink() or source.stat().st_size != entry['bytes'] or sha(source) != entry['sha256']:
                    raise SystemExit(f'Rank artifact hash mismatch: {name}')
                if name in manifest['files']:
                    if manifest['files'][name] != entry:
                        raise SystemExit(f'Conflicting rank artifact: {name}')
                    continue
                target = tmp / 'payload' / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                manifest['files'][name] = entry
            manifest['b12x_device_identities'] = sorted(set(manifest['b12x_device_identities'] + doc['b12x_device_identities']))
        manifest['additional_rank_provenance'] = [doc for _, doc in additions]
        (tmp / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        os.rename(tmp, output)
    print(json.dumps({'output': str(output), 'files': len(manifest['files'])}))


def seed(args):
    bundle = Path(args.bundle)
    doc = json.loads((bundle / 'manifest.json').read_text())
    machine = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
    if doc['schema'] != 1 or machine != doc['architecture']:
        raise SystemExit('Cache bundle schema or native architecture mismatch')
    if os.environ.get('CUTE_DSL_ARCH') != doc['cute_arch']:
        raise SystemExit('Cache bundle CUTE_DSL_ARCH mismatch')
    # Keep paths used by embedded Triton group metadata and vLLM compilation.
    if os.environ.get('TRITON_CACHE_DIR') != str(ROOT / 'triton'):
        raise SystemExit('Release requires the original TRITON_CACHE_DIR')
    if os.environ.get('VLLM_CACHE_ROOT') != str(ROOT / 'vllm'):
        raise SystemExit('Release requires the original VLLM_CACHE_ROOT')
    if os.environ.get('B12X_COMPILE_CACHE_DIR') != str(ROOT / 'b12x/compile'):
        raise SystemExit('Release requires its B12X_COMPILE_CACHE_DIR')
    runtime = json.loads(run('python3', '-c', PROBE))
    if runtime != doc['runtime']:
        raise SystemExit('Cache runtime source/dependency fingerprint mismatch')
    if doc.get('roce_proxy_bundled') and os.environ.get('B12X_ROCE_CACHE_DIR') != str(ROOT / 'b12x/roce'):
        raise SystemExit('Release requires its seeded B12X_ROCE_CACHE_DIR')
    copied = existing = 0
    # Verify the complete trusted image payload before making any runtime changes.
    for name, entry in doc['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] not in {'b12x', 'triton', 'vllm'}:
            raise SystemExit(f'Invalid cache path: {name}')
        source = bundle / 'payload' / relative
        if source.is_symlink() or source.stat().st_size != entry['bytes'] or sha(source) != entry['sha256']:
            raise SystemExit(f'Cache bundle hash mismatch: {name}')
    for name in doc['files']:
        source = bundle / 'payload' / name
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing += 1
            continue
        # Atomic publish; hard link fails rather than overwriting concurrent writers.
        fd, tmp = tempfile.mkstemp(prefix='.seed-', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as output, source.open('rb') as input_file:
                shutil.copyfileobj(input_file, output)
            try:
                os.link(tmp, target)
                copied += 1
            except FileExistsError:
                existing += 1
        finally:
            os.unlink(tmp)
    print(json.dumps({'cache_seed_copied': copied, 'cache_seed_existing': existing}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    exp = sub.add_parser('export')
    exp.add_argument('--container', required=True)
    exp.add_argument('--platform', choices=['rtx', 'spark'], required=True)
    exp.add_argument('--recipe-revision', required=True)
    exp.add_argument('--evidence', required=True, help='Accepted qualification report path or immutable URL')
    exp.add_argument('--output', required=True)
    me = sub.add_parser('merge')
    me.add_argument('--bundle', required=True)
    me.add_argument('--add-b12x-from', action='append', required=True)
    me.add_argument('--output', required=True)
    se = sub.add_parser('seed')
    se.add_argument('--bundle', default='/opt/dots3/cache-seed')
    args = parser.parse_args()
    {'export': export, 'seed': seed, 'merge': merge}[args.command](args)
