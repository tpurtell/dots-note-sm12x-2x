#!/usr/bin/env python3
"""Archive a completed native qualification and extract measured report data.

CPU/filesystem only. Never calls Docker, SSH, or the model API. Does not set
release profiles qualified or establish anonymous registry access.
"""
import argparse
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def contained(root, relative):
    p = (root / relative).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f'Missing or escaping artifact: {relative}')
    return p


def flag(args, key):
    values = [args[i+1] for i, value in enumerate(args[:-1]) if value == key]
    values += [value.split('=', 1)[1] for value in args if value.startswith(key+'=')]
    require(len(values) == 1, f'Require one explicit {key}')
    return values[0]


def load_artifact(path, fmt):
    return read(path) if fmt == 'json' else [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stage_metrics(name, data):
    if name == 'prefix':
        return {key: data[key] for key in ['warm_prefix_hits', 'warm_prefix_queries', 'xgrammar_json', 'cold', 'warm', 'forced_tool']}
    if name == 'multimodal':
        return {'examples': data['source_examples']}
    if name == 'clients':
        return {'method': data['method'], 'output_tokens_per_sequence': data['output_tokens_per_sequence'],
            'points': [{'concurrency': p['concurrency'],
                'aggregate_decode_tokens_per_second': p['aggregate_decode_tokens_per_second'],
                'minimum_overlap': min(r['peak_overlapping_stream_intervals'] for r in p['runs'])} for p in data['points']]}
    if name in ('seven', 'coding'):
        result = dict(data[-1])
        if name == 'coding':
            result['quality_misses'] = [{'concurrency': wave['concurrency'], 'run': wave['run'],
                'task': r['task'], 'truncated': r['truncated'], 'finish_reason': r['finish_reason'],
                'content_result': r['content_result']} for wave in data if wave.get('record') == 'wave' and wave['timed']
                for r in wave['request_results'] if r['truncated'] or not r['content_result']['static_checks_passed']]
        return result
    if name == 'reasoning-api':
        s = data[-1]
        return {key: s[key] for key in ['passed', 'cases_passed', 'cases_total', 'failed_cases', 'evidence_issues',
            'speculative_draft_token_delta', 'structured_boundary_chunks', 'boundary_evidence_scope']}
    rows = [r for r in data if r.get('record') == 'measurement' and r.get('timed', True)]
    if name.startswith('retrieval-'):
        return {'passed': all(r['passed'] for r in rows), 'requests': len(rows),
            'positions': [{'position_fraction': r['position_fraction'], 'passed': r['passed'],
                          'usage': r['usage']} for r in rows]}
    result = []
    for depth in sorted({r['depth'] for r in rows}):
        group = [r for r in rows if r['depth'] == depth]
        result.append({'depth': depth, 'samples': len(group),
            'actual_prompt_tokens': sorted({r['usage']['prompt_tokens'] for r in group}),
            'completion_tokens': sorted({r['usage']['completion_tokens'] for r in group}),
            'ttft_seconds_median': statistics.median(r['ttft_seconds'] for r in group),
            'decode_tokens_per_second_median': statistics.median(r['decode_tps'] for r in group),
            'effective_prefill_tokens_per_second_median': statistics.median(r['usage']['prompt_tokens']/r['ttft_seconds'] for r in group)})
    return {'points': result, 'generation_scope': 'Fixed-length throughput probe; natural completion and code correctness are not established.'}


def export(args):
    source = args.input.resolve()
    target = args.output.resolve()
    require(not target.is_relative_to('/mnt/scratch'), 'Use native project storage, not /mnt/scratch')
    require(not target.exists(), 'Output must be a new directory')
    manifest = read(source/'manifest.json')
    require(manifest['schema'] == f'dots3-{args.platform}-release-qualification-v1', 'Wrong runner schema/platform')
    completions = sorted(source.glob('attempt-*/complete.json'))
    require(bool(completions), 'No ending complete.json: qualification is incomplete; no report written')
    complete_path = completions[-1]
    completion = read(complete_path)
    require(completion.get('completed') is True, 'Ending receipt is not complete')
    require(completion['requests_in_full_plan'] == sum(s['requests'] for s in manifest['plan']), 'Plan count mismatch')
    for relative, expected in manifest['source_sha256'].items():
        require(sha(contained(ROOT, relative)) == expected, f'Runner/validator source drift: {relative}; use exact recorded checkout')
    validator_name = f'qualify_{args.platform}'
    require(f'serving/release/{validator_name}.py' in manifest['source_sha256'], 'Validator source is not bound')
    validator = importlib.import_module(validator_name)
    identities = {'rtx': manifest['identity']} if args.platform == 'rtx' else manifest['identity']
    image_ids = {i['image_id'] for i in identities.values()}
    require(len(image_ids) == 1, 'Ranks use different image IDs')
    require(re.fullmatch(r'ghcr\.io/[a-z0-9_./-]+@sha256:[a-f0-9]{64}', args.image), 'Require immutable GHCR digest')
    registry = read(args.registry_receipt) if args.registry_receipt else None
    if registry:
        require(registry['wrapper_image_id'] in image_ids, 'Registry receipt image ID mismatch')
        require(registry['registry_digest'] == args.image.split('@')[1], 'Registry digest mismatch')
        require(registry['image_tag'].rsplit(':', 1)[0] == args.image.split('@')[0], 'Registry repository mismatch')
    else:
        require(args.platform == 'spark' and all(args.image in digests for digests in manifest['repo_digests'].values()),
            'RTX needs registry receipt binding local image ID to published digest')
    cache = read(args.cache_manifest)
    if registry:
        require(cache['source_image_id'] == registry['parent_image_id'], 'Cache provenance parent mismatch')
        require(cache['model_revision'] == registry['model_revision'], 'Cache checkpoint mismatch')
    profiles = {}
    for host, identity in identities.items():
        command = identity['args']
        require(any(cache['model_revision'] in item for item in command), 'Checkpoint revision absent from launch args')
        profiles[host] = {key[2:].replace('-', '_'): flag(command, key) for key in
            ['--max-model-len', '--max-num-seqs', '--max-num-batched-tokens', '--gpu-memory-utilization',
             '--kv-cache-dtype', '--reasoning-parser', '--tool-call-parser', '--speculative-config', '--tensor-parallel-size']}
    artifacts = {source/'manifest.json', complete_path}
    results = {}
    require(len({s['name'] for s in manifest['plan']}) == len(manifest['plan']), 'Duplicate stage names')
    for step in manifest['plan']:
        stage = source/step['name']
        require(stage.resolve().is_relative_to(source), 'Stage path escapes source')
        receipt_path = stage/'receipt.json'
        receipt = read(receipt_path)
        artifact = contained(stage, receipt['artifact'])
        require(sha(artifact) == receipt['sha256'], f'Artifact hash mismatch: {step["name"]}')
        require(receipt['validated'].get('complete') is True, 'Unvalidated stage receipt')
        if args.platform == 'spark':
            validator.validate(step, artifact, manifest['limits']['max_model_len'])
        else:
            validator.validate(step, artifact)
        results[step['name']] = stage_metrics(step['name'], load_artifact(artifact, step['format']))
        artifacts.add(receipt_path)
        artifacts.update(p for p in artifact.parent.iterdir() if p.is_file())
    # Keep monitoring and before/after snapshots from all attempts, including
    # earlier resume attempts; stage artifacts include only accepted attempts.
    for attempt in source.glob('attempt-*'):
        artifacts.update(p for p in attempt.iterdir() if p.is_file())
    hardware = {}
    if args.platform == 'rtx':
        runtime = read(complete_path.parent/'runtime-after.json')
        require(runtime['image_id'] in image_ids, 'Final runtime image mismatch')
        hardware['rtx'] = {k: runtime[k] for k in ['host', 'architecture', 'gpu_inventory_csv', 'meminfo', 'docker_stats']}
        profiles['rtx']['selected_environment'] = runtime['selected_environment']
    else:
        for host in identities:
            snapshot = read(complete_path.parent/f'{host}-after.json')
            require(snapshot['identity'] == identities[host], 'Final host identity changed')
            hardware[host] = {k: v for k, v in snapshot.items() if k not in ('container_log', 'guard_log', 'identity')}
    report = {'schema': 'dots3-release-report-v1', 'status': 'runner-completed; release-profile approval separate',
        'platform': args.platform, 'image': args.image, 'image_id': next(iter(image_ids)),
        'model_revision': cache['model_revision'], 'recipe_revision': cache['recipe_revision'],
        'parent_image_id': cache['source_image_id'], 'source_sha256': manifest['source_sha256'],
        'hardware': hardware, 'profile': profiles, 'stage_results': results,
        'completion': completion, 'registry_receipt': {k: registry[k] for k in ('registry_digest', 'registry', 'image_tag') if k in registry} if registry else None,
        'quality_scope': 'Static code/content checks are not executed-code correctness. Misses/truncations remain in stage results. Fixed-output timing probes are not natural-completion tests.',
        'prefill_scope': 'Actual prompt tokens / client TTFT, including first-token handoff; not kernel-only prefill speed.'}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.report-', dir=target.parent) as temp:
        temp = Path(temp); archive = []
        sources = [(p, 'runner/'+str(p.relative_to(source))) for p in sorted(artifacts)]
        sources.extend((contained(ROOT, path), 'provenance/source/'+path) for path in manifest['source_sha256'])
        sources.append((Path(__file__).resolve(), 'provenance/source/serving/release/report.py'))
        sources.append((args.cache_manifest.resolve(), 'provenance/cache-seed-manifest.json'))
        if args.registry_receipt:
            sources.append((args.registry_receipt.resolve(), 'provenance/registry-receipt.json'))
        for path, relative in sources:
            raw = path.read_bytes()
            require(not re.search(rb'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Authorization: (?:Bearer|Basic) [A-Za-z0-9])', raw), f'Potential credential in {path}')
            packed = gzip.compress(raw, compresslevel=9, mtime=0)
            dest = temp/(relative+'.gz');dest.parent.mkdir(parents=True, exist_ok=True);dest.write_bytes(packed)
            archive.append({'path': str(dest.relative_to(temp)), 'source': str(path), 'raw_bytes': len(raw),
                'raw_sha256': hashlib.sha256(raw).hexdigest(), 'gzip_sha256': hashlib.sha256(packed).hexdigest()})
        (temp/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        (temp/'archive-manifest.json').write_text(json.dumps({'schema': 1, 'report_sha256': sha(temp/'report.json'), 'files': archive}, indent=2)+'\n')
        os.rename(temp, target)
    print(json.dumps({'report': str(target/'report.json'), 'sha256': sha(target/'report.json'), 'archived_files': len(archive)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--platform', choices=['rtx', 'spark'], required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--cache-manifest', type=Path, required=True)
    p.add_argument('--registry-receipt', type=Path)
    export(p.parse_args())


if __name__ == '__main__':
    main()
