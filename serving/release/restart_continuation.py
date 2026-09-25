"""CPU-only provenance checks for explicit, same-profile RTX restart continuation."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEDGER = 'restart-continuation.json'
TOOLS = ('serving/release/continue_rtx.py', 'serving/release/restart_continuation.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    assert path.is_relative_to(Path(root).resolve()), 'provenance path escapes directory'
    return path


def same_profile(old, new):
    # Only process/container lifecycle identifiers may change. All existing and
    # future identity fields otherwise compare exactly, including launch args.
    lifecycle = {'id', 'started_at', 'restart_count'}
    assert old.keys() == new.keys(), 'runtime identity fields changed'
    assert {k:v for k,v in old.items() if k not in lifecycle} == {
        k:v for k,v in new.items() if k not in lifecycle}, 'image/model/profile changed'
    assert old != new, 'explicit restart continuation requires a changed lifecycle identity'


def verify_sources(manifest, root=ROOT):
    for relative, expected in manifest['source_sha256'].items():
        assert sha(contained(root, relative)) == expected, f'workload/validator source changed: {relative}'


def prepare(source, manifest, new_identity, validator, reason, evidence, root=ROOT):
    source = Path(source).resolve()
    assert manifest['schema'] == 'dots3-rtx-release-qualification-v2'
    assert reason.strip(), 'provide interruption reason'
    assert not list(source.glob('attempt-*/complete.json')), 'qualification already complete'
    same_profile(manifest['identity'], new_identity)
    verify_sources(manifest, root)
    completed = []
    for step in manifest['plan']:
        stage = source / step['name']
        receipt = stage / 'receipt.json'
        if not receipt.exists():
            continue
        saved = read(receipt)
        assert saved['identity'] == manifest['identity'], 'prior stage identity mismatch'
        artifact = contained(stage, saved['artifact'])
        assert sha(artifact) == saved['sha256'], 'prior artifact changed'
        assert saved['validated'].get('complete') is True, 'prior stage not validated'
        validator.validate(step, artifact, manifest['limits']['max_model_len'])
        completed.append(step['name'])
    assert completed, 'no finished stages to preserve'
    pending = [s['name'] for s in manifest['plan'] if s['name'] not in completed]
    assert pending, 'no unfinished stages'
    evidence = Path(evidence).resolve()
    assert evidence.is_file(), 'missing interruption evidence'
    return {
        'schema': 'dots3-rtx-restart-continuation-v1',
        'interrupted': True, 'uninterrupted_run': False,
        'reason': reason, 'failure_evidence': str(evidence),
        'failure_evidence_sha256': sha(evidence),
        'prior_manifest_sha256': sha(source/'manifest.json'),
        'prior_identity': manifest['identity'], 'continuation_identity': new_identity,
        'preserved_stages': completed, 'remaining_stages': pending,
        'preserved_files': {str(p.relative_to(source)):sha(p)
                            for p in sorted(source.rglob('*')) if p.is_file()},
        'continuation_source_sha256': {p:sha(contained(root,p)) for p in TOOLS},
    }


def verify(source, manifest, current_identity=None, root=ROOT):
    source = Path(source).resolve()
    ledger = read(source/LEDGER)
    assert ledger['schema'] == 'dots3-rtx-restart-continuation-v1'
    assert ledger['interrupted'] is True and ledger['uninterrupted_run'] is False
    assert sha(source/'manifest.json') == ledger['prior_manifest_sha256'], 'prior manifest changed'
    assert ledger['prior_identity'] == manifest['identity'], 'prior identity changed'
    same_profile(ledger['prior_identity'], ledger['continuation_identity'])
    if current_identity is not None:
        assert current_identity == ledger['continuation_identity'], 'another restart requires a new audit'
    verify_sources(manifest, root)
    for relative, expected in ledger['preserved_files'].items():
        assert sha(contained(source,relative)) == expected, f'preserved evidence changed: {relative}'
    for relative, expected in ledger['continuation_source_sha256'].items():
        assert sha(contained(root,relative)) == expected, f'continuation source changed: {relative}'
    assert sha(ledger['failure_evidence']) == ledger['failure_evidence_sha256'], 'failure evidence changed'
    names = [s['name'] for s in manifest['plan']]
    assert len(set(names)) == len(names)
    assert set(ledger['preserved_stages']).isdisjoint(ledger['remaining_stages'])
    assert sorted(ledger['preserved_stages'] + ledger['remaining_stages']) == sorted(names)
    for name in ledger['preserved_stages']:
        assert f'{name}/receipt.json' in ledger['preserved_files'], 'unbound prior receipt'
    return ledger


def stage_identity(ledger, name):
    return ledger['prior_identity'] if name in ledger['preserved_stages'] else ledger['continuation_identity']
