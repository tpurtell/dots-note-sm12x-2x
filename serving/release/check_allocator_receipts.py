#!/usr/bin/env python3
"""CPU-only integrity checks for optional Spark allocator receipts."""
import hashlib
import json
from report import spark_allocator_policies


def fixture():
    out = {}
    for rank, host in enumerate(('rhea', 'moa')):
        receipt = {'rank': rank, 'world_size': 2, 'allocator_backend': 'native', 'phase': 'after_compile_and_warmup_before_requests',
                   'created_at_unix': 2, 'per_process_fraction': .9,
                   'previous_settings': {'PYTORCH_CUDA_ALLOC_CONF': '', 'garbage_collection_threshold': 0., 'expandable_segments': False},
                   'settings': {'PYTORCH_CUDA_ALLOC_CONF': 'garbage_collection_threshold:0.90', 'garbage_collection_threshold': .9, 'expandable_segments': False},
                   'total_device_bytes': 1000, 'allocator_ceiling_bytes': 900, 'reclamation_threshold_bytes': 810}
        raw = json.dumps(receipt)
        out[host] = {'image_id': 'same-image', 'args': ['--headless'] if rank else [],
                     'started_at': '1970-01-01T00:00:01Z', 'captured_unix_seconds': 3,
                     'selected_environment': ['DOTS3_ALLOCATOR_FRACTION=0.90', 'PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.90'],
                     'allocator_policy': {'status': 'captured', 'path': f'/root/.cache/vllm-runtime/allocator-policy-rank{rank}.json',
                                          'raw_json': raw, 'sha256': hashlib.sha256(raw.encode()).hexdigest()}}
    return out


def reject(data):
    try:
        spark_allocator_policies(data)
    except ValueError:
        return
    raise AssertionError('Accepted invalid receipt')


def main():
    assert spark_allocator_policies({'rhea': {}, 'moa': {}}) == {'rhea': None, 'moa': None}
    assert len(spark_allocator_policies(fixture())) == 2
    for mutation in ('missing', 'stale', 'tamper', 'rank', 'fraction', 'gc', 'extra_change', 'image'):
        data = fixture(); runtime = data['moa']; proof = runtime['allocator_policy']; receipt = json.loads(proof['raw_json'])
        if mutation == 'missing': proof['status'] = 'pending'
        elif mutation == 'stale': receipt['created_at_unix'] = 0
        elif mutation == 'tamper': proof['sha256'] = 'bad'
        elif mutation == 'rank': receipt['rank'] = 0
        elif mutation == 'fraction': receipt['per_process_fraction'] = .8
        elif mutation == 'gc': receipt['settings']['garbage_collection_threshold'] = .8
        elif mutation == 'extra_change': receipt['settings']['expandable_segments'] = True
        elif mutation == 'image': runtime['image_id'] = 'other'
        if mutation not in ('tamper', 'missing', 'image'):
            proof['raw_json'] = json.dumps(receipt); proof['sha256'] = hashlib.sha256(proof['raw_json'].encode()).hexdigest()
        reject(data)
    print('Optional Spark allocator receipt CPU checks passed; no GPU/API activity')


if __name__ == '__main__':
    main()
