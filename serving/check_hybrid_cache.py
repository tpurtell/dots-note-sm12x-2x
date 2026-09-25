#!/usr/bin/env python3
"""CPU check of installed vLLM cache projection for hybrid layer ownership.

Exercises real grouping/allocation code, not model execution or GPU kernels.
The MTP SWA cache is assigned to the final owner. All expert weights remain
outside this cache test; ownership here applies only to attention state.
"""
import json
import os
from types import SimpleNamespace as NS

import torch
from vllm.v1.core.kv_cache_utils import (
    _get_packed_kv_cache_groups,
    _project_kv_cache_groups_to_worker,
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
)
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.kv_cache_interface import KVCacheLayout


def main():
    os.environ['DOTS3_COMPACT_DSA_CACHE'] = '1'
    config = NS(
        cache_config=NS(get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
                        num_gpu_blocks_override=None,
                        prefix_cache_retention_interval=None),
        model_config=NS(hf_text_config=NS(model_type='dots3_note')),
        attention_config=NS(hisparse_config=None), speculative_config=None,
    )
    dsa = {0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45}
    specs, layers = {}, {}
    for layer in range(47):
        base = dict(block_size=64, num_kv_heads=1, dtype=torch.uint8,
                    cache_dtype_str='fp8')
        name = f'layer.{layer}.attention'
        specs[name] = (MLAAttentionSpec(head_size=576, **base) if layer in dsa
                       else SlidingWindowMLASpec(head_size=1088,
                                                 sliding_window=513, **base))
        layers[name] = layer
        if layer in dsa:
            name = f'layer.{layer}.indexer'
            specs[name] = MLAAttentionSpec(head_size=132, **base)
            layers[name] = layer
    groups = _get_packed_kv_cache_groups(config, specs)
    budget = 6 * 1024**3
    baseline = get_kv_cache_config_from_groups(config, groups, budget)
    rows = []
    for cut in (18, 22, 23, 26):
        worker_specs = [{n: s for n, s in specs.items()
                         if (layers[n] >= cut) == bool(rank)} for rank in (0, 1)]
        assert not (worker_specs[0].keys() & worker_specs[1].keys())
        assert worker_specs[0].keys() | worker_specs[1].keys() == specs.keys()
        projected = [_project_kv_cache_groups_to_worker(groups, s)
                     for s in worker_specs]
        configs = [get_kv_cache_config_from_groups(config, g, budget)
                   for g in projected]
        blocks = min(c.num_blocks for c in configs)
        # Match the production cross-worker minimum-capacity reconciliation.
        strides = [c.kv_cache_tensors[0].size // c.num_blocks for c in configs]
        configs = [get_kv_cache_config_from_groups(config, g, blocks * stride)
                   for g, stride in zip(projected, strides)]
        scheduler = generate_scheduler_kv_cache_config(configs)
        assert scheduler.num_blocks == blocks
        for rank, cache in enumerate(configs):
            allocated = [n for t in cache.kv_cache_tensors for n in t.layers]
            assert set(allocated) == set(worker_specs[rank])
            assert len(allocated) == len(set(allocated))
            assert strides[rank] % 576 == 0
            for i, group in enumerate(cache.kv_cache_groups):
                assert set(group.layer_names) == (
                    set(groups[i].layer_names) & set(worker_specs[rank]))
        rows.append(dict(cut=cut, block_bytes=strides, blocks=blocks,
                         capacity_ratio=blocks / baseline.num_blocks))
    print(json.dumps({'scope': 'real CPU cache grouping/projection/allocation',
                      'cuts': rows}, indent=2))


if __name__ == '__main__':
    main()
