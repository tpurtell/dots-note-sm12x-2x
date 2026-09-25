#!/usr/bin/env python3
"""CPU gates for owner-aware grouping using installed vLLM pool/scheduler code."""
import os
import json
from types import SimpleNamespace as NS
import torch
from vllm.v1.core.kv_cache_utils import (_get_packed_kv_cache_groups,
    _project_kv_cache_groups_to_worker,get_kv_cache_config_from_groups,generate_scheduler_kv_cache_config)
from vllm.v1.kv_cache_interface import MLAAttentionSpec,SlidingWindowMLASpec,KVCacheLayout
from hybrid_cache_groups import refine_owner_state_groups

os.environ['DOTS3_COMPACT_DSA_CACHE']='1'
config=NS(cache_config=NS(get_resolved_kv_cache_layout=lambda:KVCacheLayout.BLHNC,
    num_gpu_blocks_override=None,prefix_cache_retention_interval=None),
    model_config=NS(hf_text_config=NS(model_type='dots3_note')),
    attention_config=NS(hisparse_config=None),speculative_config=None)
dsa={0,1,5,9,13,17,21,25,29,33,37,41,45}
specs={};layers={}
for i in range(47):
    base=dict(block_size=64,num_kv_heads=1,dtype=torch.uint8,cache_dtype_str='fp8')
    name=f'layer.{i}.attention'
    specs[name]=MLAAttentionSpec(head_size=576,**base) if i in dsa else SlidingWindowMLASpec(head_size=1088,sliding_window=513,**base)
    layers[name]=i
    if i in dsa:
        name=f'layer.{i}.indexer';specs[name]=MLAAttentionSpec(head_size=132,**base);layers[name]=i
global_groups=_get_packed_kv_cache_groups(config,specs)
records=[]
for cut in (1,17,18,22,23,26,45):
    worker_specs=[{n:s for n,s in specs.items() if (layers[n]<cut)==(rank==0)} for rank in (0,1)]
    refined=refine_owner_state_groups(global_groups,worker_specs)
    # Existing IDs retain only their own original layer subset; full-history
    # group object stays unchanged. New overflow IDs are global and appended.
    assert refined[0] is global_groups[0]
    assert all(set(new.layer_names)<=set(old.layer_names) for new,old in zip(refined,global_groups))
    assert set(n for g in refined for n in g.layer_names)==set(specs)
    assert sum(len(g.layer_names) for g in refined)==len(specs)
    projected=[_project_kv_cache_groups_to_worker(refined,worker) for worker in worker_specs]
    cache=[get_kv_cache_config_from_groups(config,g,6*1024**3) for g in projected]
    strides=[c.kv_cache_tensors[0].size//c.num_blocks for c in cache]
    common=min(c.num_blocks for c in cache)
    reconciled=[get_kv_cache_config_from_groups(config,g,common*stride) for g,stride in zip(projected,strides)]
    assert generate_scheduler_kv_cache_config(reconciled).num_blocks==common
    for rank,c in enumerate(reconciled):
        names=[n for t in c.kv_cache_tensors for n in t.layers]
        assert len(names)==len(set(names)) and set(names)==set(worker_specs[rank])
        full=sum(s.page_size_bytes for s in worker_specs[rank].values() if not isinstance(s,SlidingWindowMLASpec))
        largest_state=max((s.page_size_bytes for s in worker_specs[rank].values() if isinstance(s,SlidingWindowMLASpec)),default=0)
        assert strides[rank]==((max(full,largest_state)+575)//576)*576
        for i,g in enumerate(c.kv_cache_groups):
            assert set(g.layer_names)==set(refined[i].layer_names)&set(worker_specs[rank])
    if cut==18:assert strides==[271872,317376],strides
    records.append({'cut':cut,'groups_before':len(global_groups),'groups_after':len(refined),'physical_bytes_per_block':strides})
# Replicated TP specs require no ownership refinement of already capped groups.
unchanged=refine_owner_state_groups(global_groups,[specs,specs])
assert [g.layer_names for g in unchanged]==[g.layer_names for g in global_groups]
print(json.dumps({'passed':True,'scope':'actual CPU global grouping/projection/scheduler/pool allocation; no GPU or serving claim','cases':records},indent=2))
