#!/usr/bin/env python3
"""CPU estimate of Dots owner cuts using measured storage and actual KV planner.

Run inside the candidate image with no GPU allocation. Requires a successful
startup attestation containing owner_dense_storage (f9484d6 or later). Supply
per-rank pre-allocation KV budgets from that SAME startup, not allocated pools
already reduced to a cross-rank minimum. Results predict persistent-storage
changes only; owner-dependent activation/workspace peaks require admission
and workload measurement on the proposed cut.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

GIB=1024**3


def parse_pair(value):
    values=tuple(float(x) for x in value.split(','))
    if len(values)!=2 or any(x<0 for x in values):
        raise argparse.ArgumentTypeError('expected two nonnegative GiB values')
    return values


def storage_plan(receipt, kv_budgets, mm_owners, reserves):
    workers=receipt['workers']
    if not receipt.get('passed') or len(workers)!=2 or {w['rank'] for w in workers}!={0,1}:
        raise ValueError('requires passed two-worker startup receipt')
    workers=sorted(workers,key=lambda w:w['rank'])
    if not all(w.get('passed') for w in workers):
        raise ValueError('worker attestation failed')
    layers={}; old_dense=[0,0]; old_mm=[0,0]
    dsa=set()
    for worker in workers:
        rank=worker['rank']
        for layer in worker['layers']:
            if not layer['local_owner']: continue
            index=layer['layer']
            if index in layers: raise ValueError('duplicate dense owner')
            layers[index]=layer['owner_dense_storage']['unique_storage_bytes']
            old_dense[rank]+=layers[index]
            if layer['expected_full_attention_heads']==128: dsa.add(index)
            elif layer['expected_full_attention_heads']!=64: raise ValueError('unsupported attention geometry')
        old_mm[rank]=sum(worker['multimodal_storage'][name]['unique_storage_bytes']
                         for name in ('visual','audio_tower'))
    if set(layers)!=set(range(46)) or len(dsa)!=13:
        raise ValueError('receipt must describe all 46 Dots layers and 13 DSA layers')
    new_mm=[0,0]
    tower_bytes={}
    for name,owner in zip(('visual','audio_tower'),mm_owners):
        present=[w['multimodal_storage'][name]['unique_storage_bytes'] for w in workers
                 if w['multimodal_storage'][name]['present']]
        if not present or len(set(present))!=1:
            raise ValueError(f'{name}: missing or inconsistent full native tower storage')
        tower_bytes[name]=present[0];new_mm[owner]+=present[0]
    candidates=[]
    for cut in range(1,46):
        new_dense=[sum(size for i,size in layers.items() if (i<cut)==(rank==0)) for rank in range(2)]
        budgets=[int(kv_budgets[r]*GIB)+old_dense[r]+old_mm[r]-new_dense[r]-new_mm[r]-int(reserves[r]*GIB)
                 for r in range(2)]
        candidates.append(dict(cut=cut, partition=[cut,46-cut],
            estimated_kv_budget_bytes=budgets, moved_dense_storage_bytes=[new_dense[r]-old_dense[r] for r in range(2)],
            moved_mm_storage_bytes=[new_mm[r]-old_mm[r] for r in range(2)]))
    return candidates,dsa,tower_bytes


def apply_real_cache_planner(candidates,dsa):
    import torch
    from vllm.v1.core.kv_cache_utils import (_get_packed_kv_cache_groups,
        _project_kv_cache_groups_to_worker,get_kv_cache_config_from_groups)
    from vllm.v1.kv_cache_interface import MLAAttentionSpec,SlidingWindowMLASpec,KVCacheLayout
    os.environ['DOTS3_COMPACT_DSA_CACHE']='1'
    config=NS(cache_config=NS(get_resolved_kv_cache_layout=lambda:KVCacheLayout.BLHNC,
        num_gpu_blocks_override=None,prefix_cache_retention_interval=None),
        model_config=NS(hf_text_config=NS(model_type='dots3_note')),
        attention_config=NS(hisparse_config=None),speculative_config=None)
    specs={};layer_ids={}
    for i in range(47):
        base=dict(block_size=64,num_kv_heads=1,dtype=torch.uint8,cache_dtype_str='fp8')
        name=f'layer.{i}.attention'
        specs[name]=(MLAAttentionSpec(head_size=576,**base) if i in dsa else
            SlidingWindowMLASpec(head_size=1088,sliding_window=513,**base))
        layer_ids[name]=i
        if i in dsa:
            name=f'layer.{i}.indexer';specs[name]=MLAAttentionSpec(head_size=132,**base);layer_ids[name]=i
    groups=_get_packed_kv_cache_groups(config,specs)
    for row in candidates:
        if min(row['estimated_kv_budget_bytes'])<=0:
            row.update(feasible_estimate=False,common_blocks=0);continue
        projected=[_project_kv_cache_groups_to_worker(groups,{name:spec for name,spec in specs.items()
                    if (layer_ids[name]<row['cut'])==(rank==0)}) for rank in (0,1)]
        cache=[get_kv_cache_config_from_groups(config,g,budget)
               for g,budget in zip(projected,row['estimated_kv_budget_bytes'])]
        # BLHNC tensor descriptors alias the same physical pool. Each size
        # is the whole pool; summing descriptors would multiply real memory.
        if any(len({t.size for t in c.kv_cache_tensors}) != 1 for c in cache):
            raise ValueError('unexpected nonuniform cache pools; estimator requires audited BLHNC layout')
        strides=[c.kv_cache_tensors[0].size//c.num_blocks for c in cache]
        blocks=[c.num_blocks for c in cache]
        common=min(blocks)
        row.update(feasible_estimate=True,worker_blocks=blocks,common_blocks=common,
                   block_bytes=strides,limiting_rank=blocks.index(common),
                   dsa_layers_per_rank=[sum((i<row['cut'])==(rank==0) for i in dsa) for rank in (0,1)],
                   unused_budget_after_common_pool_bytes=[b-common*s for b,s in zip(row['estimated_kv_budget_bytes'],strides)])
    return sorted(candidates,key=lambda r:r['common_blocks'],reverse=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attestation',type=Path,required=True)
    parser.add_argument('--kv-budgets-gib',type=parse_pair,help='Optional explicit override; default exact worker receipt budgets')
    parser.add_argument('--extra-workspace-reserve-gib',type=parse_pair,default=(0.,0.))
    parser.add_argument('--mm-owners',default='0,1')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    mm=tuple(int(x) for x in args.mm_owners.split(','))
    if len(mm)!=2 or any(x not in (0,1) for x in mm): parser.error('MM owners must be two worker ranks')
    receipt=json.loads(args.attestation.read_text())
    budgets = args.kv_budgets_gib
    if budgets is None:
        measured = [w.get('available_kv_cache_memory_bytes') for w in sorted(receipt['workers'],key=lambda w:w['rank'])]
        if len(measured)!=2 or any(not isinstance(x,int) or x<=0 for x in measured):
            parser.error('receipt lacks exact worker KV budgets; provide --kv-budgets-gib explicitly')
        budgets = tuple(x/GIB for x in measured)
    candidates,dsa,towers=storage_plan(receipt,budgets,mm,args.extra_workspace_reserve_gib)
    result={'schema':'dots3-hybrid-placement-estimate-v1','measured_performance':False,
        'source_attestation':str(args.attestation.resolve()),'source_kv_budgets_gib':budgets,
        'kv_budget_source':'override' if args.kv_budgets_gib else 'exact_worker_receipt',
        'mm_owners':mm,'tower_storage_bytes':towers,'extra_workspace_reserve_gib':args.extra_workspace_reserve_gib,
        'assumptions':['Same model/MTP3/compact-cache/block64/batch and CUDA graph settings as receipt startup.',
          'Per-rank budgets are pre-allocation available KV memory, not the allocated shared-minimum pools.',
          'Only persistent registered owner weights/buffers move; attention workspaces and peak activations may change.',
          'Target embedding/vocabulary and draft owner remain unchanged; final runtime admission must be tested.'],
        'candidates':apply_real_cache_planner(candidates,dsa)}
    import hashlib
    result['source_attestation_sha256']=hashlib.sha256(args.attestation.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'best_estimates':result['candidates'][:5],'output':str(args.output)},indent=2))

if __name__=='__main__':main()
