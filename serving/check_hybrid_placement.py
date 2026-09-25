#!/usr/bin/env python3
"""CPU synthetic bookkeeping plus real installed vLLM allocation checks."""
from plan_hybrid_placement import storage_plan,apply_real_cache_planner,GIB

dsa={0,1,5,9,13,17,21,25,29,33,37,41,45}
workers=[]
for rank in (0,1):
    workers.append({'rank':rank,'passed':True,'layers':[
        {'layer':i,'local_owner':(i<23)==(rank==0),
         'expected_full_attention_heads':128 if i in dsa else 64,
         'owner_dense_storage':{'unique_storage_bytes':128*1024**2}}
        for i in range(46)],'multimodal_storage':{
            'visual':{'present':True,'unique_storage_bytes':8*GIB},
            'audio_tower':{'present':True,'unique_storage_bytes':2*GIB}}})
receipt={'passed':True,'workers':workers}
rows,actual_dsa,towers=storage_plan(receipt,(10,10),(0,1),(0,0))
assert actual_dsa==dsa and len(rows)==45
unchanged=next(row for row in rows if row['cut']==23)
assert unchanged['moved_dense_storage_bytes']==[0,0]
assert unchanged['moved_mm_storage_bytes']==[-2*GIB,-8*GIB]
assert unchanged['estimated_kv_budget_bytes']==[12*GIB,18*GIB]
ranked=apply_real_cache_planner(rows,dsa)
assert ranked[0]['cut']<23
for row in ranked:
    assert row['common_blocks']==min(row['worker_blocks'])
    assert all(b>0 and b%576==0 for b in row['block_bytes'])
    assert sum(row['dsa_layers_per_rank'])==13
    assert all(b>=0 for b in row['unused_budget_after_common_pool_bytes'])
receipt['passed']=False
try:storage_plan(receipt,(10,10),(0,1),(0,0))
except ValueError:pass
else:raise AssertionError('failed receipt accepted')
print({'passed':True,'scope':'synthetic storage plus actual CPU vLLM cache planner',
       'candidate_count':len(ranked),'synthetic_best_cut':ranked[0]['cut']})
