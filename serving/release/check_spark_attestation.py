#!/usr/bin/env python3
"""CPU-only centralized EngineCore ownership receipt validation."""
import copy
import hashlib
import json
from report import spark_hybrid_attestations

raw=json.dumps({'passed':True,'workers':[{'rank':i,'world_size':2,'passed':True,'errors':[]} for i in range(2)]})
args=['--max-model-len','524288','--max-num-seqs','16','--max-num-batched-tokens','512',
      '--gpu-memory-utilization','0.8','--kv-cache-dtype','fp8','--tensor-parallel-size','2']
head={'image_id':'sha256:test','args':args,
      'selected_environment':['VLLM_HYBRID_LAYER_PARTITION=23,23','VLLM_HYBRID_ATTESTATION_PATH=/cache/owners.json'],
      'hybrid_attestation':{'status':'captured','path':'/cache/owners.json','raw_json':raw,'sha256':hashlib.sha256(raw.encode()).hexdigest()}}
worker=copy.deepcopy(head);worker['args']=args+['--headless'];worker['hybrid_attestation']={'status':'pending','path':'/cache/owners.json'}
base={'rhea':head,'moa':worker}
assert spark_hybrid_attestations(base)['moa']['status']=='aggregate-head-reference'
worker['hybrid_attestation']['status']='headless-worker'
assert spark_hybrid_attestations(base)['rhea']['receipt']['passed']
for failure in ['missing-head','missing-rank','tamper','image','profile','duplicate-head']:
    test=copy.deepcopy(base)
    if failure=='missing-head':test['rhea']['hybrid_attestation']['status']='pending'
    if failure=='missing-rank':
        proof=test['rhea']['hybrid_attestation'];data=json.loads(proof['raw_json']);data['workers'].pop();proof['raw_json']=json.dumps(data);proof['sha256']=hashlib.sha256(proof['raw_json'].encode()).hexdigest()
    if failure=='tamper':test['rhea']['hybrid_attestation']['sha256']='bad'
    if failure=='image':test['moa']['image_id']='sha256:other'
    if failure=='profile':test['moa']['args'][1]='262144'
    if failure=='duplicate-head':test['moa']['hybrid_attestation']=copy.deepcopy(head['hybrid_attestation'])
    try:spark_hybrid_attestations(test)
    except ValueError:pass
    else:raise AssertionError('accepted '+failure)
print('PASS: centralized head receipt/worker reference; missing head/rank, hash tamper, image/profile drift and duplicate receipt rejected')
