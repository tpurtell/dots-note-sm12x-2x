#!/usr/bin/env python3
"""CPU optional attention flag/config compatibility checks."""
import copy
import json
from run import attention_config,attention_flags
from report import spark_hybrid_attestations
import check_spark_attestation as fixture
assert attention_flags({})==[] and attention_flags({'sparse_mla_force_mqa':False})==[]
assert json.loads(attention_flags({'sparse_mla_force_mqa':True})[1])=={'sparse_mla_force_mqa':True}
assert attention_config(None)==attention_config('{}')==attention_config('{"sparse_mla_force_mqa":false}')=={}
assert attention_config('{"sparse_mla_force_mqa":true}')=={'sparse_mla_force_mqa':True}
for bad in ['[]','{"sparse_mla_force_mqa":1}','{"other":true}']:
    try:attention_config(bad)
    except SystemExit:pass
    else:raise AssertionError('accepted '+bad)
for mode in ('matched','mismatch','duplicate'):
    data=copy.deepcopy(fixture.base)
    for runtime in data.values():runtime['args']+=['--attention-config','{"sparse_mla_force_mqa":true}']
    if mode=='mismatch':data['moa']['args'][-1]='{"sparse_mla_force_mqa":false}'
    if mode=='duplicate':data['moa']['args']+=['--attention-config={}']
    try:spark_hybrid_attestations(data)
    except ValueError:
        assert mode!='matched'
    else:assert mode=='matched'
print('PASS optional true/false/absent configuration, host mismatch and duplicate rejection')
