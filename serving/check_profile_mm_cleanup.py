#!/usr/bin/env python3
"""CPU check of the real port placement and no-peak-reset cleanup contract."""
from types import SimpleNamespace as NS
import tempfile
from pathlib import Path
from profile_mm_cleanup import cleanup_profile_mm_cache
from port_profile_mm_cleanup import patch
state={'reserved':4096,'calls':0}
def stats():return {'allocated_bytes.all.current':1024,'allocated_bytes.all.peak':8192,'reserved_bytes.all.current':state['reserved'],'inactive_split_bytes.all.current':512}
def empty_cache():state['reserved']=1536;state['calls']+=1
fake=NS(cuda=NS(memory_stats=stats,mem_get_info=lambda:(10000-state['reserved'],10000)),accelerator=NS(empty_cache=empty_cache))
x=cleanup_profile_mm_cache(fake)
assert x['released_reserved_bytes']==2560 and x['active_bytes_unchanged'] and x['peak_counter_unchanged'] and state['calls']==1
with tempfile.TemporaryDirectory() as tmp:
 r=Path(tmp);p=r/'v1/worker/gpu/model_runner.py';p.parent.mkdir(parents=True)
 p.write_bytes(Path('.cache/vllm-v0.30.0/vllm/v1/worker/gpu/model_runner.py').read_bytes());patch(r);s=p.read_text()
 a=s.index('self.hybrid_mm_cleanup_profile = cleanup_profile_mm_cache(torch)')
 assert s.rfind('profile_encoder_cache(',0,a)>s.rfind('def profile_run(',0,a)
 assert a<s.index('self.max_num_tokens, skip_attn=True, is_profile=True',a)
 assert 'reset_peak' not in s[a-250:a+250] and 'reset_encoder_cache' not in s[a-250:a+250]
print('MM cleanup preserves active bytes, peaks and encoder outputs; port lies before decoder dummy')
