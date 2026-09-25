#!/usr/bin/env python3
"""Exercise patched real profiling branch lifetime without CUDA allocation."""
import ast, gc, os, tempfile, weakref
from pathlib import Path
from types import SimpleNamespace as NS
from port_indexer_profile_lifetime import patch

class Buffer:
    device = 'cuda:0'
    def numel(self): return 512
allocations=[]
def empty(*args, **kwargs):
    x=Buffer(); allocations.append(weakref.ref(x)); return x
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp); path=root/'model_executor/layers/sparse_attn_indexer.py'
    path.parent.mkdir(parents=True)
    path.write_bytes(Path('.cache/vllm-v0.30.0/vllm/model_executor/layers/sparse_attn_indexer.py').read_bytes())
    patch(root)
    source=path.read_text(); start=source.index('        import os as _profile_os')
    end=source.index('\n        return sparse_attn_indexer_fake(',start)
    import textwrap
    code=compile('def profile(forward_context):\n'+textwrap.indent(textwrap.dedent(source[start:end]),'    '),'profile','exec')
    scope=dict(torch=NS(empty=empty,uint8='uint8'),max_logits_elems=512,hidden_states=NS(device='cuda:0'))
    exec(code,scope); profile=scope['profile']
    os.environ['VLLM_INDEXER_PROFILE_HOLD_LOGITS']='1'
    ctx=NS();profile(ctx);profile(ctx)
    assert len(allocations)==1 and allocations[0]() is not None
    del ctx;gc.collect();assert allocations[0]() is None
    ctx=NS();profile(ctx);assert len(allocations)==2
    del ctx;gc.collect();assert allocations[1]() is None
    os.environ.pop('VLLM_INDEXER_PROFILE_HOLD_LOGITS')
    profile(NS());profile(NS());gc.collect()
    assert len(allocations)==4 and all(x() is None for x in allocations)
print('One buffer per profiling forward, release on context exit, default per-call lifetime passed')
