#!/usr/bin/env python3
"""CPU ownership/protocol/source checks; actual image/audio GPU gate separate."""
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
from threading import Barrier, local
from concurrent.futures import ThreadPoolExecutor
import torch
import hybrid_parallel
state=local()
distributed=ModuleType('vllm.distributed')
distributed.get_tp_group=lambda:NS(rank_in_group=state.rank,world_size=2)
sys.modules['vllm.distributed']=distributed
sys.modules['vllm.distributed.hybrid_parallel']=hybrid_parallel
from dots3_hybrid_multimodal import create_owned_tower,broadcast_embeddings
from port_hybrid_multimodal import patch

os.environ['VLLM_HYBRID_LAYER_PARTITION']='23,23'
os.environ['VLLM_HYBRID_MM_OWNERS']='0,1'
for rank in (0,1):
    state.rank=rank
    for name,owner in (('visual',0),('audio_tower',1)):
        called=[]
        result=create_owned_tower(lambda:called.append(True) or 'constructed',name)
        assert bool(called)==(rank==owner)
        assert result==('constructed' if rank==owner else None)
os.environ.pop('VLLM_HYBRID_MM_OWNERS')
assert create_owned_tower(lambda:'native','visual')=='native'

for owner in (0,1):
    for lengths in ((3,5),(0,7,1),(0,0)):
        barrier=Barrier(2,timeout=10); holder={}; calls=[]
        class Transport:
            def __init__(self,rank):self.rank=rank;self.step=0
            def broadcast(self,tensor,root):
                step=self.step;self.step+=1
                if self.rank==root:holder[step]=tensor.clone()
                barrier.wait();tensor.copy_(holder[step]);barrier.wait()
        def run(rank):
            transport=Transport(rank)
            def encode():
                calls.append(rank)
                return tuple(torch.full((n,7),i+.5,dtype=torch.bfloat16) for i,n in enumerate(lengths))
            output=broadcast_embeddings(transport=transport,owner=owner,item_count=len(lengths),
                                        device=torch.device('cpu'),encode=encode)
            for i,(n,tensor) in enumerate(zip(lengths,output)):
                assert tensor.shape==(n,7) and tensor.dtype==torch.bfloat16
                assert torch.all(tensor==i+.5)
            return transport.step
        with ThreadPoolExecutor(2) as pool:steps=list(pool.map(run,(0,1)))
        assert calls==[owner]
        assert steps==[2 if sum(lengths) else 1]*2
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);name='models/dots3_note/nvidia/multimodal.py'
    path=root/name;path.parent.mkdir(parents=True)
    path.write_bytes((Path(__file__).resolve().parents[1]/'.cache/vllm-v0.30.0/vllm'/name).read_bytes())
    patch(root)
    assert path.read_text().count('_create_owned_tower(lambda:')==2
print('Owner MM CPU checks passed: no peer construction, native unset behavior, owner-only encode, ragged BF16 outputs, zero-length items, two-rank collective order, source syntax')
