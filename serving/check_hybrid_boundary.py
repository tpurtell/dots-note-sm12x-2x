#!/usr/bin/env python3
"""CPU actual native embedding/head loader plus emulated owner collectives."""
import os
import sys
import tempfile
from pathlib import Path
from threading import Barrier,local
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import torch
import hybrid_parallel
sys.modules['vllm.distributed.hybrid_parallel']=hybrid_parallel
import dots3_hybrid_boundary as boundary
from vllm.config import set_current_vllm_config
from port_hybrid_boundary import patch

state=local()
boundary.get_tp_group=lambda:NS(rank_in_group=state.rank,world_size=2)
os.environ['VLLM_HYBRID_LAYER_PARTITION']='23,23'
os.environ['VLLM_HYBRID_BOUNDARY_OWNERS']='0,1'
barrier=Barrier(2,timeout=20); holder={}
class Transport:
    def __init__(self,group):self.rank=group.rank_in_group;self.step=0
    def broadcast(self,tensor,owner):
        key=self.step;self.step+=1
        if self.rank==owner:holder[key]=tensor.detach().clone()
        barrier.wait();tensor.copy_(holder[key]);barrier.wait()
boundary.PyNcclOwnerTransport=Transport

# Native vLLM constructors/loaders run unmodified with explicit disable_tp.
modules=[]
for rank in (0,1):
    state.rank=rank
    embedding=boundary.embedding(19,4,role='target',params_dtype=torch.float32)
    head=boundary.head(19,4,params_dtype=torch.float32)
    draft=boundary.embedding(19,4,role='draft',params_dtype=torch.float32)
    assert (sum(p.numel() for p in embedding.parameters())>0)==(rank==0)
    assert (sum(p.numel() for p in head.parameters())>0)==(rank==1)
    assert (sum(p.numel() for p in draft.parameters())>0)==(rank==1)
    for module in (embedding,head,draft):
        if hasattr(module,'weight'):
            assert module.tp_size==1
            weight=torch.arange(19*4,dtype=torch.float32).view(19,4)/100
            module.weight_loader(module.weight,weight)
            assert torch.equal(module.weight[:19],weight)
    modules.append((embedding,head,draft))

config=NS(model_config=NS(head_dtype=None))
with set_current_vllm_config(config):
    processors=[]
    for rank in (0,1):
        state.rank=rank
        processors.append(boundary.logits_processor(19,scale=.7,soft_cap=3.0))
    def run(rank):
        state.rank=rank
        embedding,head,draft=modules[rank]
        processor=processors[rank]
        tokens=torch.tensor([0,3,18])
        x=embedding(tokens)
        expected=torch.arange(19*4,dtype=torch.float32).view(19,4)/100
        assert torch.equal(x,expected[tokens])
        logits=processor(head,x)
        expected_logits=torch.tanh((x@expected.T)/3)*3*.7
        assert torch.allclose(logits,expected_logits)
        ids=processor.get_top_tokens(head,x)
        assert torch.equal(ids,expected_logits.argmax(dim=-1))
        assert processor.transport.step==2
        if rank==0:
            assert not hasattr(draft,'weight')
            assert draft(tokens).shape==(3,4)  # no peer collective for draft
        else:assert torch.equal(draft(tokens),expected[tokens])
    # Processor and embedding transports have independent step counters;
    # clear completed broadcast keys is unnecessary because barriers order use.
    with torch.no_grad(),ThreadPoolExecutor(2) as pool:list(pool.map(run,(0,1)))

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    names=('model_executor/models/deepseek_v2.py','models/dots3_note/nvidia/model.py',
           'models/dots3_note/nvidia/mtp.py','v1/worker/gpu/spec_decode/eagle/utils.py')
    source=Path(__file__).resolve().parents[1]/'.cache/vllm-v0.30.0/vllm'
    for name in names:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((source/name).read_bytes())
    patch(root)
    for name in names:compile((root/name).read_text(),name,'exec')
print('Boundary CPU checks passed: native full-weight loaders, zero peer parameters, identical target embeddings/logits/token IDs, owner-only draft lookup, syntax and constructor source anchors')
