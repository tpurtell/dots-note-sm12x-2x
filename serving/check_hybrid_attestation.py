#!/usr/bin/env python3
"""CPU validation of attestation failures; does not simulate GPU correctness."""
import sys
from types import SimpleNamespace as NS, ModuleType
from hybrid_attestation import attest_model

class Tensor:
    shape=(2, 4)
    dtype='uint8'
    device='cuda:0'
    def __init__(self, pointer): self.pointer=pointer
    def stride(self): return (4,1)
    def storage_offset(self): return 0
    def untyped_storage(self): return self
    def nbytes(self): return 8
    def data_ptr(self): return self.pointer
    def numel(self): return 8
    def element_size(self): return 1
class Module:
    def __init__(self, owned=False): self.owned=owned
    def parameters(self): return iter([Tensor(1)] if self.owned else [])
    def buffers(self): return iter([])
    def named_modules(self): return iter([('projection',NS(tp_size=1,tp_rank=0))])

class ReplicatedLinear: pass
linear=ModuleType('vllm.model_executor.layers.linear');linear.ReplicatedLinear=ReplicatedLinear
sys.modules['vllm.model_executor.layers.linear']=linear

fake=ModuleType('vllm.distributed');fake.get_tp_group=lambda:NS(world_size=2,rank_in_group=0)
sys.modules['vllm.distributed']=fake
config=NS(first_k_dense_replace=1,num_hidden_layers=3,swa_num_attention_heads=64,num_attention_heads=128,
          layer_types=['sliding_attention','deepseek_sparse_attention','sliding_attention'],n_routed_experts=256,moe_intermediate_size=1536)
plan=NS(owners=(0,0,1),output_owner=1)
layers=[]
for i in range(3):
    owned=i<2
    attn=Module(owned);attn.mla_attn=NS(num_heads=128 if i==1 else 64)
    prepared=NS(num_experts=256,hidden_size=5120,intermediate_size=768,_impl=NS(w1_fp4=Tensor(10+i),w2_fp4=Tensor(20+i)))
    expert=NS(exl3_tp_size=2,exl3_tp_rank=0,dots3_b12x_experts=prepared)
    layers.append(NS(layer_idx=i,owner=0 if owned else 1,context=NS(owns_parameters=owned),
        self_attn=attn,input_layernorm=Module(owned),post_attention_layernorm=Module(owned),
        is_moe=i>0,mlp=NS(gate=Module(owned),shared_experts=Module(owned),experts=NS(routed_experts=expert)),plan=plan))
layers[0].mlp=Module(True)
model=NS(language_model=NS(model=NS(config=config,layers=layers)))
context={'model.layers.0.attn':NS(kv_cache=Tensor(100)), 'model.layers.1.attn':NS(kv_cache=Tensor(100))}
vconfig=NS(compilation_config=NS(static_forward_context=context))
receipt=attest_model(model,vconfig)
assert receipt['passed'],receipt['errors']
assert receipt['unique_kv_storage_bytes']==8  # aliases counted once
layers[2].mlp.experts.routed_experts.dots3_b12x_experts.intermediate_size=1536
assert not attest_model(model,vconfig)['passed']
layers[2].mlp.experts.routed_experts.dots3_b12x_experts.intermediate_size=768
context['model.layers.2.attn']=NS(kv_cache=Tensor(200))
assert any('nonowner registered cache' in x for x in attest_model(model,vconfig)['errors'])
print('Hybrid attestation CPU checks: geometry failure, nonowner KV rejection, pool alias byte dedup passed')

# Actual storage aliases count once; parameter and buffer logical bytes remain
# separate, so this also detects accidental double-counting in MM reporting.
from hybrid_attestation import _module_storage, write_startup_receipt
mm = NS(parameters=lambda: iter([Tensor(1), Tensor(2)]), buffers=lambda: iter([Tensor(1)]))
summary = _module_storage(mm)
assert summary['parameter_bytes'] == 16 and summary['buffer_bytes'] == 8
assert summary['unique_storage_bytes'] == 16

import tempfile
import json
from pathlib import Path
class Executor:
    def __init__(self, passed): self.passed = passed
    def collective_rpc(self, method):
        assert method == 'hybrid_ownership_receipt'
        return [{'rank':rank,'passed':self.passed,'errors':[] if self.passed else ['fixture']} for rank in range(2)]
with tempfile.TemporaryDirectory() as tmp:
    target=Path(tmp)/'receipt.json'
    write_startup_receipt(Executor(True),target)
    assert json.loads(target.read_text())['passed']
    try: write_startup_receipt(Executor(False),target)
    except RuntimeError: pass
    else: raise AssertionError('failed ownership did not stop startup')
    assert not json.loads(target.read_text())['passed']
print('MM storage alias accounting and atomic startup RPC receipt/error gates passed')

from hybrid_attestation import _parallel_module_evidence
replicated=ReplicatedLinear()
replicated.tp_size=2; replicated.tp_rank=1
replicated.input_size=2; replicated.output_size=4
replicated.output_partition_sizes=[4]; replicated.weight=Tensor(300)
assert _parallel_module_evidence(replicated)[1]
replicated.output_partition_sizes=[2]
assert not _parallel_module_evidence(replicated)[1]
assert not _parallel_module_evidence(NS(tp_size=2,tp_rank=1))[1]
print('Replicated indexer full-weight evidence passes; partial weight and nonreplicated TP2 still rejected')

# CPU buffers remain visible in total storage but cannot inflate modeled CUDA
# weights. Draft ownership checks include norms/projection and target head alias.
cpu_tensor=Tensor(777);cpu_tensor.device='cpu'
summary=_module_storage(NS(parameters=lambda:iter([Tensor(778)]),buffers=lambda:iter([cpu_tensor])))
assert summary['unique_storage_bytes']==16 and summary['cuda_storage_bytes']==8
assert summary['unique_storage_bytes_by_device']=={'cuda:0':8,'cpu':8}
from hybrid_attestation import attest_draft_and_boundaries
head=Module(True)
language=NS(model=NS(embed_tokens=Module(True)),lm_head=head)
draft_layer=NS(mtp_block=layers[2],enorm=Module(),hnorm=Module(),eh_proj=Module(),
               shared_head=NS(norm=Module(),head=head))
draft=NS(model=NS(embed_tokens=Module(True),layers={'46':draft_layer}))
extra,errors=attest_draft_and_boundaries(NS(language_model=language),draft,0)
assert not errors,errors
assert extra['draft_layers'][0]['dense_storage']['cuda_storage_bytes']==0
draft_layer.eh_proj=Module(True)
assert any('nonowner dense' in e for e in attest_draft_and_boundaries(NS(language_model=language),draft,0)[1])
draft_layer.eh_proj=Module();draft_layer.shared_head.head=Module(True)
assert any('not shared' in e for e in attest_draft_and_boundaries(NS(language_model=language),draft,0)[1])
print('CUDA-only accounting, draft nonowner storage and target/draft head alias checks passed')

# Profile telemetry copies exact counters without retaining live snapshots or
# modifying native accounting. This fake has no allocation/sync APIs, so any
# accidental GPU operation in the helper fails this CPU check.
from hybrid_attestation import capture_profile_evidence
fields = ('torch_peak', 'torch_allocated', 'free_memory', 'total_memory',
          'cuda_memory', 'torch_memory', 'non_torch_memory')
snapshots = {name: NS(**{key: index * 100 + k for k, key in enumerate(fields)}, device_='cuda:1')
             for index, name in enumerate(('before_create','before_profile','after_profile'))}
stats = {'allocated_bytes.all.current': 17, 'reserved_bytes.all.current': 39,
         'inactive_split_bytes.all.current': 22, 'num_alloc_retries': 0}
def memory_stats(device):
    assert device == 'cuda:1'
    return stats
fake_torch = NS(cuda=NS(memory_stats=memory_stats, get_allocator_backend=lambda:'native'))
worker = NS()
result = NS(**snapshots, total_consumed=12345)
capture_profile_evidence(worker, result, fake_torch)
evidence = worker.hybrid_profile_evidence
assert evidence['snapshots_bytes']['after_profile']['torch_memory'] == 205
assert evidence['allocator_counters']['inactive_split_bytes.all.current'] == 22
snapshots['after_profile'].torch_memory = 0
stats['inactive_split_bytes.all.current'] = 0
assert evidence['snapshots_bytes']['after_profile']['torch_memory'] == 205
assert evidence['allocator_counters']['inactive_split_bytes.all.current'] == 22
assert result.total_consumed == 12345
json.dumps(evidence)
print('Profile snapshots and allocator counters copied without allocation, retained snapshots, or accounting changes')

# Native TP boundary tables expose storage identity too; module-local summaries
# alone cannot establish whether target and draft embeddings share allocation.
from hybrid_attestation import _boundary_storage, _modules_storage
shared=Tensor(901)
m1=NS(weight=shared,parameters=lambda:iter([shared]),buffers=lambda:iter([]))
m2=NS(weight=shared,parameters=lambda:iter([shared]),buffers=lambda:iter([]))
assert _boundary_storage(m1,0,[],'native')['weight']['storage_pointer']==901
assert _modules_storage([m1,m2])['cuda_storage_bytes']==8
print('Native boundary weight identity and combined alias dedup passed')
