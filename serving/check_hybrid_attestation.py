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
    def named_modules(self): return iter([('projection',NS(tp_size=1,tp_rank=0))])

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
