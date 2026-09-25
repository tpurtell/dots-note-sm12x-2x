"""Optional owner-local embeddings and vocabulary projection for Dots hybrid.

Target embeddings broadcast hidden states. Target logits broadcast the complete
native vocabulary to preserve both TP samplers. Draft embeddings live only on
the draft decoder owner; greedy draft output broadcasts token IDs only.
"""
import os
import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.distributed.hybrid_parallel import PyNcclOwnerTransport
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.layernorm import RMSNorm


def placement():
    value=os.environ.get('VLLM_HYBRID_BOUNDARY_OWNERS','')
    if not value:return None
    partition=os.environ.get('VLLM_HYBRID_LAYER_PARTITION','')
    if not partition:raise ValueError('boundary ownership requires hybrid decoder ownership')
    pair=tuple(int(x) for x in value.split(','))
    if len(pair)!=2 or any(not 0<=x<get_tp_group().world_size for x in pair):
        raise ValueError('boundary owners must be target-embedding,head worker ranks')
    return {'target':pair[0],'head':pair[1],'draft':len(partition.split(','))-1}


class RemoteBoundary(PPMissingLayer):
    """Parameter-free checkpoint-loader sentinel with explicit owner metadata."""
    def __init__(self, num_embeddings, embedding_dim, *, owner, role, dtype):
        super().__init__()
        self._hybrid_owned_boundary=True
        self.hybrid_owner=owner;self.hybrid_role=role
        self.num_embeddings=num_embeddings;self.embedding_dim=embedding_dim
        self.output_dtype=dtype
        self.transport=PyNcclOwnerTransport(get_tp_group())
    def forward(self, input_ids):
        if self.hybrid_role=='head':raise RuntimeError('remote head must use owner logits processor')
        output=torch.empty((*input_ids.shape,self.embedding_dim),device=input_ids.device,dtype=self.output_dtype)
        if self.hybrid_role=='target':self.transport.broadcast(output,self.hybrid_owner)
        # Draft peer data is not consumed: its decoder owner computes the full
        # embedding table gather and nonowner dense state is deliberately absent.
        return output


class OwnerEmbedding(VocabParallelEmbedding):
    def __init__(self,*args,owner,role,**kwargs):
        kwargs['disable_tp']=True
        super().__init__(*args,**kwargs)
        self._hybrid_owned_boundary=True
        self.hybrid_owner=owner;self.hybrid_role=role
        self.transport=PyNcclOwnerTransport(get_tp_group())
    def forward(self,input_ids):
        output=super().forward(input_ids)
        if self.hybrid_role=='target':self.transport.broadcast(output,self.hybrid_owner)
        return output


def embedding(*args,role,**kwargs):
    plan=placement()
    if plan is None:return VocabParallelEmbedding(*args,**kwargs)
    owner=plan[role]
    if get_tp_group().rank_in_group==owner:
        return OwnerEmbedding(*args,owner=owner,role=role,**kwargs)
    return RemoteBoundary(args[0],args[1],owner=owner,role=role,
                          dtype=kwargs.get('params_dtype') or torch.get_default_dtype())


def head(*args,model_type='dots3_note',**kwargs):
    plan=placement()
    if plan is None:return ParallelLMHead(*args,**kwargs)
    if model_type!='dots3_note':raise ValueError('boundary adapter currently supports Dots only')
    owner=plan['head']
    if get_tp_group().rank_in_group!=owner:
        return RemoteBoundary(args[0],args[1],owner=owner,role='head',
                              dtype=kwargs.get('params_dtype') or torch.get_default_dtype())
    kwargs['disable_tp']=True
    result=ParallelLMHead(*args,**kwargs)
    result._hybrid_owned_boundary=True;result.hybrid_owner=owner;result.hybrid_role='head'
    return result


class OwnerLogitsProcessor(LogitsProcessor):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.transport=PyNcclOwnerTransport(get_tp_group())
    def forward(self,lm_head,hidden_states,embedding_bias=None,skip_gather=False):
        if not getattr(lm_head,'_hybrid_owned_boundary',False):
            return super().forward(lm_head,hidden_states,embedding_bias,skip_gather)
        if skip_gather:raise ValueError('owner full-vocabulary path does not expose local logits shards')
        owner=lm_head.hybrid_owner
        if self.transport.rank==owner:
            logits=super().forward(lm_head,hidden_states,embedding_bias).contiguous()
        else:
            logits=torch.empty((*hidden_states.shape[:-1],self.org_vocab_size),
                device=hidden_states.device,dtype=self.head_dtype or hidden_states.dtype)
        self.transport.broadcast(logits,owner)
        return logits
    def get_top_tokens(self,lm_head,hidden_states,embedding_bias=None):
        if not getattr(lm_head,'_hybrid_owned_boundary',False):
            return super().get_top_tokens(lm_head,hidden_states,embedding_bias)
        owner=lm_head.hybrid_owner
        if self.transport.rank==owner:
            result=super().get_top_tokens(lm_head,hidden_states,embedding_bias)
        else:
            result=torch.empty(hidden_states.shape[:-1],device=hidden_states.device,dtype=torch.int64)
        self.transport.broadcast(result,owner)
        return result


def logits_processor(*args,**kwargs):
    return (OwnerLogitsProcessor if placement() is not None else LogitsProcessor)(*args,**kwargs)


def shared_head(*,config,prefix,quant_config=None):
    if placement() is None:
        from vllm.model_executor.models.deepseek_mtp import SharedHead
        return SharedHead(config=config,prefix=prefix,quant_config=quant_config)
    # Native MTP load_eagle_model shares target.lm_head after loading. A missing
    # placeholder avoids allocating an unused full draft copy beforehand.
    result=nn.Module()
    result.norm=RMSNorm(config.hidden_size,eps=config.rms_norm_eps)
    result.head=RemoteBoundary(config.vocab_size,config.hidden_size,
        owner=placement()['head'],role='head',dtype=torch.get_default_dtype())
    return result
