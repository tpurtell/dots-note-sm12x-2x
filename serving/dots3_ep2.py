"""Experimental static EP2 adapter; emits local partials for vLLM to reduce.

No automatic registration. Select Dots3B12xEp2Method only for routed EXL3
experts with --enable-expert-parallel, retaining attention TP2 and DP1/DCP1.
"""
import threading
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.fused_moe import MoEActivation
from vllm.model_executor.layers.quantization.exl3 import Exl3MoEParameter, Exl3LinearMethod, _MCG_SENTINEL
try:
    from .dots3_exl3_fp8 import Dots3B12xExl3MoEMethod
except ImportError:
    from dots3_exl3_fp8 import Dots3B12xExl3MoEMethod

_PRIMERS = {}
_SESSIONS = {}


def validate_placement(expert_map, local_experts, global_experts=256):
    if expert_map.ndim != 1 or expert_map.numel() != global_experts:
        raise ValueError('EP map must cover the complete global namespace')
    values = tuple(int(x) for x in expert_map.detach().cpu().tolist())
    if any(x < -1 or x >= local_experts for x in values):
        raise ValueError('invalid EP local expert index')
    if sorted(x for x in values if x >= 0) != list(range(local_experts)):
        raise ValueError('EP placement must own each local slot exactly once')
    inverse = [None] * local_experts
    for global_id, local_id in enumerate(values):
        if local_id >= 0:
            inverse[local_id] = global_id
    return values, tuple(inverse)


def make_global_loader(global_to_local):
    """Parameter loader receives GLOBAL IDs directly from model expert mapping."""
    def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        del weight_name
        if not 0 <= expert_id < len(global_to_local):
            raise ValueError('global expert ID outside checkpoint namespace')
        local_id = global_to_local[expert_id]
        if local_id < 0:
            return False if return_success else None
        param.load_exl3_weight(loaded_weight, expert_id=local_id, shard_id=shard_id)
        return True if return_success else None
    load.supports_moe_loading = True
    return load


def route_plan(experts, capacity, top_k, global_experts=256):
    from b12x.moe import fused_moe
    return fused_moe.plan_execution(experts=experts, capacity=fused_moe.ExecutionCapacity(
        max_tokens=capacity, top_k=top_k, route_num_experts=global_experts))


class Dots3B12xEp2Method(Dots3B12xExl3MoEMethod):
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        cfg = get_current_vllm_config()
        parallel = self.moe.moe_parallel_config
        if (not parallel.use_ep or parallel.ep_size != 2 or parallel.tp_size != 1
                or cfg.parallel_config.tensor_parallel_size != 2
                or cfg.parallel_config.data_parallel_size != 1
                or cfg.parallel_config.decode_context_parallel_size != 1
                or cfg.parallel_config.prefill_context_parallel_size != 1
                or parallel.enable_eplb or self.moe.is_sequence_parallel):
            raise ValueError('Dots3 EP candidate requires static TP2+EP2, DP1, DCP1, PCP1, no SP/EPLB')
        if (num_experts, hidden_size, intermediate_size_per_partition, params_dtype) != (128,5120,1536,torch.bfloat16):
            raise ValueError('Dots3 EP requires128 whole experts, H5120,N1536,BF16 per rank')
        if self.quant_config.rank_sliced_metadata is not None or self.moe.has_bias:
            raise ValueError('EP candidate requires ordinary bias-free uniform K4 checkpoint')
        if attrs.get('global_num_experts',256) != 256 or layer.expert_map is None:
            raise ValueError('missing canonical256-expert placement')
        forward, inverse = validate_placement(layer.expert_map,128)
        layer.dots3_ep_global_to_local, layer.dots3_ep_local_to_global = forward, inverse
        layer.exl3_tp_rank, layer.exl3_tp_size = 0,1
        layer.exl3_hidden_size = hidden_size
        layer.exl3_intermediate_size_per_partition = intermediate_size_per_partition
        layer.exl3_params_dtype = params_dtype
        layer.dots3_b12x_capacity = cfg.scheduler_config.max_num_batched_tokens
        loader = make_global_loader(forward)
        for prefix, shards in (('w13',('w1','w3')),('w2',('w2',))):
            for suffix in ('suh','svh','trellis','mcg','mul1'):
                layer.register_parameter(f'{prefix}_{suffix}',Exl3MoEParameter(
                    weight_loader=loader,num_experts=num_experts,shard_ids=shards,preallocate=False))

    def _validate_codebooks(self, layer):
        for local_id,global_id in enumerate(layer.dots3_ep_local_to_global):
            for shard,projection in (('w1',layer.ckpt_gate_proj_name),('w3',layer.ckpt_up_proj_name),('w2',layer.ckpt_down_proj_name)):
                prefix = f'{layer.layer_name}.{global_id}.{projection}'
                if self.quant_config.codebook_for_prefix(prefix) != 'mcg':
                    raise ValueError(f'EP source is not MCG: {prefix}')
                group = 'w2' if shard == 'w2' else 'w13'
                key = (local_id,shard)
                marker = getattr(layer,f'{group}_mcg').exl3_tensors.get(key)
                if marker is None or key in getattr(layer,f'{group}_mul1').exl3_tensors:
                    raise ValueError(f'EP codebook marker mismatch: {prefix}')
                Exl3LinearMethod._validate_marker(marker,_MCG_SENTINEL,'mcg')

    @classmethod
    def _shard_tensors_for_tensor_parallel(cls,layer):
        # All three projections, including rotation vectors, remain whole.
        if layer.exl3_tp_size != 1 or layer.exl3_intermediate_size_per_partition !=1536:
            raise ValueError('EP expert tensors must not undergo TP slicing')

    def process_weights_after_loading(self,layer):
        super().process_weights_after_loading(layer)
        device = layer.w13_trellis.device
        layer.register_buffer('dots3_ep_route_map',torch.tensor(
            layer.dots3_ep_global_to_local,dtype=torch.int32,device=device),persistent=False)
        layer.dots3_b12x_plan = route_plan(layer.dots3_b12x_experts,
            layer.dots3_b12x_capacity,int(layer.top_k))

    @staticmethod
    def _prepare_plan(layer,x):
        from b12x.preparation import PreparedCall,PreparationSession
        plan = layer.dots3_b12x_plan
        if plan.prepared is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('EP plan must be prepared before graph capture')
        capacity,hidden,topk = layer.dots3_b12x_capacity,layer.exl3_hidden_size,int(layer.top_k)
        key = (x.device,capacity,hidden,topk,threading.get_ident())
        if key not in _PRIMERS:
            a = torch.randn((capacity,hidden),dtype=x.dtype,device=x.device)*.125
            ids = torch.arange(capacity*topk,dtype=torch.int32,device=x.device).reshape(capacity,topk)%256
            weights = torch.full((capacity,topk),1/topk,dtype=torch.float32,device=x.device)
            output = torch.empty((capacity,hidden),dtype=torch.float32,device=x.device)
            _PRIMERS[key] = (a,ids,weights,output)
        a,ids,weights,output = _PRIMERS[key]
        def prepare(state):
            scratch = Dots3B12xExl3MoEMethod._scratch(state.scratch,x.device)
            binding = state.bind(scratch=scratch,a=a,experts=layer.dots3_b12x_experts,
                topk_ids=ids,topk_weights=weights,route_expert_map=layer.dots3_ep_route_map,output=output)
            return PreparedCall(run=binding.run,output=output,owners=(scratch,a,ids,weights,binding))
        sessionkey = (x.device,threading.get_ident())
        if sessionkey not in _SESSIONS:
            _SESSIONS[sessionkey] = PreparationSession(device=x.device,autotune=False,compile_workers=2)
        _SESSIONS[sessionkey].prepare((plan.request(name=f'dots3-ep:{layer.layer_name}',prepare_call=prepare),))

    def apply(self,layer,x,topk_weights,topk_ids,shared_experts,shared_experts_input):
        del shared_experts,shared_experts_input
        if layer.activation != MoEActivation.SILU or layer.apply_router_weight_on_input:
            raise ValueError('EP candidate requires SiLU with output router weighting')
        from b12x.moe import fused_moe
        shape = x.shape
        inputs = x.reshape(-1,shape[-1]).contiguous()
        ids = topk_ids.reshape(inputs.shape[0],-1).to(torch.int32).contiguous()
        weights = topk_weights.reshape_as(ids).to(torch.float32).contiguous()
        self._prepare_plan(layer,inputs)
        binding = fused_moe.bind(layer.dots3_b12x_plan,
            scratch=self._scratch(layer.dots3_b12x_plan,inputs.device),a=inputs,
            experts=layer.dots3_b12x_experts,topk_ids=ids,topk_weights=weights,
            route_expert_map=layer.dots3_ep_route_map)
        # The vLLM runner owns the one final reduction, including shared experts.
        return fused_moe.run(binding=binding).to(inputs.dtype).reshape(shape)
