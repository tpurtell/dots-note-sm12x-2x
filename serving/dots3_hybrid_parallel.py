"""Dots decoder adapter for layer ownership with routed expert TP.

Opt-in VLLM_HYBRID_LAYER_PARTITION contains contiguous decoder counts per
rank. Boundary embedding/vocabulary interfaces keep the runner's ordinary TP
contract. Every routed expert remains tensor-sharded over all workers.
"""
import os
import re
import torch
from torch import nn
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.distributed.hybrid_parallel import (
    LayerOwnerPlan, DenseParallelContext, PyNcclOwnerTransport,
    RoutedLayerBuffers, allocate_packed_routing, execute_routed_layer, transfer_owner_state,
)
from vllm.model_executor.models.utils import PPMissingLayer

_NATIVE = None
_ARENAS = {}
_LOG = init_logger(__name__)


def enabled():
    return bool(os.environ.get('VLLM_HYBRID_LAYER_PARTITION', ''))


def owner_plan(config):
    counts = tuple(int(x) for x in os.environ['VLLM_HYBRID_LAYER_PARTITION'].split(','))
    if any(x <= 0 for x in counts) or sum(counts) != config.num_hidden_layers:
        raise ValueError('hybrid partition must assign every decoder layer exactly once')
    if len(counts) != get_tp_group().world_size:
        raise ValueError('hybrid partition must have one count per expert-TP rank')
    owners = tuple(rank for rank, count in enumerate(counts) for _ in range(count))
    return LayerOwnerPlan(owners, len(counts), owners[-1])


def _arena(device, dtype, capacity, hidden, topk):
    packed = os.environ.get("VLLM_HYBRID_PACKED_ROUTING", "0") == "1"
    key = (device, dtype, capacity, hidden, topk, packed)
    if key not in _ARENAS:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('hybrid buffers must be prepared before graph capture')
        if packed:
            _ARENAS[key] = allocate_packed_routing(capacity=capacity, hidden=hidden,
                                                  topk=topk, dtype=dtype, device=device)
            return _ARENAS[key]
        _ARENAS[key] = RoutedLayerBuffers(
            torch.empty((capacity, hidden), device=device, dtype=dtype),
            torch.empty((capacity, topk), device=device, dtype=torch.int32),
            torch.empty((capacity, topk), device=device, dtype=torch.float32),
            torch.empty((capacity, hidden), device=device, dtype=dtype),
        )
    return _ARENAS[key]


class Dots3HybridDecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, config=None, topk_indices_buffer=None):
        super().__init__()
        native = _NATIVE
        config = config or vllm_config.model_config.hf_config
        parallel = vllm_config.parallel_config
        if (parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel
                or parallel.decode_context_parallel_size != 1
                or parallel.prefill_context_parallel_size != 1
                or parallel.enable_eplb or parallel.use_sequence_parallel_moe):
            raise ValueError('hybrid layer ownership requires expert TP, PP1, DCP1, PCP1, no EP/EPLB/SP')
        self.layer_idx = int(prefix.rsplit('.', 1)[1])
        self.plan = owner_plan(config)
        self.owner = self.plan.owners[self.layer_idx] if self.layer_idx < len(self.plan.owners) else self.plan.output_owner
        self.transport = PyNcclOwnerTransport(get_tp_group())
        self.rank = self.transport.rank
        self.context = DenseParallelContext(self.owner, self.rank)
        self.hidden_size = config.hidden_size
        self.capacity = vllm_config.scheduler_config.max_num_batched_tokens
        self.top_k = config.num_experts_per_tok
        self.pack_owner = None
        if os.environ.get('VLLM_HYBRID_FUSED_PACK', '0') == '1':
            if os.environ.get('VLLM_HYBRID_PACKED_ROUTING', '0') != '1':
                raise ValueError('fused routing pack requires packed routing')
            from vllm.distributed.hybrid_pack_kernel import pack_routing
            self.pack_owner = pack_routing
        self.overlap_shared = os.environ.get('VLLM_HYBRID_OVERLAP_SHARED', '0') == '1' 
        self.use_sequence_parallel = False
        self.use_sequence_parallel_moe = False
        self.use_mha = False
        self.tp_size = parallel.tensor_parallel_size
        self.routed_scaling_factor = config.routed_scaling_factor
        self.is_hybrid_owner_layer = True
        self.is_draft = self.layer_idx >= config.num_hidden_layers
        self.is_moe = (not self.is_draft and config.n_routed_experts is not None
            and self.layer_idx >= config.first_k_dense_replace
            and self.layer_idx % getattr(config, 'moe_layer_freq', 1) == 0)
        self.has_checkpoint_decoder_parameters = self.context.owns_parameters or self.is_moe
        quant = vllm_config.quant_config
        if self.context.owns_parameters:
            attention = native['Dots3NoteSlidingAttention'] if config.layer_types[self.layer_idx] == 'sliding_attention' else native['Dots3NoteFullAttention']
            self.self_attn = attention(vllm_config=vllm_config, config=config,
                prefix=f'{prefix}.self_attn', topk_indices_buffer=topk_indices_buffer,
                dense_parallel_context=self.context)
            self.input_layernorm = native['RMSNorm'](config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = native['RMSNorm'](config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.self_attn = PPMissingLayer()
            self.input_layernorm = PPMissingLayer()
            self.post_attention_layernorm = PPMissingLayer()
        if self.is_moe:
            # Construct expert shards under the unchanged global TP2 context.
            self.mlp = native['Dots3NoteMoE'](config=config, parallel_config=parallel,
                quant_config=quant, reduce_results=False, prefix=f'{prefix}.mlp',
                apply_routed_scale_to_output=False)
            if self.context.owns_parameters:
                self.mlp.shared_experts = native['DeepseekV2MLP'](
                    hidden_size=config.hidden_size,
                    intermediate_size=native['_padded_mlp_size'](config.moe_intermediate_size * config.n_shared_experts, quant),
                    hidden_act=config.hidden_act, quant_config=quant,
                    is_sequence_parallel=True, reduce_results=False,
                    prefix=f'{prefix}.mlp.shared_experts')
            else:
                self.mlp.shared_experts = PPMissingLayer()
                self.mlp.gate = PPMissingLayer()
                self.mlp.experts.gate = None
                self.mlp.experts.router.e_score_correction_bias = None
                if hasattr(self.mlp.experts.routed_experts, "e_score_correction_bias"):
                    self.mlp.experts.routed_experts.e_score_correction_bias = None
        elif self.context.owns_parameters:
            self.mlp = native['DeepseekV2MLP'](hidden_size=config.hidden_size,
                intermediate_size=native['_padded_mlp_size'](config.intermediate_size, quant),
                hidden_act=config.hidden_act, quant_config=quant,
                is_sequence_parallel=True, reduce_results=False, prefix=f'{prefix}.mlp')
        else:
            self.mlp = PPMissingLayer()

    def forward(self, positions, hidden_states, residual, attn_in=None):
        rows = positions.shape[0]
        if rows > self.capacity:
            raise ValueError('hybrid active rows exceed prepared capacity')
        arena = _arena(hidden_states.device, hidden_states.dtype, self.capacity, self.hidden_size, self.top_k)
        if self.layer_idx > 0 and not self.is_draft:
            previous = self.plan.owners[self.layer_idx - 1]
            if previous != self.owner:
                if residual is None: raise RuntimeError('missing owner residual at partition boundary')
                transfer_owner_state(self.transport, previous_owner=previous, next_owner=self.owner,
                    hidden=hidden_states, residual=residual)
        owner = self.context.owns_parameters
        if owner:
            if residual is None:
                residual = hidden_states
                hidden_states = attn_in if attn_in is not None else self.input_layernorm(hidden_states)
            else:
                # Prior routed output is already reduced to this owner.
                hidden_states, residual = self.input_layernorm(hidden_states, residual)
            hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        if self.is_moe:
            runner = self.mlp.experts
            shared = None
            def prepare():
                nonlocal shared
                logits, _ = self.mlp.gate(hidden_states)
                weights, ids = runner.router.select_experts(hidden_states, logits,
                    topk_indices_dtype=runner.routed_experts.quant_method.topk_indices_dtype)
                if self.overlap_shared:
                    from vllm.distributed.hybrid_shared_overlap import launch_shared
                    shared = launch_shared(self.mlp.shared_experts, hidden_states)
                else:
                    shared = self.mlp.shared_experts(hidden_states)
                return hidden_states, ids, weights
            def experts(x, ids, weights):
                runner.routed_experts._ensure_moe_quant_config_init()
                return runner.routed_experts.forward_modular(x, weights, ids)
            def finish(reduced):
                return reduced + (shared.join() if self.overlap_shared else shared)
            output = execute_routed_layer(owner=self.owner, transport=self.transport,
                buffers=arena.active(rows),
                prepare_owner=prepare, execute_local_experts=experts, finish_owner=finish,
                pack_owner=self.pack_owner)
        else:
            output = self.mlp(hidden_states) if owner else None
        if not owner:
            # Peer values carry shape/dtype only; boundaries explicitly replace
            # them from the previous owner before a different owner consumes them.
            output = arena.reduced_output[:rows]
            if residual is None: residual = torch.empty_like(output)
        return output, residual


def hybrid_model_forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
    if intermediate_tensors is not None:
        raise ValueError('hybrid layer ownership does not use ordinary PP intermediates')
    hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
    residual = None
    for layer in self.layers:
        hidden, residual = layer(positions, hidden, residual)
    last = self.layers[-1]
    if last.context.owns_parameters:
        hidden, _ = self.norm(hidden, residual)
    # Preserve vLLM's external TP2 vocabulary/sampling contract.
    last.transport.broadcast(hidden, last.owner)
    return hidden


def _filter_weights(self, weights):
    plan = self.layers[0].plan
    rank = get_tp_group().rank_in_group
    for name, value in weights:
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', name)
        if match:
            index = int(match.group(1))
            if index < len(plan.owners) and plan.owners[index] != rank and '.mlp.experts.' not in name:
                continue
        elif name.startswith('norm.') and rank != plan.output_owner:
            continue
        yield name, value


def install_model(namespace):
    global _NATIVE
    _NATIVE = namespace
    if not enabled(): return
    namespace['Dots3NoteDecoderLayer'] = Dots3HybridDecoderLayer
    model = namespace['Dots3NoteModel']
    original_init, original_load = model.__init__, model.load_weights
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.layers[-1].context.owns_parameters:
            self.norm = PPMissingLayer()
        owned = [layer.layer_idx for layer in self.layers if layer.context.owns_parameters]
        routed = [layer.layer_idx for layer in self.layers if layer.is_moe]
        _LOG.info("Hybrid layer ownership rank=%d owners=%s local_dense_layers=%s routed_expert_tp_layers=%s boundary_embedding_vocab=TP",
                  get_tp_group().rank_in_group, self.layers[0].plan.owners, owned, routed)
    def load(self, weights):
        return original_load(self, _filter_weights(self, weights))
    model.__init__, model.load_weights, model.forward = init, load, hybrid_model_forward


def install_mtp(namespace):
    if not enabled(): return
    layer_cls = namespace['Dots3NoteMultiTokenPredictorLayer']
    original_init = layer_cls.__init__
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not self.mtp_block.context.owns_parameters:
            self.enorm = PPMissingLayer()
            self.hnorm = PPMissingLayer()
            self.eh_proj = PPMissingLayer()
            self.shared_head.norm = PPMissingLayer()
    def forward(self, input_ids, positions, previous_hidden_states,
                inputs_embeds=None, embed_table=None, spec_step_index=0):
        block = self.mtp_block
        if block.context.owns_parameters:
            if embed_table is not None:
                eh_input = namespace['fused_embed_eh_norm'](positions, input_ids, embed_table,
                    previous_hidden_states, self.enorm.weight, self.hnorm.weight,
                    self.enorm.variance_epsilon)
            else:
                if inputs_embeds is None: raise ValueError('missing MTP embedding input')
                eh_input = namespace['fused_eh_norm'](positions, inputs_embeds,
                    previous_hidden_states, self.enorm.weight, self.hnorm.weight,
                    self.enorm.variance_epsilon)
            hidden = self.eh_proj(eh_input)[0]
        else:
            hidden = torch.empty_like(previous_hidden_states)
        hidden, residual = block(positions, hidden, None)
        if block.context.owns_parameters:
            # The complete dense MTP block lives on its owner. No TP sum.
            hidden, _ = self.shared_head.norm(hidden, residual)
        block.transport.broadcast(hidden, block.owner)
        return hidden
    layer_cls.__init__, layer_cls.forward = init, forward
    model_cls = namespace['Dots3NoteMTP']
    original_adapt = model_cls._adapt_weights
    def adapt(self, weights):
        from vllm.model_executor.models.utils import is_pp_missing_parameter
        from vllm.model_executor.models.deepseek_mtp import get_spec_layer_idx_from_weight_name
        for name, weight in original_adapt(self, weights):
            spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
            rewritten = self._rewrite_spec_layer_name(spec_layer, name) if spec_layer is not None else name
            if is_pp_missing_parameter(rewritten, self): continue
            yield name, weight
    model_cls._adapt_weights = adapt
