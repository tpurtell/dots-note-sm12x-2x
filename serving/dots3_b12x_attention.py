"""Dots3 padded FP8 sparse MLA through B12x's planned strided-record path."""
import math

import torch

from b12x.attention.sparse_mla import strided
from b12x.preparation import PreparationSession, PreparedCall
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.attention.sparse_mla_attention import SparseMLACommonImpl
from vllm.models.dots3_note.nvidia.attention import (
    Dots3NotePaddedSparseBackend, Dots3NotePaddedSparseImpl,
)


class Dots3B12xSparseBackend(Dots3NotePaddedSparseBackend):
    supported_kv_cache_dtypes = ['fp8', 'fp8_e4m3']

    @staticmethod
    def get_name():
        return 'B12X_DOTS3_MLA_SPARSE'

    @staticmethod
    def get_impl_cls():
        return Dots3B12xSparseImpl

    @classmethod
    def supports_compute_capability(cls, capability):
        return capability.major == 12 and capability.minor in (0, 1)

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size,
                             use_mla, has_sink, use_sparse, use_mm_prefix,
                             device_capability):
        if head_size != 576 or dtype != torch.bfloat16:
            return 'Dots3 B12x requires BF16 queries with 576 QK dimensions'
        if kv_cache_dtype not in cls.supported_kv_cache_dtypes:
            return 'Dots3 B12x requires ordinary E4M3 FP8 cache'
        config = get_current_vllm_config()
        if config.model_config.hf_text_config.model_type != 'dots3_note':
            return 'Dots3 model only'
        if config.parallel_config.tensor_parallel_size != 2:
            return 'Dots3 recipe requires TP2'
        if config.parallel_config.decode_context_parallel_size != 1:
            return 'Dots3 B12x DCP integration is not implemented'
        return None


# Layer calls execute serially on each TP rank. Share planned workspace across
# sparse layers with identical geometry/capacity/cache layout.
_STATES = {}


class Dots3B12xSparseImpl(Dots3NotePaddedSparseImpl):
    def __init__(self, *args, **kwargs):
        SparseMLACommonImpl.__init__(self, *args, **kwargs)
        self.supports_quant_query_input = False
        config = get_current_vllm_config()
        self.capacity = config.scheduler_config.max_num_batched_tokens
        if self.num_heads != 64 or not math.isclose(self.scale, strided.SM_SCALE):
            raise ValueError('Dots3 B12x attention geometry/softmax scale differs')
        if self.kv_cache_dtype not in ('fp8', 'fp8_e4m3'):
            raise ValueError('Dots3 B12x requires E4M3 cache')

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        q_nope, q_rope = q
        cache = kv_c_and_k_pe_cache
        rows = q_rope.shape[0]
        physical_records = (cache.shape[0] - 1) * (cache.stride(0) // 1088) + 64
        key = (cache.device, self.capacity, cache.shape[0], physical_records)
        state = _STATES.get(key)
        if state is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Dots3 sparse attention must be prepared before capture')
            plan = strided.plan(strided.Caps(
                device=cache.device, num_q_heads=64, tp_size=2,
                max_q_rows=self.capacity, num_cache_blocks=cache.shape[0],
                max_physical_records=physical_records, use_cuda_graph=True,
            ))
            query = torch.zeros((self.capacity, 64, 576), dtype=torch.bfloat16, device=cache.device)
            output = torch.empty((self.capacity, 64, 512), dtype=torch.bfloat16, device=cache.device)
            starts = torch.arange(self.capacity + 1, dtype=torch.int32, device=cache.device)
            q_scale = torch.ones((), dtype=torch.float32, device=cache.device)
            scratch = torch.empty(plan.scratch_specs()[0].shape, dtype=torch.uint8, device=cache.device)

            def prepare(runtime):
                binding = runtime.bind_indexed(
                    scratch=scratch, q=query[:rows], kv_cache=cache, output=output[:rows],
                    logical_indices=self.topk_indices_buffer[:rows],
                    request_ids=attn_metadata.req_id_per_token[:rows],
                    block_table=attn_metadata.block_table, cu_seqlens_q=starts[:rows+1],
                    kv_scale=layer._k_scale, q_scale=q_scale,
                )
                runtime.prime(binding)
                return PreparedCall(run=lambda: runtime.run(binding), owners=(binding,))

            session = PreparationSession(device=cache.device, autotune=False)
            session.prepare((plan.request(name='dots3-sparse-tp2', prepare_call=prepare),))
            state = (plan, scratch, query, output, starts, q_scale, session)
            _STATES[key] = state
        plan, scratch, query, output, starts, q_scale, _ = state
        query[:rows, :, :512].copy_(q_nope)
        query[:rows, :, 512:].copy_(q_rope)
        # A batch-wide dynamic scale is consumed as a GPU scalar by the planned
        # quantizer. It is not a compile specialization or a host synchronization.
        torch.amax(query[:rows].abs().float(), out=q_scale)
        q_scale.div_(448).clamp_(min=1e-12)
        binding = strided.bind_indexed(
            plan, scratch=scratch, q=query[:rows], kv_cache=cache, output=output[:rows],
            logical_indices=self.topk_indices_buffer[:rows],
            request_ids=attn_metadata.req_id_per_token[:rows],
            block_table=attn_metadata.block_table, cu_seqlens_q=starts[:rows+1],
            kv_scale=layer._k_scale, q_scale=q_scale,
        )
        result, _ = strided.run_decode(binding=binding)
        return result, None
