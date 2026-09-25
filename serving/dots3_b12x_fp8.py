"""Experimental exact-scale FP8 projections, explicitly opt-in and preplanned.

Integration: call ``maybe_exact_fp8_method(fp8_config, prefix)`` before the native
``fp8_config.get_quant_method`` and use its result only when non-None. No runtime
patch is installed by importing this module. This candidate retains original
weights alongside the native layout; include that memory in admission profiling.
"""
import os
import threading

import torch
from vllm.model_executor.layers.linear import register_weight_loader_v2_supported_method
from vllm.model_executor.layers.quantization.fp8 import Fp8LinearMethod
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape


_PLANS = {}
_LOCK = threading.Lock()
_GEOMETRIES = {
    'q_b_proj': {(12288, 1024), (8192, 1024)},
}


def maybe_exact_fp8_method(config, prefix):
    """Return a candidate only for named attention projections; disabled by default.

    DOTS3_B12X_EXACT_FP8=q_b_proj
    DOTS3_B12X_EXACT_FP8_ROWS=4,16,64,512
    Rows are deployment-time specialization declarations, never request-driven.
    """
    enabled = os.getenv('DOTS3_B12X_EXACT_FP8', '')
    if not enabled:
        return None
    names = set(enabled.split(','))
    if not names <= set(_GEOMETRIES):
        raise ValueError('DOTS3_B12X_EXACT_FP8 accepts q_b_proj only')
    projection = prefix.rsplit('.', 1)[-1]
    if projection not in names or '.self_attn.' not in prefix:
        return None
    rows = tuple(sorted(set(int(v) for v in os.getenv(
        'DOTS3_B12X_EXACT_FP8_ROWS', '4,16,64,512').split(','))))
    if not rows or min(rows) <= 0:
        raise ValueError('exact FP8 planned row counts must be positive')
    return Dots3ExactFp8Method(config, projection, rows)


@register_weight_loader_v2_supported_method
class Dots3ExactFp8Method(Fp8LinearMethod):
    # The additional source tensors/plans are not in vLLM's tensorizer format.
    supports_pre_processed_weights = False

    def __init__(self, config, projection, rows):
        super().__init__(config)
        self.projection, self.rows = projection, rows
        self.plans = None
        if config.weight_block_size != [128, 128] or config.activation_scheme != 'dynamic':
            raise ValueError('exact Dots3 FP8 requires dynamic activations and 128x128 weights')

    def process_weights_after_loading(self, layer):
        from b12x.gemm import blockscaled
        from b12x.preparation import PreparationSession, PreparedCall

        weight, scale = layer.weight, layer.weight_scale_inv
        n, k = weight.shape
        if (weight.device.type != 'cuda' or torch.cuda.get_device_capability(weight.device) not in ((12, 0), (12, 1))
                or (n, k) not in _GEOMETRIES[self.projection]
                or weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32
                or tuple(scale.shape) != (n // 128, k // 128)
                or self.input_dtype != torch.bfloat16 or self.out_dtype != torch.bfloat16):
            raise ValueError('exact Dots3 FP8 requires the qualified TP2 BF16 projection geometry')
        # Native preparation may repack/requantize either operand. Own immutable
        # copies first, preserving checkpoint values and arbitrary FP32 scales.
        self.source_weight = weight.detach().clone().contiguous()
        self.source_scale = scale.detach().clone().contiguous()
        super().process_weights_after_loading(layer)
        self.quantizer = QuantFP8(static=False, group_shape=GroupShape(1, 128),
                                 num_token_padding=None, use_ue8m0=False)
        self.plans = {}
        for rows in self.rows:
            key = (weight.device, n, k, rows, threading.get_ident())
            with _LOCK:
                cached = _PLANS.get(key)
                if cached is None:
                    av = torch.zeros((rows, k), dtype=torch.float8_e4m3fn, device=weight.device)
                    asc = torch.ones((rows, k // 128), dtype=torch.float32, device=weight.device)
                    plan = blockscaled.plan(blockscaled.query_from_call(
                        (av, asc), (self.source_weight, self.source_scale),
                        ab_dtype='float8_e4m3fn', sf_dtype='float32', sf_vec_size=128,
                        block_fp8=True, c_dtype='bfloat16'))
                    # One primer per static geometry/count, shared across layers.
                    w, s = self.source_weight, self.source_scale
                    def prepare(state, av=av, asc=asc, w=w, s=s):
                        return PreparedCall(run=lambda: state.run_serialized(
                            av, asc, w, s, None, ab_dtype='float8_e4m3fn', sf_dtype='float32',
                            c_dtype='bfloat16', sf_vec_size=128, block_fp8=True, stream=None),
                            owners=(av, asc, w, s))
                    session = PreparationSession(device=weight.device, autotune=False, compile_workers=2)
                    session.prepare((plan.request(name=f'dots3-exact-fp8:{n}:{k}:{rows}', prepare_call=prepare),))
                    cached = (plan, session)
                    _PLANS[key] = cached
                self.plans[rows] = cached[0]

    def apply(self, layer, x, bias=None):
        if (self.plans is not None and bias is None and x.ndim == 2
                and x.dtype == torch.bfloat16 and x.device == self.source_weight.device
                and x.shape[1] == self.source_weight.shape[1] and x.is_contiguous()
                and x.shape[0] in self.plans):
            from b12x.gemm import blockscaled
            values, scales = self.quantizer(x, None, None, use_triton=False)
            return blockscaled.mm_block_fp8(values, scales, self.source_weight,
                                           self.source_scale, plan=self.plans[x.shape[0]])
        return super().apply(layer, x, bias)
