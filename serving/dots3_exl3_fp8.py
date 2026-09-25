"""vLLM quantization config for Dots3's FP8 core and EXL3 routed experts."""

from __future__ import annotations

import os
import threading

import torch
from transformers import PretrainedConfig

from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import MoEActivation, RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.exl3 import Exl3Config, Exl3MoEMethod
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.transformers_utils.repo_utils import get_hf_file_to_dict


_SCRATCH_LOCK = threading.Lock()
_SCRATCH_BY_SPEC = {}
_PRIMERS_BY_SPEC = {}
_SESSION_LOCK = threading.Lock()
_SESSIONS_BY_DEVICE = {}


class Dots3B12xVocabMethod(UnquantizedEmbeddingMethod):
    """Use the exact BF16 head shard for single-token B12x projection."""

    def process_weights_after_loading(self, layer: ParallelLMHead) -> None:
        super().process_weights_after_loading(layer)
        weight = layer.weight
        if (weight.device.type != "cuda" or weight.dtype != torch.bfloat16
                or tuple(weight.shape) != (76032, 5120)
                or not weight.is_contiguous()):
            raise ValueError("Dots3 B12x vocabulary head requires a TP2 BF16 shard")
        from b12x.gemm import bf16_vocab_projection as vocab
        from b12x.preparation import PreparedCall, PreparationSession

        plan = vocab.plan(vocab.Caps(
            device=weight.device, max_tokens=1,
            in_features=5120, out_features=76032,
        ))
        session_key = (weight.device.type, weight.device.index, threading.get_ident())
        with _SESSION_LOCK:
            session = _SESSIONS_BY_DEVICE.get(session_key)
            if session is None:
                session = PreparationSession(
                    device=weight.device, autotune=False, compile_workers=2
                )
                _SESSIONS_BY_DEVICE[session_key] = session

        def prepare(state):
            source = torch.zeros((1, 5120), dtype=torch.bfloat16, device=weight.device)
            binding = state.bind(plan=plan, source=source, weight=weight)
            return PreparedCall(
                run=lambda: state.run(binding.source, binding.weight),
                owners=(source, binding),
            )

        session.prepare((
            plan.request(name=f"dots3-vocab:{id(layer)}", prepare_call=prepare),
        ))
        if plan.selection is None or plan.selection.config.backend != "triton":
            raise ValueError("Dots3 B12x vocabulary did not select a GPU kernel")
        layer.dots3_b12x_vocab_plan = plan

    def apply(self, layer, x, bias=None):
        plan = getattr(layer, "dots3_b12x_vocab_plan", None)
        if (plan is not None and bias is None and x.ndim == 2
                and tuple(x.shape) == (1, 5120)
                and x.dtype == torch.bfloat16 and x.is_contiguous()):
            from b12x.gemm import bf16_vocab_projection as vocab

            return vocab.run(vocab.bind(plan, source=x, weight=layer.weight))
        return super().apply(layer, x, bias)


class Dots3B12xExl3MoEMethod(Exl3MoEMethod):
    """Load ordinary per-expert EXL3 K4 weights into B12x's fused MoE owner."""

    def create_weights(self, layer: RoutedExperts, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        config = get_current_vllm_config_or_none()
        scheduler = getattr(config, "scheduler_config", None)
        capacity = getattr(scheduler, "max_num_batched_tokens", None)
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Dots3 B12x MoE requires max_num_batched_tokens")
        layer.dots3_b12x_capacity = capacity

    @staticmethod
    def _slab(param, *, experts: int, shard_ids: tuple[str, ...]):
        first = param.exl3_tensors[(0, shard_ids[0])]
        shape = ((len(shard_ids), experts, *first.shape)
                 if len(shard_ids) > 1 else (experts, *first.shape))
        slab = torch.empty(shape, dtype=first.dtype, device=first.device)
        for shard_index, shard_id in enumerate(shard_ids):
            for expert in range(experts):
                tensor = param.exl3_tensors.pop((expert, shard_id))
                destination = slab[shard_index, expert] if len(shard_ids) > 1 else slab[expert]
                destination.copy_(tensor)
        return slab

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        super().process_weights_after_loading(layer)
        if self.quant_config.rank_sliced_metadata is not None:
            raise ValueError("Dots3 uses uniform unsliced K4 expert checkpoints")
        from b12x.moe import fused_moe

        experts = int(layer.local_num_experts)
        hidden = int(layer.exl3_hidden_size)
        intermediate = int(layer.exl3_intermediate_size_per_partition)
        marker = layer.w13_mcg.exl3_tensors[(0, "w1")]
        for expert in range(experts):
            for shard_id in ("w1", "w3"):
                if layer.w13_trellis.exl3_tensors[(expert, shard_id)].shape[2] != 64:
                    raise ValueError("Dots3 routed experts must all be K4")
            if layer.w2_trellis.exl3_tensors[(expert, "w2")].shape[2] != 64:
                raise ValueError("Dots3 routed experts must all be K4")

        gate_suh = self._slab(layer.w13_suh, experts=experts, shard_ids=("w1",))
        up_suh = self._slab(layer.w13_suh, experts=experts, shard_ids=("w3",))
        gate_svh = self._slab(layer.w13_svh, experts=experts, shard_ids=("w1",))
        up_svh = self._slab(layer.w13_svh, experts=experts, shard_ids=("w3",))
        down_suh = self._slab(layer.w2_suh, experts=experts, shard_ids=("w2",))
        down_svh = self._slab(layer.w2_svh, experts=experts, shard_ids=("w2",))
        rotations = torch.cat((gate_svh, up_svh, down_suh), dim=1).contiguous()
        w13 = self._slab(layer.w13_trellis, experts=experts, shard_ids=("w1", "w3"))
        w2 = self._slab(layer.w2_trellis, experts=experts, shard_ids=("w2",))
        if tuple(w13.shape) != (2, experts, hidden // 16, intermediate // 16, 64):
            raise ValueError("Dots3 K4 gate/up slab geometry differs from the model")
        if tuple(w2.shape) != (experts, intermediate // 16, hidden // 16, 64):
            raise ValueError("Dots3 K4 down slab geometry differs from the model")

        weight_plan = fused_moe.plan_weights(
            source=fused_moe.Exl3TrellisSource(bits=4),
            activation=fused_moe.ActivationSpec(
                mode=fused_moe.ActivationMode.A16,
                nonlinearity="silu",
                io_dtype=layer.exl3_params_dtype,
            ),
            geometry=fused_moe.MoEGeometry(
                num_experts=experts,
                hidden_size=hidden,
                intermediate_size=intermediate,
            ),
        )
        layer.dots3_b12x_experts = fused_moe.prepare_weights(
            plan=weight_plan,
            weights=fused_moe.Exl3TrellisWeights(
                w13=w13, w2=w2,
                gate_suh=gate_suh, up_suh=up_suh,
                intermediate_rotations=rotations,
                down_svh=down_svh, mcg=marker,
            ),
        )
        layer.dots3_b12x_plan = fused_moe.plan_execution(
            experts=layer.dots3_b12x_experts,
            capacity=fused_moe.ExecutionCapacity(
                max_tokens=layer.dots3_b12x_capacity,
                top_k=int(layer.top_k),
                route_num_experts=0,
            ),
        )
        for prefix in ("w13", "w2"):
            for field in ("suh", "svh", "trellis", "mcg", "mul1"):
                getattr(layer, f"{prefix}_{field}").exl3_tensors.clear()

    @staticmethod
    def _prepare_plan(layer: RoutedExperts, x: torch.Tensor) -> None:
        from b12x.preparation import PreparedCall, PreparationSession

        plan = layer.dots3_b12x_plan
        if plan.prepared is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Dots3 B12x MoE was not prepared before CUDA graph capture")
        tokens = int(layer.dots3_b12x_capacity)
        hidden = int(layer.exl3_hidden_size)
        top_k = int(layer.top_k)
        device = x.device
        # Preparation is serial on the rank's stream, just like the shared
        # scratch arena. Retaining primers per layer wastes ~0.67 GiB at 512
        # tokens across 45 routed layers, and doubles with each capacity step.
        primer_key = (device.type, device.index, x.dtype, tokens, hidden,
                      top_k, int(layer.local_num_experts))
        with _SCRATCH_LOCK:
            primers = _PRIMERS_BY_SPEC.get(primer_key)
            if primers is None:
                dummy_x = torch.randn((tokens, hidden), dtype=x.dtype, device=device) * 0.125
                ids = torch.arange(tokens * top_k, dtype=torch.int32, device=device)
                ids = ids.reshape(tokens, top_k).remainder_(layer.local_num_experts).contiguous()
                weights = torch.full((tokens, top_k), 1 / top_k, dtype=torch.float32, device=device)
                output = torch.empty((tokens, hidden), dtype=torch.float32, device=device)
                primers = (dummy_x, ids, weights, output)
                _PRIMERS_BY_SPEC[primer_key] = primers
        dummy_x, ids, weights, output = primers

        def prepare(state):
            # PreparedCall owners survive preparation. Allocating here would
            # retain a full workspace for every routed layer. Preparation and
            # execution are serial on the rank's stream, so use the same
            # capacity-planned arena as live execution.
            scratch = Dots3B12xExl3MoEMethod._scratch(state.scratch, device)
            binding = state.bind(
                scratch=scratch,
                a=dummy_x,
                experts=layer.dots3_b12x_experts,
                topk_weights=weights,
                topk_ids=ids,
                output=output,
            )
            return PreparedCall(
                run=binding.run, output=output,
                owners=(scratch, dummy_x, ids, weights, binding),
            )

        session_key = (device.type, device.index, threading.get_ident())
        with _SESSION_LOCK:
            session = _SESSIONS_BY_DEVICE.get(session_key)
            if session is None:
                session = PreparationSession(
                    device=device, autotune=False, compile_workers=2
                )
                _SESSIONS_BY_DEVICE[session_key] = session
        session.prepare((
            plan.request(name=f"dots3-exl3:{layer.layer_name}", prepare_call=prepare),
        ))

    @staticmethod
    def _scratch(plan, device):
        specs = plan.scratch_specs()
        key = (device.type, device.index, tuple((tuple(spec.shape), spec.dtype) for spec in specs))
        with _SCRATCH_LOCK:
            scratch = _SCRATCH_BY_SPEC.get(key)
            if scratch is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("Dots3 B12x MoE scratch was not allocated before capture")
                scratch = tuple(
                    torch.empty(spec.shape, dtype=spec.dtype, device=device)
                    for spec in specs
                )
                _SCRATCH_BY_SPEC[key] = scratch
            return scratch

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input):
        del shared_experts, shared_experts_input
        if layer.activation != MoEActivation.SILU:
            raise ValueError("Dots3 B12x MoE requires SiLU")
        if layer.expert_map is not None or layer.apply_router_weight_on_input:
            raise ValueError("Dots3 B12x MoE does not accept expert maps or input weighting")
        from b12x.moe import fused_moe

        original_shape = x.shape[:-1]
        inputs = x.reshape(-1, x.shape[-1]).contiguous()
        ids = topk_ids.reshape(inputs.shape[0], -1).to(torch.int32).contiguous()
        weights = topk_weights.reshape_as(ids).to(torch.float32).contiguous()
        plan = layer.dots3_b12x_plan
        self._prepare_plan(layer, inputs)
        binding = fused_moe.bind(
            plan,
            scratch=self._scratch(plan, inputs.device),
            a=inputs,
            experts=layer.dots3_b12x_experts,
            topk_weights=weights,
            topk_ids=ids,
        )
        output = fused_moe.run(binding=binding).to(inputs.dtype)
        return output.reshape(*original_shape, output.shape[-1])


class Dots3HybridExl3Config(Exl3Config):
    """Keep source block-FP8 methods for every non-EXL3 module.

    The exported model uses EXL3 only for 45 × 256 language routed experts.
    Its attention, shared experts, dense first layer, and multimodal towers
    retain the source model's serialized 128×128 block-FP8 checkpoint.
    """

    fp8_core: Fp8Config | None = None

    def maybe_update_config(
        self,
        model_name: str,
        hf_config: PretrainedConfig | None = None,
        revision: str | None = None,
    ) -> None:
        is_dots3 = getattr(hf_config, "model_type", None) in (
            "dots3_note", "dots3_note_mtp",
        )
        if is_dots3 and not self.tensor_storage:
            # GPTQModel writes its full EXL3 map under quantize_config.json;
            # config.json deliberately contains only the summary.
            payload = get_hf_file_to_dict(
                "quantize_config.json", model_name, revision=revision,
            )
            if not payload or payload.get("bits") != 4 or payload.get("codebook") != "mcg":
                raise ValueError("Dots3 requires GPTQModel uniform MCG K4 metadata")
            storage = payload.get("tensor_storage", {})
            if len(storage) != 34560 or any(
                entry.get("quant_format") != "exl3" or entry.get("bits_per_weight") != 4
                for entry in storage.values()
            ):
                raise ValueError("Dots3 requires 34,560 uniform K4 projection records")
            self.tensor_storage = storage
        super().maybe_update_config(model_name, hf_config, revision)
        if not is_dots3:
            return
        source = getattr(hf_config, "dots3_fp8_core", None)
        if source is not None and source != {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        }:
            raise ValueError("Dots3 FP8 core contract differs from the source")
        self.fp8_core = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        # Dots3's shared-expert and first-layer geometry uses this attribute.
        self.weight_block_size = [128, 128]

    def get_quant_method(self, layer, prefix):
        fp8 = self.fp8_core
        if fp8 is None:
            return super().get_quant_method(layer, prefix)
        if isinstance(layer, ParallelLMHead):
            if os.getenv("DOTS3_B12X_VOCAB", "0") == "1":
                return Dots3B12xVocabMethod()
            return UnquantizedEmbeddingMethod()
        if isinstance(layer, LinearBase) and not self._linear_prefix_is_exl3(prefix):
            from vllm.model_executor.layers.quantization.dots3_b12x_fp8 import maybe_exact_fp8_method

            candidate = maybe_exact_fp8_method(fp8, prefix)
            return candidate if candidate is not None else fp8.get_quant_method(layer, prefix)
        if isinstance(layer, RoutedExperts) and not self._moe_prefix_is_exl3(
            prefix, layer
        ):
            return fp8.get_quant_method(layer, prefix)
        if isinstance(layer, RoutedExperts):
            return Dots3B12xExl3MoEMethod(self, layer.moe_config)
        if isinstance(layer, Attention):
            return fp8.get_quant_method(layer, prefix)
        return None
