"""vLLM quantization config for Dots3's FP8 core and EXL3 routed experts."""

from __future__ import annotations

from transformers import PretrainedConfig

from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.exl3 import Exl3Config
from vllm.model_executor.layers.quantization.fp8 import Fp8Config


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
        super().maybe_update_config(model_name, hf_config, revision)
        if getattr(hf_config, "model_type", None) != "dots3_note":
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
        if isinstance(layer, LinearBase) and not self._linear_prefix_is_exl3(prefix):
            return fp8.get_quant_method(layer, prefix)
        if isinstance(layer, RoutedExperts) and not self._moe_prefix_is_exl3(
            prefix, layer
        ):
            return fp8.get_quant_method(layer, prefix)
        if isinstance(layer, Attention):
            return fp8.get_quant_method(layer, prefix)
        return super().get_quant_method(layer, prefix)
