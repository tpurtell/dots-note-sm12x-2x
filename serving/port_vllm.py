#!/usr/bin/env python3
"""Install the Dots3 EXL3/FP8 adapter into pinned vLLM v0.30.0."""

import hashlib
import sys
from pathlib import Path


ROOT = Path(sys.argv[1])
TARGETS = {
    "model_executor/layers/quantization/__init__.py":
        "55f1d9d309d9caee6b0d47a95004dfa380e504c6cca061a6766c92c2654e11e4",
    "model_executor/layers/fused_moe/routed_experts.py":
        "68c48e5df0ea4a7cc1f04436b7b9153eb53802ecd9bb55d38db8738508daac6d",
}


def replace(source: str, old: str, new: str, *, count: int = 1) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f"expected {count} matches for {old!r}, got {actual}")
    return source.replace(old, new)


for name, digest in TARGETS.items():
    path = ROOT / name
    source = path.read_text()
    if hashlib.sha256(source.encode()).hexdigest() != digest:
        raise RuntimeError(f"vLLM v0.30.0 source identity changed: {name}")
    if name.endswith("quantization/__init__.py"):
        updated = replace(
            source,
            '    "deepseek_v4_fp8",\n',
            '    "deepseek_v4_fp8",\n    "exl3",\n',
        )
        updated = replace(
            updated,
            "    from .experts_int8 import ExpertsInt8Config\n",
            "    from .experts_int8 import ExpertsInt8Config\n"
            "    from .dots3_exl3_fp8 import Dots3HybridExl3Config\n",
        )
        updated = replace(
            updated,
            '        "deepseek_v4_fp8": deepseek_config,\n',
            '        "deepseek_v4_fp8": deepseek_config,\n'
            '        "exl3": Dots3HybridExl3Config,\n',
        )
    else:
        updated = replace(
            source,
            "            is_fused = loaded_weight.dim() == 3\n",
            "            # EXL3 per-expert Trellis tiles also have rank three.\n"
            "            is_fused = (\n"
            "                loaded_weight.dim() == 3\n"
            "                and not getattr(self.quant_method, \"exl3_per_expert_trellis\", False)\n"
            "            )\n",
        )
    compile(updated, str(path), "exec")
    path.write_text(updated)
path = ROOT / "transformers_utils/configs/dots3_note.py"
source = path.read_text()
source = replace(
    source,
    '        kwargs.setdefault("n_group", 1)\n',
    '''        # The exported HF config uses per-attention-type RoPE. vLLM's
        # Dots3 implementation consumes a flat DSA config and swa_rope_theta.
        # Translate at the serving boundary without changing checkpoint files.
        rope = kwargs.get("rope_parameters")
        if isinstance(rope, dict) and "deepseek_sparse_attention" in rope:
            if set(rope) != {"deepseek_sparse_attention", "sliding_attention"}:
                raise ValueError("Unexpected Dots3 per-layer RoPE keys")
            dsa = dict(rope["deepseek_sparse_attention"])
            swa = dict(rope["sliding_attention"])
            for parameters in (dsa, swa):
                if parameters.get("rope_type") != "default":
                    raise ValueError("Dots3 serving requires default RoPE")
            for key, parameters in (("rope_theta", dsa), ("swa_rope_theta", swa)):
                if key in kwargs and kwargs[key] != parameters["rope_theta"]:
                    raise ValueError(f"Conflicting Dots3 {key}")
                kwargs[key] = parameters["rope_theta"]
            kwargs["rope_parameters"] = dsa
            # Transformers treats legacy rope_scaling as an alias.
            kwargs.pop("rope_scaling", None)
        # vLLM explicitly selects swa_* dimensions for sliding layers. Its
        # generic HF heterogeneity converter does not support MLA. Validate
        # the duplicate HF overrides before using the native Dots3 fields.
        overrides = kwargs.pop("per_layer_config", None)
        if overrides:
            expected = {
                "kv_lora_rank": kwargs["swa_kv_lora_rank"],
                "num_attention_heads": kwargs["swa_num_attention_heads"],
                "num_key_value_heads": kwargs["swa_num_key_value_heads"],
                "qk_nope_head_dim": kwargs["swa_qk_nope_head_dim"],
            }
            expected_layers = {
                i for i, kind in enumerate(kwargs["layer_types"])
                if kind == "sliding_attention"
            }
            if {int(i) for i in overrides} != expected_layers:
                raise ValueError("Unexpected Dots3 per-layer override coverage")
            if any(value != expected for value in overrides.values()):
                raise ValueError("Dots3 per-layer dimensions differ from swa_* fields")
        kwargs.setdefault("n_group", 1)
''',
)
compile(source, str(path), "exec")
path.write_text(source)
print("Dots3 EXL3/FP8 vLLM v0.30.0 port applied")

path = ROOT / 'v1/attention/backends/registry.py'
source = replace(path.read_text(), '    FLASH_ATTN_MLA_SPARSE = (\n',
    '    B12X_DOTS3_MLA_SPARSE = (\n'
    '        "vllm.models.dots3_note.nvidia.b12x_attention.Dots3B12xSparseBackend"\n'
    '    )\n'
    '    FLASH_ATTN_MLA_SPARSE = (\n')
path.write_text(source)
path = ROOT / 'platforms/cuda.py'
source = replace(path.read_text(),
    '                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n',
    '                AttentionBackendEnum.B12X_DOTS3_MLA_SPARSE,\n'
    '                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,\n')
path.write_text(source)
path = ROOT / 'models/dots3_note/nvidia/model.py'
source = replace(path.read_text(), '\n\ndef _padded_mlp_size(',
    '\n\nfrom .b12x_attention import Dots3B12xSparseBackend as Dots3NotePaddedSparseBackend\n'
    '\n\ndef _padded_mlp_size(')
source = replace(source,
    '    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MLAAttentionSpec:\n',
    '    def _use_sparse_mha(self, attn_metadata):\n'
    '        # B12x consumes the causal indexer selections for both prefill and decode.\n'
    '        return False\n\n'
    '    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> MLAAttentionSpec:\n')
path.write_text(source)
path = ROOT / 'models/dots3_note/common/processor.py'
source = replace(path.read_text(),
    '    value = (config or {}).get(section)\n',
    '''    if (config or {}).get("quantization_config", {}).get("quant_method") == "exl3":
        # HF config normalization dropped legacy multimodal architecture fields.
        # Restore the exact FP8-source sections only for this audited export.
        import hashlib
        import json
        from pathlib import Path
        source_config = json.loads(
            Path(__file__).with_name("source_multimodal_config.json").read_text()
        )
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if digest != source_config["export_config_sha256"]:
            raise ValueError("Dots3 multimodal source configuration requires the pinned export")
        if section in ("vision_config", "audio_config"):
            return source_config[section]
    value = (config or {}).get(section)
''')
path.write_text(source)

path = ROOT / 'models/dots3_note/nvidia/attention.py'
source = replace(path.read_text(),
    '                ops.gather_and_maybe_dequant_cache(\n',
    '                from .cache_gather import gather_and_dequant_cache\n'
    '                gather_and_dequant_cache(\n')
path.write_text(source)

# Match the live runner's input contract during multimodal graph capture.
# The model consumes embeddings; keeping an extra input_ids tensor in capture
# gives breakable CUDA graphs a different argument set at replay.
path = ROOT / 'v1/worker/gpu/model_states/default.py'
source = replace(path.read_text(),
    '            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)\n',
    '            model_inputs["inputs_embeds"] = self.dummy_inputs_embeds(num_tokens)\n'
    '            from vllm.model_executor.models.interfaces import requires_raw_input_tokens\n'
    '            if not requires_raw_input_tokens(self.model):\n'
    '                model_inputs["input_ids"] = None\n')
compile(source, str(path), 'exec')
path.write_text(source)

# v0.30.0 guides named/required tool choices to standard JSON. Dots' XML
# extractor is for automatic tool choice; use the serving layer's JSON
# extraction for constrained named/required responses (including streaming).
path = ROOT / 'tool_parsers/dots_tool_parser.py'
source = replace(path.read_text(),
    '    supports_required_and_named = False\n',
    '    supports_required_and_named = True\n')
path.write_text(source)
