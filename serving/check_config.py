#!/usr/bin/env python3
"""Validate the real checkpoint's serving RoPE translation without loading weights."""
import copy
import json
import sys
from pathlib import Path

from vllm.transformers_utils.configs.dots3_note import Dots3NoteConfig

raw = json.loads((Path(sys.argv[1]) / 'config.json').read_text())
original = copy.deepcopy(raw)
config = Dots3NoteConfig(**copy.deepcopy(raw))
assert config.rope_parameters['rope_theta'] == 80000000.0
assert config.swa_rope_theta == 50000.0
assert config.rope_parameters['rope_type'] == 'default'
assert config.n_group == config.topk_group == 1
assert config.kv_lora_rank == 512
assert config.swa_kv_lora_rank == 1024
assert config.num_attention_heads == raw['num_attention_heads']
assert not config.is_heterogeneous
assert config.layer_types == raw['layer_types']
assert Dots3NoteConfig.from_dict(config.to_dict()).rope_parameters == config.rope_parameters
assert raw == original
bad = copy.deepcopy(raw)
bad['swa_rope_theta'] = 1.0
try:
    Dots3NoteConfig(**bad)
except ValueError as error:
    assert 'Conflicting Dots3 swa_rope_theta' in str(error)
else:
    raise AssertionError('Conflicting SWA RoPE was accepted')
bad = copy.deepcopy(raw)
bad['per_layer_config']['02']['kv_lora_rank'] = 512
try:
    Dots3NoteConfig(**bad)
except ValueError as error:
    assert 'per-layer dimensions differ' in str(error)
else:
    raise AssertionError('Conflicting per-layer dimensions were accepted')
print('Dots3 real configuration: DSA/SWA RoPE, roundtrip, routing and conflict checks passed')

from vllm.model_executor.layers.quantization.dots3_exl3_fp8 import Dots3HybridExl3Config
quant = Dots3HybridExl3Config.from_config(raw['quantization_config'])
quant.maybe_update_config(sys.argv[1], config)
assert len(quant.tensor_storage) == 34560
assert quant.fp8_core.weight_block_size == [128, 128]
print('GPTQModel metadata: 34,560 uniform MCG K4 projections and FP8 core validated')

from vllm.models.dots3_note.nvidia.audio import Dots3NoteAudioConfig
from transformers import WhisperConfig
from vllm.models.dots3_note.common.processor import load_note_config_section
audio = Dots3NoteAudioConfig(**load_note_config_section(sys.argv[1], None, 'audio_config'))
vision = load_note_config_section(sys.argv[1], None, 'vision_config')
assert vision['adapter_type'] == 'patch_merger' and vision['pre_pixel_shuffle']
assert vision['hidden_size'] == 5120
encoder = WhisperConfig(**audio.whisper_config)
assert (audio.whisper_adapter_in_dim, audio.whisper_adapter_out_dim) == (1280, 5120)
assert (encoder.d_model, encoder.encoder_layers, encoder.encoder_attention_heads) == (1280, 32, 20)
assert encoder.encoder_ffn_dim == 5120
assert encoder.activation_function == 'swiglu'
print('Dots3 multimodal configuration: source audio SwiGLU and vision patch-merger restored')
