#!/usr/bin/env python3
"""Run multimodal processor warmup with a visible exception trace."""
import sys
from vllm.engine.arg_utils import EngineArgs
from vllm.renderers import renderer_from_config

config = EngineArgs(model=sys.argv[1], tensor_parallel_size=2,
                    max_model_len=32768, max_num_seqs=16,
                    max_num_batched_tokens=512, kv_cache_dtype='fp8',
                    limit_mm_per_prompt={'image':1,'audio':1,'video':0}).create_engine_config()
renderer = renderer_from_config(config)
renderer._warmup_mm_processor(renderer.mm_processor, log_prefix='Dots3 diagnostic')
print('Multimodal processor warmup passed')
