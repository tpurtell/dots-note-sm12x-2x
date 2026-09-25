#!/usr/bin/env python3
"""CPU selector/fallback ownership checks; no CUDA or numeric qualification."""
import ast
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch


def main():
    source=Path(__file__).with_name('dots3_b12x_fp8.py')
    parsed=ast.parse(source.read_text())
    nodes=[node for node in parsed.body if
           isinstance(node,(ast.FunctionDef,ast.ClassDef)) or
           (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('_GEOMETRIES','_SELECTORS') for t in node.targets))]
    class Native:
        def __init__(self,config):pass
        def process_weights_after_loading(self,layer):layer.native_prepared=True
        def apply(self,layer,x,bias=None):return ('native',layer,x,bias)
    scope={'os':os,'re':re,'Fp8LinearMethod':Native,'register_weight_loader_v2_supported_method':lambda cls:cls}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    factory=scope['maybe_exact_fp8_method'];config=NS(weight_block_size=[128,128],activation_scheme='dynamic')
    model=NS(layer_types=['deepseek_sparse_attention','sliding_attention'],num_hidden_layers=2)
    cfg_module=NS(get_current_vllm_config=lambda:NS(model_config=NS(hf_text_config=model)))
    with patch.dict(sys.modules,{'vllm.config':cfg_module}):
        with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':''}):
            assert factory(config,'model.layers.0.self_attn.q_b_proj') is None
        with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'q_b_proj','DOTS3_B12X_EXACT_FP8_ROWS':'4,16'}):
            method=factory(config,'model.layers.0.self_attn.q_b_proj')
            assert method.geometry_filter is None and method.rows==(4,16)
            assert factory(config,'model.layers.0.self_attn.o_proj') is None
        with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'swa_q_b_proj,dsa_o_proj','DOTS3_B12X_EXACT_FP8_ROWS':'1,2,4'}):
            for prefix in ['language_model.model.layers.1.self_attn.q_b_proj',
                           'model.layers.2.mtp_block.self_attn.q_b_proj']:
                assert factory(config,prefix).rows==(1,2,4)
            assert factory(config,'model.layers.0.self_attn.o_proj') is not None
            # Owner SWA-O shares shape with TP2 DSA-O. Layer identity prevents
            # the old silent misclassification before any tensors are retained.
            assert factory(config,'model.layers.1.self_attn.o_proj') is None
            assert factory(config,'model.layers.0.self_attn.q_b_proj') is None
            assert factory(config,'model.layers.2.mtp_block.self_attn.o_proj') is None
            assert factory(config,'vision.q_b_proj') is None
            assert factory(config,'model.layers.0.mlp.o_proj') is None
            try:factory(config,'model.layers.9.self_attn.o_proj')
            except ValueError:pass
            else:raise AssertionError('unknown layer silently classified')
        for shape in [(12288,1024),(8192,1024),(24576,1024),(16384,1024)]:
            assert shape in scope['_GEOMETRIES']['q_b_proj']
        assert scope['_GEOMETRIES']['o_proj']=={(5120,8192),(5120,16384)}
        with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'o_proj'}):
            try:factory(config,'model.layers.0.self_attn.o_proj')
            except ValueError:pass
            else:raise AssertionError('unsupported broad O selector accepted')
    print('PASS CPU: native/owner geometry allowlists; DSA/SWA layer identity; owner SWA-O collision rejected; MTP SWA recognized; disabled/general-QB preserved; unknown identity rejected.')
    print('GPU owner numeric/graph/full-model memory and performance qualification remains required.')


if __name__=='__main__':main()
