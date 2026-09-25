#!/usr/bin/env python3
"""CPU selector/fallback ownership checks; no CUDA or numeric qualification."""
import ast
import os
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
    scope={'os':os,'Fp8LinearMethod':Native,'register_weight_loader_v2_supported_method':lambda cls:cls}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),scope)
    factory=scope['maybe_exact_fp8_method'];config=NS(weight_block_size=[128,128],activation_scheme='dynamic')
    with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':''}):
        assert factory(config,'model.layers.2.self_attn.q_b_proj') is None
    with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'q_b_proj','DOTS3_B12X_EXACT_FP8_ROWS':'4,16'}):
        method=factory(config,'model.layers.0.self_attn.q_b_proj')
        assert method.geometry_filter is None and method.rows==(4,16)
        assert factory(config,'model.layers.0.self_attn.o_proj') is None
    with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'swa_q_b_proj,dsa_o_proj','DOTS3_B12X_EXACT_FP8_ROWS':'1,2,4'}):
        qb=factory(config,'model.layers.2.self_attn.q_b_proj')
        output=factory(config,'model.layers.0.self_attn.o_proj')
        assert qb.geometry_filter==frozenset({(8192,1024)})
        assert output.geometry_filter==frozenset({(5120,8192)})
        assert qb.rows==output.rows==(1,2,4)
        assert factory(config,'vision.q_b_proj') is None
        assert factory(config,'model.layers.0.mlp.o_proj') is None
        # Unselected geometry must use native preparation before importing any
        # B12x module or retaining cloned FP8 source tensors.
        for method,shape in ((qb,(12288,1024)),(output,(5120,4096))):
            layer=NS(weight=NS(shape=shape));method.process_weights_after_loading(layer)
            assert layer.native_prepared and method.plans is None
            assert not hasattr(method,'source_weight') and not hasattr(method,'source_scale')
            assert method.apply(layer,object())[0]=='native'
    with patch.dict(os.environ,{'DOTS3_B12X_EXACT_FP8':'o_proj'}):
        try:factory(config,'model.layers.0.self_attn.o_proj')
        except ValueError:pass
        else:raise AssertionError('unqualified broad O selector accepted')
    print('PASS CPU: disabled/legacy mode preserved; SWA-QB and DSA-O exact geometry filters; unselected geometry stays native without source ownership; explicit draft rows retained.')
    print('GPU numeric/graph/full-model memory and performance qualification remains required.')


if __name__=='__main__':main()
