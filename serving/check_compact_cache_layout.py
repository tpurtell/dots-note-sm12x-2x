#!/usr/bin/env python3
"""CPU source-level check of actual vLLM grouping/allocation formulas.

Executes patched upstream grouping functions with minimal spec/config fixtures;
this is not a GPU backend or real scheduler integration test.
"""
import argparse
import ast
import collections
import dataclasses
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from port_compact_cache import patch

@dataclasses.dataclass(frozen=True)
class Spec:
    head_size: int
    block_size: int = 64
    @property
    def page_size_bytes(self): return self.head_size * self.block_size
class Sliding(Spec): pass
class Other(Spec): pass
@dataclasses.dataclass
class Uniform:
    kv_cache_specs: dict
    @property
    def first_spec(self): return next(iter(self.kv_cache_specs.values()))
    @staticmethod
    def is_uniform_type(specs):
        return len({isinstance(s, Sliding) for s in specs.values()}) == 1
    @classmethod
    def from_specs(cls, specs): return cls(specs)
    def get_max_layers_per_page_size(self):
        return max(collections.Counter(s.page_size_bytes for s in self.kv_cache_specs.values()).values())
@dataclasses.dataclass
class Group:
    layer_names: list
    kv_cache_spec: Uniform


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=Path,default=Path('.cache/vllm-v0.30.0/vllm')); args=ap.parse_args()
    files=['models/dots3_note/nvidia/model.py','v1/core/kv_cache_utils.py']
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for f in files:
            p=root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((args.source/f).read_bytes())
        patch(root)
        tree=ast.parse((root/files[1]).read_text())
        wanted={'_approximate_gcd','_get_packed_kv_cache_groups','_get_kv_cache_bytes_per_block'}
        src='from __future__ import annotations\n'+'\n'.join(ast.unparse(n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted)
    env=dict(os=os,math=math,defaultdict=collections.defaultdict,UniformTypeKVCacheSpecs=Uniform,KVCacheGroupSpec=Group,SlidingWindowSpec=Sliding,MambaSpec=Other,CircularBufferSpec=Other,HiSparseHotSpec=Other,cdiv=lambda x,y:(x+y-1)//y,round_up=lambda x,y:(x+y-1)//y*y,_annotate_eagle_groups=lambda *a,**k:None,_warn_if_unannotated_eagle_mamba=lambda *a:None,_is_deepseek_v4_eagle=lambda *a:False,_glm5_next_tensor_layout=lambda *a:None,_get_per_layer_spec=lambda g,n:g.kv_cache_spec.kv_cache_specs[n])
    exec(src,env)
    config=NS(cache_config=NS(get_resolved_kv_cache_layout=lambda:NS(is_block_outermost=True)),model_config=NS(hf_text_config=NS(model_type='dots3_note')))
    output={}
    for compact in (False,True):
        os.environ['DOTS3_COMPACT_DSA_CACHE']='1' if compact else '0'
        specs={**{f'dsa{i}':Spec(576 if compact else 1088) for i in range(14)},**{f'idx{i}':Spec(132) for i in range(14)},**{f'swa{i}':Sliding(1088) for i in range(33)}}
        groups=env['_get_packed_kv_cache_groups'](config,specs)
        stride=env['_get_kv_cache_bytes_per_block'](groups)
        assert len({n for g in groups for n in g.layer_names})==61
        assert sum(len(g.layer_names) for g in groups)==61
        assert stride%(576 if compact else 1088)==0
        # Regions within each group's allocated block never overlap. Different
        # groups alias the arena, but scheduler allocates distinct pool IDs.
        for g in groups:
            end=sum(specs[n].page_size_bytes for n in g.layer_names)
            assert end<=stride
        counts=[len(g.layer_names) for g in groups]
        assert counts==([28,9,8,8,8] if compact else [28,17,16]),counts
        row=dict(groups=counts,pool_block_bytes=stride,contexts={})
        for context in (262144,524288):
            # Async two512-token batches +window512; +1 boundaryblock.
            sw_blocks=math.ceil((512+2*512)/64)+1
            full_blocks=math.ceil(context/64)
            required_blocks=full_blocks+(len(groups)-1)*sw_blocks
            row['contexts'][context]=dict(required_blocks=required_blocks,bytes=required_blocks*stride,gib=required_blocks*stride/2**30)
        output['compact' if compact else'legacy']=row
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
