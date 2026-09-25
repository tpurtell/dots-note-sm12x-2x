#!/usr/bin/env python3
"""Opt-in owner-aware SWA grouping before global worker projection/admission."""
import sys
from pathlib import Path

def patch(root):
    path=root/'v1/core/kv_cache_utils.py';source=path.read_text()
    anchor='    global_kv_cache_groups = get_kv_cache_groups(vllm_config, merged_kv_cache_specs)'
    if source.count(anchor)!=1:raise RuntimeError('global KV grouping source anchor changed')
    source=source.replace(anchor,anchor+'\n'
        '    if os.environ.get("VLLM_HYBRID_BALANCE_KV_GROUPS", "0") == "1":\n'
        '        if not os.environ.get("VLLM_HYBRID_LAYER_PARTITION"):\n'
        '            raise ValueError("owner-aware KV grouping requires hybrid ownership")\n'
        '        from vllm.v1.core.hybrid_cache_groups import refine_owner_state_groups\n'
        '        global_kv_cache_groups = refine_owner_state_groups(global_kv_cache_groups, kv_cache_specs)\n')
    old = '            for layer_name, spec in group_spec.kv_cache_specs.items():\n                layers_by_spec[spec].append(layer_name)'
    new = ('            for layer_name in group.layer_names:\n'
           '                spec = group_spec.kv_cache_specs[layer_name]\n'
           '                layers_by_spec[spec].append(layer_name)')
    if source.count(old) != 1:
        raise RuntimeError('projected cache tensor allocation source anchor changed')
    source = source.replace(old, new)
    compile(source,str(path),'exec');path.write_text(source)

if __name__=='__main__':patch(Path(sys.argv[1]))
