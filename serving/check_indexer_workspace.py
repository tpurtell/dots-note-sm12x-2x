#!/usr/bin/env python3
"""CPU check actual indexer cap helper and chunk planner without GPU imports."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
from port_compact_cache import replace_once

source = Path('.cache/vllm-v0.30.0/vllm/v1/attention/backends/mla/indexer.py').read_text()
# Get exact injected replacement from the port script by running it on a temp
# source tree; share the same path as the layout test rather than copy formula.
import tempfile
from port_compact_cache import patch
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name in ('models/dots3_note/nvidia/model.py', 'v1/core/kv_cache_utils.py', 'v1/attention/backends/mla/indexer.py'):
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((Path('.cache/vllm-v0.30.0/vllm') / name).read_bytes())
    patch(root)
    source = (root / 'v1/attention/backends/mla/indexer.py').read_text()

tree = ast.parse(source)
functions = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ('get_max_prefill_buffer_size', '_split_indexer_prefill_chunks'):
        node.decorator_list = []
        functions[node.name] = ast.unparse(node)
namespace = {}
exec('from __future__ import annotations\n' + '\n'.join(functions.values()), namespace)
helper = namespace['get_max_prefill_buffer_size']
planner = namespace['_split_indexer_prefill_chunks']
config = NS(model_config=NS(max_model_len=524288, hf_text_config=NS(model_type='dots3_note')))
for value in (None, '0', '4'):
    if value is None: os.environ.pop('DOTS3_INDEXER_PREFILL_CONTEXTS', None)
    else: os.environ['DOTS3_INDEXER_PREFILL_CONTEXTS'] = value
    assert helper(config) == 524288 * (4 if value == '4' else 40)

class Integer(int):
    def item(self): return int(self)

for lengths, queries in (([524288], [512]), ([524288] * 9, [32] * 9), ([1, 200000, 524288, 524000, 262144, 500000], [1, 16, 32, 64, 128, 16])):
    lengths = list(map(Integer, lengths))
    queries = list(map(Integer, queries))
    cap = helper(config)
    chunks = planner(lengths, queries, cap, 512 * 1024**2)
    visited = []
    for req, rows in chunks:
        count = sum(lengths[req])
        assert count <= cap
        assert (rows.stop - rows.start) * count * 4 <= 512 * 1024**2
        flat = [(i, j) for i in range(req.start, req.stop) for j in range(queries[i])]
        visited.extend(flat[rows])
    expected = [(i, j) for i in range(len(queries)) for j in range(queries[i])]
    assert visited == expected
    if len(lengths) == 9: assert len(chunks) >= 3
print('Indexer workspace: absent/zero default,4-context allocation/planner agreement, full-context and >4-request chunk coverage passed')
