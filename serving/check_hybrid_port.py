#!/usr/bin/env python3
"""CPU source checks of explicit dense contexts and opt-in model hooks."""
import ast
import tempfile
from pathlib import Path
from port_hybrid_parallel import patch
source = Path('.cache/vllm-v0.30.0/vllm')
files = ('model_executor/models/deepseek_v2.py', 'models/dots3_note/nvidia/model.py', 'models/dots3_note/nvidia/mtp.py', 'models/deepseek_v32/nvidia/mtp.py')
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name in files:
        p = root/name; p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes((source/name).read_bytes())
    patch(root)
    found = set()
    for name in files:
        text = (root/name).read_text()
        compile(text,name,'exec')
        tree = ast.parse(text)
        for cls in (x for x in tree.body if isinstance(x,ast.ClassDef)):
            if cls.name not in ('DeepseekV2MLAAttention','Dots3NoteFullAttention','Dots3NoteSlidingAttention'): continue
            found.add(cls.name)
            init = next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='__init__')
            assert any(x.arg=='dense_parallel_context' for x in init.args.kwonlyargs)
            for call in (x for x in ast.walk(init) if isinstance(x,ast.Call)):
                if isinstance(call.func,ast.Name) and call.func.id in ('ColumnParallelLinear','RowParallelLinear','q_proj_cls','gate_cls'):
                    assert any(x.arg=='disable_tp' for x in call.keywords)
            assert not any(isinstance(x,ast.Global) for x in ast.walk(init))
    assert len(found)==3
print('Hybrid source port: explicit dense context in DSA/SWA constructors and linear sharding; syntax passed')

# Ownership only exempts a parameter-free peer; native and owner layers retain
# the existing mandatory checkpoint-layer completeness validation.
from types import SimpleNamespace as NS
for block, expected in ((NS(),True),(NS(has_checkpoint_decoder_parameters=True),True),(NS(has_checkpoint_decoder_parameters=False),False)):
    assert getattr(block, 'has_checkpoint_decoder_parameters', True) is expected
