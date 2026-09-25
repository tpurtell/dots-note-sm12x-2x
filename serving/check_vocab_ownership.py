#!/usr/bin/env python3
"""CPU-only adapter ownership test; does not qualify B12x kernels or CUDA graphs.

Execute the real adapter class without importing CUDA/vLLM, using a preparation
stub with the same retained PreparedCall.owners semantics as B12x. Verify that
MTP head replacement releases its old shard while the prepared target plan stays
alive, and runtime calls bind the current weight rather than the warmup weight.
"""
import ast
import gc
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import weakref
from unittest.mock import patch


def main():
    path=Path(__file__).with_name('dots3_exl3_fp8.py')
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dots3B12xVocabMethod')
    class Weight:
        dtype='bf16';shape=(76032,5120)
        def __init__(self):self.device=NS(type='cuda',index=0)
        def is_contiguous(self):return True
    class Base:
        def process_weights_after_loading(self,layer):pass
        def apply(self,layer,x,bias=None):return 'native'
    class Session:
        def __init__(self,**kwargs):self.plans=[]
        def prepare(self,requests):
            for request in requests:
                plan=request.plan
                state=NS(bind=lambda **kw:NS(**kw),run=lambda source,weight:(source,weight))
                call=request.prepare_call(state)
                call.run()
                plan.owners=call.owners
                self.plans.append(plan)
    class Plan:
        selection=NS(config=NS(backend='triton'))
        def request(self,**kwargs):return NS(plan=self,**kwargs)
    vocab=NS(Caps=lambda **kwargs:NS(**kwargs),plan=lambda caps:Plan(),
             bind=lambda plan,**kwargs:NS(plan=plan,**kwargs),run=lambda binding:binding)
    modules={name:ModuleType(name) for name in ('b12x','b12x.gemm','b12x.preparation')}
    modules['b12x.gemm'].bf16_vocab_projection=vocab
    modules['b12x.preparation'].PreparedCall=lambda **kwargs:NS(**kwargs)
    modules['b12x.preparation'].PreparationSession=Session
    scope={'UnquantizedEmbeddingMethod':Base,'ParallelLMHead':object,'threading':threading,
           'torch':NS(bfloat16='bf16',zeros=lambda *a,**k:object()),
           '_SESSION_LOCK':threading.Lock(),'_SESSIONS_BY_DEVICE':{},'_VOCAB_PLANS_BY_SPEC':{}}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
    with patch.dict(sys.modules,modules):
        method=scope[cls.name]();target=NS(weight=Weight());draft=NS(weight=Weight())
        old_ref=weakref.ref(draft.weight)
        method.process_weights_after_loading(target)
        method.process_weights_after_loading(draft)
        assert target.dots3_b12x_vocab_plan is draft.dots3_b12x_vocab_plan
        session=next(iter(scope['_SESSIONS_BY_DEVICE'].values()))
        assert len(session.plans)==1
        assert session.plans[0].owners[1].weight is target.weight
        del draft
        gc.collect()
        assert old_ref() is None, 'obsolete MTP shard retained'
        x=NS(ndim=2,shape=(1,5120),dtype='bf16',is_contiguous=lambda:True)
        replacement=Weight();target.weight=replacement
        binding=method.apply(target,x)
        assert binding.weight is replacement and binding.source is x
        assert binding.plan is session.plans[0]
        assert method.apply(target,NS(ndim=2,shape=(2,5120)))=='native'
    print('PASS CPU ownership: one prepared plan; obsolete MTP shard collectible; current weight rebound; multirow fallback')
    print('Scope: adapter object lifetime only; GPU numerics, graph replay and memory recovery require live qualification.')


if __name__=='__main__':main()
