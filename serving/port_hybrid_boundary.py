#!/usr/bin/env python3
"""Separate owner boundary port, installed after the decoder ownership port."""
import ast
import sys
from pathlib import Path

IMPORTS='''
from vllm.models.dots3_note.nvidia.hybrid_boundary import (
    embedding as _hybrid_embedding, head as _hybrid_head,
    logits_processor as _hybrid_logits_processor, shared_head as _hybrid_shared_head,
)
'''

class Calls(ast.NodeTransformer):
    def __init__(self,mapping):self.mapping=mapping;self.current=None;self.counts={}
    def visit_ClassDef(self,node):
        old=self.current;self.current=node.name
        node=self.generic_visit(node);self.current=old;return node
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if not isinstance(node.func,ast.Name):return node
        name=node.func.id;rule=self.mapping.get(self.current,{}).get(name)
        if rule is None:return node
        replacement,extras=rule
        self.counts[(self.current,name)]=self.counts.get((self.current,name),0)+1
        node.func.id=replacement
        for key,value in extras.items():
            node.keywords.append(ast.keyword(arg=key,value=ast.parse(value,mode='eval').body))
        return node


def patch(root):
    targets={
      'model_executor/models/deepseek_v2.py':{
        'DeepseekV2ForCausalLM':{'ParallelLMHead':('_hybrid_head',{'model_type':'config.model_type'}),
                               'LogitsProcessor':('_hybrid_logits_processor',{})}},
      'models/dots3_note/nvidia/model.py':{
        'Dots3NoteModel':{'VocabParallelEmbedding':('_hybrid_embedding',{'role':repr('target')})}},
      'models/dots3_note/nvidia/mtp.py':{
        'Dots3NoteMultiTokenPredictor':{'VocabParallelEmbedding':('_hybrid_embedding',{'role':repr('draft')}),
                                      'LogitsProcessor':('_hybrid_logits_processor',{})},
        'Dots3NoteMultiTokenPredictorLayer':{'SharedHead':('_hybrid_shared_head',{})}},
    }
    for name,mapping in targets.items():
        path=root/name;tree=ast.parse(path.read_text());transform=Calls(mapping)
        tree=transform.visit(tree);ast.fix_missing_locations(tree)
        expected={(cls,call):1 for cls,calls in mapping.items() for call in calls}
        if transform.counts!=expected:raise RuntimeError(f'boundary constructor anchors changed: {name}: {transform.counts}')
        source=ast.unparse(tree)+'\n'+IMPORTS
        compile(source,str(path),'exec');path.write_text(source)
    # The draft owns a distinct embedding checkpoint. Cross-rank ownership
    # makes comparing draft.weight to target.weight invalid on a remote rank;
    # preserve the explicit has_own_embed_tokens declaration for this feature.
    path=root/'v1/worker/gpu/spec_decode/eagle/utils.py';source=path.read_text()
    anchor='    draft_embed = getattr(draft_inner, "embed_tokens", None)'
    if source.count(anchor)!=1:raise RuntimeError('draft embedding sharing anchor changed')
    source=source.replace(anchor,anchor+'\n'
        '    if getattr(draft_embed, "_hybrid_owned_boundary", False) and getattr(draft_model, "has_own_embed_tokens", False):\n'
        '        return\n')
    compile(source,str(path),'exec');path.write_text(source)

if __name__=='__main__':patch(Path(sys.argv[1]))
