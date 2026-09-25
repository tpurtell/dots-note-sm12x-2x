#!/usr/bin/env python3
"""Opt-in owner-local MM tower construction; install after hybrid model port."""
import ast
import sys
from pathlib import Path

class TowerCalls(ast.NodeTransformer):
    def __init__(self): self.count = 0
    def visit_Call(self, node):
        node = self.generic_visit(node)
        names = {'DotsMoEVitModel':'visual', 'Dots3NoteAudioModel':'audio_tower'}
        if isinstance(node.func,ast.Name) and node.func.id in names:
            self.count += 1
            return ast.Call(func=ast.Name(id='_create_owned_tower',ctx=ast.Load()),
                args=[ast.Lambda(args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
                                 body=node),ast.Constant(names[node.func.id])],keywords=[])
        return node


def patch(root):
    path = root/'models/dots3_note/nvidia/multimodal.py'
    tree = ast.parse(path.read_text())
    transform = TowerCalls(); tree = transform.visit(tree)
    if transform.count != 2:
        raise RuntimeError('expected exactly native vision and audio constructors')
    ast.fix_missing_locations(tree)
    source = ast.unparse(tree)+'\n'
    source += ('\nfrom .hybrid_multimodal import create_owned_tower as _create_owned_tower\n'
               'from .hybrid_multimodal import install as _install_hybrid_mm\n'
               '_install_hybrid_mm(globals())\n')
    compile(source,str(path),'exec'); path.write_text(source)

if __name__ == '__main__': patch(Path(sys.argv[1]))
