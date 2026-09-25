#!/usr/bin/env python3
"""Thread explicit local dense projection context through Dots constructors.

Install after normal Dots ports. No distributed-group globals are changed.
"""
import ast
import sys
from pathlib import Path


class ConstructorPort(ast.NodeTransformer):
    def __init__(self, classes): self.classes = classes; self.active = False
    def visit_ClassDef(self, node):
        old = self.active
        self.active = node.name in self.classes
        node = self.generic_visit(node)
        self.active = old
        return node
    def visit_FunctionDef(self, node):
        if not self.active or node.name != '__init__': return node
        node.args.kwonlyargs.append(ast.arg(arg='dense_parallel_context'))
        node.args.kw_defaults.append(ast.Constant(None))
        for child in ast.walk(node):
            if isinstance(child, ast.Assign) and any(isinstance(t,ast.Name) and t.id=='tp_size' for t in child.targets):
                child.value = ast.parse('dense_parallel_context.tensor_parallel_size if dense_parallel_context is not None else get_tensor_model_parallel_world_size()',mode='eval').body
            if isinstance(child, ast.Call):
                name = child.func.id if isinstance(child.func,ast.Name) else None
                if name in ('ColumnParallelLinear','RowParallelLinear','q_proj_cls','q_proj_cls','gate_cls'):
                    if not any(k.arg=='disable_tp' for k in child.keywords):
                        child.keywords.append(ast.keyword(arg='disable_tp',value=ast.parse('dense_parallel_context is not None',mode='eval').body))
                # Dots full-attention superclass has the new explicit context.
                if isinstance(child.func,ast.Attribute) and child.func.attr=='__init__' and isinstance(child.func.value,ast.Call) and isinstance(child.func.value.func,ast.Name) and child.func.value.func.id=='super':
                    if any(k.arg=='q_lora_rank' for k in child.keywords):
                        child.keywords.append(ast.keyword(arg='dense_parallel_context',value=ast.Name(id='dense_parallel_context',ctx=ast.Load())))
        return node


def patch(root):
    targets = {
        'model_executor/models/deepseek_v2.py': {'DeepseekV2MLAAttention'},
        'models/dots3_note/nvidia/model.py': {'Dots3NoteFullAttention','Dots3NoteSlidingAttention'},
    }
    for name, classes in targets.items():
        path = root/name
        tree = ast.parse(path.read_text())
        present = {n.name for n in ast.walk(tree) if isinstance(n,ast.ClassDef)}
        if not classes <= present: raise RuntimeError(f'missing expected classes in {path}')
        tree = ConstructorPort(classes).visit(tree)
        ast.fix_missing_locations(tree)
        source = ast.unparse(tree)+'\n'
        if name.endswith('dots3_note/nvidia/model.py'):
            source += '\nfrom .hybrid_parallel import install_model as _install_hybrid_model\n_install_hybrid_model(globals())\n'
        compile(source,str(path),'exec')
        path.write_text(source)

    path = root / 'models/dots3_note/nvidia/mtp.py'
    source = path.read_text() + '\nfrom .hybrid_parallel import install_mtp as _install_hybrid_mtp\n_install_hybrid_mtp(globals())\n'
    compile(source, str(path), 'exec')
    path.write_text(source)


    path = root / 'models/deepseek_v32/nvidia/mtp.py'
    source = path.read_text()
    anchor = '            if layer_idx not in loaded_layers and is_mtp_completeness_check_enabled():'
    if source.count(anchor) != 1:
        raise RuntimeError('MTP completeness source anchor changed')
    source = source.replace(anchor,
        '            owns_parameters = getattr(self.model.layers[str(layer_idx)].mtp_block, "has_checkpoint_decoder_parameters", True)\n'
        '            if owns_parameters and layer_idx not in loaded_layers and is_mtp_completeness_check_enabled():')
    compile(source, str(path), 'exec')
    path.write_text(source)

    path = root / 'v1/engine/core.py'
    source = path.read_text()
    anchor = '        self.model_executor.initialize_from_config(kv_cache_configs)'
    if source.count(anchor) != 1:
        raise RuntimeError('engine KV initialization source anchor changed')
    source = source.replace(anchor, anchor + '\n'
        '        import os as _hybrid_os\n'
        '        _hybrid_receipt_path = _hybrid_os.environ.get("VLLM_HYBRID_ATTESTATION_PATH")\n'
        '        if _hybrid_receipt_path:\n'
        '            from hybrid_attestation import write_startup_receipt\n'
        '            write_startup_receipt(self.model_executor, _hybrid_receipt_path)')
    compile(source, str(path), 'exec')
    path.write_text(source)


if __name__ == '__main__': patch(Path(sys.argv[1]))
