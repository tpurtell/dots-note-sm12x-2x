#!/usr/bin/env python3
"""CPU-only verification of a release parent's installed Dots3 parser source."""
import argparse
import ast
import hashlib
from importlib import metadata
from pathlib import Path


def verify(root, expected):
    source = root / 'reasoning/dots3_reasoning_parser.py'
    if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise SystemExit('Release parent lacks the exact recipe Dots3 reasoning parser')
    tree = ast.parse((root / 'reasoning/__init__.py').read_text())
    registries = [node.value for node in tree.body if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == '_REASONING_PARSERS_TO_REGISTER' for t in node.targets)]
    if len(registries) != 1 or ast.literal_eval(registries[0]).get('dots3') != ('dots3_reasoning_parser', 'Dots3ReasoningParser'):
        raise SystemExit('Release parent has no matching dots3 reasoning registration')
    ast.parse(source.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    verify(Path(metadata.distribution('vllm').locate_file('vllm')), args.sha256)
    print('Verified installed dots3 reasoning parser and registry; live API qualification remains required')
