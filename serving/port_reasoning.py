#!/usr/bin/env python3
"""Register the Dots3 reasoning parser without changing other parser behavior."""
import ast
from pathlib import Path
import sys


def port(root):
    target = root / 'reasoning/__init__.py'
    text = target.read_text()
    old = '_REASONING_PARSERS_TO_REGISTER = {\n'
    if text.count(old) != 1 or '"Dots3ReasoningParser"' in text:
        raise RuntimeError(f'{target}: unsupported or already patched reasoning registry')
    new = old + '    "dots3": ("dots3_reasoning_parser", "Dots3ReasoningParser"),\n'
    text = text.replace(old, new, 1)
    ast.parse(text)
    target.write_text(text)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} VLLM_PACKAGE_ROOT')
    port(Path(sys.argv[1]))
    print('Dots3 reasoning parser registered')
