#!/usr/bin/env python3
"""Opt-in one-forward dummy-logits lifetime; leaves native accounting unchanged."""
from pathlib import Path
import sys


def patch(root):
    path = root/'model_executor/layers/sparse_attn_indexer.py'
    source = path.read_text()
    old = '''        _ = torch.empty(
            max_logits_elems, dtype=torch.uint8, device=hidden_states.device
        )'''
    new = '''        # Keep one dummy logits allocation alive through the profiling forward.
        # This prevents later persistent workspace allocations from splitting its
        # temporary segment. The ForwardContext releases it before native profile
        # cleanup; live inference, logits size and peak accounting are unchanged.
        import os as _profile_os
        if _profile_os.environ.get("VLLM_INDEXER_PROFILE_HOLD_LOGITS", "0") == "1":
            _ = getattr(forward_context, "_indexer_profile_logits", None)
            if _ is None:
                _ = torch.empty(
                    max_logits_elems, dtype=torch.uint8, device=hidden_states.device
                )
                forward_context._indexer_profile_logits = _
            elif _.numel() != max_logits_elems or _.device != hidden_states.device:
                raise RuntimeError("indexer profile logits geometry changed within one forward")
        else:
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )'''
    if source.count(old) != 1:
        raise RuntimeError('indexer profile dummy allocation source anchor changed')
    source = source.replace(old, new)
    compile(source, str(path), 'exec')
    path.write_text(source)


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
