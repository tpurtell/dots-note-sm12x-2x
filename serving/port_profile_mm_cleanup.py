#!/usr/bin/env python3
"""Opt-in native V2 profiling cleanup; preserves encoder outputs and peak usage."""
from pathlib import Path
import sys

def patch(root):
    path=root/'v1/worker/gpu/model_runner.py'
    source=path.read_text()
    anchor='''                self.model_state.encoder_runner.profile_encoder_cache(
                    dummy_mm_inputs, mm_budget
                )'''
    addition='''
                import os as _profile_mm_os
                if _profile_mm_os.environ.get("VLLM_PROFILE_MM_EMPTY_CACHE", "0") == "1":
                    from profile_mm_cleanup import cleanup_profile_mm_cache
                    self.hybrid_mm_cleanup_profile = cleanup_profile_mm_cache(torch)
'''
    if source.count(anchor)!=1:raise RuntimeError('V2 MM profile source anchor changed')
    source=source.replace(anchor,anchor+addition)
    compile(source,str(path),'exec');path.write_text(source)

if __name__=='__main__':patch(Path(sys.argv[1]))
