#!/usr/bin/env python3
"""Add opt-in compact Dots3 DSA pages to already ported vLLM 0.30.0.

DOTS3_COMPACT_DSA_CACHE=1 preserves logical FP8 MLA width576. SWA stays1088.
Default behavior is unchanged. Requires B12x physical_record_width support.
"""
import sys
from pathlib import Path


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f'expected one source anchor: {old!r}')
    return source.replace(old, new)


def patch(root):
    path = root / 'models/dots3_note/nvidia/model.py'
    source = path.read_text()
    source = replace_once(source, '        return replace(spec, head_size=self.physical_head_size)',
        '        if __import__("os").environ.get("DOTS3_COMPACT_DSA_CACHE") == "1":\n'
        '            if (spec.head_size != 576 or spec.cache_dtype_str not in ("fp8", "fp8_e4m3") or spec.dtype not in (torch.uint8, torch.float8_e4m3fn)):\n'
        '                raise ValueError("Compact Dots3 DSA requires logical576 E4M3 cache")\n'
        '            return spec\n'
        '        return replace(spec, head_size=self.physical_head_size)')
    compile(source, str(path), 'exec')
    path.write_text(source)

    path = root / 'v1/core/kv_cache_utils.py'
    source = path.read_text()
    source = replace_once(source,
        '        return repeats_per_group is None and isinstance(\n'
        '            spec.first_spec, SlidingWindowSpec\n'
        '        )',
        '        compact_dots = (os.environ.get("DOTS3_COMPACT_DSA_CACHE") == "1"\n'
        '                        and vllm_config.model_config.hf_text_config.model_type == "dots3_note")\n'
        '        return (compact_dots or repeats_per_group is None) and isinstance(\n'
        '            spec.first_spec, SlidingWindowSpec\n'
        '        )')
    # B12x physical-record addressing requires each pool block to start on a
    # whole576-byte record. Indexer pages retain their own132-byte row stride.
    # All pool size/admission/allocation callers share this single divisor.
    source = replace_once(source,
        '    assert bytes_per_block > 0\n    hot_page_sizes = [',
        '    assert bytes_per_block > 0\n'
        '    if os.environ.get("DOTS3_COMPACT_DSA_CACHE") == "1":\n'
        '        bytes_per_block = round_up(bytes_per_block, 576)\n'
        '    hot_page_sizes = [')
    compile(source, str(path), 'exec')
    path.write_text(source)


if __name__ == '__main__':
    patch(Path(sys.argv[1]))
