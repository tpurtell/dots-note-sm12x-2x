"""Ragged cache gather for Dots3's 1088-wide sliding-window MLA rows."""
import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _gather(src, dst, table, cu, token_to_seq, starts, scale,
            src_block_stride, src_token_stride, dst_row_stride, table_stride,
            WIDTH: tl.constexpr, PAGE: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    request = tl.load(token_to_seq + row).to(tl.int64)
    logical = (row - tl.load(cu + request) + tl.load(starts + request)).to(tl.int64)
    page = tl.load(table + request * table_stride + logical // PAGE).to(tl.int64)
    dim = tl.program_id(1) * TILE + tl.arange(0, TILE)
    values = tl.load(src + page * src_block_stride + logical % PAGE * src_token_stride + dim,
                     mask=(page >= 0) & (dim < WIDTH), other=0.0).to(tl.float32)
    values *= tl.load(scale)
    tl.store(dst + row.to(tl.int64) * dst_row_stride + dim, values, mask=dim < WIDTH)


def gather_and_dequant_cache(src_cache, dst, block_table, cu_seq_lens,
                            token_to_seq, num_tokens, kv_cache_dtype, scale,
                            seq_starts):
    if kv_cache_dtype not in ('fp8', 'fp8_e4m3'):
        raise ValueError('Dots3 gather requires ordinary E4M3 cache')
    if src_cache.dtype != torch.float8_e4m3fn:
        src_cache = src_cache.view(torch.float8_e4m3fn)
    if src_cache.shape[-1] != 1088 or src_cache.stride(-1) != 1:
        raise ValueError('Dots3 gather requires contiguous 1088-element rows')
    if dst.dtype != torch.bfloat16 or dst.stride(-1) != 1 or dst.shape[-1] != 1088:
        raise ValueError('Dots3 gather requires BF16 destination rows')
    _gather[(num_tokens, triton.cdiv(1088, 256))](
        src_cache, dst, block_table, cu_seq_lens, token_to_seq, seq_starts, scale,
        src_cache.stride(0), src_cache.stride(1), dst.stride(0), block_table.stride(0),
        WIDTH=1088, PAGE=src_cache.shape[1], TILE=256,
    )
