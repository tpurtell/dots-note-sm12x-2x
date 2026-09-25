"""Optional native-dtype routing pack; no communication or allocation.

The caller supplies the same aligned typed views used by packed owner routing.
This kernel replaces three independent copies with one launch. Source tensors
may have arbitrary positive row/column strides; destination views are dense.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pack(A, I, W, OA, OI, OW, ROWS: tl.constexpr, HIDDEN: tl.constexpr,
          TOPK: tl.constexpr, AS0: tl.constexpr, AS1: tl.constexpr,
          IS0: tl.constexpr, IS1: tl.constexpr, WS0: tl.constexpr,
          WS1: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A + (offsets // HIDDEN) * AS0 + (offsets % HIDDEN) * AS1,
                offsets < ROWS * HIDDEN, other=0)
    tl.store(OA + offsets, a, offsets < ROWS * HIDDEN)
    indices = (offsets // TOPK) * IS0 + (offsets % TOPK) * IS1
    weights = (offsets // TOPK) * WS0 + (offsets % TOPK) * WS1
    ids = tl.load(I + indices, offsets < ROWS * TOPK, other=0)
    values = tl.load(W + weights, offsets < ROWS * TOPK, other=0)
    tl.store(OI + offsets, ids, offsets < ROWS * TOPK)
    tl.store(OW + offsets, values, offsets < ROWS * TOPK)


def pack_routing(buffers, activation, route_ids, route_weights):
    sources = (activation, route_ids, route_weights)
    targets = (buffers.activation, buffers.route_ids, buffers.route_weights)
    if buffers.packed_payload is None:
        raise ValueError('fused routing pack requires prepared packed storage')
    if activation.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError('unsupported activation dtype')
    if route_ids.dtype != torch.int32 or route_weights.dtype != torch.float32:
        raise ValueError('routing pack requires native int32 IDs and FP32 weights')
    for source, target in zip(sources, targets):
        if (source.ndim != 2 or source.shape != target.shape or source.dtype != target.dtype
                or source.device != target.device or source.device.type != 'cuda'
                or not target.is_contiguous()):
            raise ValueError('routing pack source/destination geometry mismatch')
    rows, hidden = activation.shape
    if route_ids.shape != route_weights.shape or route_ids.shape[0] != rows:
        raise ValueError('routing pack route shape mismatch')
    topk = route_ids.shape[1]
    if not rows:
        return
    _pack[(triton.cdiv(max(activation.numel(), route_ids.numel()), 256),)](
        *sources, *targets, rows, hidden, topk,
        *activation.stride(), *route_ids.stride(), *route_weights.stride(),
        BLOCK=256)
