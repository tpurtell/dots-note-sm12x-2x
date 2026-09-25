#!/usr/bin/env python3
"""Exercise the real Dots3 vLLM-to-B12x sparse attention adapter."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from b12x.attention.sparse_mla import strided
from vllm.models.dots3_note.nvidia.b12x_attention import Dots3B12xSparseImpl

parser = argparse.ArgumentParser()
parser.add_argument("--compact", action="store_true", help="576-wide DSA in actual compact mixed-page block stride")
parser.add_argument('--heads', type=int, choices=[64,128], default=64)
parser.add_argument('--rows', type=int, default=4)
parser.add_argument('--selected', type=int, default=64, help='Maximum selected tokens per query, up to2048')
parser.add_argument('--requests', type=int, default=4)
parser.add_argument('--block-stride', type=int, default=589248, help='Compact physical block stride in bytes')
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.rows < 1 or not 1 <= args.selected <= 2048 or not 1 <= args.requests <= args.rows:
    parser.error('require positive rows,1..2048 selected tokens,1..rows requests')
if args.compact and (args.block_stride % 576 or args.block_stride < 2 * 64 * 576):
    parser.error('compact stride must fit two64-token pages and be divisible by576')

torch.manual_seed(42)
device = torch.device('cuda')
impl = object.__new__(Dots3B12xSparseImpl)
impl.capacity = args.rows
impl.num_heads = args.heads
impl.topk_indices_buffer = torch.full((args.rows, 2048), -1, dtype=torch.int32, device=device)
positions = torch.arange(args.selected, dtype=torch.int32, device=device)
counts = args.selected - torch.arange(args.rows, device=device, dtype=torch.int32).remainder(4) * (args.selected // 4)
impl.topk_indices_buffer[:, :args.selected] = torch.where(positions[None, :] < counts[:, None], positions[None, :], -1)
blocks_per_request = (args.selected + 63) // 64
num_blocks = args.requests * blocks_per_request
if args.compact:
    # Nonzero layer offset and interleaved neighboring pages, including the
    # smaller physical strides used by owner-local cache pools.
    backing = (torch.randn(num_blocks * args.block_stride, device=device) * 10).to(torch.float8_e4m3fn)
    cache = backing.as_strided((num_blocks, 64, 576), (args.block_stride, 576, 1), 64 * 576)
else:
    backing = (torch.randn(num_blocks, 3, 64, 1088, device=device) * 10).to(torch.float8_e4m3fn)
    cache = backing[:, 1]
meta = SimpleNamespace(
    req_id_per_token=torch.arange(args.rows, dtype=torch.int32, device=device).remainder(args.requests),
    block_table=torch.randperm(num_blocks, dtype=torch.int32, device=device).reshape(args.requests, blocks_per_request),
)
layer = SimpleNamespace(_k_scale=torch.tensor(0.01, device=device))
q = (torch.randn(args.rows, args.heads, 576, device=device) * 0.1).to(torch.bfloat16)
original_backing = backing.clone()

def run():
    return impl.forward_mqa((q[..., :512], q[..., 512:]), cache, meta, layer)[0]

def expected():
    logical = impl.topk_indices_buffer.clamp_min(0).long()
    tables = meta.block_table[meta.req_id_per_token.long()]
    selected = tables.gather(1, logical // 64) * 64 + logical.remainder(64)
    return strided.reference(q, cache, selected, counts, kv_scale=layer._k_scale,
                             q_scale=q.abs().float().amax().div(448).clamp(min=1e-12))[0]

actual = run()
torch.testing.assert_close(actual, expected(), rtol=2e-2, atol=2e-2)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    run()
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream):
    replay_output = run()
for _ in range(2):
    q.normal_(0, 0.1)
    # Captured bindings must read changing selection and request page tables,
    # rather than retaining the first prefix's translated physical indices.
    logical = impl.topk_indices_buffer
    logical.copy_(torch.where(logical >= 0, (logical + 17) % args.selected, logical))
    meta.block_table.copy_(meta.block_table.flip(1).clone())
    reference = expected()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, reference, rtol=2e-2, atol=2e-2)
assert torch.equal(backing.view(torch.uint8), original_backing.view(torch.uint8)), 'adapter modified cache or neighboring pages'
result = {'passed': True, 'rows':args.rows, 'heads':args.heads, 'selected_max':args.selected,
          'selected_min':int(counts.min().item()), 'requests':args.requests,
          'cache_shape':list(cache.shape), 'cache_stride':list(cache.stride()),
          'changed_input_graph_replays':2, 'cache_storage_unchanged':True,
          'reference':'B12x FP8-query mathematical reference; not native BF16-prefill equivalence'}
if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
