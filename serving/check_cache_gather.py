#!/usr/bin/env python3
"""Qualify Dots3's ragged FP8 gather, including high physical page offsets."""
import torch
from vllm.models.dots3_note.nvidia.cache_gather import gather_and_dequant_cache

torch.manual_seed(42)
device = 'cuda'
width, page, layers = 1088, 64, 3
high = 2**31 // (page * layers * width) + 2
backing = torch.empty((high + 2, layers, page, width), device=device, dtype=torch.float8_e4m3fn)
cache = backing[:, 1]
assert high * cache.stride(0) > 2**31
for idx in (0, 2, high, high+1):
    cache[idx].copy_((torch.randn(page, width, device=device)*10).to(cache.dtype))
table = torch.tensor([[high, 2], [0, high+1]], dtype=torch.int32, device=device)
cu = torch.tensor([0, 10, 23], dtype=torch.int32, device=device)
starts = torch.tensor([60, 57], dtype=torch.int32, device=device)
seqs = torch.tensor([0]*10+[1]*13, dtype=torch.int32, device=device)
scale = torch.tensor(0.017, device=device)
out = torch.empty(23, width, dtype=torch.bfloat16, device=device)

def run():
    gather_and_dequant_cache(cache, out, table, cu, seqs, 23, 'fp8', scale, starts)

def reference():
    seq = seqs.long()
    positions = torch.arange(23, device=device) - cu[seq] + starts[seq]
    return (cache[table[seq, positions//page].long(), positions%page].float()*scale).to(out.dtype)

run()
torch.testing.assert_close(out, reference(), rtol=0, atol=0)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream): run()
stream.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream): run()
scale.fill_(0.03)
torch.cuda.synchronize()
graph.replay()
torch.cuda.synchronize()
torch.testing.assert_close(out, reference(), rtol=0, atol=0)
print('Dots3 gather: ragged boundaries, interleaved layers, >2GB addressing and graph replay exact')
