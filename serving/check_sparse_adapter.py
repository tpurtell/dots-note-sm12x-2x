#!/usr/bin/env python3
"""Exercise the real Dots3 vLLM-to-B12x sparse attention adapter."""
from types import SimpleNamespace
import torch
from b12x.attention.sparse_mla import strided
from vllm.models.dots3_note.nvidia.b12x_attention import Dots3B12xSparseImpl

torch.manual_seed(42)
device = torch.device('cuda')
impl = object.__new__(Dots3B12xSparseImpl)
impl.capacity = 4
impl.topk_indices_buffer = torch.full((4, 2048), -1, dtype=torch.int32, device=device)
impl.topk_indices_buffer[:, :64] = torch.arange(64, dtype=torch.int32, device=device)
backing = (torch.randn(4, 3, 64, 1088, device=device) * 10).to(torch.float8_e4m3fn)
cache = backing[:, 1]
meta = SimpleNamespace(
    req_id_per_token=torch.arange(4, dtype=torch.int32, device=device),
    block_table=torch.tensor([[0], [3], [1], [2]], dtype=torch.int32, device=device),
)
layer = SimpleNamespace(_k_scale=torch.tensor(0.01, device=device))
selected = torch.full((4, 2048), -1, dtype=torch.int32, device=device)
selected[:, :64] = meta.block_table * 64 + torch.arange(64, device=device)
counts = torch.full((4,), 64, dtype=torch.int32, device=device)
q = (torch.randn(4, 64, 576, device=device) * 0.1).to(torch.bfloat16)

def run():
    return impl.forward_mqa((q[..., :512], q[..., 512:]), cache, meta, layer)[0]

def expected():
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
    reference = expected()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replay_output, reference, rtol=2e-2, atol=2e-2)
print('Dots3 vLLM sparse adapter: interleaved pages and changed-query graph replay passed')
