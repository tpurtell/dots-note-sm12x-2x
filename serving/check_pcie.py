#!/usr/bin/env python3
"""Compare Dots3 B12x PCIe graph collectives with native vLLM and NCCL."""
import json
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.b12x_pcie_all_reduce import B12xPcieAllReduce
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator


def timing(graph):
    for _ in range(10):
        graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(500):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / 500


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    cpu = dist.new_group(backend='gloo')
    device = torch.device('cuda', rank)
    b12x = B12xPcieAllReduce(cpu, dist.group.WORLD, device)
    native = CustomAllreduce(cpu, device)
    nccl = PyNcclCommunicator(cpu, device)
    assert not native.disabled
    graphs = []
    records = []
    for rows in (1, 2, 4, 8, 16, 32):
        inp = torch.ones((rows, 5120), device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        cases = [('b12x', b12x.custom_all_reduce, b12x.capture),
                 ('native', native.custom_all_reduce, native.capture),
                 ('nccl', nccl.all_reduce, nullcontext)]
        outputs = []
        for name, run, capture in cases:
            graph = torch.cuda.CUDAGraph()
            with capture():
                with torch.cuda.graph(graph):
                    out = run(inp)
            assert out is not None
            outputs.append((name, graph, out))
            graphs.append(graph)
        for iteration in range(3):
            torch.manual_seed(100 + rank * 10 + iteration)
            inp.normal_()
            expected = inp.clone()
            dist.all_reduce(expected)
            for name, graph, out in outputs:
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(out, expected, rtol=0, atol=0)
        local = {name: timing(graph) for name, graph, out in outputs}
        gathered = [None, None]
        dist.all_gather_object(gathered, local, group=cpu)
        if rank == 0:
            row = {'rows': rows, 'changed_input_exact_parity': True,
                   'graph_us': {name: max(item[name] for item in gathered) for name in local}}
            records.append(row)
            print(json.dumps(row), flush=True)
    # Destroy graphs before the resources and IPC mappings they reference.
    torch.cuda.synchronize()
    for graph in graphs:
        graph.reset()
    b12x.close()
    native.close()
    nccl.destroy()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({'event': 'dots3-pcie-qualified', 'records': records}), flush=True)


if __name__ == '__main__':
    main()
