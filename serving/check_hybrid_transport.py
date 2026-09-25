#!/usr/bin/env python3
"""Two-rank GPU check of the real hybrid owner transport, eager and captured.

Run with torchrun --nproc-per-node=2 (RTX) or one process per Spark host.
Checks changed inputs and both owners; does not benchmark model performance.
"""
import argparse
import json
import os
from types import SimpleNamespace as NS

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from hybrid_parallel import PyNcclOwnerTransport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reduction-dtype', choices=['float32', 'bfloat16'], default='float32')
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    cpu = dist.new_group(backend='gloo')
    communicator = PyNcclCommunicator(cpu, torch.device('cuda', local_rank))
    transport = PyNcclOwnerTransport(NS(
        rank_in_group=rank, world_size=world,
        device_communicator=NS(pynccl_comm=communicator)))
    records = []
    for rows in (1, 4, 16, 512):
        for owner in (0, 1):
            x = torch.empty((rows, 5120), device='cuda', dtype=torch.bfloat16)
            partial = torch.empty_like(x, dtype=getattr(torch, args.reduction_dtype))
            output = torch.empty_like(partial)
            ids = torch.empty((rows, 8), device='cuda', dtype=torch.int32)
            weights = torch.empty((rows, 8), device='cuda', dtype=torch.float32)

            def run():
                transport.broadcast(x, owner)
                transport.broadcast(ids, owner)
                transport.broadcast(weights, owner)
                # Both tensor shards contribute, with distinct partial values.
                torch.mul(x, rank + 1, out=partial)
                transport.reduce_sum(output, partial, owner)

            x.fill_(1); ids.fill_(1); weights.fill_(1)
            run()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.graph(graph, stream=stream):
                run()
            torch.cuda.current_stream().wait_stream(stream)
            for iteration in range(3):
                value = iteration + 2
                x.fill_(value if rank == owner else -19)
                ids.fill_(value if rank == owner else -1)
                weights.fill_(value / 8 if rank == owner else -1)
                graph.replay()
                torch.cuda.synchronize()
                assert torch.all(x == value).item()
                assert torch.all(ids == value).item()
                assert torch.all(weights == value / 8).item()
                if rank == owner:
                    assert torch.all(output == value * 3).item()
            graph.reset()
            records.append(dict(rows=rows, owner=owner, changed_input_replays=3))
    dist.barrier()
    communicator.destroy()
    dist.destroy_process_group(cpu)
    dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({'passed': True, 'reduction_dtype': args.reduction_dtype,
                          'checks': records}), flush=True)


if __name__ == '__main__':
    main()
