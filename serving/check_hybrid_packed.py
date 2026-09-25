#!/usr/bin/env python3
"""CPU byte-layout check; --gpu adds two-rank PyNccl parity/graph timings.

GPU invocation: torchrun --nproc-per-node=2 check_hybrid_packed.py --gpu.
Timings cover copies, broadcast(s), a synthetic shard operation and reduce;
they are component evidence, not model-performance estimates.
"""
import argparse
import json
import os
from types import SimpleNamespace as NS
import torch
from hybrid_parallel import (RoutedLayerBuffers, allocate_packed_routing,
                             execute_routed_layer, PyNcclOwnerTransport)


def cpu_checks():
    for hidden in (3, 5120):
        base = allocate_packed_routing(capacity=512, hidden=hidden, topk=8,
                                      dtype=torch.bfloat16, device='cpu')
        for rows in (1, 2, 4, 16, 511, 512):
            active = base.active(rows)
            active.activation.fill_(3.5)
            active.route_ids.fill_(255)
            active.route_weights.fill_(0.125)
            assert torch.all(active.activation == 3.5)
            assert torch.all(active.route_ids == 255)
            assert torch.all(active.route_weights == .125)
            assert active.packed_payload.is_contiguous()
            assert active.route_ids.data_ptr() % 4 == 0
            assert active.route_weights.data_ptr() % 4 == 0
            assert active.packed_payload.numel() == ((rows*hidden*2+3)//4*4 + rows*8*8)
            assert active.packed_payload.untyped_storage().data_ptr() == base.packed_payload.data_ptr()
    return {'passed': True, 'geometries': 12, 'checks': 'typed byte aliases, alignment, active-only message, shared allocation'}


def gpu_checks():
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    dist.init_process_group('nccl')
    assert dist.get_world_size() == 2
    rank = dist.get_rank()
    cpu = dist.new_group(backend='gloo')
    comm = PyNcclCommunicator(cpu, device)
    transport = PyNcclOwnerTransport(NS(rank_in_group=rank, world_size=2,
        device_communicator=NS(pynccl_comm=comm)))
    records = []
    for rows in (1, 2, 4, 8, 16, 512):
        for owner in (0, 1):
            record = {'rows': rows, 'owner': owner, 'timings_us': {}}
            for packed in (False, True):
                if packed:
                    buffers = allocate_packed_routing(capacity=512, hidden=5120,
                        topk=8, dtype=torch.bfloat16, device=device).active(rows)
                else:
                    buffers = RoutedLayerBuffers(torch.empty((rows,5120),device=device,dtype=torch.bfloat16),
                        torch.empty((rows,8),device=device,dtype=torch.int32),
                        torch.empty((rows,8),device=device,dtype=torch.float32),
                        torch.empty((rows,5120),device=device,dtype=torch.bfloat16))
                x = torch.full_like(buffers.activation, 2)
                ids = torch.full_like(buffers.route_ids, 7)
                weights = torch.full_like(buffers.route_weights, .125)
                partial = torch.empty_like(x)
                def run():
                    return execute_routed_layer(owner=owner, transport=transport, buffers=buffers,
                        prepare_owner=lambda: (x,ids,weights),
                        execute_local_experts=lambda a,i,w: torch.mul(a,rank+1,out=partial),
                        finish_owner=lambda result: result)
                for _ in range(3): run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.graph(graph, stream=stream): run()
                torch.cuda.current_stream().wait_stream(stream)
                for iteration in range(3):
                    value = iteration + 3
                    x.fill_(value if rank==owner else -1)
                    ids.fill_(value+17 if rank==owner else -2)
                    weights.fill_(value/16 if rank==owner else -3)
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.all(buffers.activation == value).item()
                    assert torch.all(buffers.route_ids == value+17).item()
                    assert torch.all(buffers.route_weights == value/16).item()
                    if rank == owner: assert torch.all(buffers.reduced_output == value*3).item()
                timings = []
                for _ in range(5):
                    dist.barrier(group=cpu)
                    begin,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(40): graph.replay()
                    end.record(); end.synchronize()
                    timings.append(begin.elapsed_time(end)*1000/40)
                all_times = [None,None]
                dist.all_gather_object(all_times,timings,group=cpu)
                record['timings_us']['packed' if packed else 'unpacked'] = all_times
                graph.reset()
            records.append(record)
    dist.barrier(group=cpu)
    comm.destroy(); dist.destroy_process_group(cpu); dist.destroy_process_group()
    return {'passed':True,'changed_input_replays':3,'records':records} if rank==0 else None


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--gpu',action='store_true'); args=parser.parse_args()
    result={'cpu':cpu_checks()}
    if args.gpu:
        result['gpu']=gpu_checks()
        if result['gpu'] is None: raise SystemExit(0)
    print(json.dumps(result,indent=2))
