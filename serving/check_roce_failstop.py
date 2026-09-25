#!/usr/bin/env python3
"""Disposable two-host adapter watchdog test; never run inside a model worker.

Run with torchrun, one supervisor per host. Only the disposable child's own
proxy is stopped. No NIC, link, driver, or host setting is changed.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    from check_roce import mem_available
    if mem_available() < args.min_available_gib * 2**30:
        raise RuntimeError('Insufficient physical memory headroom')
    os.environ.update(DOTS3_B12X_ROCE='1', DOTS3_B12X_ROCE_ROWS='1-64',
        DOTS3_B12X_ROCE_EAGER='0', B12X_ROCE_HCA=args.hcas,
        B12X_ROCE_GID_INDEX=str(args.gid_index), B12X_ROCE_SPIN_LIMIT='2000000',
        NCCL_IB_HCA='='+args.hcas, NCCL_IB_GID_INDEX=str(args.gid_index),
        NCCL_IB_DISABLE='0', VLLM_ALLREDUCE_USE_FLASHINFER='0',
        VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC='0', VLLM_ALLREDUCE_USE_SYMM_MEM='0')
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
    from vllm.distributed.device_communicators.b12x_roce_all_reduce import B12xRoceAllReduce
    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.05, 0)
    dist.init_process_group('gloo', timeout=timedelta(seconds=120))
    rank = dist.get_rank()
    def barrier():
        dist.monitored_barrier(timeout=timedelta(seconds=120), wait_all_ranks=True)
    configuration = [None, None]
    dist.all_gather_object(configuration, (args.hcas, args.gid_index, args.fault_timeout))
    if configuration[0] != configuration[1]:
        raise RuntimeError('Rank configuration mismatch')
    comm = CudaCommunicator(cpu_group=dist.group.WORLD, device=torch.device('cuda:0'),
                           unique_name='tp:roce-disposable-failstop')
    adapter = comm.ca_comm
    if not isinstance(adapter, B12xRoceAllReduce) or adapter.disabled:
        raise RuntimeError('Actual vLLM communicator has no active RoCE adapter')
    x = torch.ones((4, 5120), dtype=torch.bfloat16, device='cuda')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), adapter.capture(stream=stream):
        comm.all_reduce(x)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream), adapter.capture(stream=stream):
        a = comm.all_reduce(x)
        b = comm.all_reduce(a)
        c = comm.all_reduce(b)
    barrier()
    graph.replay()
    torch.cuda.synchronize()
    adapter.check_health()
    torch.testing.assert_close(c, x * 8, rtol=0, atol=0)
    barrier()
    if rank == 1:
        # Test-only private fault injection, matching the vendored transport's
        # own timeout tests. This proxy belongs exclusively to this child.
        adapter.runtime._proxy.stop()
    barrier()
    print(json.dumps({'event':'fault_armed', 'rank':rank, 'time':time.time()}), flush=True)
    graph.replay()
    # No device reads, health polling, synchronize, cleanup, or native fallback:
    # the adapter's independent host watchdog must detect mapped poison and
    # terminate this worker itself, including the non-output TP rank.
    time.sleep(args.fault_timeout)
    os.write(2, b'FAIL: adapter watchdog did not terminate disposable worker\n')
    os._exit(71)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hcas', default='roceP2p1s0f0')
    p.add_argument('--gid-index', type=int, default=3)
    p.add_argument('--min-available-gib', type=float, default=16)
    p.add_argument('--abort-available-gib', type=float, default=1)
    p.add_argument('--deadline', type=int, default=1200)
    p.add_argument('--fault-timeout', type=int, default=60)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 < args.fault_timeout < args.deadline <= 1200:
        p.error('Require 0 < fault timeout < deadline <= 1200 seconds')
    if not 0 < args.abort_available_gib < args.min_available_gib:
        p.error('Require 0 < abort headroom < initial headroom')
    if int(os.environ.get('RANK', '-1')) not in (0,1) or os.environ.get('WORLD_SIZE') != '2' or int(os.environ.get('LOCAL_WORLD_SIZE','1')) != 1:
        p.error('Use torchrun with two hosts and one process per host')
    if args.output.resolve().is_relative_to('/mnt/scratch'):
        p.error('Use project .cache output')
    if args.worker:
        worker(args)
        return
    from check_roce import mem_available
    args.output.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = args.output.with_suffix('.worker.stdout')
    stderr_path = args.output.with_suffix('.worker.stderr')
    started = time.monotonic()
    reason = None
    with args.output.open('x') as receipt, stdout_path.open('x') as stdout, stderr_path.open('x') as stderr:
        child = subprocess.Popen([sys.executable, __file__, *sys.argv[1:], '--worker'], stdout=stdout, stderr=stderr, start_new_session=True)
        low = 0
        try:
            while child.poll() is None:
                low = low + 1 if mem_available() < args.abort_available_gib * 2**30 else 0
                if low >= 3 or time.monotonic() - started > args.deadline:
                    reason = 'physical headroom' if low >= 3 else 'deadline'
                    child.kill()
                    break
                time.sleep(1)
            code = child.wait(timeout=10)
        finally:
            if child.poll() is None:
                child.kill(); child.wait(timeout=10)
        stdout.flush(); stderr.flush()
        out, err = stdout_path.read_text(), stderr_path.read_text()
        passed = code == 70 and '"event": "fault_armed"' in out and 'FATAL B12x RoCE worker poison:' in err and reason is None
        result = dict(passed=passed, rank=int(os.environ['RANK']), worker_exit=code,
            failure=reason, elapsed_seconds=time.monotonic()-started,
            stdout=str(stdout_path), stderr=str(stderr_path), fault='rank1 child proxy stopped')
        json.dump(result, receipt, indent=2); receipt.write('\n')
        print(json.dumps(result), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
