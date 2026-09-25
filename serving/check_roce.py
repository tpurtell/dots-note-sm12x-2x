#!/usr/bin/env python3
"""Two-host BF16 Dots3 RoCE probe; launch once per Spark with torchrun.

No serving adapter is changed. Uses Gloo for setup and vLLM's PyNccl for the
native NCCL oracle, avoiding a separate torch NCCL process group. Do not run
alongside a model benchmark. See check_roce.md for complete two-host commands.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import threading
import time
import traceback


def mem_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read physical MemAvailable')


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', nargs='+', type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--hidden', type=int, default=5120)
    parser.add_argument('--hcas', help='Comma-separated local HCA device names; auto-discover if omitted')
    parser.add_argument('--gid-index', type=int, default=3)
    parser.add_argument('--peer-hosts', nargs=2, required=True, help='Rank-ordered host labels/IPs for provenance')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=120, help='Gloo process-group timeout seconds')
    parser.add_argument('--deadline', type=int, default=1200, help='Hard process deadline, including cleanup')
    parser.add_argument('--min-available-gib', type=float, default=16)
    parser.add_argument('--abort-available-gib', type=float, default=8)
    parser.add_argument('--spin-limit', type=int, default=20_000_000)
    parser.add_argument('--output', type=Path, required=True, help='Local append-only JSONL receipt')
    args = parser.parse_args()
    for name in ['hidden', 'iterations', 'samples', 'trials', 'timeout', 'deadline', 'spin_limit']:
        if getattr(args, name) <= 0:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.warmup < 0 or not args.rows or any(n <= 0 for n in args.rows):
        parser.error('Rows must be positive; warmup must be nonnegative')
    if args.hidden % 8 or max(args.rows) * args.hidden * 2 > 1 << 20:
        parser.error('This bounded probe requires 16-byte row alignment and at most 1 MiB per payload')
    if not 0 < args.abort_available_gib < args.min_available_gib:
        parser.error('Require 0 < abort headroom < initial headroom')
    if args.hcas and not 1 <= len(args.hcas.split(',')) <= 2:
        parser.error('Specify one or two HCA devices')
    return args


def main():
    args = arguments()
    rank = int(os.environ.get('RANK', '-1'))
    if rank not in (0, 1) or int(os.environ.get('WORLD_SIZE', '0')) != 2:
        raise SystemExit('Use torchrun with exactly two nodes and one process per node')
    if int(os.environ.get('LOCAL_WORLD_SIZE', '1')) != 1:
        raise SystemExit('One probe process per Spark is required')
    # The project keeps all new files on its Linux filesystem, never /mnt/scratch.
    if args.output.resolve().is_relative_to(Path('/mnt/scratch')):
        raise SystemExit('Write the receipt under the project .cache directory')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    receipt = args.output.open('x', buffering=1)
    emit_lock = threading.Lock()

    def emit(event, **values):
        row = {'event': event, 'rank': rank, 'time': time.time(), **values}
        with emit_lock:
            line = json.dumps(row, sort_keys=True)
            receipt.write(line + '\n')
            print(line, flush=True)

    initial_available = mem_available()
    if initial_available < args.min_available_gib * 2**30:
        emit('refused', available_bytes=initial_available, reason='Insufficient initial physical headroom')
        receipt.close()
        raise SystemExit(2)
    if args.hcas:
        os.environ['B12X_ROCE_HCA'] = args.hcas
        # Compare the same physical transport choice, not a different NCCL rail.
        os.environ['NCCL_IB_HCA'] = '=' + args.hcas
    os.environ['B12X_ROCE_GID_INDEX'] = str(args.gid_index)
    os.environ['NCCL_IB_GID_INDEX'] = str(args.gid_index)
    os.environ['B12X_ROCE_SPIN_LIMIT'] = str(args.spin_limit)
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_IB_DISABLE', '0')
    stopped = threading.Event()
    started = time.monotonic()

    def watchdog():
        low = 0
        while not stopped.wait(1):
            available = mem_available()
            low = low + 1 if available < args.abort_available_gib * 2**30 else 0
            if low >= 3 or time.monotonic() - started > args.deadline:
                # A CUDA synchronization or communicator teardown can block Python's
                # main thread. Exit only this probe process; torchrun reports failure.
                emit('hard_abort', available_bytes=available,
                     reason='physical headroom' if low >= 3 else 'process deadline')
                os._exit(124)

    thread = threading.Thread(target=watchdog, daemon=True)
    thread.start()
    emit('starting', available_bytes=initial_available, deadline_seconds=args.deadline)

    def interrupted(signum, frame):
        raise InterruptedError(f'Probe received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    torch = dist = runtime = session = nccl = None
    graphs = []
    complete = False
    try:
        import torch
        import torch.distributed as dist
        from b12x.comm import roce
        from b12x.preparation import PreparationSession, PreparedCall
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
        if not roce.is_supported():
            raise RuntimeError('RoCE requires an integrated GPU and active RDMA devices')
        # Bound PyTorch allocations; transport pinned memory is separately bounded below.
        torch.cuda.set_per_process_memory_fraction(0.05, device)
        dist.init_process_group('gloo', timeout=timedelta(seconds=args.timeout))
        group = dist.group.WORLD
        peer_configs = [None, None]
        distributed_config = {k: v for k, v in vars(args).items() if k not in {'output', 'hcas'}}
        dist.all_gather_object(peer_configs, distributed_config, group=group)
        if peer_configs[0] != peer_configs[1]:
            raise RuntimeError('Both ranks must use identical probe settings (local HCA names may differ)')

        def barrier():
            dist.monitored_barrier(group=group, timeout=timedelta(seconds=args.timeout), wait_all_ranks=True)

        capacity = max(args.rows) * args.hidden * 2
        # Never inherit the library's 16 MiB default gather capacity. Although this
        # probe uses only reduce, public preparation allocates padded-gather scratch.
        runtime = roce.AllReduce(exchange_group=group, device=device,
            max_size=capacity, max_gather_bytes=16,
            hca_names=tuple(args.hcas.split(',')) if args.hcas else None,
            gid_index=args.gid_index)
        query = roce.query_from_runtime(runtime, surface='AllReduce.all_reduce',
            call={'dtypes': ('bfloat16',)}, topology='roce_rdma', peer_hosts=tuple(args.peer_hosts))
        declaration = roce.plan(query, runtime=runtime)
        primer = torch.zeros((max(args.rows), args.hidden), dtype=torch.bfloat16, device=device)
        primer_out = torch.empty_like(primer)
        session = PreparationSession(device=device, autotune=False, compile_workers=0)
        def prepare_collective(state):
            # Materialization/compilation can take different times on the ranks.
            # Meet on CPU before launching the first bounded-spin GPU collective.
            barrier()
            return PreparedCall(run=lambda: state.all_reduce(primer, out=primer_out), output=primer_out)
        session.prepare((declaration.request(name='dots3-roce-bf16', prepare_call=prepare_collective),))
        torch.cuda.synchronize()
        runtime.check_health()
        nccl = PyNcclCommunicator(group, device)
        if not nccl.available or nccl.disabled:
            raise RuntimeError('Native vLLM PyNccl communicator is unavailable')
        emit('setup', args={**vars(args), 'output': str(args.output)},
             versions={n: importlib.metadata.version(n) for n in ['torch', 'vllm', 'nvidia-cutlass-dsl']},
             gpu=torch.cuda.get_device_name(device), gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
             roce_api=roce.API_VERSION, stats=runtime.stats(), available_bytes=mem_available(),
             probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(),
             nccl_env={k: v for k, v in os.environ.items() if k.startswith('NCCL_')})

        def sync_health():
            torch.cuda.synchronize()
            runtime.check_health()

        def reduce(inp, out):
            result = runtime.all_reduce(inp, out=out, plan=declaration)
            if result is None:
                raise RuntimeError('Eligible RoCE collective unexpectedly declined')
            return result

        def native(inp, out):
            result = nccl.all_reduce(inp, out_tensor=out)
            if result is None:
                raise RuntimeError('Native NCCL unexpectedly declined')
            return result

        def exact(output, oracle):
            torch.testing.assert_close(output, oracle, rtol=0, atol=0)
            digest = hashlib.sha256(output.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            peers = [None, None]
            dist.all_gather_object(peers, digest, group=group)
            if peers[0] != peers[1]:
                raise AssertionError('RoCE output is not bit-identical across ranks')

        def measure(call):
            for _ in range(args.warmup):
                call()
            sync_health()
            samples = []
            for _ in range(args.samples):
                barrier()
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.iterations):
                    call()
                end.record()
                end.synchronize()
                runtime.check_health()
                samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
            peers = [None, None]
            dist.all_gather_object(peers, samples, group=group)
            return {'rank_samples_us': peers,
                    'slowest_rank_median_us': max(statistics.median(values) for values in peers)}

        cases = []
        for case_number, rows in enumerate(args.rows):
            inp = torch.zeros((rows, args.hidden), dtype=torch.bfloat16, device=device)
            out = torch.empty_like(inp)
            oracle = torch.empty_like(inp)
            if not runtime.should_allreduce(inp):
                raise RuntimeError(f'RoCE declined aligned BF16 {rows} x {args.hidden}')
            eager = {'roce': lambda: reduce(inp, out), 'nccl': lambda: native(inp, oracle)}
            for trial in range(args.trials):
                torch.manual_seed(1100 + rank * 100 + trial)
                inp.normal_()
                native(inp, oracle)
                reduce(inp, out)
                sync_health()
                exact(out, oracle)
            pair = {}
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream())
            for name in ('nccl', 'roce'):
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        eager[name]()
                sync_health()
                barrier()
                graph = torch.cuda.CUDAGraph()
                with (runtime.capture(stream=stream) if name == 'roce' else nullcontext()):
                    with torch.cuda.graph(graph, stream=stream):
                        eager[name]()
                graphs.append(graph)
                pair[name] = graph
                sync_health()
                barrier()
            for trial in range(args.trials):
                torch.manual_seed(2200 + rank * 100 + trial)
                inp.normal_()
                pair['nccl'].replay()
                pair['roce'].replay()
                sync_health()
                exact(out, oracle)
            # Reverse arm ordering at alternating shapes to expose order bias.
            order = ['nccl_eager', 'roce_eager', 'nccl_graph', 'roce_graph']
            if case_number % 2:
                order.reverse()
            timings = {}
            for name in order:
                backend, mode = name.split('_')
                timings[name] = measure(pair[backend].replay if mode == 'graph' else eager[backend])
            emit('shape', rows=rows, hidden=args.hidden, payload_bytes=rows * args.hidden * 2,
                 eager_exact=True, changed_input_graph_exact=True, timing_order=order,
                 timings=timings, stats=runtime.stats(), available_bytes=mem_available())
            cases.append((inp, out, oracle, pair))

        # Revisit small and large captured grids after the whole size sweep.
        for cycle in range(args.trials):
            for index in (0, len(cases) - 1, 0):
                inp, out, oracle, pair = cases[index]
                torch.manual_seed(3300 + rank * 100 + cycle * 10 + index)
                inp.normal_()
                native(inp, oracle)
                pair['roce'].replay()
                sync_health()
                exact(out, oracle)
        # Ordered eager transitions between different streams use runtime events.
        inp, out, oracle, _ = cases[-1]
        streams = [torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)]
        for trial in range(args.trials):
            inp.fill_(rank + trial + 1)
            native(inp, oracle)
            sync_health()
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    reduce(inp, out)
            sync_health()
            exact(out, oracle)
        barrier()
        emit('correctness_complete', alternating_graph_sizes_exact=True, ordered_multistream_exact=True,
             stats=runtime.stats(), available_bytes=mem_available(),
             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
             peak_reserved_bytes=torch.cuda.max_memory_reserved())
        complete = True
    except BaseException as exc:
        emit('failed', error=repr(exc), traceback=traceback.format_exc(),
             poisoned=bool(runtime.poisoned) if runtime is not None else None)
        raise
    finally:
        # No success-path barriers after an error: the peer may already be gone.
        # Hard watchdog remains active through CUDA synchronization and teardown.
        cleanup_errors = []
        def cleanup(name, call):
            try:
                call()
            except BaseException as exc:
                cleanup_errors.append({'step': name, 'error': repr(exc)})
        if torch is not None:
            cleanup('cuda_synchronize', torch.cuda.synchronize)
        for graph in graphs:
            cleanup('graph_reset', graph.reset)
        if session is not None:
            cleanup('session_close', session.close)
        if runtime is not None:
            cleanup('roce_close', runtime.close)
        if nccl is not None:
            cleanup('nccl_destroy', nccl.destroy)
        if dist is not None and dist.is_initialized():
            cleanup('destroy_groups', dist.destroy_process_group)
        emit('complete' if complete and not cleanup_errors else 'incomplete', cleanup_errors=cleanup_errors)
        stopped.set()
        thread.join(timeout=2)
        receipt.close()
        if complete and cleanup_errors:
            raise RuntimeError(f'Probe cleanup failed: {cleanup_errors}')


if __name__ == '__main__':
    main()
