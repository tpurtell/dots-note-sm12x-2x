#!/usr/bin/env python3
"""Isolated two-host qualification of the installed Spark CUDA communicator.

Uses the real patched CudaCommunicator and its native PyNccl fallback. No
engine or model is loaded. This is not a transport failure-injection test.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

from check_roce import mem_available


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',nargs='+',type=int,default=[1,2,4,8,16,32,48,64])
    p.add_argument('--hcas',default='roceP2p1s0f0')
    p.add_argument('--gid-index',type=int,default=3)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--timeout',type=int,default=120)
    p.add_argument('--deadline',type=int,default=1200)
    p.add_argument('--min-available-gib',type=float,default=16)
    p.add_argument('--abort-available-gib',type=float,default=8)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if any(r<1 or r>64 for r in args.rows) or min(args.trials,args.timeout,args.deadline)<1:
        p.error('Require rows1..64 and positive trials/timeouts')
    if not 0<args.abort_available_gib<args.min_available_gib:
        p.error('Require0<abort<initial physical headroom')
    rank=int(os.environ.get('RANK','-1'))
    if rank not in (0,1) or int(os.environ.get('WORLD_SIZE','0'))!=2 or int(os.environ.get('LOCAL_WORLD_SIZE','1'))!=1:
        p.error('Launch with torchrun two nodes, one process per host')
    if args.output.resolve().is_relative_to('/mnt/scratch'):
        p.error('Use project .cache for output')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=args.output.open('x',buffering=1)
    lock=threading.Lock()
    def emit(event,**data):
        with lock:
            line=json.dumps({'event':event,'rank':rank,'unix_time':time.time(),**data})
            receipt.write(line+'\n');print(line,flush=True)
    if mem_available()<args.min_available_gib*2**30:
        emit('refused',reason='initial physical headroom');receipt.close();raise SystemExit(2)
    os.environ.update(DOTS3_B12X_ROCE='1',DOTS3_B12X_ROCE_ROWS='1-64',DOTS3_B12X_ROCE_EAGER='0',
        B12X_ROCE_HCA=args.hcas,B12X_ROCE_GID_INDEX=str(args.gid_index),
        NCCL_IB_HCA='='+args.hcas,NCCL_IB_GID_INDEX=str(args.gid_index),NCCL_IB_DISABLE='0',
        VLLM_ALLREDUCE_USE_FLASHINFER='0',VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC='0',VLLM_ALLREDUCE_USE_SYMM_MEM='0')
    stopped=threading.Event();started=time.monotonic()
    def watchdog():
        low=0
        while not stopped.wait(1):
            available=mem_available();low=low+1 if available<args.abort_available_gib*2**30 else 0
            if low>=3 or time.monotonic()-started>args.deadline:
                emit('hard_abort',available_bytes=available,reason='physical headroom' if low>=3 else 'deadline')
                os._exit(124)
    watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
    torch=dist=comm=None;graphs=[];complete=False
    try:
        import torch
        import torch.distributed as dist
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
        from vllm.distributed.device_communicators import b12x_roce_all_reduce as implementation
        torch.cuda.set_device(0);device=torch.device('cuda:0')
        dist.init_process_group('gloo',timeout=timedelta(seconds=args.timeout))
        group=dist.group.WORLD
        configs=[None,None]
        dist.all_gather_object(configs, {'rows':args.rows,'trials':args.trials,
            'hcas':args.hcas,'gid_index':args.gid_index,'timeout':args.timeout})
        if configs[0]!=configs[1]:
            raise RuntimeError(f'Rank probe configuration mismatch: {configs}')
        def barrier():
            dist.monitored_barrier(group=group,timeout=timedelta(seconds=args.timeout),wait_all_ranks=True)
        barrier()
        comm=CudaCommunicator(cpu_group=group,device=device,unique_name='tp:roce-adapter-probe')
        adapter=comm.ca_comm;nccl=comm.pynccl_comm
        if not isinstance(adapter,implementation.B12xRoceAllReduce) or adapter.disabled or nccl is None or nccl.disabled:
            raise RuntimeError('Actual communicator did not install active B12x/native backends')
        calls={'custom':0,'native':0}
        original_custom,original_native=adapter.custom_all_reduce,nccl.all_reduce
        def counted_custom(*a,**kw):
            calls['custom']+=1;return original_custom(*a,**kw)
        def counted_native(*a,**kw):
            calls['native']+=1;return original_native(*a,**kw)
        adapter.custom_all_reduce=counted_custom;nccl.all_reduce=counted_native
        emit('setup',args=vars(args)|{'output':str(args.output)},gpu=torch.cuda.get_device_name(),
            adapter_sha256=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest(),
            communicator_sha256=hashlib.sha256(Path(__import__(CudaCommunicator.__module__,fromlist=['']).__file__).read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            runtime=adapter.runtime.stats(),available_bytes=mem_available())
        def sync():
            torch.cuda.synchronize();implementation.check_roce_health()
        def native(x):
            out=original_native(x)
            if out is None:raise RuntimeError('Native PyNccl declined input')
            return out
        def same(a,b):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        cases=[(f'bf16-r{r}',(r,5120),torch.bfloat16,True) for r in args.rows]
        cases.extend([('rows65',(65,5120),torch.bfloat16,False),
                      ('wrong-width',(4,5128),torch.bfloat16,False),
                      ('float32',(4,5120),torch.float32,False)])
        retained=[]
        for name,shape,dtype,eligible in cases:
            barrier()
            torch.manual_seed(812+rank)
            x=torch.randint(-16,17,shape,device=device).to(dtype)
            # Graph-only integration must use native outside capture, even for
            # eligible geometry. Instrumentation observes actual dispatch.
            before=dict(calls);eager=comm.all_reduce(x);oracle=native(x);sync();same(eager,oracle)
            assert calls['custom']==before['custom'] and calls['native']==before['native']+1
            with adapter.capture():
                assert adapter.should_custom_ar(x)==eligible
                before=dict(calls);a=comm.all_reduce(x);b=comm.all_reduce(x+1)
                sync();same(a,oracle);same(b,native(x+1));sync()
                selected='custom' if eligible else 'native'
                assert calls[selected]==before[selected]+2
                assert a.data_ptr()!=b.data_ptr(), 'consecutive outputs alias'
                # Warm the same expressions before graph capture.
                barrier()
                graph=torch.cuda.CUDAGraph()
                before=dict(calls)
                with torch.cuda.graph(graph):
                    first=comm.all_reduce(x)
                    second=comm.all_reduce(x+1)
                assert calls[selected]==before[selected]+2
            graphs.append(graph);retained.append((x,first,second,graph,eligible,name))
            assert first.data_ptr()!=second.data_ptr()
            for trial in range(args.trials):
                x.copy_(torch.randint(-16,17,shape,device=device).to(dtype)+rank+trial)
                barrier();graph.replay();sync()
                same(first,native(x));same(second,native(x+1));sync()
            emit('case_passed',case=name,eligible=eligible,shape=list(shape),dtype=str(dtype),
                changed_input_trials=args.trials,eager_native_exact=True,graph_native_exact=True,
                independent_output_addresses=True,dispatch_calls=dict(calls))
        # Explicitly decline unsupported noncontiguous input; no claim that
        # the native raw-pointer API supports strided tensors.
        view=torch.empty((4,10240),device=device,dtype=torch.bfloat16)[:,::2]
        with adapter.capture():
            assert not adapter.should_custom_ar(view)
            assert adapter.custom_all_reduce(view) is None
        emit('noncontiguous_declined')
        for x,first,second,graph,eligible,name in reversed(retained):
            x.add_(1);barrier();graph.replay();sync()
            same(first,native(x));same(second,native(x+1));sync()
        barrier();complete=True
        emit('correctness_complete',alternating_shapes_exact=True,runtime=adapter.runtime.stats(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),available_bytes=mem_available(),
            scope='real CUDA communicator dispatch; no model, throughput claim, or failure injection')
    except BaseException as exc:
        emit('failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        errors=[]
        def cleanup(name,fn):
            try:fn()
            except BaseException as exc:errors.append({'step':name,'error':repr(exc)})
        if torch is not None:cleanup('synchronize',torch.cuda.synchronize)
        for graph in graphs:cleanup('graph_reset',graph.reset)
        if comm is not None:cleanup('communicator_destroy',comm.destroy)
        if dist is not None and dist.is_initialized():cleanup('group_destroy',dist.destroy_process_group)
        emit('complete' if complete and not errors else 'incomplete',cleanup_errors=errors)
        stopped.set();watcher.join(timeout=2);receipt.close()
        if complete and errors:raise RuntimeError(errors)


if __name__=='__main__':
    main()
