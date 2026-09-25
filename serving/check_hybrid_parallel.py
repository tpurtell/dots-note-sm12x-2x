#!/usr/bin/env python3
"""CPU protocol checks with two concurrent emulated tensor-parallel ranks."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Lock
from hybrid_parallel import (LayerOwnerPlan, ExpertParallelContext,
    HybridModelCapabilities, RoutedLayerBuffers, execute_routed_layer)

class Buffer:
    def __init__(self, value): self.value = list(value)
    def copy_(self, other): self.value = list(other.value)

class Group:
    def __init__(self):
        self.barrier = Barrier(2, timeout=5)
        self.lock = Lock()
        self.values = {}
        self.events = [[], []]
    def rank(self, rank):
        group = self
        class Transport:
            world_size = 2
            def __init__(self): self.rank = rank; self.step = 0
            def broadcast(self, tensor, owner):
                step = self.step; self.step += 1
                group.events[rank].append(('broadcast', owner))
                if rank == owner: group.values[step] = list(tensor.value)
                group.barrier.wait()
                tensor.value = list(group.values[step])
                group.barrier.wait()
            def reduce_sum(self, out, partial, owner):
                step = self.step; self.step += 1
                group.events[rank].append(('reduce_sum', owner))
                with group.lock: group.values.setdefault(step, {})[rank] = list(partial.value)
                group.barrier.wait()
                if rank == owner:
                    out.value = [sum(x) for x in zip(*group.values[step].values())]
                group.barrier.wait()
        return Transport()

assert LayerOwnerPlan.contiguous(46, 2).owner_changes() == (23,)
assert LayerOwnerPlan.contiguous(46, 2).dense_context(24, 1).owns_parameters
assert ExpertParallelContext(2, 1).tensor_parallel_rank == 1
try:
    HybridModelCapabilities(True, False, True, True).validate()
except ValueError: pass
else: raise AssertionError('missing cache ownership accepted')
for owner in (0, 1):
    for activation in (2.0, -3.0, 7.0):
        group = Group(); calls = []
        def run(rank):
            buffers = RoutedLayerBuffers(*(Buffer([0]) for _ in range(4)))
            def prepare():
                calls.append(('prepare', rank))
                return Buffer([activation]), Buffer([5]), Buffer([0.25])
            def expert(x, ids, weights):
                assert ids.value == [5] and weights.value == [0.25]
                # Distinct tensor partitions of the same selected expert.
                return Buffer([x.value[0] * weights.value[0] * (rank + 1)])
            def finish(reduced):
                calls.append(('finish', rank))
                return reduced.value[0] + activation * 4  # full shared expert once
            return execute_routed_layer(owner=owner, transport=group.rank(rank), buffers=buffers,
                prepare_owner=prepare, execute_local_experts=expert, finish_owner=finish)
        with ThreadPoolExecutor(2) as pool: results = list(pool.map(run, (0, 1)))
        assert results[owner] == activation * (0.25 * (1 + 2) + 4)
        assert results[1 - owner] is None
        assert calls == [('prepare', owner), ('finish', owner)]
        assert group.events[0] == group.events[1] == [('broadcast', owner)] * 3 + [('reduce_sum', owner)]
print('Hybrid expert-TP protocol: both owner ranks, changed activations, identical collectives, single routing/shared contribution passed')
