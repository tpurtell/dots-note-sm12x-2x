"""Model-independent layer ownership with tensor-parallel routed experts.

The global worker group stays an expert tensor-parallel group. Dense modules
use explicit local contexts; this module never changes distributed globals.
All ranks execute identical collective sequences. Buffers are supplied by the
model adapter and must be allocated before graph capture.
"""
from dataclasses import dataclass
from typing import Protocol, Any, Callable


@dataclass(frozen=True)
class DenseParallelContext:
    owner_rank: int
    rank: int
    @property
    def owns_parameters(self):
        return self.rank == self.owner_rank
    @property
    def linear_kwargs(self):
        return {'disable_tp': True}
    tensor_parallel_size: int = 1
    tensor_parallel_rank: int = 0


@dataclass(frozen=True)
class ExpertParallelContext:
    """Tensor partitions of every expert; this is not expert parallelism."""
    tensor_parallel_size: int
    tensor_parallel_rank: int
    def __post_init__(self):
        if self.tensor_parallel_size < 2 or not 0 <= self.tensor_parallel_rank < self.tensor_parallel_size:
            raise ValueError('invalid expert tensor-parallel context')


@dataclass(frozen=True)
class LayerOwnerPlan:
    owners: tuple[int, ...]
    worker_count: int
    output_owner: int = 0
    def __post_init__(self):
        if self.worker_count < 2 or not self.owners:
            raise ValueError('ownership requires layers and at least two workers')
        if any(not 0 <= owner < self.worker_count for owner in (*self.owners, self.output_owner)):
            raise ValueError('owner outside worker group')
    @classmethod
    def contiguous(cls, layer_count, worker_count, output_owner=0):
        if layer_count < worker_count:
            raise ValueError('each owner must receive at least one layer')
        return cls(tuple(min(i * worker_count // layer_count, worker_count - 1) for i in range(layer_count)), worker_count, output_owner)
    def dense_context(self, layer, rank):
        return DenseParallelContext(self.owners[layer], rank)
    def owner_changes(self):
        return tuple(i for i in range(1, len(self.owners)) if self.owners[i] != self.owners[i - 1])


@dataclass(frozen=True)
class HybridModelCapabilities:
    explicit_dense_parallel_context: bool
    owner_only_kv_cache: bool
    routed_expert_tensor_parallel: bool
    owner_routing: bool
    def validate(self):
        missing = [name for name, value in vars(self).items() if not value]
        if missing:
            raise ValueError('model lacks hybrid ownership capabilities: ' + ', '.join(missing))


class OwnerTransport(Protocol):
    rank: int
    world_size: int
    def broadcast(self, tensor: Any, owner: int) -> None: ...
    def reduce_sum(self, output: Any, local_partial: Any, owner: int) -> None: ...


class PyNcclOwnerTransport:
    """Graph-capturable device collectives using an existing TP communicator."""
    def __init__(self, group):
        self.rank = group.rank_in_group
        self.world_size = group.world_size
        self.communicator = group.device_communicator.pynccl_comm
        if self.communicator is None or self.communicator.disabled:
            raise ValueError('hybrid ownership requires an enabled PyNccl communicator')
    def broadcast(self, tensor, owner):
        self.communicator.broadcast(tensor, src=owner)
    def reduce_sum(self, output, local_partial, owner):
        self.communicator.reduce(output, local_partial, root=owner)


@dataclass
class RoutedLayerBuffers:
    activation: Any
    route_ids: Any
    route_weights: Any
    reduced_output: Any


def execute_routed_layer(
    *, owner: int, transport: OwnerTransport, buffers: RoutedLayerBuffers,
    prepare_owner: Callable[[], tuple[Any, Any, Any]],
    execute_local_experts: Callable[[Any, Any, Any], Any],
    finish_owner: Callable[[Any], Any],
):
    """Route once, broadcast inputs/routes, run every TP shard, reduce to owner.

    ``prepare_owner`` returns activation, route IDs, and route weights using
    the model's native routing semantics. ``finish_owner`` adds the complete
    owner-local shared expert output exactly once and updates owner state.
    The peer result is deliberately None: it does not own a reduced state.
    """
    if not 0 <= owner < transport.world_size:
        raise ValueError('invalid owner rank')
    if transport.rank == owner:
        activation, ids, weights = prepare_owner()
        buffers.activation.copy_(activation)
        buffers.route_ids.copy_(ids)
        buffers.route_weights.copy_(weights)
    transport.broadcast(buffers.activation, owner)
    transport.broadcast(buffers.route_ids, owner)
    transport.broadcast(buffers.route_weights, owner)
    partial = execute_local_experts(buffers.activation, buffers.route_ids, buffers.route_weights)
    transport.reduce_sum(buffers.reduced_output, partial, owner)
    if transport.rank == owner:
        return finish_owner(buffers.reduced_output)
    return None


def transfer_owner_state(transport, *, previous_owner, next_owner, hidden, residual):
    """Move the two residual-stream tensors only at ownership boundaries."""
    if previous_owner == next_owner:
        return
    transport.broadcast(hidden, previous_owner)
    transport.broadcast(residual, previous_owner)
