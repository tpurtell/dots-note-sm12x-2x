# SPDX-License-Identifier: Apache-2.0
"""Optional prepared Spark TP2 RoCE transport with fail-stop health handling.

No peer buffer or output is borrowed from another process. The runtime owns
its registered pinned transport, the session owns executable preparation, and
PyTorch owns each out-of-place result (including graph-pool allocations).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
import os
from pathlib import Path
import threading
import weakref

import torch
import torch.distributed as dist
from vllm.logger import init_logger

logger = init_logger(__name__)
_ACTIVE = weakref.WeakSet()


def _enabled(name, default='0'):
    return os.getenv(name, default).lower() not in ('', '0', 'false', 'no', 'off')


def check_roce_health():
    """Host-only check, also called after the runner's existing output-copy wait."""
    for adapter in tuple(_ACTIVE):
        adapter.check_health()


def maybe_create_roce(group, device, unique_name):
    """Negotiate the opt-in on TP2's CPU group, including inconsistent flags."""
    if unique_name.split(':', 1)[0] != 'tp' or dist.get_world_size(group) != 2:
        return None
    local_enabled = _enabled('DOTS3_B12X_ROCE')
    flags = [None, None]
    dist.all_gather_object(flags, local_enabled, group=group)
    if not any(flags):
        return None
    if not all(flags):
        raise RuntimeError('DOTS3_B12X_ROCE must be enabled identically on both TP ranks')
    return B12xRoceAllReduce(group, device)


class B12xRoceAllReduce:
    """vLLM's CustomAllreduce slot, restricted to two distinct GB10 hosts."""

    def __init__(self, group, device):
        from b12x.comm import roce
        from b12x.preparation import PreparationSession, PreparedCall

        device = torch.device(f'cuda:{device}') if isinstance(device, int) else torch.device(device)
        if device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        self.group, self.device = group, device
        self.disabled, self._closed, self._capturing = True, False, False
        self.runtime = self.session = self.plan = None
        self._stop = threading.Event()
        self._watchdog = None
        # Serialize source selection and failures over the CPU group before RDMA
        # construction so a locally invalid config cannot choose native fallback.
        try:
            if dist.get_backend(group) != 'gloo':
                raise ValueError('B12x RoCE requires a Gloo CPU control group')
            rows_text = os.getenv('DOTS3_B12X_ROCE_ROWS', '1-64')
            if rows_text == '1-64':
                self.rows = frozenset(range(1, 65))
            else:
                self.rows = frozenset(int(value) for value in rows_text.split(','))
            if not self.rows or min(self.rows) < 1 or max(self.rows) > 64:
                raise ValueError('DOTS3_B12X_ROCE_ROWS must select rows within 1..64')
            self.enable_eager = _enabled('DOTS3_B12X_ROCE_EAGER')
            self.max_size = max(self.rows) * 5120 * 2
            properties = torch.cuda.get_device_properties(device)
            if torch.cuda.get_device_capability(device) != (12, 1) or not roce.is_supported():
                raise ValueError('B12x RoCE adapter requires SM121 integrated GPUs with active RDMA')
            local = {'error': None, 'rows': sorted(self.rows), 'eager': self.enable_eager,
                     'gpu_uuid': str(properties.uuid),
                     'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
        except Exception as error:
            local = {'error': repr(error)}
        peers = [None, None]
        dist.all_gather_object(peers, local, group=group)
        if any(peer['error'] is not None for peer in peers):
            raise RuntimeError(f'RoCE TP configuration rejected: {peers}')
        if any(peers[0][key] != peers[1][key] for key in ('rows', 'eager')):
            raise RuntimeError('RoCE shapes and eager setting must match across TP ranks')
        if peers[0]['boot_id'] == peers[1]['boot_id'] or peers[0]['gpu_uuid'] == peers[1]['gpu_uuid']:
            raise RuntimeError('RoCE TP2 requires one integrated GPU on each of two distinct hosts')
        os.environ.setdefault('B12X_ROCE_CACHE_DIR', '/root/.cache/vllm-runtime/b12x/roce')
        try:
            # Constructor honors explicit HCA/GID arguments. Its legacy factory
            # drops extra kwargs, so do not pass transport configuration there.
            self.runtime = roce.AllReduce(exchange_group=group, device=device,
                max_size=self.max_size, max_gather_bytes=16,
                hca_names=roce.discover_hcas(roce.default_gid_index()),
                gid_index=roce.default_gid_index())
            query = roce.query_from_runtime(self.runtime, surface='AllReduce.all_reduce',
                call={'dtypes': ('bfloat16',), 'hidden': 5120, 'rows': tuple(sorted(self.rows))},
                topology='roce_rdma', peer_hosts=tuple(peer['boot_id'] for peer in peers))
            self.plan = roce.plan(query, runtime=self.runtime)
            primer = torch.zeros((max(self.rows), 5120), device=device, dtype=torch.bfloat16)
            output = torch.empty_like(primer)
            self.session = PreparationSession(device=device, autotune=False, compile_workers=0)

            def prepare(state):
                # Meet after rank-local compilation; GPU spin wait must not be
                # consumed while the other host is compiling its launcher.
                dist.monitored_barrier(group=group, timeout=timedelta(seconds=120), wait_all_ranks=True)
                return PreparedCall(run=lambda: state.all_reduce(primer, out=output),
                                    output=output, owners=(primer, output))

            self.session.prepare((self.plan.request(name='dots3-spark-tp-roce', prepare_call=prepare),))
            torch.cuda.synchronize(device)
            self.runtime.check_health()
            self.disabled = False
            _ACTIVE.add(self)
            self._watchdog = threading.Thread(target=self._watch_health, daemon=True,
                                              name='dots3-roce-health')
            self._watchdog.start()
            logger.info('B12x RoCE enabled: rows=%s, hidden=5120, BF16, eager=%s, transport_capacity=%d, hcas=%s',
                        sorted(self.rows), self.enable_eager, self.max_size, self.runtime.hca_names)
        except BaseException:
            self.close()
            raise

    def backend_name(self):
        return 'B12X_ROCE_ONESHOT'

    def check_health(self):
        if not self._closed and self.runtime is not None:
            self.runtime.check_health()

    def _watch_health(self):
        # Non-output TP ranks may never unwrap AsyncOutput. Reading the mapped
        # host error words also catches a poisoned idle rank after graph replay.
        # vLLM can swallow non-output RPC exceptions; fail this worker process
        # so the executor detects its death and tears down the complete engine.
        while not self._stop.wait(0.05):
            try:
                self.check_health()
            except BaseException as error:
                os.write(2, f'FATAL B12x RoCE worker poison: {error}\n'.encode())
                os._exit(70)

    def should_custom_ar(self, inp):
        self.check_health()  # Poison never becomes an unsupported/native fallback.
        if self.disabled or self._closed:
            return False
        if not (self.enable_eager or self._capturing or torch.cuda.is_current_stream_capturing()):
            return False
        return (inp.device == self.device and inp.dtype == torch.bfloat16
                and inp.ndim == 2 and inp.shape[1] == 5120 and inp.shape[0] in self.rows
                and inp.is_contiguous() and self.runtime.should_allreduce(inp))

    def custom_all_reduce(self, inp):
        if not self.should_custom_ar(inp):
            return None
        # Do not share one scratch output between layers: residuals can keep
        # earlier results live. Graph capture gives each result a stable address
        # in its graph pool, exactly as native out-of-place NCCL does.
        out = torch.empty_like(inp)
        self.runtime.all_reduce(inp, out=out, plan=self.plan)
        return out

    def custom_all_gather(self, inp):
        self.check_health()
        return None

    def custom_reduce_scatter(self, inp):
        self.check_health()
        return None

    @contextmanager
    def capture(self, stream=None):
        self.check_health()
        if self.disabled or self._closed:
            yield
            return
        old = self._capturing
        self._capturing = True
        try:
            with self.runtime.capture(stream=stream):
                yield
        finally:
            self._capturing = old

    def close(self):
        if self._closed:
            return
        self._stop.set()
        if self._watchdog is not None:
            self._watchdog.join(timeout=2)
        _ACTIVE.discard(self)
        try:
            # Explicitly quiesce references before releasing the prepared state.
            if self.runtime is not None:
                torch.cuda.synchronize(self.device)
        finally:
            try:
                if self.session is not None:
                    self.session.close()
            finally:
                if self.runtime is not None:
                    self.runtime.close()
                self._closed = True
                self.disabled = True
