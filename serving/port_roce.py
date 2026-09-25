#!/usr/bin/env python3
"""Apply the optional Spark RoCE slot and fail-stop output checks to vLLM0.30."""
from pathlib import Path
import sys


def port(root):
    edits = {}

    def replace(relative, old, new, count=1):
        path = root / relative
        text = edits.get(path, path.read_text())
        if text.count(old) != count:
            raise RuntimeError(f'{path}: expected {count} source matches, found {text.count(old)}: {old[:100]!r}')
        edits[path] = text.replace(old, new)

    comm = 'distributed/device_communicators/cuda_communicator.py'
    replace(comm,
        '        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:\n'
        '            # Initialize a custom fast all-reduce implementation.\n'
        '            self.ca_comm = CustomAllreduce(\n'
        '                group=self.cpu_group,\n'
        '                device=self.device,\n'
        '                symm_mem_enabled=(\n'
        '                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled\n'
        '                ),\n'
        '            )\n',
        '        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:\n'
        '            from vllm.distributed.device_communicators.b12x_roce_all_reduce import maybe_create_roce\n'
        '            self.ca_comm = maybe_create_roce(self.cpu_group, self.device, unique_name)\n'
        '            if self.ca_comm is None:\n'
        '                self.ca_comm = CustomAllreduce(\n'
        '                    group=self.cpu_group,\n'
        '                    device=self.device,\n'
        '                    symm_mem_enabled=(\n'
        '                        self.symm_mem_comm is not None and not self.symm_mem_comm.disabled\n'
        '                    ),\n'
        '                )\n')
    replace(comm,
        '        if self.ca_comm is not None and not self.ca_comm.disabled:\n'
        '            enabled_ar_backends.append("CUSTOM")\n',
        '        if self.ca_comm is not None and not self.ca_comm.disabled:\n'
        '            name = getattr(self.ca_comm, "backend_name", lambda: "CUSTOM")\n'
        '            enabled_ar_backends.append(name())\n')
    replace(comm,
        '    def all_reduce(self, input_):\n'
        '        fi_ar_comm = self.fi_ar_comm\n',
        '    def all_reduce(self, input_):\n'
        '        # A poisoned custom transport cannot be hidden by another backend.\n'
        '        if self.ca_comm is not None:\n'
        '            health = getattr(self.ca_comm, "check_health", None)\n'
        '            if health is not None:\n'
        '                health()\n'
        '        fi_ar_comm = self.fi_ar_comm\n')
    replace(comm,
        '    def destroy(self):\n'
        '        if self.pynccl_comm is not None:\n'
        '            self.pynccl_comm.destroy()\n'
        '            self.pynccl_comm = None\n'
        '        if self.ca_comm is not None:\n'
        '            self.ca_comm = None\n',
        '    def destroy(self):\n'
        '        # Quiesce and release registered RoCE memory while groups live.\n'
        '        if self.ca_comm is not None:\n'
        '            self.ca_comm.close()\n'
        '            self.ca_comm = None\n'
        '        if self.pynccl_comm is not None:\n'
        '            self.pynccl_comm.destroy()\n'
        '            self.pynccl_comm = None\n')
    # CUDA graph replay skips Python communicator hooks. Check after the existing
    # output-copy event wait and before token/pooling data is published. No new
    # synchronization is introduced in either runner generation.
    for path, event in [
        ('v1/worker/gpu/async_utils.py', 'self.copy_event'),
        ('v1/worker/gpu_model_runner.py', 'self.async_copy_ready_event'),
    ]:
        old = f'        {event}.synchronize()\n'
        replace(path, old, old +
                '        from vllm.distributed.device_communicators.b12x_roce_all_reduce import check_roce_health\n'
                '        check_roce_health()\n', count=2)
    # Check all source anchors before touching any file, then syntax-check them.
    import ast
    for path, text in edits.items():
        ast.parse(text, filename=str(path))
    for path, text in edits.items():
        path.write_text(text)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit(f'usage: {sys.argv[0]} VLLM_PACKAGE_ROOT')
    port(Path(sys.argv[1]))
    print('Optional Spark B12x RoCE TP2 adapter and output health checks applied')
