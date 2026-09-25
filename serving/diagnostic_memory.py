"""Disposable worker telemetry for Spark context growth; no allocator changes."""
import json
import os
from pathlib import Path
import threading
import time

from hybrid_attestation import HybridAttestationWorkerExtension


def sample(torch, device):
    counters = torch.cuda.memory_stats(device)
    result = {
        'time': time.time(), 'pid': os.getpid(),
        'cuda': {key: int(value) for key, value in counters.items()},
        'meminfo': Path('/proc/meminfo').read_text(),
        'smaps_rollup': Path('/proc/self/smaps_rollup').read_text(),
    }
    host_stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    if host_stats is not None:
        try:
            result['pinned'] = {key: int(value) for key, value in host_stats().items()}
        except Exception as error:
            result['pinned_error'] = repr(error)
    return result


class MemoryDiagnosticWorkerExtension(HybridAttestationWorkerExtension):
    def hybrid_ownership_receipt(self):
        receipt = super().hybrid_ownership_receipt()
        if getattr(self, '_memory_diagnostic_thread', None) is None:
            import torch
            rank = receipt['rank']
            device = torch.cuda.current_device()
            path = Path('/root/.cache/vllm-runtime') / f'memory-diagnostic-rank{rank}.jsonl'

            def monitor():
                with path.open('a', buffering=1) as output:
                    while True:
                        try:
                            output.write(json.dumps(sample(torch, device)) + '\n')
                        except Exception as error:
                            output.write(json.dumps({'time': time.time(), 'error': repr(error)}) + '\n')
                        time.sleep(2)

            self._memory_diagnostic_thread = threading.Thread(target=monitor, daemon=True)
            self._memory_diagnostic_thread.start()
            receipt['diagnostic_memory_path'] = str(path)
        return receipt
