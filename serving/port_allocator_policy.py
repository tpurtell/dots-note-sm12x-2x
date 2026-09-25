"""Apply allocator policy only after native worker profiling/warmup finishes."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / 'v1/worker/gpu_worker.py'
source = path.read_text()
anchor = '        return CompilationTimes(\n'
if source.count(anchor) != 1 or 'apply_allocator_policy(self)' in source:
    raise RuntimeError('Allocator policy worker anchor changed or already patched')
source = source.replace(anchor,
    '        from allocator_policy import apply_allocator_policy\n'
    '        apply_allocator_policy(self)\n\n' + anchor)
compile(source, str(path), 'exec')
path.write_text(source)
