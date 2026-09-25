"""Opt-in native CUDA cache reclamation after profiling and graph warmup."""
import json
import os
import re


def apply_allocator_policy(worker, torch_module=None):
    raw = os.environ.get('DOTS3_ALLOCATOR_FRACTION', '')
    if not raw:
        return
    if torch_module is None:
        import torch as torch_module
    if torch_module.cuda.get_allocator_backend() != 'native':
        raise ValueError('Cache reclamation policy requires the native CUDA allocator')
    fraction = float(raw)
    if not 0 < fraction < 1:
        raise ValueError('DOTS3_ALLOCATOR_FRACTION must be between zero and one')
    config = os.environ.get('PYTORCH_ALLOC_CONF', '')
    if not config or os.environ.get('PYTORCH_CUDA_ALLOC_CONF'):
        raise ValueError('Allocator policy requires unambiguous PYTORCH_ALLOC_CONF')
    desired = re.search(r'(?:^|,)garbage_collection_threshold:([^,]+)', config)
    if desired is None or not 0 < float(desired.group(1)) < 1:
        raise ValueError('Allocator policy requires a GC threshold between zero and one')
    before = torch_module.cuda.memory._snapshot()['allocator_settings']
    active_config = before['PYTORCH_CUDA_ALLOC_CONF']
    active_config = re.sub(r'(?:^|,)garbage_collection_threshold:[^,]+', '', active_config).strip(',')
    applied_config = ','.join(filter(None, [active_config,
        'garbage_collection_threshold:' + desired.group(1)]))
    torch_module._C._accelerator_setAllocatorSettings(applied_config)
    torch_module.cuda.set_per_process_memory_fraction(fraction)
    settings = torch_module.cuda.memory._snapshot()['allocator_settings']
    actual = torch_module.cuda.get_per_process_memory_fraction()
    gc = settings['garbage_collection_threshold']
    untouched = set(before) - {'PYTORCH_CUDA_ALLOC_CONF', 'garbage_collection_threshold'}
    if any(settings[key] != before[key] for key in untouched):
        raise RuntimeError(f'Unrequested allocator settings changed: {before=}, {settings=}')
    if abs(actual - fraction) > 1e-8 or gc != float(desired.group(1)):
        raise RuntimeError(f'Allocator policy did not activate: {actual=}, {settings=}')
    total = torch_module.cuda.get_device_properties(torch_module.cuda.current_device()).total_memory
    receipt = {'phase': 'after_compile_and_warmup_before_requests',
               'per_process_fraction': actual, 'settings': settings,
               'previous_settings': before,
               'total_device_bytes': total, 'allocator_ceiling_bytes': int(total * actual),
               'reclamation_threshold_bytes': int(total * actual * gc)}
    worker.dots3_allocator_policy = receipt
    print('DOTS3_ALLOCATOR_POLICY ' + json.dumps(receipt), flush=True)
    return receipt
