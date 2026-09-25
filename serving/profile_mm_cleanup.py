"""Experimental release of idle MM profiling segments before decoder profiling.

Preserve live encoder outputs and allocator peak counters. This does not change
KV accounting or remove any active allocation. Not enabled in release images.
"""
def cleanup_profile_mm_cache(torch_module):
    def snapshot():
        stats = torch_module.cuda.memory_stats()
        free, total = torch_module.cuda.mem_get_info()
        return {
            'free_bytes': int(free), 'total_bytes': int(total),
            'allocated_bytes': int(stats['allocated_bytes.all.current']),
            'reserved_bytes': int(stats['reserved_bytes.all.current']),
            'inactive_split_bytes': int(stats['inactive_split_bytes.all.current']),
            'peak_allocated_bytes': int(stats['allocated_bytes.all.peak']),
        }
    before = snapshot()
    torch_module.accelerator.empty_cache()
    after = snapshot()
    return {'phase':'after_mm_profile_before_decoder_profile',
            'before':before,'after':after,
            'released_reserved_bytes':before['reserved_bytes']-after['reserved_bytes'],
            'active_bytes_unchanged':before['allocated_bytes']==after['allocated_bytes'],
            'peak_counter_unchanged':before['peak_allocated_bytes']==after['peak_allocated_bytes']}
