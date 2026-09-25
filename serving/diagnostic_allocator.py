"""Targeted native allocator cache reclamation trial after KV admission."""
from diagnostic_memory import MemoryDiagnosticWorkerExtension


class AllocatorDiagnosticWorkerExtension(MemoryDiagnosticWorkerExtension):
    def hybrid_ownership_receipt(self):
        import torch
        torch.cuda.set_per_process_memory_fraction(0.90)
        receipt = super().hybrid_ownership_receipt()
        settings = torch.cuda.memory._snapshot()['allocator_settings']
        fraction = torch.cuda.get_per_process_memory_fraction()
        if abs(fraction - 0.90) > 1e-8 or settings['garbage_collection_threshold'] != 0.90:
            raise RuntimeError('Requested diagnostic allocator policy is not active')
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
        receipt['diagnostic_allocator'] = {
            'per_process_fraction': fraction, 'settings': settings,
            'total_device_bytes': total, 'allocator_ceiling_bytes': int(total * fraction),
            'reclamation_threshold_bytes': int(total * fraction * 0.90),
            'applied_at': 'ownership_attestation_after_KV_admission_before_requests',
        }
        return receipt
