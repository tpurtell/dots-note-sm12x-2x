# Owner-aware bounded-state grouping candidate

The hybrid decoder splits full-history DSA/indexer layers between workers.
SWA remains bounded. The existing global packed groups, however, are formed
before the per-worker projection and can contain too many SWA layers on one
worker. That widens **every physical pool block**, including blocks assigned
to full-history attention.

For the observed 18/28 decoder split, rank 1 has seven DSA/indexer pairs:

- Useful history page: `7 × (576 + 132) × 64 = 317184` bytes.
- An existing projected SWA group has five layers: `5 × 1088 × 64 = 348160`
  bytes; the B12x whole-record alignment raises the physical stride to348480.
- Owner-aware regrouping gives stride317376, retaining only192 bytes of
  record alignment beyond useful history. Rank0 remains271872 bytes.

This saves physical headroom on rank1. It does **not** establish a higher
whole-model capacity: rank0 was the limiting worker in the measured18/28
startup, and an additional bounded SWA group slightly increases shared block
admission per request. Other placements and boundary weight ownership can
change the limiting worker.

## Candidate implementation

`VLLM_HYBRID_BALANCE_KV_GROUPS=1` invokes the model-independent
`refine_owner_state_groups` before worker projection and memory admission.
Existing global group IDs retain their own layer subsets; overflow compatible
SWA layers receive appended global IDs. Every worker projects the same list.
The function changes neither layer cache formats nor attention/retention
semantics. Kernel physical record width stays576 for DSA and1088 for SWA.

The source port also fixes a real empty-group allocation issue: the original
allocator iterates a `UniformTypeKVCacheSpecs` dictionary even when the worker's
projected `group.layer_names` is empty. It can emit tensor descriptors for
remote layers. Iterating the projected names fixes that while retaining empty
global IDs for scheduler agreement.

CPU gates exercise the actual installed vLLM grouping, projection, tensor
allocation and scheduler reconciliation for cuts1/17/18/22/23/26/45, including
empty worker groups, complete ownership, no duplicate or remote tensor
allocations, preserved original IDs and aligned physical strides. Replicated
TP groups stay unchanged. These are CPU gates; GPU metadata, prefix caching,
SWA/DSA numerics and full-context admission still require runtime qualification.

## Accounting

The placement utility reports useful full-history bytes, useful bounded SWA
bytes, history padding and bounded-state arena overhead separately. It reads
the actual `max_in_flight_tokens` from worker receipts. The current V2 async
configuration has two concurrent batches, so batch512 means1024 in-flight
tokens; with window513 and block64, native SWA admission is25 blocks/request,
not17. All candidate ranking includes the number of resulting SWA groups.

Storage estimates use CUDA-only registered persistent tensors, not CPU
buffers. Profiling, allocator workspace and graph peaks can still change by
placement, so estimated cuts must be checked against measured startup budgets.

## Source anchors

- Upstream `vllm/v1/core/kv_cache_utils.py::_get_packed_kv_cache_groups` constructs
  global state groups; `_project_kv_cache_groups_to_worker` keeps global IDs.
- `get_kv_cache_config_from_groups` lays out the physical pool; the candidate
  fixes its `layers_by_spec` construction for empty worker groups.
- `SlidingWindowSpec.max_admission_blocks_per_request` in
  `vllm/v1/kv_cache_interface.py` is the native bounded-state admission formula.
- Recipe sources: `serving/hybrid_cache_groups.py`,
  `serving/port_hybrid_cache_groups.py`, `serving/check_hybrid_cache_groups.py`,
  `serving/plan_hybrid_placement.py`.
