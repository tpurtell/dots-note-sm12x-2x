# Compact DSA cache candidate

This candidate is opt-in with `DOTS3_COMPACT_DSA_CACHE=1`; it has CPU layout checks but requires GPU component and complete-model qualification. Existing published images remain unchanged.

## Measured versus calculated memory

The published RTX startup reports 79.60–79.62 GiB model loading, 82.49–82.52 GiB consumed memory, 2.68 GiB peak activation, 0.32 GiB final graph pool, and 5.03 GiB allocated KV. The activation field already includes a 0.63 GiB graph estimate (`gpu_worker.py:595,615`); these fields must not be summed with a second graph estimate or a separately reported draft graph. The consumed-memory difference is not isolated to a specific workspace or communicator.

Upstream `models/dots3_note/nvidia/model.py:230–239,306` deliberately changes DSA's logical 576-element FP8 cache record to 1088 elements to match SWA. The 13 target DSA layers plus one DSA MTP layer retain complete history. At 524,288 tokens their unused 512-byte record tails alone occupy **3.5 GiB per rank**. TP2 replicates the single MLA latent head; the two ranks' KV budgets do not add context capacity.

The indexer retains another 132 bytes per history token per DSA layer (128 FP8 values plus one FP32 scale; `model_executor/models/deepseek_v2.py:724`). DSA selects 2048 historical positions for attention computation, but retains history for that selection. SWA's 33 layers retain only the window plus in-flight work; `SlidingWindowSpec.max_admission_blocks_per_request` bounds this correctly.

The upstream packed allocator groups full-history MLA and indexer pages together and SWA into groups of 17 and 16 layers. The larger SWA group sets every shared pool block's byte stride, producing further avoidable padding of full-history blocks. Simply narrowing DSA records is insufficient.

## Candidate changes

`port_compact_cache.py` makes three opt-in changes to already ported vLLM 0.30.0:

- Preserve logical 576-byte DSA FP8 records. SWA remains 1088; indexer remains 132.
- Treat SWA as bounded state when packing mixed-page groups, so SWA groups fit within the full-history block budget. This produces SWA groups of 9, 8, 8, and 8 layers.
- Round the shared block byte stride to a multiple of 576 for B12x physical-record addressing. The additional alignment costs 384 bytes per block.

The B12x adapter passes actual record width and keys prepared state by exact cache shape/strides. The kernel must support `Caps.physical_record_width` with legacy default 1088 and candidate 576. Prefix blocks remain group-specific; groups alias the arena only through distinct allocator block IDs, as upstream already requires.

CPU execution of the actual upstream grouping functions with the model's cache specs gives:

| Layout | Shared block bytes | 262,144 admission GiB/rank | 524,288 admission GiB/rank |
|---|---:|---:|---:|
| Published padded | 1,183,744 | 4.57075 | 9.08637 |
| Compact candidate | 634,752 | 2.48050 | 4.90189 |

Calculation includes 64-token pages, a 513-token SWA window, two in-flight 512-token batches, and an extra boundary block per SWA group. The legacy 262K result agrees with the observed startup's 4.57 GiB requirement. The compact 524K requirement is below the observed 5.03 GiB allocation, but startup profiling and execution must confirm the remaining margin.

## Validation and installation

CPU: `python3 serving/check_compact_cache_layout.py` extracts and executes the patched source functions with minimal spec fixtures. It verifies exact grouping, complete layer ownership, non-overlapping regions within a block, and stride alignment. It does not execute the actual GPU allocator/scheduler.

Image build: copy the new B12x library and `dots3_b12x_attention.py`, then run `python3 serving/port_compact_cache.py /usr/local/lib/python3.12/dist-packages/vllm` after the ordinary port. Enable the env only for the candidate launch.

GPU component: run `python3 serving/check_sparse_adapter.py --compact` and the legacy invocation. The compact fixture uses the actual 634752-byte shared block stride and a nonzero layer7 storage offset, with eager/reference checks and changed-query graph replay. Full-model validation must additionally cover cache writes, indexer selection, SWA, prefill/decode, prefix reuse, MTP, and near/exact524K boundaries. No published result or default is changed by this proposal.
