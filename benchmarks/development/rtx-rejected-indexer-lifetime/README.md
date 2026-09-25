# Rejected indexer profiling lifetime experiment

The same cut18 workload started healthily with the experimental dummy-logits lifetime change, but allocator results worsened. No performance traffic was sent and the candidate was stopped.

- Rank0 inactive splits increased1.547→5.047GiB; available KV decreased6.654→2.654GiB.
- Rank1 inactive splits stayed3.025GiB; peak activation increased512MiB, reducing KV9.076→8.576GiB.

`VLLM_INDEXER_PROFILE_HOLD_LOGITS=1` is rejected and not adopted. No additional GPU benchmarks are warranted by these results. The experimental port is retained only for reproducibility and is not integrated into native Dockerfiles or launchers. Exact bytes, baseline image, candidate source/build and lossless raw hashes are archived.
