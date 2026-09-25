# Packed candidate first attestation failure

Startup was rejected by the ownership attestation: replicated `indexer.wq_b` retained global TP2 metadata despite full weights and local execution. Fix `362edab` validates actual ReplicatedLinear full output partitions and loaded weight element count, retaining strict TP1 checks for sharded linear classes. This is a helper-only correction.

Image `6915af02` used source `fc7551a`. Failed startup, ownership JSON, runtime/container snapshots and build provenance are losslessly archived with SHA-256 hashes. No model performance or successful qualification is claimed.
