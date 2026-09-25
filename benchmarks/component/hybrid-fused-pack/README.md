# Fused routing copy component

Both RTX GPUs passed 13 cases each. CUDA-graph replay copies BF16 activations, int32 expert IDs and FP32 weights into the prepared packed payload with one Triton launch. For 1–16 rows, median component time was about 2.1 µs versus 4.1 µs for three tensor copies; at 512 contiguous rows both were about 6.15 µs. These measurements exclude routing, communication and model execution. See the manifest for source and raw-log hashes.
