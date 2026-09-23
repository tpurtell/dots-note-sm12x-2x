# Two-Spark vLLM integration (in progress)

`Dockerfile.spark` pins the multi-architecture vLLM v0.30.0 image, vendors the
Apache-2.0 EXL3 adapter source, installs the Dots3 FP8-core/EXL3-expert
configuration, and includes the pinned B12x submodule. It builds on ARM64 and
imports the native Dots3 model and hybrid quantization config on moa.

```bash
docker build -f serving/Dockerfile.spark -t dots3-vllm-spark:dev .
```

This is an integration image, not a qualified serving release. The generic
EXL3 routed-expert path in the vendored adapter uses the ExLlamaV3 parity
kernel. The uniform K4 B12x fused-MoE path, block-FP8 core parity, padded DSA
attention, prefix-cache hits, xgrammar requests, and the 85% memory target
must be tested with the completed checkpoint before starting the service.
The vLLM base currently ships Torch 2.13 and CuTe DSL 4.7.1 while the pinned
B12x package declares CuTe DSL 4.6.2; import succeeds, kernel parity is pending.
