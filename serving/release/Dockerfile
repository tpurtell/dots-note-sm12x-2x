# syntax=docker/dockerfile:1.7
# Build natively on the qualified platform. The qualified local parent image ID
# must equal manifest.source_image_id (checked by build.py before Docker build).
ARG RUNTIME_IMAGE
FROM ${RUNTIME_IMAGE}
ARG RECIPE_REVISION
ARG PLATFORM
ARG REASONING_PARSER=dots3
ARG REASONING_PARSER_SHA256
COPY cache-seed /opt/dots3/cache-seed
COPY cache_bundle.py entrypoint.sh verify_reasoning.py /opt/dots3/
RUN if [ "$REASONING_PARSER" = dots3 ]; then \
      python3 /opt/dots3/verify_reasoning.py --sha256 "$REASONING_PARSER_SHA256"; \
    elif [ "$REASONING_PARSER" != none ]; then exit 1; fi
ENV B12X_ROCE_CACHE_DIR=/root/.cache/vllm-runtime/b12x/roce \
    B12X_COMPILE_CACHE_DIR=/root/.cache/vllm-runtime/b12x/compile \
    TRITON_CACHE_DIR=/root/.cache/vllm-runtime/triton \
    VLLM_CACHE_ROOT=/root/.cache/vllm-runtime/vllm \
    CUDA_CACHE_PATH=/root/.cache/vllm-runtime/cuda
LABEL org.opencontainers.image.source="https://github.com/tpurtell/dots-note-sm12x-2x" \
      org.opencontainers.image.revision="${RECIPE_REVISION}" \
      io.tpurtell.dots3.platform="${PLATFORM}" \
      io.tpurtell.dots3.reasoning-parser="${REASONING_PARSER}"
ENTRYPOINT ["bash", "/opt/dots3/entrypoint.sh"]
