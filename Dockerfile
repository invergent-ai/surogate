# Surogate Docker Image
# High-performance LLM pre-training/fine-tuning framework
#
# Build Args:
#   CUDA_RUNTIME_IMAGE: CUDA runtime image (e.g., nvidia/cuda:12.9.1-runtime-ubuntu24.04)
#   CUDA_MAJOR: CUDA major version for package dependencies (e.g., 12, 13)
#   PACKAGE_VERSION: Surogate package version (e.g., 0.1.1)
#   WHEEL_TAG: Wheel CUDA tag (e.g., cu129, cu13)
#
# Build:
#   docker build \
#     --build-arg CUDA_MAJOR=13 \
#     --build-arg PACKAGE_VERSION=0.1.1 \
#     --build-arg WHEEL_TAG=cu13 \
#     -t ghcr.io/invergent-ai/surogate:0.1.1-cu13 .
#
# Run:
#   docker run --gpus all ghcr.io/invergent-ai/surogate:0.1.1-cu13 --help
#   docker run --gpus all \
#     -v /path/to/config.yaml:/config.yaml \
#     -v /path/to/output:/output \
#     ghcr.io/invergent-ai/surogate:0.1.1-cu13 sft /config.yaml

FROM ubuntu:noble-20260210.1
LABEL org.opencontainers.image.source=https://github.com/invergent-ai/surogate
ARG CUDA_MAJOR=13
ARG PACKAGE_VERSION=0.1.1
ARG WHEEL_TAG=cu13
ARG WHEEL_URL=https://github.com/invergent-ai/surogate/releases/download/v${PACKAGE_VERSION}/surogate-${PACKAGE_VERSION}+${WHEEL_TAG}-cp312-abi3-manylinux_2_39_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_LIB_ROOT="/home/surogate/.venv/lib/python3.12/site-packages/nvidia"

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN echo "${CUDA_LIB_ROOT}/cu13/lib" >> /etc/ld.so.conf.d/cuda-13-1.conf \
    && echo "${CUDA_LIB_ROOT}/cudnn/lib" >> /etc/ld.so.conf.d/cuda-13-1.conf \
    && echo "${CUDA_LIB_ROOT}/nccl/lib" >> /etc/ld.so.conf.d/cuda-13-1.conf \
    && echo "${CUDA_LIB_ROOT}/cuda_runtime/lib" >> /etc/ld.so.conf.d/cuda-12.conf \
    && echo "${CUDA_LIB_ROOT}/cuda_nvrtc/lib" >> /etc/ld.so.conf.d/cuda-12.conf \
    && echo "${CUDA_LIB_ROOT}/cublas/lib" >> /etc/ld.so.conf.d/cuda-12.conf \
    && echo "${CUDA_LIB_ROOT}/cufile/lib" >> /etc/ld.so.conf.d/cuda-12.conf \
    && ldconfig

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv* /usr/local/bin

# Create non-root user
RUN useradd -m -s /bin/bash surogate
USER surogate
WORKDIR /home/surogate

# Create virtual environment
RUN uv venv /home/surogate/.venv --python=3.12

ENV PATH="/home/surogate/.venv/bin:$PATH" \
    VIRTUAL_ENV="/home/surogate/.venv"

# Install wheel from URL
RUN uv pip install ${WHEEL_URL}

# Default entrypoint
ENTRYPOINT [".venv/bin/surogate"]
CMD ["--help"]
