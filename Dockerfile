# Surogate Docker Image
# High-performance LLM pre-training/fine-tuning framework
#
# Build Args:
#   PACKAGE_VERSION: Surogate package version (e.g., 0.1.1)
#   WHEEL_TAG: Wheel CUDA tag. One of cu128, cu129, cu130
#
# Build:
#   docker build \
#     --build-arg PACKAGE_VERSION=0.1.1 \
#     --build-arg CUDA_TAG=cu130 \
#     -t ghcr.io/invergent-ai/surogate:0.1.1-cu130 .
#
# Run:
#   docker run --gpus all ghcr.io/invergent-ai/surogate:0.1.1-cu130 --help
#   docker run --gpus all \
#     -v /path/to/config.yaml:/config.yaml \
#     -v /path/to/output:/output \
#     ghcr.io/invergent-ai/surogate:0.1.1-cu130 sft /config.yaml

FROM ubuntu:noble-20260210.1
LABEL org.opencontainers.image.source=https://github.com/invergent-ai/surogate
ARG PACKAGE_VERSION=0.1.1
ARG CUDA_TAG=cu129
ARG WHEEL_URL=https://github.com/invergent-ai/surogate/releases/download/v${PACKAGE_VERSION}/surogate-${PACKAGE_VERSION}+${CUDA_TAG}-cp312-abi3-manylinux_2_39_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_LIB_ROOT="/root/.venv/lib/python3.12/site-packages/nvidia"

RUN apt-get update \
    && apt-get install -y --no-install-recommends --allow-change-held-packages ca-certificates curl libdw-dev python3-dev build-essential rsync wget netcat-openbsd gcc patch pciutils fuse3 openssh-server \
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
RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh


WORKDIR /root

# Create virtual environment for Surogate
RUN ~/.local/bin/uv venv /root/.venv --python=3.12

# Create virutal environment for SkyPilot
RUN ~/.local/bin/uv venv --seed ~/sky_runtime/skypilot-runtime --python 3.10

ENV PATH="/root/.venv/bin:$PATH" \
    VIRTUAL_ENV="/root/.venv"

# Install wheel from URL
RUN ~/.local/bin/uv pip install vllm==0.18.0 --index-strategy unsafe-best-match --torch-backend=${CUDA_TAG}}
RUN ~/.local/bin/uv pip install ${CUDA_TAG}

RUN VIRTUAL_ENV=~/sky_runtime/skypilot-runtime UV_LINK_MODE=copy UV_SYSTEM_PYTHON=false \
    ~/.local/bin/uv pip install "setuptools<70" \
    "ray[default]==2.9.3" "skypilot[kubernetes,remote]"

# Default entrypoint
ENTRYPOINT [".venv/bin/surogate"]
CMD ["--help"]
