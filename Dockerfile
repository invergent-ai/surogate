# Surogate Docker Image
# High-performance LLM pre-training/fine-tuning framework
#
# Build:
#   docker build -t invergent-ai/surogate:0.0.2 .
#
# Run:
#   docker run --gpus all ghcr.io/invergent-ai/surogate:0.0.2 --help
#   docker run --gpus all -v /path/to/config.yaml:/config.yaml -v path_to_output:/output ghcr.io/invergent-ai/surogate:0.0.2 sft --config /config.yaml

ARG BUILD_IMAGE=nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04
ARG RUNTIME_IMAGE=nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04
ARG CMAKE_JOBS=8

FROM ${BUILD_IMAGE} AS builder

# Build arguments
ARG CUDAARCHS="80;86;89;90;100a;103a;120a"

# Environment variables for build
ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_GENERATOR=Ninja \
    CMAKE_C_COMPILER=gcc-13 \
    CMAKE_CXX_COMPILER=g++-13 \
    CUDAARCHS=${CUDAARCHS}

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    g++-13 \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    ccache \
    libopenmpi-dev \
    ca-certificates

RUN apt-get remove -y --allow-change-held-packages libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 libcudnn9-cuda-13 \
    && apt-get install -y cudnn9-cuda-13=9.17.1.4-1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for Python package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv /root/.local/bin/uv* /usr/local/bin

WORKDIR /build

# Copy source code
COPY . .

# Build the wheel
RUN uv venv --python=3.12
RUN CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_JOBS} NVCC_THREADS=${CMAKE_JOBS} uv build --wheel

# ----- Runtime stage -----
FROM ${RUNTIME_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    libopenmpi3t64 \
    ca-certificates \
    wget \
    curl

RUN apt-get remove -y --allow-change-held-packages libcudnn9-dev-cuda-13 libcudnn9-headers-cuda-13 libcudnn9-cuda-13 \
    && apt-get install -y cudnn9-cuda-13=9.17.1.4-1 \
    && rm -rf /var/lib/apt/lists/*

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

# Copy wheel from builder stage
COPY --from=builder --chown=surogate:surogate /build/dist/*.whl /tmp/

# Install surogate with CUDA 13.0 dependencies
RUN uv pip install /tmp/*.whl \
    && rm -f /tmp/*.whl

# Set default working directory for user data
WORKDIR /workspace

# Default entrypoint
ENTRYPOINT ["surogate"]
CMD ["--help"]
