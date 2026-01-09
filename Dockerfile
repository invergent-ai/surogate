# Surogate Docker Image
# High-performance LLM pre-training/fine-tuning framework
#
# Build:
#   docker build -t invergent-ai/surogate:0.0.2 .
#
# Run:
#   docker run --gpus all ghcr.io/invergent-ai/surogate:0.0.2 --help
#   docker run --gpus all -v /path/to/config.yaml:/config.yaml -v path_to_output:/output ghcr.io/invergent-ai/surogate:0.0.2 sft /config.yaml


FROM nvidia/cuda:13.1.0-runtime-ubuntu24.04

ARG WHL=https://github.com/invergent-ai/surogate/releases/download/v0.0.3/surogate-0.0.3+cu129-cp312-abi3-manylinux_2_39_x86_64.whl

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

# Install surogate with CUDA 13 dependencies
RUN --mount=type=cache,target=/root/.cache/uv uv pip install ${WHL} \
    && rm -f ${WHL}

# Set default working directory for user data
WORKDIR /workspace

# Default entrypoint
ENTRYPOINT ["surogate"]
CMD ["--help"]
