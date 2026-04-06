FROM ghcr.io/ggml-org/llama.cpp:full-cuda

RUN apt-get update && apt-get install -y curl unzip \
    fuse openssh-server \
    && apt install -y fuse3 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$HOME/.local/bin" sh

# Install rclone
RUN curl https://rclone.org/install.sh | bash
