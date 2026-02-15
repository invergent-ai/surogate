#!/bin/bash
# install.sh - Auto-detect CUDA and install appropriate surogate package

set -e

# Ensure HOME is set (cloud-init and minimal environments may not have it)
if [ -z "$HOME" ]; then
    export HOME=$(getent passwd "$(id -u)" | cut -d: -f6)
    # Fallback to /root if running as root
    [ -z "$HOME" ] && [ "$(id -u)" -eq 0 ] && export HOME="/root"
fi

REPO="invergent-ai/surogate"
VENV_DIR=".venv"

# Check for required tools
if ! command -v curl &> /dev/null; then
    echo "Error: curl is required but not installed."
    exit 1
fi

# Install uv if not available
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the uv environment to make it available in this session
    # Try both $HOME and common root locations
    export PATH="$HOME/.local/bin:/root/.local/bin:$PATH"
    # Also source the env file if it exists
    if [ -f "$HOME/.local/bin/env" ]; then
        source "$HOME/.local/bin/env"
    elif [ -f "/root/.local/bin/env" ]; then
        source "/root/.local/bin/env"
    fi

    if ! command -v uv &> /dev/null; then
        echo "Error: Failed to install uv. Please install it manually from https://github.com/astral-sh/uv"
        exit 1
    fi
    echo "uv installed successfully."
else
    echo "uv is already installed."
fi

# Create virtual environment with Python 3.12 if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with Python 3.12..."
    echo "Using uv at: $(which uv)"
    uv venv --python 3.12 "$VENV_DIR"
    if [ ! -d "$VENV_DIR" ]; then
        echo "Error: Failed to create virtual environment at $VENV_DIR"
        exit 1
    fi
else
    echo "Using existing virtual environment: $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VENV_DIR"

# Check if surogate is already installed and get current version
INSTALLED_VERSION=""
if python -c "import surogate" 2>/dev/null; then
    INSTALLED_VERSION=$(python -c "import surogate; print(surogate.__version__)" 2>/dev/null || echo "")
    # Strip any CUDA suffix from version for comparison (e.g., "0.0.1+cu129" -> "0.0.1")
    INSTALLED_VERSION_BASE="${INSTALLED_VERSION%%+*}"
    echo "Currently installed surogate version: $INSTALLED_VERSION"
fi

# --- CUDA version-specific install functions ---
# Each function installs the appropriate packages for its CUDA version.
# Uses CPU torch wheels to avoid pulling in nvidia-* pip packages;
# CUDA kernels in surogate link against the system CUDA toolkit.

install_cu128() {
    local version="$1"
    echo "Installing packages for CUDA 12.8..."
    uv pip install "nvidia-cudnn-cu12==9.19.0.56"
    install_surogate_wheel "$version" "cu128"
}

install_cu129() {
    local version="$1"
    echo "Installing packages for CUDA 12.9..."
    uv pip install "nvidia-cudnn-cu12==9.19.0.56"
    install_surogate_wheel "$version" "cu129"
}

install_cu13() {
    local version="$1"
    echo "Installing packages for CUDA 13+..."
    uv pip install "nvidia-cudnn-cu13==9.19.0.56"
    install_surogate_wheel "$version" "cu129"
}

# --- Helper: download and install the surogate wheel ---

install_surogate_wheel() {
    local version="$1"
    local cuda_suffix="$2"

    local wheel_name="surogate-${version}+${cuda_suffix}-cp312-abi3-manylinux_2_39_x86_64.whl"
    local wheel_pattern="surogate-${version}%2B${cuda_suffix}-cp312-abi3-manylinux_2_39_x86_64.whl"

    local download_url
    download_url=$(echo "$RELEASE_JSON" | grep -oP '"browser_download_url":\s*"\K[^"]+' | grep "$wheel_pattern" || true)

    if [ -z "$download_url" ]; then
        echo "Error: Could not find wheel for CUDA $cuda_suffix (looking for $wheel_name)"
        echo "Available wheels:"
        echo "$RELEASE_JSON" | grep -oP '"browser_download_url":\s*"\K[^"]+' | grep '\.whl$' || echo "  (none found)"
        exit 1
    fi

    echo "Downloading: $wheel_name"
    echo "URL: $download_url"

    local temp_dir
    temp_dir=$(mktemp -d)
    local wheel_path="${temp_dir}/${wheel_name}"

    curl -L -o "$wheel_path" "$download_url"

    if [ ! -f "$wheel_path" ]; then
        echo "Error: Failed to download wheel."
        rm -rf "$temp_dir"
        exit 1
    fi

    if [ -n "$INSTALLED_VERSION" ]; then
        echo "Upgrading surogate from $INSTALLED_VERSION to $version..."
        uv pip install --reinstall "$wheel_path"
    else
        echo "Installing surogate..."
        uv pip install "$wheel_path"
    fi

    rm -rf "$temp_dir"
}

# --- Detect CUDA version ---

CUDA_VERSION=""
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
elif [ -f /usr/local/cuda/version.txt ]; then
    CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep -oP '\d+\.\d+')
elif command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
fi

if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA version. Please ensure CUDA is installed."
    exit 1
fi

CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)

echo "Detected CUDA version: $CUDA_VERSION"

# --- Fetch latest release ---

echo "Fetching latest release from GitHub..."
RELEASE_JSON=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest")

if [ -z "$RELEASE_JSON" ] || echo "$RELEASE_JSON" | grep -q '"message": "Not Found"'; then
    echo "Error: Could not fetch release information from GitHub."
    exit 1
fi

TAG_NAME=$(echo "$RELEASE_JSON" | grep -oP '"tag_name":\s*"\K[^"]+')
VERSION="${TAG_NAME#v}"
echo "Latest version: $VERSION"

# --- Dispatch to the right install function ---

if [[ "$CUDA_MAJOR" -ge 13 ]]; then
    install_cu13 "$VERSION"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 9 ]]; then
    install_cu129 "$VERSION"
elif [[ "$CUDA_MAJOR" -eq 12 && "$CUDA_MINOR" -ge 8 ]]; then
    install_cu128 "$VERSION"
else
    echo "Error: CUDA $CUDA_VERSION is not compatible with Surogate. Aborting."
    exit 1
fi

echo ""
if [ -n "$INSTALLED_VERSION" ]; then
    echo "Successfully upgraded surogate from $INSTALLED_VERSION to $VERSION"
else
    echo "Successfully installed surogate $VERSION"
fi

# Clone examples
EXAMPLES_DIR="examples"
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo ""
    echo "Downloading examples..."
    mkdir -p "$EXAMPLES_DIR"

    # Download SFT examples using GitHub API to get directory contents
    EXAMPLES_API_URL="https://api.github.com/repos/${REPO}/contents/examples/sft?ref=main"
    EXAMPLES_JSON=$(curl -sL "$EXAMPLES_API_URL")

    if echo "$EXAMPLES_JSON" | grep -q '"download_url"'; then
        mkdir -p "$EXAMPLES_DIR/sft"
        # Extract file URLs and download each file
        echo "$EXAMPLES_JSON" | grep -oP '"download_url":\s*"\K[^"]+' | while read -r file_url; do
            filename=$(basename "$file_url")
            echo "  Downloading $filename..."
            curl -sL "$file_url" -o "$EXAMPLES_DIR/sft/$filename"
        done
        echo "Examples downloaded to $EXAMPLES_DIR/sft/"
    else
        echo "Warning: Could not download examples from GitHub."
    fi
else
    echo "Examples directory already exists: $EXAMPLES_DIR"
fi

echo ""
echo "To run your first Qwen3-0.6B fine-tune run:"
echo "  source $VENV_DIR/bin/activate"
echo "  surogate sft examples/sft/qwen3-lora-bf16.yaml"
echo ""
