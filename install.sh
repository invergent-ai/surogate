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

# Pinned by the release workflow. When set, install.sh uses this version and
# constructs download URLs directly from the release tag, skipping the GitHub
# API lookup. Leave empty in-tree so install.sh from main still resolves the
# latest release dynamically.
VERSION_OVERRIDE=""

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
    uv pip install pip
else
    echo "Using existing virtual environment: $VENV_DIR"
fi

# Activate the virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VENV_DIR"

# Check if surogate is already installed and get current version. The package metadata is the
# authority: `surogate.__version__` does not exist (the package's __init__ is empty), so asking
# for it reported every install as a fresh one.
INSTALLED_VERSION=$(python -c "from importlib.metadata import version; print(version('surogate'))" 2>/dev/null || true)
if [ -n "$INSTALLED_VERSION" ]; then
    echo "Currently installed surogate version: $INSTALLED_VERSION"
fi

# --- Install the CUDA 13 packages this wheel needs ---
# One wheel, because the serving engine needs CUDA 13.0 or newer: the 12.x toolkits cap a
# block's static shared memory at 48 KB and cannot assemble the engine's W8 and NVFP4 GEMM
# kernels for the RTX line. torch comes straight from PyPI, whose Linux build is CUDA 13.

install_cu130_deps() {
    local version="$1"
    echo "Installing packages for CUDA 13..."
    pip install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
    install_surogate_wheel "$version" "cu130"
    pip install "nvidia-cuda-runtime==13.1.80" "nvidia-cudnn-cu13>=9.10.2.21" "nvidia-nccl-cu13==2.29.3" "nvidia-cufile==1.16.1.26" "nvidia-cuda-nvrtc==13.1.115"
}

# --- Helper: download and install the surogate wheel ---

install_surogate_wheel() {
    local version="$1"
    local cuda_suffix="$2"

    local wheel_name="surogate-${version}+${cuda_suffix}-cp312-abi3-manylinux_2_39_x86_64.whl"
    local wheel_pattern="surogate-${version}%2B${cuda_suffix}-cp312-abi3-manylinux_2_39_x86_64.whl"

    local download_url
    if [ -n "$VERSION_OVERRIDE" ]; then
        download_url="https://github.com/${REPO}/releases/download/v${version}/surogate-${version}%2B${cuda_suffix}-cp312-abi3-manylinux_2_39_x86_64.whl"
    else
        download_url=$(echo "$RELEASE_JSON" | grep -oP '"browser_download_url":\s*"\K[^"]+' | grep "$wheel_pattern" || true)
    fi

    if [ -z "$download_url" ]; then
        echo "Error: Could not find wheel for CUDA $cuda_suffix (looking for $wheel_name)"
        if [ -z "$VERSION_OVERRIDE" ]; then
            echo "Available wheels:"
            echo "$RELEASE_JSON" | grep -oP '"browser_download_url":\s*"\K[^"]+' | grep '\.whl$' || echo "  (none found)"
        fi
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
        # Reinstall the package, not the environment: a plain --reinstall re-resolves every
        # dependency, which a wheel upgrade has no reason to disturb.
        uv pip install --reinstall-package surogate "$wheel_path"
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

echo "Detected CUDA version: $CUDA_VERSION"

# Fail here rather than after a download: the serving engine cannot be built by a CUDA 12
# toolkit (it caps a block's static shared memory at 48 KB and rejects the engine's W8 and
# NVFP4 kernels for the RTX line), so there is no CUDA 12 package to fall back to.
if [[ "$CUDA_MAJOR" -lt 13 ]]; then
    echo "Error: surogate needs CUDA 13.0 or newer; this host reports CUDA $CUDA_VERSION."
    echo "Update the NVIDIA driver to a 580 series or newer release and run this installer again."
    exit 1
fi

# --- Resolve target version ---

if [ -n "$VERSION_OVERRIDE" ]; then
    VERSION="$VERSION_OVERRIDE"
    echo "Using pinned version: $VERSION"
else
    echo "Fetching latest release from GitHub..."
    RELEASE_JSON=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest")

    if [ -z "$RELEASE_JSON" ] || echo "$RELEASE_JSON" | grep -q '"message": "Not Found"'; then
        echo "Error: Could not fetch release information from GitHub."
        exit 1
    fi

    TAG_NAME=$(echo "$RELEASE_JSON" | grep -oP '"tag_name":\s*"\K[^"]+')
    VERSION="${TAG_NAME#v}"
    echo "Latest version: $VERSION"
fi

# --- Install ---

install_cu130_deps "$VERSION"

# --- Verify the install before calling it one ---
# Two questions the download cannot answer: does the package import with its extension, and
# can the serving engine start on this host. `ldd` asks the second without a GPU; the driver
# (libcuda.so.1) is the one permitted miss, since a headless install host may not have it.
verify_install() {
    local report
    report=$(python - <<'VERIFY'
import subprocess, sys
try:
    import surogate._surogate  # noqa: F401  (the training extension)
except ImportError as error:
    # The extension links the driver; a host with the toolkit but no driver (an image build,
    # a login node) cannot load it, and that is not a broken install.
    if "libcuda.so.1" not in str(error):
        sys.exit(f"the training extension does not import: {error}")
    print("no CUDA driver on this host; skipped the import check", file=sys.stderr)
from surogate.cli.serve import _resolve_binary
DRIVER = "libcuda.so.1"
problems = []
for mode in ("server", "generate", "embed"):
    path = _resolve_binary(mode)
    if path is None:
        problems.append(f"{mode}: no engine binary")
        continue
    ldd = subprocess.run(["ldd", path], capture_output=True, text=True).stdout
    missing = [ln.split("=>")[0].strip() for ln in ldd.splitlines() if "not found" in ln and DRIVER not in ln]
    if missing:
        problems.append(f"{mode}: {path} cannot load " + ", ".join(missing))
if problems:
    print("\n".join(problems))
VERIFY
    ) || { echo "Error: $report"; exit 1; }
    if [ -n "$report" ]; then
        echo "Error: the serving engine is not whole on this host:"
        echo "$report" | sed 's/^/  /'
        echo "The wheel carries its own FFmpeg, numa and ICU; it relies on the system for glib and the"
        echo "X11 client libraries, which a minimal server image may lack. On Ubuntu/Debian:"
        echo "  sudo apt-get install -y libglib2.0-0t64 libx11-6 libxext6 libxrender1"
        echo "then re-run this installer."
        exit 1
    else
        echo "Serving engine verified: surogate serve can start on this host."
    fi
}
verify_install

echo ""
if [ -n "$INSTALLED_VERSION" ]; then
    echo "Successfully upgraded surogate from $INSTALLED_VERSION to $VERSION"
else
    echo "Successfully installed surogate $VERSION"
fi

# --- Install the jackalope dashboard binary (best effort) ---
# jackalope is a standalone, Node-free TUI shipped on its own release channel
# (independent of the CUDA wheel). If the fetch fails, `surogate jackalope` will
# download it on first run, so this step is non-fatal.
install_jackalope() {
    local arch os asset
    case "$(uname -m)" in
        x86_64 | amd64) arch="x64" ;;
        aarch64 | arm64) arch="arm64" ;;
        *) echo "jackalope: no binary for $(uname -m), skipping"; return 0 ;;
    esac
    case "$(uname -s)" in
        Linux) os="linux" ;;
        Darwin) os="darwin" ;;
        *) echo "jackalope: unsupported OS $(uname -s), skipping"; return 0 ;;
    esac
    asset="jackalope-${os}-${arch}"
    # Honor the same pin the runtime uses (SUROGATE_JACKALOPE_VERSION), so a
    # version pinned at install time matches `surogate jackalope`'s.
    local tag="${SUROGATE_JACKALOPE_VERSION:-${JACKALOPE_VERSION:-jackalope-latest}}"
    local url="https://github.com/${REPO}/releases/download/${tag}/${asset}"
    local dest="${VENV_DIR}/bin/jackalope"
    echo ""
    echo "Installing jackalope dashboard (${asset})..."
    if curl -fsSL "$url" -o "$dest"; then
        chmod +x "$dest"
        echo "  installed — run it with: surogate jackalope"
    else
        rm -f "$dest"
        echo "  not published yet — 'surogate jackalope' will fetch it on first run"
    fi
}
install_jackalope || true

# Download examples
EXAMPLES_DIR="examples"
if [ ! -d "$EXAMPLES_DIR" ]; then
    echo ""
    echo "Downloading examples..."

    # Extract just the examples/ directory from the repo tarball
    REPO_NAME=$(echo "$REPO" | cut -d/ -f2)
    if curl -sL "https://github.com/${REPO}/archive/refs/heads/main.tar.gz" \
        | tar -xz --strip-components=1 "${REPO_NAME}-main/examples"; then
        echo "Examples downloaded to $EXAMPLES_DIR/"
    else
        echo "Warning: Could not download examples from GitHub."
    fi
else
    echo "Examples directory already exists: $EXAMPLES_DIR"
fi

echo ""
echo "To run your first Qwen3-0.6B fine-tune run:"
echo "  source $VENV_DIR/bin/activate"
echo "  surogate sft examples/sft/qwen3/qwen3-lora-bf16.yaml"
echo ""
