# Installation

## Supported platforms

- Linux x86_64
- NVIDIA GPU

## GPU / CUDA

Surogate requires a recent NVIDIA driver and CUDA libraries.

Commonly used CUDA versions:
- CUDA 12.8
- CUDA 12.9
- CUDA 13.x

Multi-GPU training requires NCCL.

## Python

Python 3.12 is the default for the published wheels and the install script

## Option A: Install via script (recommended)

```bash
curl -LsSf https://surogate.ai/install.sh | sh
```

What the script does (high level):
- Installs `uv` if missing
- Creates a local `.venv` using Python 3.12
- Detects your CUDA version and installs a matching Surogate wheel
- Downloads example configs into `./examples/` (if not already present)

Activate the environment:

```bash
source .venv/bin/activate
```

## Option B: Build from source (developers)

Prerequisites:
- CUDA toolkit (12.8/12.9/13.x)
- NCCL development libraries

From the repository root:

```bash
uv pip install -e .
```

## Verify installation

After install, these should work:

```bash
surogate --help
surogate sft --help
surogate pt --help
```

## See also

- [Quickstart: SFT](quickstart-sft.md)
- [Back to docs index](../index.mdx)
