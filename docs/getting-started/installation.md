# Installation

## Supported platforms

- Linux x86_64
- NVIDIA GPU

## GPU / CUDA

Surogate requires a recent NVIDIA driver and CUDA libraries.

Supported CUDA versions:
- CUDA 12.8 or newer in the 12.x line
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

- CUDA toolkit (12.8+ or 13.x)
- NCCL development libraries
- Rust 1.87 or newer and Cargo (not needed when installing a wheel)

From the repository root:

```bash
uv pip install -e .
```

## aarch64 (NVIDIA DGX Spark / GB10)

The published wheel and container are x86_64 only. On an aarch64 host with CUDA 13 the
source build above works, with these settings (the GB10 is compute capability 12.1):

```bash
export CUDAARCHS=121a
export SKBUILD_CMAKE_DEFINE="SUROGATE_SERVE_CUDA_ARCHS=121a"
uv pip install -e . --no-build-isolation
```

`SUROGATE_SERVE_CUDA_ARCHS` must name the device's architecture-specific target
(`120a` for RTX 50 / RTX PRO, `121a` for GB10): the NVFP4 kernel families are built for
exactly that target and refuse other devices at runtime. The CPU expert-compute and
CPU embedding paths have no NEON port, so they run their reference code on aarch64.
`SUROGATE_BUILD_SPEECH=OFF` skips the STT server on a host without FFmpeg development
libraries; `SINFER_ENABLE_FFMPEG=OFF` does the same for media decoding in the engine.
`SKBUILD_CMAKE_DEFINE` is one semicolon-separated list, so add them to the line above rather
than exporting it a second time, which would drop the architecture:

```bash
export SKBUILD_CMAKE_DEFINE="SUROGATE_SERVE_CUDA_ARCHS=121a;SUROGATE_BUILD_SPEECH=OFF;SINFER_ENABLE_FFMPEG=OFF"
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
