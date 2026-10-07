# Installation

## Supported platforms

- Linux x86_64
- Linux aarch64 on a DGX Spark (GB10)
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

## DGX Spark (GB10)

The install script detects the Arm host and installs the aarch64 wheel, which is built for
GB10 (SM121) only. Training and serving use the same commands as on any other GPU.

To build from source on the Spark, install Rust with `rustup` and the build prerequisites
above. The Makefile finds the CUDA toolkit in `/usr/local/cuda`, where DGX OS keeps it off
`PATH`. `make build` detects the GPU and builds for `121a`, and `make serve-build` defaults to
`121a` on aarch64.

The GPU shares the machine's memory with the system. Surogate counts as free what the system
can still give it (`MemAvailable`, which includes reclaimable page cache) less 8 GiB left to
the host. Set `SUROGATE_UNIFIED_MEMORY_RESERVE_MIB` to change that reserve.

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
