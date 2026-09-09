# Compatibility

This page summarizes practical compatibility constraints for typical installs.

## Python

The installer script creates a Python 3.12 virtual environment and the published wheels target CPython 3.12.

## CUDA

The installer selects a CUDA-specific wheel build. It currently maps to:

- `cu128` for CUDA 12.8+
- `cu130` for CUDA 13+

Two builds cover every supported runtime. A binary compiled against 12.8 runs on any 12.x
under CUDA minor version compatibility, and nothing in the wheel links torch, so no ABI ties
a wheel to the toolkit that built it.

`cu130` installs with a plain `pip install <wheel-url>`: it asks for `torch==2.11.0`, which
PyPI serves as a CUDA 13 build, plus the CUDA runtime libraries our own binaries link
(`nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cufile`). `cu128` still needs the installer
or `--index-url https://download.pytorch.org/whl/cu128`, because the only Linux torch on
PyPI is the CUDA 13 one.

If CUDA cannot be detected, installation fails.

## OS / platform

Wheels are built for Linux x86_64.

---

## Back

- [Back to docs index](../index.mdx)
