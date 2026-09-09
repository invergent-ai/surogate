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

If CUDA cannot be detected, installation fails.

## OS / platform

Wheels are built for Linux x86_64.

---

## Back

- [Back to docs index](../index.mdx)
