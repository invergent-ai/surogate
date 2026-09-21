# Compatibility

This page summarizes practical compatibility constraints for typical installs.

## Python

The installer script creates a Python 3.12 virtual environment and the published wheels target CPython 3.12.

## CUDA

CUDA 13 only, with an NVIDIA driver from the 580 series or newer. There is one wheel,
`cu130`, and it both trains and serves.

There is no CUDA 12 build. The serving engine's W8 and NVFP4 GEMM kernels declare between 49
and 97 KB of static shared memory per block; the 12.8 and 12.9 toolkits cap a block at 48 KB
and refuse to assemble them for the RTX line, while 13.0 and 13.1 accept them. Since a
package that serves is the product, CUDA 13.0 is the floor for the whole package rather than
for the engine alone.

The wheel installs with a plain `pip install <wheel-url>`: it asks for `torch==2.11.0`, which
PyPI serves as a CUDA 13 build, plus the CUDA runtime libraries our own binaries link
(`nvidia-cuda-runtime`, `nvidia-cublas`, `nvidia-cufile`).

If CUDA cannot be detected, installation fails.

## OS / platform

Wheels are built for Linux x86_64.

---

## Back

- [Back to docs index](../index.mdx)
