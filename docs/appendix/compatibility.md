# Compatibility

This page summarizes practical compatibility constraints for typical installs.

## Python

The installer script creates a Python 3.12 virtual environment and the published wheels target CPython 3.12.

## CUDA

The installer selects a CUDA-specific wheel build. It currently maps to:

- `cu128` for CUDA 12.8 and 12.9
- `cu130` for CUDA 13+

A CUDA 12.9 host installs the `cu128` wheel. There is no `cu129` build: a binary compiled
against 12.8 runs on any 12.x runtime, and nothing in the wheel links torch, so no ABI ties a
wheel to one toolkit. On 12.9 the installer still fetches torch and the CUDA libraries built
for 12.9 — only the surogate wheel is shared.

If CUDA cannot be detected, installation fails.

## OS / platform

Wheels are built for Linux x86_64.

---

## Back

- [Back to docs index](../index.mdx)
