# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# Compile Triton or CuTe DSL kernels to cubin + JSON manifest for loading by
# the C++ JitKernel loader (csrc/src/runtime/jit/jit_kernel.h).
#
# Two compiler functions:
#
#   compile_triton_kernel  — compiles a @triton.jit function (requires triton >= 3.0)
#   compile_cute_kernel    — compiles a CuTe DSL kernel (requires cutlass + quack)
#
# Usage (Triton):
#   from surogate.kernels.compiler import compile_triton_kernel
#
#   @triton.jit
#   def my_kernel(X, Out, N, BLOCK: tl.constexpr):
#       ...
#
#   manifest = compile_triton_kernel(
#       fn=my_kernel,
#       signature={"X": "*bf16", "Out": "*bf16", "N": "i32"},
#       constants={"BLOCK": 1024},
#       output_dir="compiled_kernels/",
#   )
#
# Usage (CuTe DSL):
#   from surogate.kernels.compiler import compile_cute_kernel
#
#   kernel = quack.rmsnorm.RMSNorm(cutlass.BFloat16, 768)
#   args = (x_fake, w_fake, None, None, out_fake, None, rstd_fake, None, eps, stream)
#   manifest = compile_cute_kernel(
#       kernel, args,
#       output_dir="compiled_kernels/",
#       base_name="cute_rmsnorm_bf16_C768",
#       shared_mem=6160,
#       params=[...],     # parameter layout for C++ launcher
#       library="quack",  # extra manifest metadata
#   )

from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _detect_sm() -> int:
    """Auto-detect GPU SM version via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        # Take the first GPU, parse "9.0" -> 90
        major, minor = out.split("\n")[0].strip().split(".")
        return int(major) * 10 + int(minor)
    except Exception:
        raise RuntimeError(
            "Cannot detect GPU SM version. Pass sm= explicitly or ensure "
            "nvidia-smi is available."
        )


def compile_triton_kernel(
    fn,
    signature: dict[str, str],
    constants: dict[str, Any] | None = None,
    output_dir: str | Path = ".",
    kernel_name: str | None = None,
    num_warps: int = 4,
    num_stages: int = 2,
    sm: int | None = None,
) -> str:
    """Compile a Triton kernel to cubin + JSON manifest.

    Args:
        fn: A ``@triton.jit``-decorated function.
        signature: Maps parameter names to type strings (constexpr params excluded).
            Example: ``{"X": "*bf16", "Out": "*bf16", "N": "i32"}``
        constants: Compile-time constants for ``tl.constexpr`` parameters.
            Example: ``{"BLOCK": 1024}``
        output_dir: Directory to write the cubin and manifest into.
        kernel_name: Override the kernel name in the manifest.
            Defaults to ``fn.__name__``.
        num_warps: Number of warps per CTA (block_x = num_warps * 32).
        num_stages: Software pipelining stages.
        sm: Target SM version (e.g., 90 for H100). Auto-detected if None.

    Returns:
        Path to the generated JSON manifest file.
    """
    from triton.compiler import ASTSource, compile as triton_compile
    from triton.backends.nvidia.compiler import CUDAOptions, GPUTarget

    if sm is None:
        sm = _detect_sm()
    if constants is None:
        constants = {}

    name = kernel_name or fn.__name__
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build source, target, and options for Triton 3.x
    src = ASTSource(fn=fn, signature=signature, constexprs=constants)
    target = GPUTarget("cuda", sm, 32)
    options = {"num_warps": num_warps, "num_stages": num_stages}

    compiled = triton_compile(src, target=target, options=options)

    # Extract cubin and metadata
    cubin = compiled.asm["cubin"]
    shared_mem = getattr(compiled, "shared", 0)
    global_scratch_size = 0
    profile_scratch_size = 0
    if hasattr(compiled, "metadata"):
        meta = compiled.metadata
        shared_mem = getattr(meta, "shared", shared_mem)
        global_scratch_size = getattr(meta, "global_scratch_size", 0)
        profile_scratch_size = getattr(meta, "profile_scratch_size", 0)

    # The actual function name in the cubin (Triton mangles it)
    cubin_fn_name = compiled.name if hasattr(compiled, "name") else name

    # Write cubin
    cubin_filename = f"{name}.cubin"
    cubin_path = output_dir / cubin_filename
    cubin_path.write_bytes(cubin)

    # Triton 3.x always appends 2 extra pointer params after user params:
    # global_scratch and profile_scratch. These must be passed as null pointers
    # when launching via the CUDA Driver API (cuLaunchKernel).
    extra_null_params = 2

    # Write JSON manifest (consumed by C++ JitKernel::load_manifest)
    manifest = {
        "name": cubin_fn_name,
        "num_warps": num_warps,
        "shared_mem": shared_mem,
        "cubin": cubin_filename,
        "sm": sm,
        "num_stages": num_stages,
        "signature": signature,
        "constants": {str(k): v for k, v in constants.items()},
        "extra_null_params": extra_null_params,
    }

    manifest_path = output_dir / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Compiled %s -> %s (%d bytes, sm%d, %d warps, %d shared)",
        name, cubin_path, len(cubin), sm, num_warps, shared_mem,
    )

    return str(manifest_path)


def autotune_triton_kernel(
    fn,
    signature: dict[str, str],
    constants: dict[str, Any] | None = None,
    configs: list[dict[str, Any]] | None = None,
    bench_args: tuple | None = None,
    grid: tuple[int, ...] | None = None,
    output_dir: str | Path = ".",
    kernel_name: str | None = None,
    sm: int | None = None,
    warmup: int = 5,
    rep: int = 25,
) -> str:
    """Autotune a Triton kernel and compile the winning config.

    Benchmarks multiple configurations (num_warps, num_stages, extra constants)
    at compile time, then compiles only the fastest config to cubin + manifest.

    Args:
        fn: A ``@triton.jit``-decorated function.
        signature: Maps parameter names to type strings (constexpr params excluded).
        constants: Base compile-time constants shared by all configs.
            Config-specific constants override these.
        configs: List of config dicts, each with optional keys:
            ``num_warps`` (int), ``num_stages`` (int), and any extra
            compile-time constants to merge with *constants*.
            Example: ``[{"num_warps": 4, "num_stages": 1},
                        {"num_warps": 8, "num_stages": 2}]``
            If None, generates a default sweep over num_warps=[1,2,4,8]
            and num_stages=[1,2].
        bench_args: Positional arguments (real tensors) for benchmarking.
            Required — these are passed to ``fn[grid](*bench_args, **constexprs)``.
        grid: Launch grid as a tuple (e.g. ``(1024,)``).  Required.
        output_dir: Directory to write the cubin and manifest into.
        kernel_name: Override the kernel name in the manifest.
        sm: Target SM version (auto-detected if None).
        warmup: Benchmark warmup iterations.
        rep: Benchmark repetitions.

    Returns:
        Path to the generated JSON manifest file.
    """
    import triton.testing

    if bench_args is None:
        raise ValueError("bench_args (real tensors for benchmarking) is required")
    if grid is None:
        raise ValueError("grid (launch grid tuple) is required")
    if constants is None:
        constants = {}

    if configs is None:
        configs = [
            {"num_warps": w, "num_stages": s}
            for w in [1, 2, 4, 8]
            for s in [1, 2]
        ]

    best_time = float("inf")
    best_config = configs[0]

    for config in configs:
        num_warps = config.get("num_warps", 4)
        num_stages = config.get("num_stages", 2)
        # Merge base constants with config-specific overrides
        cfg_constants = {
            k: v for k, v in config.items()
            if k not in ("num_warps", "num_stages")
        }
        merged = {**constants, **cfg_constants}

        def _launch(_m=merged, _nw=num_warps, _ns=num_stages):
            fn[grid](*bench_args, **_m, num_warps=_nw, num_stages=_ns)

        try:
            times = triton.testing.do_bench(
                _launch, warmup=warmup, rep=rep, quantiles=(0.5, 0.2, 0.8),
            )
            median = times[0]
        except Exception as e:
            logger.warning("Config %s failed: %s", config, e)
            continue

        logger.info(
            "  config num_warps=%d num_stages=%d %s -> %.3f ms",
            num_warps, num_stages,
            {k: v for k, v in cfg_constants.items()} if cfg_constants else "",
            median,
        )

        if median < best_time:
            best_time = median
            best_config = config

    best_warps = best_config.get("num_warps", 4)
    best_stages = best_config.get("num_stages", 2)
    best_extra = {
        k: v for k, v in best_config.items()
        if k not in ("num_warps", "num_stages")
    }
    final_constants = {**constants, **best_extra}

    logger.info(
        "Autotuning %s: best config num_warps=%d num_stages=%d (%.3f ms)",
        kernel_name or fn.__name__, best_warps, best_stages, best_time,
    )

    return compile_triton_kernel(
        fn=fn,
        signature=signature,
        constants=final_constants,
        output_dir=output_dir,
        kernel_name=kernel_name,
        num_warps=best_warps,
        num_stages=best_stages,
        sm=sm,
    )


def compile_cute_kernel(
    kernel,
    compile_args: tuple,
    output_dir: str | Path = ".",
    kernel_name: str | None = None,
    base_name: str | None = None,
    shared_mem: int = 0,
    params: list[dict] | None = None,
    **extra_metadata: Any,
) -> str:
    """Compile a CuTe DSL kernel to cubin + PTX + JSON manifest.

    This is the generic compiler for any CuTe DSL kernel (quack, custom, etc.).
    It handles compilation via ``cute.compile``, PTX/cubin extraction, metadata
    parsing, and manifest generation.  Kernel-specific wrappers (e.g.
    ``compile_cute_rmsnorm``) should prepare the kernel object, fake tensors,
    shared memory, and parameter layout, then delegate here.

    Args:
        kernel: A CuTe DSL kernel callable (e.g. ``quack.rmsnorm.RMSNorm``
            instance, or any object with a ``cute.kernel``-decorated method
            invoked via ``cute.compile``).
        compile_args: Positional arguments passed to ``cute.compile`` after
            the kernel.  Typically fake tensors, ``None`` placeholders,
            scalar constants, and a CUDA stream.
        output_dir: Directory to write the compiled artifacts into.
        kernel_name: Override the kernel function name in the manifest.
            Defaults to the ``.entry`` name extracted from the PTX.
        base_name: Base filename for artifacts (``<base>.cubin``,
            ``<base>.ptx``, ``<base>.json``).  Defaults to the kernel
            function name extracted from PTX.
        shared_mem: Dynamic shared memory in bytes.  Must be provided by
            the caller (CuTe DSL uses ``SmemAllocator`` at compile time;
            the size is not recorded in the cubin).
        params: Parameter layout description for the C++ launcher — a list
            of dicts, each describing one kernel parameter (name, type,
            size_bytes, fields).  Stored verbatim in the manifest.
        **extra_metadata: Additional key-value pairs written into the
            manifest (e.g. ``library="quack"``, ``dtype="bf16"``,
            ``C=768``).

    Returns:
        Path to the generated JSON manifest file.
    """
    import cutlass.cute as cute

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- compile ----
    compiled = cute.compile[cute.KeepPTX(True), cute.KeepCUBIN(True)](
        kernel, *compile_args,
    )

    cubin_data = compiled.artifacts.CUBIN
    ptx_data = compiled.artifacts.PTX
    if not cubin_data:
        raise RuntimeError("CuTe DSL compilation produced no cubin")

    # ---- parse PTX metadata ----
    entries = re.findall(r"\.entry\s+(\S+)\(", ptx_data)
    if not entries:
        raise RuntimeError("No .entry directive found in compiled PTX")
    ptx_kernel_name = entries[0]

    reqntid = re.findall(r"\.reqntid\s+(\d+)", ptx_data)
    num_threads = int(reqntid[0]) if reqntid else 128
    num_warps = num_threads // 32

    # ---- write artifacts ----
    name = base_name or ptx_kernel_name
    cubin_file = f"{name}.cubin"
    ptx_file = f"{name}.ptx"

    (output_dir / cubin_file).write_bytes(cubin_data)
    (output_dir / ptx_file).write_text(ptx_data)

    # ---- write manifest ----
    manifest: dict[str, Any] = {
        "name": kernel_name or ptx_kernel_name,
        "backend": "cute_dsl",
        "cubin": cubin_file,
        "ptx": ptx_file,
        "num_warps": num_warps,
        "num_threads": num_threads,
        "shared_mem": shared_mem,
    }
    if params is not None:
        manifest["params"] = params
    manifest.update(extra_metadata)

    manifest_path = output_dir / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Compiled CuTe %s -> %s (%d bytes, %d threads, %d shared)",
        manifest["name"], output_dir / cubin_file,
        len(cubin_data), num_threads, shared_mem,
    )

    return str(manifest_path)
