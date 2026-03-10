# Copyright (c) 2026, Invergent SA, developed by Flavius Burca
# SPDX-License-Identifier: Apache-2.0
#
# CuTe DSL RMSNorm kernel compiler — wraps quack's production-quality kernel.
#
# Compiles quack.rmsnorm.RMSNorm to a standalone cubin + JSON manifest
# that can be loaded by the C++ JitKernel loader.
#
# The CuTe DSL kernel takes structured tensor descriptor arguments, not raw
# pointers.  The manifest records the exact parameter layout so the C++ side
# can construct the correct launch arguments.
#
# Parameter layout (from PTX analysis of quack RMSNorm, N=compile-time):
#
#   param_0 [24B, align 8] — mX  (input,  [M, N] bf16)
#       +0:  void*  data_ptr
#       +8:  int32  M  (batch/row count, dynamic)
#       +12: int32  (padding)
#       +16: int64  stride_row  (in elements, = N for contiguous)
#
#   param_1 [8B, align 8] — mW  (weight, [N] bf16)
#       +0:  void*  data_ptr
#
#   param_2 [24B, align 8] — mO  (output, [M, N] bf16)
#       +0:  void*  data_ptr
#       +8:  int32  (padding)
#       +12: int32  (padding)
#       +16: int64  stride_row
#
#   param_3 [16B, align 8] — mRstd (rstd, [M] fp32, optional)
#       +0:  void*  data_ptr
#       +8:  int64  (padding / unused stride)
#
#   param_4 [4B] — eps  (float32)
#
# Usage:
#   from surogate.kernels.cute_rmsnorm import compile_cute_rmsnorm
#   manifest = compile_cute_rmsnorm(C=768, output_dir="/tmp/cute_rmsnorm")

from __future__ import annotations

import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32


def _align(val: int, alignment: int) -> int:
    return (val + alignment - 1) & ~(alignment - 1)


def _compute_dynamic_smem(C: int, dtype_bytes: int) -> int:
    """Compute dynamic shared memory for quack RMSNorm.

    Mirrors the SmemAllocator layout in quack.rmsnorm.RMSNorm.kernel():
      sX tile (rows_per_block x cols_per_tile x dtype_bytes, align 16)
      + reduction_buffer (num_warps x cluster_n x stage x 4, align 8)
    """
    N = C

    # quack.rmsnorm.RMSNorm._threads_per_row
    threads_per_row = 256
    for limit, threads in [(64, 8), (128, 16), (3072, 32), (6144, 64), (16384, 128)]:
        if N <= limit:
            threads_per_row = threads
            break

    # quack.reduction_base.ReductionBase._num_threads
    num_threads = 128 if N <= 16384 else 256

    # quack.rmsnorm.RMSNorm._set_cluster_n (16-bit dtypes: bf16/fp16)
    if dtype_bytes == 2:
        thresholds = [(16384, 1), (32768, 2), (65536, 4), (131072, 8)]
    else:
        thresholds = [(32768, 1), (65536, 2), (131072, 4), (262144, 8)]
    cluster_n = 16
    for limit, cluster in thresholds:
        if N <= limit:
            cluster_n = cluster
            break

    # quack.reduction_base.ReductionBase._get_tiled_copy
    dtype_width_bits = dtype_bytes * 8
    vecsize = math.gcd(N, 128 // dtype_width_bits)
    num_blocks_N = -(-(N // vecsize) // (threads_per_row * cluster_n))
    rows_per_block = num_threads // threads_per_row
    cols_per_tile = vecsize * num_blocks_N * threads_per_row

    # SmemAllocator: sX tile (byte_alignment=16)
    sx_bytes = rows_per_block * cols_per_tile * dtype_bytes
    smem = _align(sx_bytes, 16)

    # Reduction buffer (byte_alignment=8): num_warps * cluster_n * stage elements of fp32
    # RMSNorm stage=1 (LayerNorm would be stage=2)
    stage = 1
    num_warps = num_threads // 32
    reduction_bytes = num_warps * cluster_n * stage * 4
    smem = _align(smem, 8) + reduction_bytes

    # mbar_ptr only allocated when cluster_n > 1
    if cluster_n > 1:
        mbar_bytes = stage * 8  # Int64
        smem = _align(smem, 8) + mbar_bytes

    return smem


def _rmsnorm_params(dtype: str, store_rstd: bool) -> list[dict]:
    """Parameter layout for quack RMSNorm (for C++ launcher)."""
    params = [
        {
            "name": "mX",
            "type": "tensor_2d",
            "dtype": dtype,
            "size_bytes": 24,
            "fields": [
                {"name": "data_ptr", "offset": 0, "type": "ptr"},
                {"name": "M", "offset": 8, "type": "int32"},
                {"name": "stride_row", "offset": 16, "type": "int64"},
            ],
        },
        {
            "name": "mW",
            "type": "tensor_1d",
            "dtype": dtype,
            "size_bytes": 8,
            "fields": [
                {"name": "data_ptr", "offset": 0, "type": "ptr"},
            ],
        },
        {
            "name": "mO",
            "type": "tensor_2d",
            "dtype": dtype,
            "size_bytes": 24,
            "fields": [
                {"name": "data_ptr", "offset": 0, "type": "ptr"},
                {"name": "stride_row", "offset": 16, "type": "int64"},
            ],
        },
    ]
    if store_rstd:
        params.append({
            "name": "mRstd",
            "type": "tensor_1d",
            "dtype": "fp32",
            "size_bytes": 16,
            "fields": [
                {"name": "data_ptr", "offset": 0, "type": "ptr"},
            ],
        })
    params.append({
        "name": "eps",
        "type": "scalar",
        "dtype": "fp32",
        "size_bytes": 4,
    })
    return params


def compile_cute_rmsnorm(
    C: int,
    output_dir: str = ".",
    dtype: str = "bf16",
    store_rstd: bool = True,
) -> str:
    """Compile quack's CuTe DSL RMSNorm kernel for a specific hidden dimension.

    Args:
        C: Hidden dimension (e.g. 768, 2048, 4096).
        output_dir: Where to write the compiled artifacts.
        dtype: Input/output data type ("bf16", "fp16", "fp32").
        store_rstd: Whether the kernel outputs reciprocal standard deviation.

    Returns:
        Path to the JSON manifest file.
    """
    import cuda.bindings.driver as cuda_drv
    from quack.compile_utils import make_fake_tensor as fake_tensor
    from quack.rmsnorm import RMSNorm
    from surogate.kernels.compiler import compile_cute_kernel

    dtype_map = {
        "bf16": cutlass.BFloat16,
        "fp16": cutlass.Float16,
        "fp32": cutlass.Float32,
    }
    cute_dtype = dtype_map[dtype]

    # ---- build kernel + fake tensors ----
    kernel = RMSNorm(cute_dtype, C)
    batch_sym = cute.sym_int()
    # Divisibility must match quack's _rmsnorm_fwd: only tensor dtypes, not rstd's fp32
    tensor_dtypes = [cute_dtype]  # x, out, weight all share the same dtype
    div = math.gcd(C, *(128 // dt.width for dt in tensor_dtypes))

    x_cute = fake_tensor(cute_dtype, (batch_sym, C), div)
    out_cute = fake_tensor(cute_dtype, (batch_sym, C), div)
    w_cute = fake_tensor(cute_dtype, (C,), div)
    rstd_cute = fake_tensor(Float32, (batch_sym,)) if store_rstd else None
    stream = cuda_drv.CUstream(0)

    compile_args = (
        x_cute,      # mX
        w_cute,      # mW
        None,        # mB  (no bias)
        None,        # mRes (no residual)
        out_cute,    # mO
        None,        # mResO
        rstd_cute,   # mRstd
        None,        # mMean
        Float32(0),  # eps
        stream,
    )

    dtype_bytes = {"bf16": 2, "fp16": 2, "fp32": 4}[dtype]

    return compile_cute_kernel(
        kernel,
        compile_args,
        output_dir=output_dir,
        base_name=f"cute_rmsnorm_{dtype}_C{C}",
        shared_mem=_compute_dynamic_smem(C, dtype_bytes),
        params=_rmsnorm_params(dtype, store_rstd),
        library="quack",
        dtype=dtype,
        C=C,
        store_rstd=store_rstd,
    )
