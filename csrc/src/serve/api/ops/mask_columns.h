#pragma once

// surogate serve — mask_columns_zero (PATCHES.md #27).
//
// Zeroes the trailing columns of a contiguous [C, T] matrix whose column index
// is >= a device-resident valid-column count. Built for bucket-padded prefill
// bodies running under CUDA graphs: the launch geometry is static (the padded
// T), while the effective length is read from device memory at execution time,
// so one captured graph serves every actual length <= T.
//
// The canonical consumers are the GDN gating tensors: g is the log-decay
// (0 => decay 1) and beta gates the rank-1 state update (0 => no update), so
// zero-filled pad columns make the recurrent state update an exact identity
// without touching the scan kernels.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mask_columns_zero
 *
 * matrix: contiguous [C, T] (ne[2] == ne[3] == 1), FP32 or BF16.
 * valid_columns: contiguous I32 scalar on device; columns with index
 *   >= max(0, *valid_columns) are zero-filled, columns below it are untouched.
 *   Values above T mask nothing.
 */
void mask_columns_zero(Tensor& matrix, const Tensor& valid_columns, cudaStream_t stream);

} // namespace ninfer::ops
