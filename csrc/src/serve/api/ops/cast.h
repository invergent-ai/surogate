#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops {

/**
 * Op: cast_fp32_to_bf16
 *
 * Math / indexing:
 *   destination[i] = round_to_bf16_rne(source[i]) for every logical element i.
 *
 * Logical shapes:
 *   Source and destination have identical shapes and contiguous element order.
 *
 * Supported domain:
 *   Source is FP32 and destination is BF16. Contiguous aligned element counts divisible by four
 *   or two use vector routes; all other contiguous shapes use the scalar route.
 *
 * Numeric:
 *   Each FP32 value is converted independently with round-to-nearest-even.
 *
 * Effects:
 *   Writes the full destination. Source and destination must not alias.
 *
 * Workspace:
 *   None. The Op has no state side effect beyond writing destination.
 */
void cast_fp32_to_bf16(const Tensor& source, Tensor& destination, cudaStream_t stream);

/// Exact BF16 promotion to FP32. Same shapes, contiguous, non-overlapping tensors.
void cast_bf16_to_fp32(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace sinfer::ops
