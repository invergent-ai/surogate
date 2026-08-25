#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: finite typed scalar state transitions
 *
 * Math / indexing:
 *   set_i32_scalar:       destination' = value
 *   assign_i32_scalar:    destination' = source
 *   add_i32_scalars:      destination' = lhs + rhs
 *   increment_i32_scalar: scalar' = scalar + 1
 *   increment_i64_scalar: scalar' = scalar + 1
 *
 * Logical shapes:
 *   Every tensor argument is one contiguous scalar element of the named dtype.
 *
 * Numeric:
 *   Add/increment callers keep the exact result within the corresponding signed integer range.
 *
 * Effects:
 *   Each call writes only its destination scalar. assign_i32_scalar and add_i32_scalars require
 *   destination storage distinct from their inputs; the other calls update their single
 *   destination in place.
 *
 * Workspace:
 *   None. There is no state side effect beyond the stated destination transition.
 */
void set_i32_scalar(Tensor& destination, std::int32_t value, cudaStream_t stream);
void assign_i32_scalar(const Tensor& source, Tensor& destination, cudaStream_t stream);
void add_i32_scalars(const Tensor& lhs, const Tensor& rhs, Tensor& destination,
                     cudaStream_t stream);
void increment_i32_scalar(Tensor& scalar, cudaStream_t stream);
void increment_i64_scalar(Tensor& scalar, cudaStream_t stream);

} // namespace ninfer::ops
