#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: fill_i32_positions
 *
 * Math / indexing:
 *   positions[i] = start + i, 0 <= i < T.
 *
 * Logical shapes:
 *   positions is a contiguous I32 vector [T].
 *
 * Numeric:
 *   T is positive, start is nonnegative, and start+T must not exceed INT32_MAX. Thus every
 *   emitted value is a nonnegative I32 position.
 *
 * Effects:
 *   Writes the full positions vector.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void fill_i32_positions(Tensor& positions, std::int32_t start, cudaStream_t stream);

/**
 * Op: offset_i32_positions
 *
 * Math / indexing:
 *   destination[i] = source[i] + delta[0], 0 <= i < T.
 *
 * Logical shapes:
 *   source and destination are contiguous I32 vectors [T]; delta is an I32 scalar [1].
 *
 * Numeric:
 *   Callers provide values whose sums are representable by I32.
 *
 * Effects:
 *   Writes the full destination. Source and destination may alias; delta must not alias a written
 *   destination element.
 *
 * Workspace:
 *   None. The Op has no other state side effect.
 */
void offset_i32_positions(const Tensor& source, const Tensor& delta, Tensor& destination,
                          cudaStream_t stream);

} // namespace ninfer::ops
