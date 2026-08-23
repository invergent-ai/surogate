#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Op: prepare a batch of finite anchor-and-mask query blocks.
 *
 * anchors, lengths, and valid_columns are distinct contiguous device I32 [B] vectors. ids and
 * positions are distinct contiguous I32 [W,B] outputs. For each batch row b:
 *
 *   ids[0,b]       = anchors[b]
 *   ids[i,b]       = mask_id,                         1 <= i < W
 *   positions[i,b] = lengths[b] + min(i, V[b] - 1),  0 <= i < W
 *
 * where V[b]=valid_columns[b] and 1 <= V[b] <= W. Repeating the last valid position makes the
 * invalid physical tail safe for downstream fixed-width execution; valid_columns remains the
 * authority that excludes those rows from mathematical work and persistent-state updates.
 *
 * The registered domain is W=1..16, B=1..8, and nonnegative mask_id. The caller guarantees
 * nonnegative lengths whose final valid positions fit I32. Inputs are unchanged, every output
 * element is overwritten, and the Op owns no workspace or persistent state.
 */
void prepare_masked_block(const Tensor& anchors, const Tensor& lengths, const Tensor& valid_columns,
                          std::int32_t mask_id, Tensor& ids, Tensor& positions,
                          cudaStream_t stream);

} // namespace ninfer::ops
