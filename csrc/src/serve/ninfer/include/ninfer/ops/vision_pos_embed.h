#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Adds a four-corner interpolated position embedding. For d,p:
 *
 *   position[d,p] = sum_{c=0..3} table[d,indices[c,p]] * weights[c,p]
 *   ideal[d,p]    = x[d,p] + position[d,p].
 *
 * table is contiguous BF16 [D,R], indices is contiguous I32 [4,P] with values in [0,R), weights
 * is contiguous FP32 [4,P], and x is contiguous BF16 [D,P]. The oracle evaluates `ideal` naively in
 * FP64 from the represented inputs. The updated BF16 x is promoted and compared directly with that
 * result; output storage rounding belongs to the Op's numerical criterion, not the oracle.
 * Production routes may choose their natural accumulator and staging precision. Inputs must not
 * overlap x. The Op updates all of x in place and uses no workspace or other persistent state.
 * D=1152 with aligned table/x storage selects the optimized small-P warp or large-P CTA route;
 * other contiguous dimensions use the scalar route.
 */
void vision_pos_embed_add(const Tensor& table, const Tensor& indices, const Tensor& weights,
                          Tensor& x, cudaStream_t stream);

} // namespace ninfer::ops
