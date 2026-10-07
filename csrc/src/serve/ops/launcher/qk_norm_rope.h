#pragma once

// sinfer::ops::detail - private launch prototype for qk_norm_rope. Included by the wrapper and
// defined by the CUDA launcher.

#include "api/ops/qk_norm_rope.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

/// The launch, for arguments the wrapper validated; false where the kernel has no form for them.
bool qk_norm_rope_launch(const QkNormRope& args, cudaStream_t stream);

} // namespace sinfer::ops::detail
