// Minimal c10 Half compatibility header for vendored kernels.
// Provides at::Half alias to CUDA __half.
#pragma once

#include <cuda_fp16.h>

namespace at {
using Half = __half;
} // namespace at
