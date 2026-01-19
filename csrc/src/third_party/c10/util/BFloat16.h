// Minimal c10 BFloat16 compatibility header for vendored kernels.
// Provides at::BFloat16 alias to CUDA nv_bfloat16.
#pragma once

#include <cuda_bf16.h>

namespace at {
using BFloat16 = nv_bfloat16;
} // namespace at
