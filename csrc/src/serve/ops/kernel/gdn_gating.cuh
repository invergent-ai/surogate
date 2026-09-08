#pragma once

// sinfer::ops - gdn_gating kernel: elementwise GDN gate prep over [H,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace sinfer::ops {

__global__ void gdn_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* A_log, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n, int heads) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h    = static_cast<int>(i % heads);
        const float av = __bfloat162float(a[i]);
        const float bv = __bfloat162float(b[i]);
        const float sp = softplus(av + dt_bias[h]);
        g[i]           = -expf(A_log[h]) * sp;
        beta[i]        = sigmoid(bv);
    }
}

/// Kimi Delta Attention's gates. One thread per channel of the decay; the update gate is one
/// value per head, so the first `H*T` threads write it too.
__global__ void kda_gating_kernel(const __nv_bfloat16* a, const __nv_bfloat16* b,
                                  const float* A_log, const float* dt_bias, float lower_bound,
                                  float* g, float* beta, int head_dim, int heads,
                                  std::int64_t decay_count, std::int64_t beta_count) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t width  = static_cast<std::int64_t>(head_dim) * heads;
    for (std::int64_t i = start; i < decay_count; i += stride) {
        const std::int64_t within = i % width;
        const int head            = static_cast<int>(within / head_dim);
        const float value         = __bfloat162float(a[i]) + dt_bias[within];
        g[i]                      = lower_bound * sigmoid(expf(A_log[head]) * value);
    }
    for (std::int64_t i = start; i < beta_count; i += stride) {
        beta[i] = sigmoid(__bfloat162float(b[i]));
    }
}

} // namespace sinfer::ops
