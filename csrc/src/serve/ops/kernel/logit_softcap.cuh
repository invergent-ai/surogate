#pragma once

// sinfer::ops — logit softcap kernel: x = tanh(x / cap) * cap, elementwise, in place.
// Vectorised over BF16 pairs. Included only by its launcher.
// See docs/op-development.md §6.

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kLogitSoftcapBlock = 256;

/// `tanh(x * inv_cap) * cap`, with the reciprocal precomputed on the host: one multiply per
/// element rather than a divide, and the same value for every element of the round.
__device__ __forceinline__ float logit_softcap_value(float x, float inv_cap, float cap) {
    return tanhf(x * inv_cap) * cap;
}

__launch_bounds__(kLogitSoftcapBlock) __global__
    void logit_softcap_kernel(__nv_bfloat16* x, std::int64_t n, float inv_cap, float cap) {
    const std::int64_t tid    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t pairs  = n / 2;

    auto* x2 = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t i = tid; i < pairs; i += stride) {
        const __nv_bfloat162 value = x2[i];
        x2[i] = __floats2bfloat162_rn(logit_softcap_value(__low2float(value), inv_cap, cap),
                                      logit_softcap_value(__high2float(value), inv_cap, cap));
    }
    if (tid == 0 && (n & 1) != 0) {
        x[n - 1] = __float2bfloat16(
            logit_softcap_value(__bfloat162float(x[n - 1]), inv_cap, cap));
    }
}

} // namespace sinfer::ops
