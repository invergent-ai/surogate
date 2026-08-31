#pragma once

// sinfer::ops — scale kernel: x *= factor, elementwise, in place.
// Vectorised over BF16 pairs. Included only by its launcher.
// See docs/op-development.md §6.

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

inline constexpr int kScaleBlock = 256;

__launch_bounds__(kScaleBlock) __global__
    void scale_kernel(__nv_bfloat16* x, std::int64_t n, float factor) {
    const std::int64_t tid    = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    const std::int64_t pairs  = n / 2;

    auto* x2 = reinterpret_cast<__nv_bfloat162*>(x);
    for (std::int64_t i = tid; i < pairs; i += stride) {
        const __nv_bfloat162 value = x2[i];
        x2[i] = __floats2bfloat162_rn(__low2float(value) * factor, __high2float(value) * factor);
    }
    if (tid == 0 && (n & 1) != 0) {
        x[n - 1] = __float2bfloat16(__bfloat162float(x[n - 1]) * factor);
    }
}

} // namespace sinfer::ops
