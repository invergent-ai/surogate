#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops {

struct alignas(8) Bf16x4 {
    __nv_bfloat162 lo;
    __nv_bfloat162 hi;
};

__global__ void cast_fp32_to_bf16_x4_kernel(const float4* source, Bf16x4* destination,
                                            std::int64_t vectors) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < vectors; i += stride) {
        const float4 value = source[i];
        destination[i]     = {__floats2bfloat162_rn(value.x, value.y),
                              __floats2bfloat162_rn(value.z, value.w)};
    }
}

__global__ void cast_fp32_to_bf16_x2_kernel(const float2* source, __nv_bfloat162* destination,
                                            std::int64_t pairs) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < pairs; i += stride) {
        const float2 value = source[i];
        destination[i]     = __floats2bfloat162_rn(value.x, value.y);
    }
}

__global__ void cast_fp32_to_bf16_scalar_kernel(const float* source, __nv_bfloat16* destination,
                                                std::int64_t count) {
    const std::int64_t start  = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < count; i += stride) {
        destination[i] = __float2bfloat16_rn(source[i]);
    }
}

__global__ void cast_bf16_to_fp32_kernel(const __nv_bfloat16* source, float* destination,
                                          std::int64_t n) {
    const auto start = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (auto i = start; i < n; i += stride) { destination[i] = __bfloat162float(source[i]); }
}

} // namespace sinfer::ops
