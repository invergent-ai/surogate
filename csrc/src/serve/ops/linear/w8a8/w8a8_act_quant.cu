#include "ops/linear/w8a8/w8a8_act_quant.h"

#include "core/device.h"

#include <cuda_bf16.h>

namespace sinfer::ops::detail {
namespace {

constexpr int kThreads = 256;

__global__ void w8a8_act_quant_kernel(const __nv_bfloat16* __restrict__ x,
                                      std::int8_t* __restrict__ codes,
                                      float* __restrict__ scales, int hidden) {
    __shared__ float red[kThreads];
    const int token = static_cast<int>(blockIdx.x);
    const int tid   = static_cast<int>(threadIdx.x);
    const __nv_bfloat16* row = x + static_cast<std::int64_t>(token) * hidden;

    float local = 0.0f;
    for (int i = tid; i < hidden; i += kThreads) {
        local = fmaxf(local, fabsf(__bfloat162float(row[i])));
    }
    red[tid] = local;
    __syncthreads();
#pragma unroll
    for (int step = kThreads / 2; step > 0; step >>= 1) {
        if (tid < step) { red[tid] = fmaxf(red[tid], red[tid + step]); }
        __syncthreads();
    }
    const float absmax = red[0];
    const float scale  = absmax > 0.0f ? absmax * (1.0f / 127.0f) : 1.0f;
    const float inv    = absmax > 0.0f ? 127.0f / absmax : 0.0f;
    if (tid == 0) { scales[token] = scale; }
    std::int8_t* out = codes + static_cast<std::int64_t>(token) * hidden;
    for (int i = tid; i < hidden; i += kThreads) {
        const float value = __bfloat162float(row[i]) * inv;
        out[i] = static_cast<std::int8_t>(
            __float2int_rn(fminf(fmaxf(value, -127.0f), 127.0f)));
    }
}

constexpr std::size_t align16(std::size_t bytes) noexcept { return (bytes + 15u) & ~std::size_t{15u}; }

} // namespace

std::size_t w8a8_act_quant_bytes(std::int32_t hidden, std::int32_t tokens) noexcept {
    return align16(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(tokens)) +
           align16(static_cast<std::size_t>(tokens) * sizeof(float));
}

W8A8QuantizedActivations w8a8_act_quant(const Tensor& x, void* workspace, cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    auto* codes  = static_cast<std::int8_t*>(workspace);
    auto* scales = reinterpret_cast<float*>(
        static_cast<std::uint8_t*>(workspace) +
        align16(static_cast<std::size_t>(hidden) * static_cast<std::size_t>(tokens)));
    w8a8_act_quant_kernel<<<tokens, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), codes, scales, hidden);
    CUDA_CHECK(cudaGetLastError());
    return {codes, scales};
}

} // namespace sinfer::ops::detail
