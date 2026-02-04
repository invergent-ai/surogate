// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdexcept>
#include <cstdint>

#include "vec.cuh"
#include "utils.cuh"

namespace quartet {

// Standalone EDEN FP4 quantization kernel
__global__ void eden_fp4_kernel(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_ptr,
    float scale_override, std::uint64_t seed, int nvecs)
{
    constexpr int HADAMARD_DIM = 128;

    using bf16x8 = GenericVector<nv_bfloat16, 8>;
    using fp32x8 = GenericVector<float, 8>;
    using fp4x8 = GenericVector<unsigned char, 4>;

    float global_abs_max = *amax_ptr;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= nvecs) return;

    float scales_max = 255.99f;
    float val_max = 6.f / scale_override;
    float scale = global_abs_max == 0 ? 1.f : global_abs_max / scales_max / val_max;

    // Per-group abs-maxes
    bf16x8 x = bf16x8::load(x_ptr + 8 * idx);
    nv_bfloat16 local_abs_max = vecReduceAbsMax(x);

    // Shuffle with neighbour for 16-element group reduction
    nv_bfloat16 other_abs_max = __shfl_xor_sync(0xffffffff, local_abs_max, 1);
    float s_group = static_cast<float>(__hmax(local_abs_max, other_abs_max)) / val_max;
    __nv_fp8_e4m3 s_as_fp8 = static_cast<__nv_fp8_e4m3>(s_group / scale);
    float s_round_fp8 = static_cast<float>(s_as_fp8);
    if (s_round_fp8 == 0) s_round_fp8 = 1.f;

    float factor = 1.f / (s_round_fp8 * scale);
    fp4x8 result;
    fp32x8 x_scaled;

    // Quantize to FP4 with RTN (not stochastic rounding)
    for (int k = 0; k < bf16x8::size; k += 2) {
        float2 src;
        src.x = static_cast<float>(x[k+0]) * factor;
        src.y = static_cast<float>(x[k+1]) * factor;
        unsigned char bits = __nv_cvt_float2_to_fp4x2(src, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
        result[k/2] = bits;
        x_scaled[k+0] = src.x;
        x_scaled[k+1] = src.y;
    }

    // Calculate EDEN correction factor: sum(x^2) / sum(x * q)
    float x_y = 0.f;
    float x_x = 0.f;
    for (int k = 0; k < bf16x8::size; k += 2) {
        x_x += x_scaled[k+0] * x_scaled[k+0];
        x_x += x_scaled[k+1] * x_scaled[k+1];
        float2 cvt_back = __nv_cvt_fp4x2_to_float2(result[k/2]);
        x_y += x_scaled[k+0] * cvt_back.x;
        x_y += x_scaled[k+1] * cvt_back.y;
    }

    // Reduce over hadamard dimension (128 elements = 16 threads * 8 elements)
    constexpr int OFFSET = HADAMARD_DIM / bf16x8::size;
    static_assert(OFFSET <= 16, "HADAMARD_DIM too large to fit in a single warp");
    for (int offset = OFFSET; offset > 0; offset /= 2) {
        x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, offset);
        x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, offset);
    }

    float correction = (x_y == 0) ? 1.f : x_x / x_y;
    float fixed_scale = s_round_fp8 * correction;

    // Stochastic rounding on corrected scale (key EDEN innovation)
    uint4 rng = philox<10>(seed, threadIdx.x, blockIdx.x);
    __nv_fp8_e4m3 sr = stochastic_rounding(fixed_scale, rng.x);

    // Store scale (one per 16 elements = every 2 threads)
    if (idx % 2 == 0) {
        scale_ptr[idx / 2] = sr;
    }

    result.store(reinterpret_cast<unsigned char*>(y_ptr) + 4 * idx);
}

void eden_fp4(
    __nv_fp4x4_e2m1* y_ptr, __nv_fp8_e4m3* scale_ptr,
    const nv_bfloat16* x_ptr, const float* amax_ptr,
    float scale_override, long seed, long nelem, cudaStream_t stream)
{
    if (nelem % 8 != 0) {
        throw std::runtime_error("eden_fp4: nelem must be divisible by 8");
    }
    int n_vecs = nelem / 8;
    int block_size = 256;
    int n_blocks = (n_vecs + block_size - 1) / block_size;
    eden_fp4_kernel<<<n_blocks, block_size, 0, stream>>>(y_ptr, scale_ptr, x_ptr, amax_ptr, scale_override, seed, n_vecs);
    QUARTET_CUDA_CHECK(cudaGetLastError());
}

}  // namespace quartet
