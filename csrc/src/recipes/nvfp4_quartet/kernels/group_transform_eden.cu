// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_bf16.h>
#include "utils.cuh"
#include "vec.cuh"

namespace quartet {

constexpr int NUM_WARPS = 12;

template<bool Transpose>
__global__ __launch_bounds__(NUM_WARPS*32, 1) void cutlass_group_transform_128_eden_kernel(
    __nv_fp4x2_storage_t* y, nv_bfloat16* scales, unsigned* max_scale,
    const nv_bfloat16* h, const nv_bfloat16* x, int rows, int cols, float inv_fp4_max)
{
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = NUM_WARPS;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ nv_bfloat16 h_smem[G * G];
    extern __shared__ uint4 dynamic_smem[];

    nv_bfloat16 local_scale_max = 0.f;
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(dynamic_smem) + T*G*warp_id;

    // Load Hadamard matrix into shared memory
    for (int k = warp_id; k < T_PER_G * T_PER_G; k += W) {
        int i = k % T_PER_G;
        int j = k / T_PER_G;
        nv_bfloat16* smem_base = h_smem + i * T * T + j * T * G;
        const nv_bfloat16* gmem_base = h + i * T + j * T * G;
        global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
    }
    __pipeline_commit();

    const int groups = rows * cols / (G * T);
    if (int i = start_i; i < groups) {
        if constexpr (Transpose) {
            const int col = (i * T) % cols;
            const int row = (i * T) / cols * G;
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
            }
        } else {
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + s * T + i * T * G;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
            }
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    for (int i = start_i; i < groups; i += W * gridDim.x) {
        __pipeline_wait_prior(0);
        for (int k = 0; k < T_PER_G; ++k) {
            if constexpr (Transpose) {
                a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T);
            } else {
                a_frags[k] = load_fragment_a_swizzle(lane_id, a_smem + k * T * T);
            }
        }

        if (const int next = i + W * gridDim.x; next < groups) {
            if constexpr (Transpose) {
                const int col = (next * T) % cols;
                const int row = (next * T) / cols * G;
                for (int s = 0; s < T_PER_G; ++s) {
                    nv_bfloat16* smem_base = a_smem + s * T * T;
                    const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
                    global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
                }
            } else {
                for (int s = 0; s < T_PER_G; ++s) {
                    nv_bfloat16* smem_base = a_smem + s * T * T;
                    const nv_bfloat16* gmem_base = x + s * T + next * T * G;
                    global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
                }
            }
        }
        __pipeline_commit();

        // Compute X @ H^T via MMA
        m16_n16_k32_c_fragment<float> acc[T_PER_G];
        for (int j = 0; j < T_PER_G; ++j) {
            for (int k = 0; k < T_PER_G; ++k) {
                const nv_bfloat16* smem_base = h_smem + k * T * T + j * T * G;
                m16_n16_b_fragment<nv_bfloat16> b_frag = load_fragment_b_swizzle(lane_id, smem_base);
                mma_m16_n16_sync(acc[j], a_frags[k], b_frag, acc[j]);
            }
        }

        // -------------------------------------------------------------------------------------------------------------
        // EDEN quantization epilogue
        using group_f_vec = GenericVector<float, 4>;
        using group_n_vec = GenericVector<__nv_fp4x2_storage_t, 2>;
        constexpr int SPT = 16;  // scales per thread

        nv_bfloat16 out_scales[SPT];

        // Process accumulator fragments and quantize
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            #pragma unroll
            for (int j = 0; j < T_PER_G; ++j) {
                int s = j + T_PER_G*k;
                group_f_vec nv_group;
                nv_group[0] = acc[j].v[0 + 2*k];
                nv_group[1] = acc[j].v[1 + 2*k];
                nv_group[2] = acc[j].v[4 + 2*k];
                nv_group[3] = acc[j].v[5 + 2*k];

                // Determine local abs-max
                float abs_max = 0.f;
                for (int g = 0; g < group_f_vec::size; ++g) {
                    abs_max = fmaxf(abs_max, fabsf(nv_group[g]));
                }
                // Reduce over the quads that collectively hold the 16 elements
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 1));
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 2));

                float scale = abs_max * inv_fp4_max;
                // RTZ mask for 4 mantissa bits (better than RTN with 3 bits)
                constexpr unsigned MASK = 0b1111'1111'1111'1u << 19u;
                float m3_scale = __uint_as_float((__float_as_uint(scale) & MASK));

                float factor = m3_scale > 0 ? 1.f / m3_scale : 0.f;

                group_n_vec converted;
                float x_y = 0.f;
                float x_x = 0.f;

                // Quantize to FP4 and compute EDEN correction factors
                for (int t = 0; t < group_f_vec::size; t += 2) {
                    float2 v = make_float2(nv_group[t] * factor, nv_group[t+1] * factor);
                    __nv_fp4x2_storage_t bits = __nv_cvt_float2_to_fp4x2(v, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
                    converted[t/2] = bits;
                    float2 back = __nv_cvt_fp4x2_to_float2(bits);
                    x_x += v.x * v.x + v.y * v.y;
                    x_y += v.x * back.x + v.y * back.y;
                }

                // Reduce EDEN factors over quad
                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 1);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 1);
                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 2);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 2);

                // Apply EDEN correction: correction = sum(x^2) / sum(x * q)
                float correction = (x_y == 0) ? 1.f : x_x / x_y;
                float fixed_scale = m3_scale * correction;
                out_scales[s] = static_cast<nv_bfloat16>(fixed_scale);

                // Store quantized FP4 values
                int t4 = lane_id % 4;
                int r4 = lane_id / 4;
                if constexpr (Transpose) {
                    const int col = (i * T) % cols;
                    const int row = (i * T) / cols * G;

                    if (s < 8) {
                        __nv_fp4x2_storage_t* y_base = y + (col + r4) * rows/2 + row/2;
                        converted.store(y_base + 2 * t4 + s*8);
                    } else {
                        __nv_fp4x2_storage_t* y_base = y + (col + r4 + 8) * rows/2 + row/2;
                        converted.store(y_base + 2 * t4 + (s-8)*8);
                    }
                } else {
                    if (s < 8) {
                        __nv_fp4x2_storage_t* y_base = y + i * T * G / 2 + r4 * G/2;
                        converted.store(y_base + 2 * t4 + s*8);
                    } else {
                        __nv_fp4x2_storage_t* y_base = y + (i * T + 8) * G / 2 + r4 * G/2;
                        converted.store(y_base + 2 * t4 + (s-8)*8);
                    }
                }
            }
        }

        // Store scales (one per lane that is 0 mod 4)
        if (lane_id % 4 == 0) {
            using scales_vec = GenericVector<nv_bfloat16, 8>;
            scales_vec sv;
            int r4 = lane_id / 4;
            for (int r = 0; r < 2; ++r) {
                for (int k = 0; k < 8; ++k) {
                    sv[k] = out_scales[k + 8 * r];
                }
                if constexpr (Transpose) {
                    const int col = (i * T) % cols;
                    const int row = (i * T) / cols * G;
                    sv.store(scales + (col + r4 + r * 8) * rows / 16 + row / 16);
                } else {
                    sv.store(scales + (i * T + r4 + r * 8) * G / 16);
                }
                local_scale_max = fmaxf(local_scale_max, vecReduceAbsMax(sv));
            }
        }
    }

    // Reduce abs-max over warp for global scale computation
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 4));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 8));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 16));

    if (lane_id == 0) {
        unsigned as_32_bits = __float_as_uint(static_cast<float>(local_scale_max));
        atomicMax(max_scale, as_32_bits);
    }
}

// Scale conversion kernel: convert BF16 scales to FP8 E4M3 with stochastic rounding
__global__ void eden_convert_scales_kernel(
    __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr,
    long seed, int groups, float inv_fp8_max)
{
    using bf16x8 = GenericVector<nv_bfloat16, 8>;
    using fp32x8 = GenericVector<float, 8>;
    using fp8x8 = GenericVector<__nv_fp8_e4m3, 8>;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= groups) return;

    uint4 rng = philox<10>(seed, threadIdx.x, blockIdx.x);
    cudaGridDependencySynchronize();

    float max_scale = __uint_as_float(*max_scale_ptr);
    float global_scale = max_scale * inv_fp8_max;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        global_scale_ptr[0] = global_scale;
    }

    float factor = global_scale != 0 ? 1.f / global_scale : 0.f;
    bf16x8 src_scales = bf16x8::load(scales_bf16 + i * 8);
    fp8x8 dst_scales;
    fp32x8 pre_sr;

    for (int k = 0; k < 8; ++k) {
        float src_scale = static_cast<float>(src_scales[k]);
        pre_sr[k] = src_scale * factor;
    }

    // Stochastic rounding on E4M3 scales (key Quartet-II innovation)
    dst_scales[0] = stochastic_rounding(pre_sr[0], rng.x);
    dst_scales[1] = stochastic_rounding(pre_sr[1], rng.y);
    dst_scales[2] = stochastic_rounding(pre_sr[2], rng.z);
    dst_scales[3] = stochastic_rounding(pre_sr[3], rng.w);
    rng = philox<10>(seed + 1, threadIdx.x, blockIdx.x);
    dst_scales[4] = stochastic_rounding(pre_sr[4], rng.x);
    dst_scales[5] = stochastic_rounding(pre_sr[5], rng.y);
    dst_scales[6] = stochastic_rounding(pre_sr[6], rng.z);
    dst_scales[7] = stochastic_rounding(pre_sr[7], rng.w);

    dst_scales.store(scales_fp8 + i * 8);
}

void launch_eden_convert_scales_kernel(
    __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr,
    long seed, int groups, float inv_fp8_max, cudaStream_t stream)
{
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t config;
    config.attrs = attribute;
    config.numAttrs = 1;
    config.blockDim = dim3(128);
    config.gridDim = dim3((groups + 127)/128);
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    QUARTET_CUDA_CHECK(cudaLaunchKernelEx(&config, eden_convert_scales_kernel, scales_fp8, global_scale_ptr, scales_bf16, max_scale_ptr, seed, groups, inv_fp8_max));
}

template<bool TransposeA>
void group_transform_128_eden_launcher(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, cudaStream_t stream)
{
    int groups = M * N / 128;
    int blocks, device;
    int smem = NUM_WARPS * 16 * 128 * 2;
    QUARTET_CUDA_CHECK(cudaGetDevice(&device));

    QUARTET_CUDA_CHECK(cudaFuncSetAttribute(cutlass_group_transform_128_eden_kernel<TransposeA>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    QUARTET_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cutlass_group_transform_128_eden_kernel<TransposeA>, 32*NUM_WARPS, smem));
    int sms;
    QUARTET_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));

    QUARTET_CUDA_CHECK(cudaMemsetAsync(max_scale, 0, sizeof(unsigned), stream));

    cutlass_group_transform_128_eden_kernel<TransposeA><<<sms * blocks, dim3(32, NUM_WARPS), smem, stream>>>(y, scratch_scales, max_scale, h, x, M, N, 1.f / fp4_max);
    QUARTET_CUDA_CHECK(cudaGetLastError());
    launch_eden_convert_scales_kernel(scales_fp8, global_scale_ptr, scratch_scales, max_scale, seed, groups, 1.f / fp8_max, stream);
}

void group_transform_128_eden(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h, const nv_bfloat16* x,
    long seed, float fp4_max, float fp8_max, int M, int N, bool transposeX, cudaStream_t stream)
{
    if (transposeX) {
        group_transform_128_eden_launcher<true>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N, stream);
    } else {
        group_transform_128_eden_launcher<false>(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, seed, fp4_max, fp8_max, M, N, stream);
    }
}

}  // namespace quartet
