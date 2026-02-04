// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_bf16.h>
#include "utils.cuh"
#include "vec.cuh"

namespace quartet {

constexpr int NUM_WARPS_DQ = 12;

// Dequantize -> Transpose -> Hadamard -> Quantize (EDEN) pipeline
// Used for re-quantizing W^T in backward pass with different randomization
__global__ __launch_bounds__(NUM_WARPS_DQ*32, 1) void cutlass_dequant_tp_quant_kernel(
    __nv_fp4x2_storage_t* y, nv_bfloat16* scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    int rows, int cols, float inv_fp4_max)
{
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = NUM_WARPS_DQ;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ nv_bfloat16 h_smem[G * G];
    extern __shared__ uint4 dynamic_smem[];

    nv_bfloat16 local_scale_max = 0.f;
    __nv_fp4x2_storage_t* a_q_smem = reinterpret_cast<__nv_fp4x2_storage_t*>(dynamic_smem) + T*G*warp_id / 2;
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(dynamic_smem + T * G * NUM_WARPS_DQ / 32) + T*G*warp_id;

    // Load Hadamard matrix
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
        const int col = (i * T) % cols;
        const int row = (i * T) / cols * G;
        for (int s = 0; s < T_PER_G; ++s) {
            __nv_fp4x2_storage_t* smem_base = a_q_smem + s * T * T / 2;
            const __nv_fp4x2_storage_t* gmem_base = x + col / 2 + (row + s * T) * cols / 2;
            global_to_shared_swizzle(smem_base, gmem_base, cols);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    for (int i = start_i; i < groups; i += W * gridDim.x) {
        __pipeline_wait_prior(0);

        // Dequantize FP4 -> BF16
        for (int k = 0; k < T_PER_G; ++k) {
            const int g_col = (i * T) % cols;
            const int g_row = (i * T) / cols * G + k * T;
            const int s_col = threadIdx.x % 2;
            const int s_row = threadIdx.x / 2;
            __nv_fp4x2_storage_t* smem_base = a_q_smem + k * T * T / 2;
            __nv_fp4x2_storage_t* addr = smem_base + swizzle_smem(s_row, s_col) / 2;
            int scale_idx = g_col / T + (g_row + s_row) * cols / 16;
            float group_scale = static_cast<float>(x_scales[scale_idx]) * (*x_global_scale);
            auto quants = GenericVector<__nv_fp4x2_storage_t, 4>::load(addr);
            GenericVector<nv_bfloat16, 8> res;
            for (int s = 0; s < 4; ++s) {
                float2 dq = __nv_cvt_fp4x2_to_float2(quants[s]);
                res[2*s+0] = dq.x * group_scale;
                res[2*s+1] = dq.y * group_scale;
            }
            res.store(a_smem + k * T * T + swizzle_smem(s_row, s_col));
        }

        for (int k = 0; k < T_PER_G; ++k) {
            a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T);
        }

        if (const int next = i + W * gridDim.x; next < groups) {
            const int col = (next * T) % cols;
            const int row = (next * T) / cols * G;
            for (int s = 0; s < T_PER_G; ++s) {
                __nv_fp4x2_storage_t* smem_base = a_q_smem + s * T * T / 2;
                const __nv_fp4x2_storage_t* gmem_base = x + col / 2 + (row + s * T) * cols / 2;
                global_to_shared_swizzle(smem_base, gmem_base, cols);
            }
        }
        __pipeline_commit();

        // Compute Hadamard transform
        m16_n16_k32_c_fragment<float> acc[T_PER_G];
        for (int j = 0; j < T_PER_G; ++j) {
            for (int k = 0; k < T_PER_G; ++k) {
                const nv_bfloat16* smem_base = h_smem + k * T * T + j * T * G;
                m16_n16_b_fragment<nv_bfloat16> b_frag = load_fragment_b_swizzle(lane_id, smem_base);
                mma_m16_n16_sync(acc[j], a_frags[k], b_frag, acc[j]);
            }
        }

        // EDEN quantization epilogue
        using group_f_vec = GenericVector<float, 4>;
        using group_n_vec = GenericVector<__nv_fp4x2_storage_t, 2>;
        constexpr int SPT = 16;

        nv_bfloat16 out_scales[SPT];
        const int col = (i * T) % cols;
        const int row = (i * T) / cols * G;

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

                float abs_max = 0.f;
                for (int g = 0; g < group_f_vec::size; ++g) {
                    abs_max = fmaxf(abs_max, fabsf(nv_group[g]));
                }
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 1));
                abs_max = fmaxf(abs_max, __shfl_xor_sync(0xffffffff, abs_max, 2));

                float scale = abs_max * inv_fp4_max;
                constexpr unsigned MASK = 0b1111'1111'1111'1u << 19u;
                float m3_scale = __uint_as_float((__float_as_uint(scale) & MASK));

                float factor = m3_scale > 0 ? 1.f / m3_scale : 0.f;

                group_n_vec converted;
                float x_y = 0.f;
                float x_x = 0.f;

                for (int t = 0; t < group_f_vec::size; t += 2) {
                    float2 v = make_float2(nv_group[t] * factor, nv_group[t+1] * factor);
                    __nv_fp4x2_storage_t bits = __nv_cvt_float2_to_fp4x2(v, __nv_fp4_interpretation_t::__NV_E2M1, cudaRoundMode::cudaRoundNearest);
                    converted[t/2] = bits;
                    float2 back = __nv_cvt_fp4x2_to_float2(bits);
                    x_x += v.x * v.x + v.y * v.y;
                    x_y += v.x * back.x + v.y * back.y;
                }

                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 1);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 1);
                x_x += __shfl_xor_sync(0xFFFFFFFFu, x_x, 2);
                x_y += __shfl_xor_sync(0xFFFFFFFFu, x_y, 2);

                float correction = (x_y == 0) ? 1.f : x_x / x_y;
                float fixed_scale = m3_scale * correction;
                out_scales[s] = static_cast<nv_bfloat16>(fixed_scale);

                int t4 = lane_id % 4;
                int r4 = lane_id / 4;

                if (s < 8) {
                    __nv_fp4x2_storage_t* y_base = y + (col + r4) * rows/2 + row/2;
                    converted.store(y_base + 2 * t4 + s*8);
                } else {
                    __nv_fp4x2_storage_t* y_base = y + (col + r4 + 8) * rows/2 + row/2;
                    converted.store(y_base + 2 * t4 + (s-8)*8);
                }
            }
        }

        if (lane_id % 4 == 0) {
            using scales_vec = GenericVector<nv_bfloat16, 8>;
            scales_vec sv;
            int r4 = lane_id / 4;
            for (int r = 0; r < 2; ++r) {
                for (int k = 0; k < 8; ++k) {
                    sv[k] = out_scales[k + 8 * r];
                }
                sv.store(scales + (col + r4 + r * 8) * rows / 16 + row / 16);
                local_scale_max = fmaxf(local_scale_max, vecReduceAbsMax(sv));
            }
        }
    }

    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 4));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 8));
    local_scale_max = fmaxf(local_scale_max, __shfl_xor_sync(0xFFFFFFFFu, local_scale_max, 16));

    if (lane_id == 0) {
        unsigned as_32_bits = __float_as_uint(static_cast<float>(local_scale_max));
        atomicMax(max_scale, as_32_bits);
    }
}

// Forward declaration
void launch_eden_convert_scales_kernel(
    __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    const nv_bfloat16* scales_bf16, const unsigned* max_scale_ptr,
    long seed, int groups, float inv_fp8_max, cudaStream_t stream);

void dequant_tp_quant_launcher(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N, cudaStream_t stream)
{
    int groups = M * N / 128;
    int blocks, device;
    int smem = NUM_WARPS_DQ * 16 * 128 * 2;
    smem += NUM_WARPS_DQ * 16 * 128 / 2;
    QUARTET_CUDA_CHECK(cudaGetDevice(&device));

    QUARTET_CUDA_CHECK(cudaFuncSetAttribute(cutlass_dequant_tp_quant_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    QUARTET_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, cutlass_dequant_tp_quant_kernel, 32*NUM_WARPS_DQ, smem));
    int sms;
    QUARTET_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    QUARTET_CUDA_CHECK(cudaMemsetAsync(max_scale, 0, sizeof(unsigned), stream));

    cutlass_dequant_tp_quant_kernel<<<sms * blocks, dim3(32, NUM_WARPS_DQ), smem, stream>>>(
        y, scratch_scales, max_scale, h, x, x_scales, x_global_scale, M, N, 1.f / fp4_max);
    QUARTET_CUDA_CHECK(cudaGetLastError());
    launch_eden_convert_scales_kernel(scales_fp8, global_scale_ptr, scratch_scales, max_scale, seed, groups, 1.f / fp8_max, stream);
}

void dequant_tp_had_quant(
    __nv_fp4x2_storage_t* y, __nv_fp8_e4m3* scales_fp8, float* global_scale_ptr,
    nv_bfloat16* scratch_scales, unsigned* max_scale, const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x, const __nv_fp8_e4m3* x_scales, const float* x_global_scale,
    long seed, float fp4_max, float fp8_max, int M, int N, cudaStream_t stream)
{
    dequant_tp_quant_launcher(y, scales_fp8, global_scale_ptr, scratch_scales, max_scale, h, x, x_scales, x_global_scale, seed, fp4_max, fp8_max, M, N, stream);
}

}  // namespace quartet
