// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cuda_bf16.h>
#include "utils.cuh"

namespace quartet {

//! Calculates X H^T efficiently for shape(H) = (128, 128)
__global__ void group_transform_128_kernel(nv_bfloat16* y, const nv_bfloat16* h, const nv_bfloat16* x, int groups) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ alignas(16) nv_bfloat16 h_smem[G * G];
    extern __shared__ uint4 dynamic_smem[];
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

    if (int i = start_i; i < groups) {
        for (int s = 0; s < T_PER_G; ++s) {
            nv_bfloat16* smem_base = a_smem + s * T * T;
            const nv_bfloat16* gmem_base = x + s * T + i * T * G;
            global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    for (int i = start_i; i < groups; i += W * gridDim.x) {

        __pipeline_wait_prior(0);
        for (int k = 0; k < T_PER_G; ++k) {
            a_frags[k] = load_fragment_a_swizzle(lane_id, a_smem + k * T * T);
        }

        if (const int next = i + W * gridDim.x; next < groups) {
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + s * T + next * T * G;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
            }
        }
        __pipeline_commit();

        for (int j = 0; j < T_PER_G; ++j) {
            m16_n16_k32_c_fragment<float> acc;
            for (int k = 0; k < T_PER_G; ++k) {
                const nv_bfloat16* smem_base = h_smem + k * T * T + j * T * G;
                m16_n16_b_fragment<nv_bfloat16> b_frag = load_fragment_b_swizzle(lane_id, smem_base);
                mma_m16_n16_sync(acc, a_frags[k], b_frag, acc);
            }
            store_fragment_row_major_sync(acc, y + j * T + i * T * G, G);
        }
    }
}

//! Calculates X^T H^T efficiently (transposed input)
__global__ void group_transform_128_tp_kernel(nv_bfloat16* y, const nv_bfloat16* h, const nv_bfloat16* x, int rows, int cols) {
    constexpr int G = 128;
    constexpr int T = 16;
    constexpr int W = 8;

    constexpr int T_PER_G = G / T;

    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;

    int start_i = warp_id + blockIdx.x * W;

    __shared__ alignas(16) nv_bfloat16 h_smem[G * G];
    extern __shared__ uint4 dynamic_smem[];
    nv_bfloat16* a_smem = reinterpret_cast<nv_bfloat16*>(dynamic_smem) + T*G*warp_id;

    for (int k = warp_id; k < T_PER_G * T_PER_G; k += W) {
        int i = k % T_PER_G;
        int j = k / T_PER_G;
        nv_bfloat16* smem_base = h_smem + i * T * T + j * T * G;
        const nv_bfloat16* gmem_base = h + i * T + j * T * G;
        global_to_shared_16_32_swizzle(smem_base, gmem_base, G);
    }
    __pipeline_commit();

    const int groups = rows * cols / (G * T);

    if (const int i = start_i; i < groups) {
        const int col = (i * T) % cols;
        const int row = (i * T) / cols * G;
        for (int s = 0; s < T_PER_G; ++s) {
            nv_bfloat16* smem_base = a_smem + s * T * T;
            const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
            global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(1);
    __syncthreads();

    m16_n16_a_fragment<nv_bfloat16> a_frags[T_PER_G];
    for (int i = start_i; i < groups; i += W * gridDim.x) {
        __pipeline_wait_prior(0);
        for (int k = 0; k < T_PER_G; ++k) {
            a_frags[k] = load_fragment_aT_swizzle(lane_id, a_smem + k * T * T);
        }

        if (const int next = i + W * gridDim.x; next < groups) {
            const int col = (next * T) % cols;
            const int row = (next * T) / cols * G;
            for (int s = 0; s < T_PER_G; ++s) {
                nv_bfloat16* smem_base = a_smem + s * T * T;
                const nv_bfloat16* gmem_base = x + col + (row + s * T) * cols;
                global_to_shared_16_32_swizzle(smem_base, gmem_base, cols);
            }
        }
        __pipeline_commit();

        const int col = (i * T) % cols;
        const int row = (i * T) / cols * G;

        for (int j = 0; j < T_PER_G; ++j) {
            m16_n16_k32_c_fragment<float> acc;
            for (int k = 0; k < T_PER_G; ++k) {
                const nv_bfloat16* smem_base = h_smem + k * T * T + j * T * G;
                m16_n16_b_fragment<nv_bfloat16> b_frag = load_fragment_b_swizzle(lane_id, smem_base);
                mma_m16_n16_sync(acc, a_frags[k], b_frag, acc);
            }

            store_fragment_row_major_sync(acc, y + j * T + col * rows + row, rows);
        }
    }
}

void group_transform_128_launcher(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, cudaStream_t stream) {
    assert(M * N % 128 == 0);
    int groups = M * N / 128;
    int blocks, device;
    int smem = 8 * 16 * 128 * 2;
    QUARTET_CUDA_CHECK(cudaGetDevice(&device));
    QUARTET_CUDA_CHECK(cudaFuncSetAttribute(group_transform_128_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    QUARTET_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, group_transform_128_kernel, 32*4, smem));
    int sms;
    QUARTET_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    group_transform_128_kernel<<<sms * blocks, dim3(32, 8), smem, stream>>>(y, H, x, groups / 16);
    QUARTET_CUDA_CHECK(cudaGetLastError());
}

void group_transform_128_tp_launcher(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, cudaStream_t stream) {
    assert(M * N % 128 == 0);
    int blocks, device;
    int smem = 8 * 16 * 128 * 2;
    QUARTET_CUDA_CHECK(cudaGetDevice(&device));
    QUARTET_CUDA_CHECK(cudaFuncSetAttribute(group_transform_128_tp_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    QUARTET_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, group_transform_128_tp_kernel, 32*4, smem));
    int sms;
    QUARTET_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    group_transform_128_tp_kernel<<<sms * blocks, dim3(32, 8), smem, stream>>>(y, H, x, M, N);
    QUARTET_CUDA_CHECK(cudaGetLastError());
}

void group_transform_128(nv_bfloat16* y, const nv_bfloat16* H, const nv_bfloat16* x, int M, int N, bool transpose, cudaStream_t stream) {
    if (transpose) {
        group_transform_128_tp_launcher(y, H, x, M, N, stream);
    } else {
        group_transform_128_launcher(y, H, x, M, N, stream);
    }
}

}  // namespace quartet
