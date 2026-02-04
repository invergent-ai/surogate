// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// Based on Quartet-II by IST Austria (Erik Schultheis et al.)
// SPDX-License-Identifier: Apache-2.0

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "quartet_kernels.h"

namespace quartet {

// Philox-based random number generation for column sign flips
static __device__ __forceinline__ unsigned int philox_round(unsigned int ctr, unsigned int key) {
    unsigned int lo, hi;
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(hi) : "r"(ctr), "r"(0xCD9E8D57u));
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(lo) : "r"(ctr), "r"(0xCD9E8D57u));
    return hi ^ key ^ lo;
}

static __device__ __forceinline__ int get_random_sign(unsigned int seed, int col) {
    unsigned int val = philox_round(col, seed);
    val = philox_round(val, seed ^ 0x9E3779B9u);
    return (val & 1) ? 1 : -1;
}

/**
 * Kernel to initialize 128x128 Hadamard matrix with random column signs
 *
 * The Hadamard matrix is constructed using the Sylvester/Walsh construction:
 * H_1 = [1]
 * H_{2n} = [H_n   H_n ]
 *          [H_n  -H_n ]
 *
 * For normalized Hadamard: H_ij = (1/sqrt(128)) * H_unnormalized_ij
 *
 * With random column signs: H[:, j] *= sign(j)
 */
__global__ void hadamard_init_kernel(nv_bfloat16* H, unsigned int seed) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= HADAMARD_DIM || col >= HADAMARD_DIM) return;

    // Sylvester construction: H[i,j] = (-1)^(popcount(i & j))
    int bit_and = row & col;
    int popcount_val = __popc(bit_and);
    int sign = (popcount_val & 1) ? -1 : 1;

    // Apply random column sign flip for re-randomization
    int col_sign = get_random_sign(seed, col);
    sign *= col_sign;

    // Normalize by 1/sqrt(128) = 1/11.3137... ≈ 0.0884
    constexpr float inv_sqrt_128 = 0.08838834764831845f;  // 1/sqrt(128)
    float value = static_cast<float>(sign) * inv_sqrt_128;

    H[row * HADAMARD_DIM + col] = static_cast<nv_bfloat16>(value);
}

void initialize_hadamard_128(nv_bfloat16* H, unsigned int seed, cudaStream_t stream) {
    // Launch kernel with 16x16 thread blocks
    dim3 threads(16, 16);
    dim3 blocks((HADAMARD_DIM + 15) / 16, (HADAMARD_DIM + 15) / 16);

    hadamard_init_kernel<<<blocks, threads, 0, stream>>>(H, seed);
}

}  // namespace quartet
