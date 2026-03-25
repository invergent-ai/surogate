// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file turboquant.cu
 * @brief TurboQuant 3.5-bit vector quantization kernels.
 *
 * Implements the TurboQuant_prod algorithm (arXiv 2504.19874) for KV cache
 * compression in LLM serving. Each vector is:
 *   1. Normalized and rotated via randomized Walsh-Hadamard transform
 *   2. Scalar-quantized with mixed 3-bit/2-bit Lloyd-Max codebooks (2.5 bits avg)
 *   3. Residual corrected with 1-bit QJL sign encoding (1 bit)
 *   Total: 3.5 bits/element + small norm overhead.
 *
 * One warp (32 threads) processes one vector. Each thread holds EPT = head_dim/32
 * elements. The Walsh-Hadamard transform uses register butterflies for the first
 * log2(EPT) stages and warp shuffles for the remaining 5 stages.
 *
 * Packing/unpacking uses shared memory to generalize across all EPT values.
 */

#include "turboquant.h"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <random>

namespace turboquant {

// ---------------------------------------------------------------------------
// Lloyd-Max codebook centroids for N(0,1)
// ---------------------------------------------------------------------------

// 3-bit (8 levels) — optimal scalar quantizer for Gaussian
__constant__ float kCodebook3[8] = {
    -2.1519f, -1.3440f, -0.7560f, -0.2451f,
     0.2451f,  0.7560f,  1.3440f,  2.1519f
};

// 2-bit (4 levels)
__constant__ float kCodebook2[4] = {
    -1.5104f, -0.4528f, 0.4528f, 1.5104f
};

// ---------------------------------------------------------------------------
// Constexpr helpers
// ---------------------------------------------------------------------------

constexpr int ct_log2(int n) { return (n <= 1) ? 0 : 1 + ct_log2(n >> 1); }

// ---------------------------------------------------------------------------
// Device functions
// ---------------------------------------------------------------------------

/// 3-bit quantization: binary search over Lloyd-Max boundaries for N(0,1).
/// Boundaries are midpoints: -1.748, -1.050, -0.501, 0, 0.501, 1.050, 1.748
__device__ __forceinline__ uint8_t quantize_3bit(float v) {
    if (v < 0.0f) {
        if (v < -1.050f) return (v < -1.748f) ? 0 : 1;
        else             return (v < -0.501f) ? 2 : 3;
    } else {
        if (v < 1.050f)  return (v < 0.501f) ? 4 : 5;
        else             return (v < 1.748f) ? 6 : 7;
    }
}

/// 2-bit quantization: boundaries at -0.982, 0, 0.982
__device__ __forceinline__ uint8_t quantize_2bit(float v) {
    if (v < 0.0f) return (v < -0.982f) ? 0 : 1;
    else          return (v <  0.982f) ? 2 : 3;
}

/// In-place unnormalized Walsh-Hadamard transform within a warp.
/// Thread `lane` holds EPT consecutive elements of a d=EPT*32 vector.
/// H*H = d*I, so applying twice and dividing by d gives identity.
template <int EPT>
__device__ __forceinline__ void warp_wht(float vals[EPT], int lane) {
    constexpr int LOG_EPT = ct_log2(EPT);

    // Within-thread butterfly stages (stride < EPT)
    #pragma unroll
    for (int s = 0; s < LOG_EPT; s++) {
        int stride = 1 << s;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            int j = i ^ stride;
            if (j > i) {
                float tmp = vals[i];
                vals[i] = tmp + vals[j];
                vals[j] = tmp - vals[j];
            }
        }
    }

    // Inter-thread butterfly stages (5 stages for 32-thread warp)
    #pragma unroll
    for (int s = 0; s < 5; s++) {
        int offset = 1 << s;
        #pragma unroll
        for (int i = 0; i < EPT; i++) {
            float partner = __shfl_xor_sync(0xFFFFFFFF, vals[i], offset);
            if (lane & offset)
                vals[i] = partner - vals[i];
            else
                vals[i] = vals[i] + partner;
        }
    }
}

/// Apply random sign flips from a packed bitmask.
template <int EPT>
__device__ __forceinline__ void apply_signs(float vals[EPT], const uint32_t* signs, int lane) {
    uint32_t word = signs[lane / (32 / EPT)];
    int bit_base = (lane % (32 / EPT)) * EPT;
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        if ((word >> (bit_base + i)) & 1u)
            vals[i] = -vals[i];
    }
}

/// Read sign word and base bit for a given lane/EPT combo.
template <int EPT>
__device__ __forceinline__ void get_sign_info(const uint32_t* signs, int lane,
                                               uint32_t& word, int& bit_base) {
    word = signs[lane / (32 / EPT)];
    bit_base = (lane % (32 / EPT)) * EPT;
}

/// Warp-wide sum reduction.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Quantize kernel
// ---------------------------------------------------------------------------

template <int EPT>
__global__ void turboquant_quantize_kernel(
    uint8_t* __restrict__ output,
    const nv_bfloat16* __restrict__ input,
    const uint32_t* __restrict__ signs_d1,
    const uint32_t* __restrict__ signs_d2,
    int num_vectors)
{
    constexpr int D = EPT * 32;
    constexpr int PACKED = 7 * D / 16 + 4;

    const int warps_per_block = blockDim.x / 32;
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int vec_idx = blockIdx.x * warps_per_block + warp_in_block;
    if (vec_idx >= num_vectors) return;

    // Shared memory: D bytes per warp for index staging
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + warp_in_block * D;

    uint8_t* out_ptr = output + (size_t)vec_idx * PACKED;
    const nv_bfloat16* in_ptr = input + (size_t)vec_idx * D;

    float vals[EPT];

    // --- Load ---
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        vals[i] = __bfloat162float(in_ptr[lane * EPT + i]);

    // --- Vector norm ---
    float norm_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) norm_sq += vals[i] * vals[i];
    norm_sq = warp_reduce_sum(norm_sq);
    float norm_x = sqrtf(norm_sq);

    // --- Normalize ---
    float inv_norm = (norm_x > 1e-12f) ? (1.0f / norm_x) : 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) vals[i] *= inv_norm;

    // --- Randomized Hadamard rotation ---
    apply_signs<EPT>(vals, signs_d1, lane);
    warp_wht<EPT>(vals, lane);
    // After unnormalized WHT of unit vector: elements ~ N(0, 1)

    // --- Scalar quantize (mixed 3/2 bit) ---
    uint8_t indices[EPT];
    float centroids_val[EPT];
    bool first_half = (lane < 16);  // elements 0..D/2-1 → 3-bit; D/2..D-1 → 2-bit

    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float v = vals[i];
        if (first_half) {
            indices[i] = quantize_3bit(v);
            centroids_val[i] = kCodebook3[indices[i]];
        } else {
            indices[i] = quantize_2bit(v);
            centroids_val[i] = kCodebook2[indices[i]];
        }
    }

    // --- Residual ---
    float residual[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++) residual[i] = vals[i] - centroids_val[i];

    // --- Residual norm ---
    float res_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < EPT; i++) res_sq += residual[i] * residual[i];
    res_sq = warp_reduce_sum(res_sq);
    float gamma = sqrtf(res_sq);

    // --- QJL: rotate residual and take signs ---
    apply_signs<EPT>(residual, signs_d2, lane);
    warp_wht<EPT>(residual, lane);

    // === Pack output ===

    // 1. QJL sign bits via ballot (EPT uint32s = D/8 bytes)
    uint32_t ballots[EPT];
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        ballots[i] = __ballot_sync(0xFFFFFFFF, residual[i] >= 0.0f);

    constexpr int OFF_QJL = 3 * D / 16 + D / 8;
    if (lane < EPT)
        reinterpret_cast<uint32_t*>(out_ptr + OFF_QJL)[lane] = ballots[lane];

    // 2. Norms (thread 0 writes)
    constexpr int OFF_NORMS = OFF_QJL + D / 8;
    if (lane == 0) {
        __half* np = reinterpret_cast<__half*>(out_ptr + OFF_NORMS);
        np[0] = __float2half(gamma);
        np[1] = __float2half(norm_x);
    }

    // 3. Stage indices in shared memory
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        smem[lane * EPT + i] = indices[i];
    __syncwarp();

    // 4. 3-bit packing: groups of 8 indices → 3 bytes (first D/2 indices)
    for (int g = lane; g < D / 16; g += 32) {
        uint8_t a = smem[g * 8 + 0], b = smem[g * 8 + 1];
        uint8_t c = smem[g * 8 + 2], d = smem[g * 8 + 3];
        uint8_t e = smem[g * 8 + 4], f = smem[g * 8 + 5];
        uint8_t gg = smem[g * 8 + 6], h = smem[g * 8 + 7];

        uint8_t* p = out_ptr + g * 3;
        p[0] = a | (b << 3) | ((c & 0x3) << 6);
        p[1] = (c >> 2) | (d << 1) | (e << 4) | ((f & 0x1) << 7);
        p[2] = (f >> 1) | (gg << 2) | (h << 5);
    }

    // 5. 2-bit packing: groups of 4 indices → 1 byte (last D/2 indices)
    constexpr int OFF_2BIT = 3 * D / 16;
    for (int g = lane; g < D / 8; g += 32) {
        int base = D / 2 + g * 4;
        uint8_t packed = smem[base] | (smem[base + 1] << 2) |
                         (smem[base + 2] << 4) | (smem[base + 3] << 6);
        out_ptr[OFF_2BIT + g] = packed;
    }
}

// ---------------------------------------------------------------------------
// Dequantize kernel
// ---------------------------------------------------------------------------

template <int EPT>
__global__ void turboquant_dequantize_kernel(
    nv_bfloat16* __restrict__ output,
    const uint8_t* __restrict__ input,
    const uint32_t* __restrict__ signs_d1,
    const uint32_t* __restrict__ signs_d2,
    int num_vectors)
{
    constexpr int D = EPT * 32;
    constexpr int PACKED = 7 * D / 16 + 4;

    const int warps_per_block = blockDim.x / 32;
    const int warp_in_block = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int vec_idx = blockIdx.x * warps_per_block + warp_in_block;
    if (vec_idx >= num_vectors) return;

    // Shared memory: D bytes per warp for index staging
    extern __shared__ uint8_t smem_raw[];
    uint8_t* smem = smem_raw + warp_in_block * D;

    const uint8_t* in_ptr = input + (size_t)vec_idx * PACKED;
    nv_bfloat16* out_ptr = output + (size_t)vec_idx * D;

    // --- Read norms ---
    constexpr int OFF_NORMS = 3 * D / 16 + D / 8 + D / 8;
    const __half* np = reinterpret_cast<const __half*>(in_ptr + OFF_NORMS);
    float gamma  = __half2float(np[0]);
    float norm_x = __half2float(np[1]);

    // --- Unpack indices into shared memory ---

    // 3-bit unpacking: 3 bytes → 8 indices (first D/2 indices)
    for (int g = lane; g < D / 16; g += 32) {
        const uint8_t* p = in_ptr + g * 3;
        uint8_t b0 = p[0], b1 = p[1], b2 = p[2];

        smem[g * 8 + 0] = b0 & 0x7;
        smem[g * 8 + 1] = (b0 >> 3) & 0x7;
        smem[g * 8 + 2] = ((b0 >> 6) | (b1 << 2)) & 0x7;
        smem[g * 8 + 3] = (b1 >> 1) & 0x7;
        smem[g * 8 + 4] = (b1 >> 4) & 0x7;
        smem[g * 8 + 5] = ((b1 >> 7) | (b2 << 1)) & 0x7;
        smem[g * 8 + 6] = (b2 >> 2) & 0x7;
        smem[g * 8 + 7] = (b2 >> 5) & 0x7;
    }

    // 2-bit unpacking: 1 byte → 4 indices (last D/2 indices)
    constexpr int OFF_2BIT = 3 * D / 16;
    for (int g = lane; g < D / 8; g += 32) {
        uint8_t byte = in_ptr[OFF_2BIT + g];
        int base = D / 2 + g * 4;
        smem[base + 0] = byte & 0x3;
        smem[base + 1] = (byte >> 2) & 0x3;
        smem[base + 2] = (byte >> 4) & 0x3;
        smem[base + 3] = (byte >> 6) & 0x3;
    }

    __syncwarp();

    // --- Look up centroids ---
    float vals[EPT];
    bool first_half = (lane < 16);
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        uint8_t idx = smem[lane * EPT + i];
        if (first_half)
            vals[i] = kCodebook3[idx];
        else
            vals[i] = kCodebook2[idx];
    }

    // --- QJL reconstruction ---
    constexpr int OFF_QJL = 3 * D / 16 + D / 8;
    float z[EPT];

    // Each ballot[k] bit j = sign of thread j's element k
    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        uint32_t ballot_word = reinterpret_cast<const uint32_t*>(in_ptr + OFF_QJL)[i];
        z[i] = ((ballot_word >> lane) & 1u) ? 1.0f : -1.0f;
    }

    // WHT on sign vector
    warp_wht<EPT>(z, lane);

    // r_approx[j] = gamma * sqrt(pi/2) / D * D2[j] * (H * z)[j]
    float qjl_scale = gamma * 1.2533141f / (float)D;  // sqrt(pi/2) ≈ 1.2533141
    uint32_t d2_word;
    int d2_bit_base;
    get_sign_info<EPT>(signs_d2, lane, d2_word, d2_bit_base);

    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float d2_sign = ((d2_word >> (d2_bit_base + i)) & 1u) ? -1.0f : 1.0f;
        vals[i] += qjl_scale * d2_sign * z[i];
    }

    // --- Inverse Hadamard rotation ---
    // x_hat = D1 * H * v_approx / D  (since H^{-1} = H/D for unnormalized WHT)
    warp_wht<EPT>(vals, lane);

    float scale = norm_x / (float)D;
    uint32_t d1_word;
    int d1_bit_base;
    get_sign_info<EPT>(signs_d1, lane, d1_word, d1_bit_base);

    #pragma unroll
    for (int i = 0; i < EPT; i++) {
        float d1_sign = ((d1_word >> (d1_bit_base + i)) & 1u) ? -1.0f : 1.0f;
        vals[i] = scale * d1_sign * vals[i];
    }

    // --- Store BF16 ---
    #pragma unroll
    for (int i = 0; i < EPT; i++)
        out_ptr[lane * EPT + i] = __float2bfloat16(vals[i]);
}

// ---------------------------------------------------------------------------
// Host launchers
// ---------------------------------------------------------------------------

void quantize(
    uint8_t* output,
    const nv_bfloat16* input,
    const uint32_t* signs_d1,
    const uint32_t* signs_d2,
    int num_vectors,
    int head_dim,
    cudaStream_t stream)
{
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int grid = (num_vectors + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    switch (head_dim) {
        case 64: {
            int smem = WARPS_PER_BLOCK * 64;
            turboquant_quantize_kernel<2><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        case 128: {
            int smem = WARPS_PER_BLOCK * 128;
            turboquant_quantize_kernel<4><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        case 256: {
            int smem = WARPS_PER_BLOCK * 256;
            turboquant_quantize_kernel<8><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        default:
            break;
    }
}

void dequantize(
    nv_bfloat16* output,
    const uint8_t* input,
    const uint32_t* signs_d1,
    const uint32_t* signs_d2,
    int num_vectors,
    int head_dim,
    cudaStream_t stream)
{
    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32;
    int grid = (num_vectors + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    switch (head_dim) {
        case 64: {
            int smem = WARPS_PER_BLOCK * 64;
            turboquant_dequantize_kernel<2><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        case 128: {
            int smem = WARPS_PER_BLOCK * 128;
            turboquant_dequantize_kernel<4><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        case 256: {
            int smem = WARPS_PER_BLOCK * 256;
            turboquant_dequantize_kernel<8><<<grid, THREADS, smem, stream>>>(
                output, input, signs_d1, signs_d2, num_vectors);
            break;
        }
        default:
            break;
    }
}

void generate_signs_host(
    uint32_t* signs_d1,
    uint32_t* signs_d2,
    int head_dim,
    uint64_t seed)
{
    int num_words = head_dim / 32;
    std::mt19937_64 rng(seed);
    for (int i = 0; i < num_words; i++) {
        uint32_t word = 0;
        for (int b = 0; b < 32; b++) {
            if (rng() & 1) word |= (1u << b);
        }
        signs_d1[i] = word;
    }
    for (int i = 0; i < num_words; i++) {
        uint32_t word = 0;
        for (int b = 0; b < 32; b++) {
            if (rng() & 1) word |= (1u << b);
        }
        signs_d2[i] = word;
    }
}

}  // namespace turboquant
