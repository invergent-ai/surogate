// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Based on "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
// arXiv:2512.02010

/**
 * @file quant_nvfp4_4o6.cu
 * @brief Four Over Six (4/6) adaptive block scaling for NVFP4 quantization.
 *
 * Implements the 4/6 algorithm which evaluates both scaling to max=6.0 (standard)
 * and max=4.0 for each block, selecting the option with lower quantization error.
 *
 * Key insight: FP4 E2M1 has non-uniform quantization steps:
 * - Near 0: steps of 0.5 (fine granularity)
 * - Near max: step from 4.0 to 6.0 is 2.0 (coarse, ~33% error)
 *
 * By scaling some blocks to max=4.0 instead of 6.0:
 * - The largest step becomes 3.0→4.0 = 1.0 (vs 2.0 for 4.0→6.0)
 * - Near-maximal values get better representation
 * - Trade-off: slightly worse representation of small values
 *
 * The selection is per-block based on error metrics (MSE, L1, or AbsMax).
 */

#include "kernels/kernel_utils.cuh"
#include "kernels/kernels.h"
#include "recipes/nvfp4/nvfp4_recipe.h"
#include "utilities/tensor.h"
#include "utilities/vec.cuh"
#include "utilities/utils.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <limits>
#include <cmath>

namespace {

// ============================================================================
// Constants for Four Over Six
// ============================================================================

constexpr float kFP4Max6 = 6.0f;          // Standard NVFP4 max
constexpr float kFP4Max4 = 4.0f;          // Alternative 4/6 max
constexpr float kFP8E4M3Max = 448.0f;     // FP8 E4M3 maximum value
constexpr float kFP8E4M3MinNormal = 0.015625f;  // Minimum positive normal FP8 E4M3
constexpr int kBlockSize = 16;            // Elements per block scale
constexpr int kTileDim = 128;             // Tile dimension for kernel
constexpr int kValuesPerByte = 2;         // 2 FP4 values packed per byte

// Global tensor scale divisor for 4/6 mode
// 384 is the largest E4M3 where 384 * (4/6) = 256 is exactly representable
constexpr float k4o6GlobalScaleDivisor = 384.0f * 4.0f;

// ============================================================================
// PTX Intrinsics
// ============================================================================

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

/**
 * @brief Quantize 8 floats to FP4 E2M1 AND simultaneously dequantize to FP16.
 *
 * This is the key optimization from the 4/6 paper: perform quantization and
 * dequantization in a single PTX sequence to avoid extra memory access.
 *
 * @param array Input array of 8 scaled float values.
 * @param[out] quant Output packed FP4 values (8 values in 32 bits).
 * @param[out] dequant Output dequantized values as half precision (4x uint32_t = 8 halfs).
 */
__device__ __forceinline__ void fp32x8_to_e2m1x8_with_dequant(
    float (&array)[8],
    uint32_t& quant,
    uint32_t (&dequant)[4])
{
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        // Quantize pairs of floats to FP4 E2M1
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
        // Pack into output
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        // Dequantize back to FP16x2 for error calculation
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
        "}"
        : "=r"(quant), "=r"(dequant[0]), "=r"(dequant[1]), "=r"(dequant[2]), "=r"(dequant[3])
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
          "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
}

/**
 * @brief Extract float from packed FP16x2 (low half).
 */
__device__ __forceinline__ float extract_half_low(uint32_t packed) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>(packed & 0xFFFF)));
}

/**
 * @brief Extract float from packed FP16x2 (high half).
 */
__device__ __forceinline__ float extract_half_high(uint32_t packed) {
    return __half2float(__ushort_as_half(static_cast<unsigned short>((packed >> 16) & 0xFFFF)));
}

/**
 * @brief Compute global encode scale for 4/6 mode.
 *
 * Uses the 4/6 specific divisor (384*4 = 1536) instead of standard (448*6 = 2688).
 */
__device__ __forceinline__ float compute_global_encode_scale_4o6(float global_amax) {
    if (global_amax == 0.0f) {
        return 1.0f;
    }
    float scale = k4o6GlobalScaleDivisor * rcp_approx_ftz(fmaxf(global_amax, 1e-10f));
    scale = fminf(scale, 3.4e38f);
    if (scale == 0.0f) {
        return 1.0f;
    }
    return scale;
}

__device__ __forceinline__ uint8_t float_to_ue4m3(float scale) {
    if (scale <= 0.0f || !isfinite(scale)) {
        return 0;
    }
    scale = fminf(fmaxf(scale, kFP8E4M3MinNormal), kFP8E4M3Max);
    __nv_fp8_e4m3 fp8_val;
    fp8_val.__x = __nv_cvt_float_to_fp8(scale, __nv_saturation_t::__NV_SATFINITE,
                                         __nv_fp8_interpretation_t::__NV_E4M3);
    return fp8_val.__x;
}

__device__ __forceinline__ float ue4m3_to_float(uint8_t ue4m3) {
    __nv_fp8_e4m3 fp8_val;
    fp8_val.__x = ue4m3;
    return __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
}

__device__ __forceinline__ size_t nvfp4_cutlass_scale_offset(
    int row, int scale_col, int num_scale_cols)
{
    constexpr int kRowsPerAtom = 128;
    constexpr int kColsPerAtom = 4;
    constexpr int kAtomSize = kRowsPerAtom * kColsPerAtom;

    int row_atom = row / kRowsPerAtom;
    int col_atom = scale_col / kColsPerAtom;
    int row_in_atom = row % kRowsPerAtom;
    int col_in_atom = scale_col % kColsPerAtom;
    int atoms_per_row = (num_scale_cols + kColsPerAtom - 1) / kColsPerAtom;

    size_t atom_offset = static_cast<size_t>(row_atom * atoms_per_row + col_atom) * kAtomSize;
    int row_in_32 = row_in_atom % 32;
    int row_blk32 = row_in_atom / 32;
    size_t intra_offset = row_in_32 * 16 + row_blk32 * 4 + col_in_atom;

    return atom_offset + intra_offset;
}

__device__ __forceinline__ uint8_t pack_fp4(uint8_t val0, uint8_t val1) {
    return (val1 << 4) | (val0 & 0xF);
}

// ============================================================================
// Error Computation Functions
// ============================================================================

/**
 * @brief Compute MSE error between original and dequantized values.
 */
__device__ __forceinline__ float compute_error_mse(
    const float* original,
    const uint32_t* dequant_packed,
    float scale_hp,
    int count)
{
    float error = 0.0f;

    #pragma unroll
    for (int i = 0; i < count / 2; ++i) {
        float dq_lo = extract_half_low(dequant_packed[i]) * scale_hp;
        float dq_hi = extract_half_high(dequant_packed[i]) * scale_hp;

        float diff_lo = dq_lo - original[i * 2];
        float diff_hi = dq_hi - original[i * 2 + 1];

        error += diff_lo * diff_lo + diff_hi * diff_hi;
    }

    return error;
}

/**
 * @brief Compute L1 error between original and dequantized values.
 */
__device__ __forceinline__ float compute_error_l1(
    const float* original,
    const uint32_t* dequant_packed,
    float scale_hp,
    int count)
{
    float error = 0.0f;

    #pragma unroll
    for (int i = 0; i < count / 2; ++i) {
        float dq_lo = extract_half_low(dequant_packed[i]) * scale_hp;
        float dq_hi = extract_half_high(dequant_packed[i]) * scale_hp;

        error += fabsf(dq_lo - original[i * 2]);
        error += fabsf(dq_hi - original[i * 2 + 1]);
    }

    return error;
}

/**
 * @brief Compute AbsMax error between original and dequantized values.
 */
__device__ __forceinline__ float compute_error_absmax(
    const float* original,
    const uint32_t* dequant_packed,
    float scale_hp,
    int count)
{
    float max_error = 0.0f;

    #pragma unroll
    for (int i = 0; i < count / 2; ++i) {
        float dq_lo = extract_half_low(dequant_packed[i]) * scale_hp;
        float dq_hi = extract_half_high(dequant_packed[i]) * scale_hp;

        float err_lo = fabsf(dq_lo - original[i * 2]);
        float err_hi = fabsf(dq_hi - original[i * 2 + 1]);

        max_error = fmaxf(max_error, fmaxf(err_lo, err_hi));
    }

    return max_error;
}

} // anonymous namespace

// ============================================================================
// Four Over Six Quantization Kernels
// ============================================================================

/**
 * @brief Four Over Six NVFP4 quantization kernel with CUTLASS-compatible scale layout.
 *
 * For each 16-element block:
 * 1. Compute scales for both max=6.0 and max=4.0 targets
 * 2. Quantize and dequantize with both scales
 * 3. Compute error metric for each
 * 4. Select the option with lower error
 *
 * @tparam TILE_SIZE Tile size (default 128).
 * @tparam ErrorMetric Error metric enum value.
 */
template<int TILE_SIZE, recipes::FourOverSixErrorMetric ErrorMetric>
__global__ void quantize_nvfp4_4o6_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int num_scale_cols)
{
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, M);
    const int col_end = min(col_start + TILE_SIZE, K);

    // Load global scale once per block
    __shared__ float s_global_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        s_global_scale = compute_global_encode_scale_4o6(ga);
    }
    __syncthreads();

    const float global_scale = s_global_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kBlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kBlockSize;
        const int block_col_end = min(block_col_start + kBlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Load values and compute block abs_max
        float block_amax = 0.0f;
        float values[kBlockSize];

        #pragma unroll
        for (int i = 0; i < kBlockSize; ++i) {
            if (i < block_width) {
                float val = (float)in[global_row * K + block_col_start + i];
                values[i] = val;
                block_amax = fmaxf(block_amax, fabsf(val));
            } else {
                values[i] = 0.0f;
            }
        }

        // Phase 2: Compute both scale factors
        // Scale for max=6: sf_6 = block_amax / (6.0 * global_scale)
        float sf_hp_6 = block_amax * rcp_approx_ftz(kFP4Max6 * global_scale);
        sf_hp_6 = fminf(fmaxf(sf_hp_6, kFP8E4M3MinNormal), kFP8E4M3Max);
        uint8_t ue4m3_scale_6 = float_to_ue4m3(sf_hp_6);
        float actual_sf_6 = ue4m3_to_float(ue4m3_scale_6);
        float sf_hp_6_final = actual_sf_6 * global_scale;

        // Scale for max=4: sf_4 = block_amax / (4.0 * global_scale)
        float sf_hp_4 = block_amax * rcp_approx_ftz(kFP4Max4 * global_scale);
        sf_hp_4 = fminf(fmaxf(sf_hp_4, kFP8E4M3MinNormal), kFP8E4M3Max);
        uint8_t ue4m3_scale_4 = float_to_ue4m3(sf_hp_4);
        float actual_sf_4 = ue4m3_to_float(ue4m3_scale_4);
        float sf_hp_4_final = actual_sf_4 * global_scale;

        // Phase 3: Quantize with both scales and compute errors
        float scaled_6[8], scaled_4[8];
        uint32_t quant_6_lo, quant_6_hi, quant_4_lo, quant_4_hi;
        uint32_t dequant_6_lo[4], dequant_6_hi[4], dequant_4_lo[4], dequant_4_hi[4];

        // Scale values for max=6
        float inv_sf_6 = rcp_approx_ftz(fmaxf(sf_hp_6_final, 1e-12f));
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_6[i] = values[i] * inv_sf_6;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_6, quant_6_lo, dequant_6_lo);

        // Scale remaining values for max=6
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_6[i] = values[i + 8] * inv_sf_6;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_6, quant_6_hi, dequant_6_hi);

        // Scale values for max=4
        float inv_sf_4 = rcp_approx_ftz(fmaxf(sf_hp_4_final, 1e-12f));
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_4[i] = values[i] * inv_sf_4;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_4, quant_4_lo, dequant_4_lo);

        // Scale remaining values for max=4
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_4[i] = values[i + 8] * inv_sf_4;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_4, quant_4_hi, dequant_4_hi);

        // Phase 4: Compute errors and select best option
        float error_6, error_4;

        if constexpr (ErrorMetric == recipes::FourOverSixErrorMetric::MSE) {
            error_6 = compute_error_mse(values, dequant_6_lo, sf_hp_6_final, 8) +
                      compute_error_mse(values + 8, dequant_6_hi, sf_hp_6_final, 8);
            error_4 = compute_error_mse(values, dequant_4_lo, sf_hp_4_final, 8) +
                      compute_error_mse(values + 8, dequant_4_hi, sf_hp_4_final, 8);
        } else if constexpr (ErrorMetric == recipes::FourOverSixErrorMetric::L1) {
            error_6 = compute_error_l1(values, dequant_6_lo, sf_hp_6_final, 8) +
                      compute_error_l1(values + 8, dequant_6_hi, sf_hp_6_final, 8);
            error_4 = compute_error_l1(values, dequant_4_lo, sf_hp_4_final, 8) +
                      compute_error_l1(values + 8, dequant_4_hi, sf_hp_4_final, 8);
        } else { // AbsMax
            error_6 = fmaxf(compute_error_absmax(values, dequant_6_lo, sf_hp_6_final, 8),
                           compute_error_absmax(values + 8, dequant_6_hi, sf_hp_6_final, 8));
            error_4 = fmaxf(compute_error_absmax(values, dequant_4_lo, sf_hp_4_final, 8),
                           compute_error_absmax(values + 8, dequant_4_hi, sf_hp_4_final, 8));
        }

        // Select the option with lower error
        bool pick_4 = (error_4 < error_6);
        uint32_t quant_lo = pick_4 ? quant_4_lo : quant_6_lo;
        uint32_t quant_hi = pick_4 ? quant_4_hi : quant_6_hi;
        uint8_t selected_scale = pick_4 ? ue4m3_scale_4 : ue4m3_scale_6;

        // Phase 5: Store scale in CUTLASS interleaved layout
        const int scale_col = (col_start / kBlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = selected_scale;

        // Phase 6: Store packed FP4 values
        const int out_col_start = block_col_start / kValuesPerByte;
        uint8_t* out_ptr = out_fp4 + global_row * (K / 2) + out_col_start;

        // Extract and pack FP4 values from uint32_t
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t byte_lo = (quant_lo >> (i * 8)) & 0xFF;
            out_ptr[i] = byte_lo;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            uint8_t byte_hi = (quant_hi >> (i * 8)) & 0xFF;
            out_ptr[4 + i] = byte_hi;
        }
    }
}

/**
 * @brief Four Over Six NVFP4 quantization with stochastic rounding.
 *
 * Uses stochastic rounding for gradient quantization while still applying
 * the 4/6 adaptive block scaling selection.
 */
template<int TILE_SIZE, recipes::FourOverSixErrorMetric ErrorMetric>
__global__ void quantize_nvfp4_4o6_stochastic_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int num_scale_cols,
    unsigned int seed)
{
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &rng_state);

    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, M);
    const int col_end = min(col_start + TILE_SIZE, K);

    __shared__ float s_global_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        s_global_scale = compute_global_encode_scale_4o6(ga);
    }
    __syncthreads();

    const float global_scale = s_global_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kBlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kBlockSize;
        const int block_col_end = min(block_col_start + kBlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        // Load values and compute block abs_max
        float block_amax = 0.0f;
        float values[kBlockSize];

        #pragma unroll
        for (int i = 0; i < kBlockSize; ++i) {
            if (i < block_width) {
                float val = (float)in[global_row * K + block_col_start + i];
                values[i] = val;
                block_amax = fmaxf(block_amax, fabsf(val));
            } else {
                values[i] = 0.0f;
            }
        }

        // Compute both scale factors
        float sf_hp_6 = block_amax * rcp_approx_ftz(kFP4Max6 * global_scale);
        sf_hp_6 = fminf(fmaxf(sf_hp_6, kFP8E4M3MinNormal), kFP8E4M3Max);
        uint8_t ue4m3_scale_6 = float_to_ue4m3(sf_hp_6);
        float actual_sf_6 = ue4m3_to_float(ue4m3_scale_6);
        float sf_hp_6_final = actual_sf_6 * global_scale;

        float sf_hp_4 = block_amax * rcp_approx_ftz(kFP4Max4 * global_scale);
        sf_hp_4 = fminf(fmaxf(sf_hp_4, kFP8E4M3MinNormal), kFP8E4M3Max);
        uint8_t ue4m3_scale_4 = float_to_ue4m3(sf_hp_4);
        float actual_sf_4 = ue4m3_to_float(ue4m3_scale_4);
        float sf_hp_4_final = actual_sf_4 * global_scale;

        // Quantize with RN first to compute errors (not stochastic for selection)
        float scaled_6[8], scaled_4[8];
        uint32_t quant_6_lo, quant_6_hi, quant_4_lo, quant_4_hi;
        uint32_t dequant_6_lo[4], dequant_6_hi[4], dequant_4_lo[4], dequant_4_hi[4];

        float inv_sf_6 = rcp_approx_ftz(fmaxf(sf_hp_6_final, 1e-12f));
        float inv_sf_4 = rcp_approx_ftz(fmaxf(sf_hp_4_final, 1e-12f));

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_6[i] = values[i] * inv_sf_6;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_6, quant_6_lo, dequant_6_lo);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_6[i] = values[i + 8] * inv_sf_6;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_6, quant_6_hi, dequant_6_hi);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_4[i] = values[i] * inv_sf_4;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_4, quant_4_lo, dequant_4_lo);

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_4[i] = values[i + 8] * inv_sf_4;
        }
        fp32x8_to_e2m1x8_with_dequant(scaled_4, quant_4_hi, dequant_4_hi);

        // Compute errors
        float error_6, error_4;

        if constexpr (ErrorMetric == recipes::FourOverSixErrorMetric::MSE) {
            error_6 = compute_error_mse(values, dequant_6_lo, sf_hp_6_final, 8) +
                      compute_error_mse(values + 8, dequant_6_hi, sf_hp_6_final, 8);
            error_4 = compute_error_mse(values, dequant_4_lo, sf_hp_4_final, 8) +
                      compute_error_mse(values + 8, dequant_4_hi, sf_hp_4_final, 8);
        } else if constexpr (ErrorMetric == recipes::FourOverSixErrorMetric::L1) {
            error_6 = compute_error_l1(values, dequant_6_lo, sf_hp_6_final, 8) +
                      compute_error_l1(values + 8, dequant_6_hi, sf_hp_6_final, 8);
            error_4 = compute_error_l1(values, dequant_4_lo, sf_hp_4_final, 8) +
                      compute_error_l1(values + 8, dequant_4_hi, sf_hp_4_final, 8);
        } else {
            error_6 = fmaxf(compute_error_absmax(values, dequant_6_lo, sf_hp_6_final, 8),
                           compute_error_absmax(values + 8, dequant_6_hi, sf_hp_6_final, 8));
            error_4 = fmaxf(compute_error_absmax(values, dequant_4_lo, sf_hp_4_final, 8),
                           compute_error_absmax(values + 8, dequant_4_hi, sf_hp_4_final, 8));
        }

        bool pick_4 = (error_4 < error_6);
        uint8_t selected_scale = pick_4 ? ue4m3_scale_4 : ue4m3_scale_6;
        float selected_sf_hp = pick_4 ? sf_hp_4_final : sf_hp_6_final;
        float inv_sf = rcp_approx_ftz(fmaxf(selected_sf_hp, 1e-12f));

        // Store scale
        const int scale_col = (col_start / kBlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = selected_scale;

        // Now do stochastic rounding quantization with selected scale
        const int out_col_start = block_col_start / kValuesPerByte;
        uint8_t* out_ptr = out_fp4 + global_row * (K / 2) + out_col_start;

        // Stochastic rounding quantization (simplified - using PTX with random bits)
        #pragma unroll
        for (int i = 0; i < kBlockSize; i += 4) {
            float4 rand_vals = curand_uniform4(&rng_state);
            float scaled_vals[4];

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                scaled_vals[j] = values[i + j] * inv_sf;
            }

            // Use stochastic rounding via PTX
            uint32_t rbits[4];
            rbits[0] = __float_as_uint(rand_vals.x);
            rbits[1] = __float_as_uint(rand_vals.y);
            rbits[2] = __float_as_uint(rand_vals.z);
            rbits[3] = __float_as_uint(rand_vals.w);

            uint16_t quant_pair;
            asm volatile(
                "cvt.rs.satfinite.e2m1x4.f32 %0, {%5, %2, %4, %1}, %6;\n"
                : "=h"(quant_pair)
                : "f"(scaled_vals[0]), "f"(scaled_vals[1]),
                  "f"(scaled_vals[2]), "f"(scaled_vals[3]),
                  "f"(scaled_vals[2]), "f"(scaled_vals[3]),
                  "r"(rbits[0]));

            out_ptr[i / 2] = quant_pair & 0xFF;
            out_ptr[i / 2 + 1] = (quant_pair >> 8) & 0xFF;
        }
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for Four Over Six NVFP4 quantization.
 */
void quantize_nvfp4_4o6_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    recipes::FourOverSixErrorMetric error_metric,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // Compute global amax first
    abs_max(global_amax, in, (long)M * K, dp, stream);

    const int num_scale_cols = div_ceil(K, kBlockSize);
    dim3 grid(div_ceil(M, kTileDim), div_ceil(K, kTileDim));
    const int threads_per_block = 256;

    switch (error_metric) {
        case recipes::FourOverSixErrorMetric::MSE:
            quantize_nvfp4_4o6_cutlass_kernel<128, recipes::FourOverSixErrorMetric::MSE>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
            break;
        case recipes::FourOverSixErrorMetric::L1:
            quantize_nvfp4_4o6_cutlass_kernel<128, recipes::FourOverSixErrorMetric::L1>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
            break;
        case recipes::FourOverSixErrorMetric::AbsMax:
            quantize_nvfp4_4o6_cutlass_kernel<128, recipes::FourOverSixErrorMetric::AbsMax>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for Four Over Six NVFP4 stochastic quantization.
 */
void quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    recipes::FourOverSixErrorMetric error_metric,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // Compute global amax first
    abs_max(global_amax, in, (long)M * K, dp, stream);

    const int num_scale_cols = div_ceil(K, kBlockSize);
    dim3 grid(div_ceil(M, kTileDim), div_ceil(K, kTileDim));
    const int threads_per_block = 256;

    switch (error_metric) {
        case recipes::FourOverSixErrorMetric::MSE:
            quantize_nvfp4_4o6_stochastic_cutlass_kernel<128, recipes::FourOverSixErrorMetric::MSE>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
            break;
        case recipes::FourOverSixErrorMetric::L1:
            quantize_nvfp4_4o6_stochastic_cutlass_kernel<128, recipes::FourOverSixErrorMetric::L1>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
            break;
        case recipes::FourOverSixErrorMetric::AbsMax:
            quantize_nvfp4_4o6_stochastic_cutlass_kernel<128, recipes::FourOverSixErrorMetric::AbsMax>
                <<<grid, threads_per_block, 0, stream>>>(
                    out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrappers
// ============================================================================

void quantize_nvfp4_4o6_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    Tensor& global_amax,
    const Tensor& in,
    recipes::FourOverSixErrorMetric error_metric,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_4o6_cutlass: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_4o6_cutlass_auto_scale(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        global_amax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, error_metric, dp, stream);
}

void quantize_nvfp4_4o6_stochastic_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    Tensor& global_amax,
    const Tensor& in,
    recipes::FourOverSixErrorMetric error_metric,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_4o6_stochastic_cutlass: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_4o6_stochastic_cutlass_auto_scale(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        global_amax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K, error_metric, seed, dp, stream);
}
