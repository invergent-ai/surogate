// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file quant_nvfp4_cutlass.cu
 * @brief CUDA kernels for NVFP4 quantization with CUTLASS-compatible scale layout.
 *
 * Provides FP4 E2M1 quantization compatible with CUTLASS block-scaled GEMM:
 * - FP4 E2M1 data values (same as cuDNN NVFP4)
 * - UE4M3 scale factors in Sm1xxBlkScaledConfig interleaved layout
 * - 16 elements per scaling group (NVFP4 standard)
 *
 * The scale layout matches CUTLASS Sm1xxBlkScaledConfig expectations for direct use
 * with matmul_cutlass_fp4(). This differs from quant_fp4.cu which produces
 * cuDNN-compatible F8_128x4 swizzled scales.
 *
 * Requires: CUDA 12.8+, Blackwell GPU (SM100+) for native FP4 instructions.
 */

#include "kernels/kernel_utils.cuh"
#include "kernels/kernels.h"
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

// Local NVFP4 constants (prefixed to avoid conflicts with other headers)
constexpr float kFP4Max = 6.0f;
constexpr float kFP8E4M3Max = 448.0f;    // FP8 E4M3 maximum value
constexpr int kNVFP4BlockSize = 16;      // Elements per block scale (NVFP4 standard)
constexpr int kNVFP4TileDim = 128;       // Tile dimension for kernel
constexpr int kNVFP4ValuesPerByte = 2;   // 2 FP4 values packed per byte

// ============================================================================
// PTX Intrinsics for Optimized FP4 Quantization (from vLLM)
// ============================================================================

/**
 * @brief Fast approximate reciprocal with flush-to-zero.
 *
 * Uses PTX rcp.approx.ftz.f32 for ~2x faster reciprocal vs __fdividef(1.0f, x).
 * Accuracy is sufficient for scale factor computation in quantization.
 */
__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

/**
 * @brief Compute global encode scale for two-level NVFP4 quantization.
 *
 * Maps the tensor's dynamic range to the FP4 range using FP8 as intermediate.
 * Formula: global_encode_scale = FP8_MAX * FP4_MAX / global_amax
 *
 * Uses rcp_approx_ftz for fast reciprocal computation.
 * This matches TransformerEngine's NVFP4 recipe scaling approach.
 */
__device__ __forceinline__ float compute_global_encode_scale(float global_amax) {
    if (global_amax == 0.0f) {
        return 1.0f;
    }
    // Use fast approximate reciprocal
    float scale = kFP8E4M3Max * kFP4Max * rcp_approx_ftz(fmaxf(global_amax, 1e-10f));
    scale = fminf(scale, 3.4e38f);
    if (scale == 0.0f) {
        return 1.0f;
    }
    return scale;
}

/**
 * @brief Compute per-block decode scale with global scaling baked in.
 *
 * decode_scale = (block_amax / FP4_MAX) * global_encode_scale
 *              = block_amax * FP8_MAX / global_amax
 *
 * Uses rcp_approx_ftz for fast reciprocal of FP4_MAX.
 */
__device__ __forceinline__ float compute_decode_scale(float block_amax, float global_encode_scale) {
    // decode_scale = block_amax / FP4_MAX * global_encode_scale
    float decode_scale = block_amax * rcp_approx_ftz(kFP4Max) * global_encode_scale;
    return fminf(decode_scale, kFP8E4M3Max);  // Clamp to FP8 max
}

/**
 * @brief Compute encode scale from decode scale for quantization.
 *
 * Uses rcp_approx_ftz for fast reciprocal computation.
 */
__device__ __forceinline__ float compute_encode_scale(float decode_scale, float global_decode_scale) {
    float denom = decode_scale * global_decode_scale;
    return (denom != 0.0f) ? fminf(rcp_approx_ftz(denom), 3.4e38f) : 0.0f;
}

/**
 * @brief Convert float scale to UE4M3 format.
 *
 * UE4M3 is an 8-bit unsigned format with 4 exponent bits and 3 mantissa bits.
 * Used as scale factors for CUTLASS NVFP4 GEMM operations.
 *
 * The format has:
 * - Bias: 7 (so exponent field of 7 = 2^0 = 1.0)
 * - No sign bit (unsigned, scales are always positive)
 * - Range: ~5.96e-8 to 480.0
 *
 * @param scale The floating-point scale factor (must be positive).
 * @return UE4M3 encoded value.
 */
__device__ __forceinline__ uint8_t float_to_ue4m3(float scale) {
    if (scale <= 0.0f || !isfinite(scale)) {
        return 0;  // Smallest representable positive
    }

    // Clamp to UE4M3 representable range
    // Max UE4M3 value: exp=14 (2^7), mantissa=0.875 => 2^7 * 1.875 = 240... actually ~480
    // Min UE4M3 value: exp=0 (2^-7), mantissa=0 => 2^-7 ≈ 0.0078
    scale = fminf(fmaxf(scale, 1.0f / 128.0f), 480.0f);

    // Convert to UE4M3 bit pattern
    // We use the FP8 E4M3 hardware conversion but need to handle the unsigned nature
    // UE4M3 has same bit layout as positive E4M3 values
    __nv_fp8_e4m3 fp8_val;
    fp8_val.__x = __nv_cvt_float_to_fp8(scale, __nv_saturation_t::__NV_SATFINITE,
                                         __nv_fp8_interpretation_t::__NV_E4M3);
    return fp8_val.__x;
}

/**
 * @brief Convert UE4M3 to float scale.
 *
 * @param ue4m3 The UE4M3 encoded value.
 * @return Floating-point scale factor.
 */
__device__ __forceinline__ float ue4m3_to_float(uint8_t ue4m3) {
    __nv_fp8_e4m3 fp8_val;
    fp8_val.__x = ue4m3;
    return __half2float(__nv_cvt_fp8_to_halfraw(fp8_val.__x, __nv_fp8_interpretation_t::__NV_E4M3));
}

/**
 * @brief Compute scale swizzled offset for CUTLASS Sm1xxBlkScaledConfig layout.
 *
 * CUTLASS SM120 block-scaled GEMM expects scales in a specific swizzled layout
 * defined by SfKMajorAtom:
 *   Layout<Shape<Shape<_32,_4>, Shape<Int<16>, _4>>,
 *         Stride<Stride<_16,_4>, Stride<_0, _1>>>
 *
 * This maps logical (row, scale_col) to physical offset within 128-row atoms.
 * The layout tiles the M dimension into 128-row blocks and K dimension into
 * 4-scale-column blocks, with swizzling within each atom.
 *
 * For K-major layout:
 * - Shape: ((32,4), (16,4)) - 128 rows x 4 scale columns per atom
 * - Stride: ((16,4), (0,1)) - row_in_32 * 16 + row_blk32 * 4 + scale_col
 *
 * @param row Row index in the data matrix.
 * @param scale_col Scale column index (K/16).
 * @param num_scale_cols Total number of scale columns.
 * @return Swizzled offset into scale buffer.
 */
__device__ __forceinline__ size_t nvfp4_cutlass_scale_offset(
    int row, int scale_col, int num_scale_cols)
{
    // CUTLASS Sm1xxBlkScaledConfig atom dimensions
    constexpr int kRowsPerAtom = 128;  // Blk_MN
    constexpr int kColsPerAtom = 4;    // Blk_SF
    constexpr int kAtomSize = kRowsPerAtom * kColsPerAtom;  // 512 scales per atom

    // Which atom are we in?
    int row_atom = row / kRowsPerAtom;
    int col_atom = scale_col / kColsPerAtom;

    // Position within the atom
    int row_in_atom = row % kRowsPerAtom;
    int col_in_atom = scale_col % kColsPerAtom;

    // How many column atoms per row of atoms?
    int atoms_per_row = (num_scale_cols + kColsPerAtom - 1) / kColsPerAtom;

    // Compute atom start offset (row-major order of atoms)
    size_t atom_offset = static_cast<size_t>(row_atom * atoms_per_row + col_atom) * kAtomSize;

    // Within-atom offset using SfKMajorAtom stride pattern: ((16,4), (0,1))
    // Decompose row_in_atom into (row_in_32, row_blk32) where row_blk32 = row_in_atom / 32
    int row_in_32 = row_in_atom % 32;
    int row_blk32 = row_in_atom / 32;  // 0-3

    // SfKMajorAtom mapping: stride<0,0> = 16, stride<0,1> = 4, stride<1,1> = 1
    // Physical offset = row_in_32 * 16 + row_blk32 * 4 + col_in_atom
    size_t intra_offset = row_in_32 * 16 + row_blk32 * 4 + col_in_atom;

    return atom_offset + intra_offset;
}

// ============================================================================
// Scale Layout Reorder Kernels (Linear <-> CUTLASS)
// ============================================================================

__global__ void nvfp4_scales_linear_to_cutlass_kernel(
    uint8_t* __restrict__ out_cutlass,
    const uint8_t* __restrict__ in_linear,
    int rows,
    int num_scale_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * num_scale_cols;
    if (idx >= total) return;

    const int row = idx / num_scale_cols;
    const int scale_col = idx - row * num_scale_cols;
    const size_t off = nvfp4_cutlass_scale_offset(row, scale_col, num_scale_cols);
    out_cutlass[off] = in_linear[idx];
}

__global__ void nvfp4_scales_cutlass_to_linear_kernel(
    uint8_t* __restrict__ out_linear,
    const uint8_t* __restrict__ in_cutlass,
    int rows,
    int num_scale_cols)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * num_scale_cols;
    if (idx >= total) return;

    const int row = idx / num_scale_cols;
    const int scale_col = idx - row * num_scale_cols;
    const size_t off = nvfp4_cutlass_scale_offset(row, scale_col, num_scale_cols);
    out_linear[idx] = in_cutlass[off];
}

/**
 * @brief FP4 E2M1 quantization using CUDA's native FP4 conversion.
 *
 * Uses `__nv_cvt_float_to_fp4(..., __NV_E2M1, cudaRoundNearest)`.
 * Returns the 4-bit storage value in the low nibble.
 */
__device__ __forceinline__ uint8_t quantize_fp4_e2m1_rn(float val) {
    const __nv_fp4_storage_t storage = __nv_cvt_float_to_fp4(val, __NV_E2M1, cudaRoundNearest);
    return static_cast<uint8_t>(storage & 0xF);
}

/**
 * @brief FP4 E2M1 stochastic rounding for gradient quantization.
 *
 * @param val Value in the scaled FP4 domain.
 * @param u Uniform random sample in (0, 1].
 * @return FP4 storage nibble in low 4 bits.
 */
__device__ __forceinline__ uint8_t quantize_fp4_e2m1_sr(float val, float u) {
    if (!isfinite(val)) {
        const float s = signbit(val) ? -1.0f : 1.0f;
        return quantize_fp4_e2m1_rn(s * kFP4Max);
    }

    const float s = (val < 0.0f) ? -1.0f : 1.0f;
    float x = fabsf(val);

    if (x >= kFP4Max) {
        return quantize_fp4_e2m1_rn(s * kFP4Max);
    }

    // Positive representable magnitudes for FP4 E2M1
    float lo = 0.0f, hi = 0.5f;
    if (x < 0.5f) { lo = 0.0f; hi = 0.5f; }
    else if (x < 1.0f) { lo = 0.5f; hi = 1.0f; }
    else if (x < 1.5f) { lo = 1.0f; hi = 1.5f; }
    else if (x < 2.0f) { lo = 1.5f; hi = 2.0f; }
    else if (x < 3.0f) { lo = 2.0f; hi = 3.0f; }
    else if (x < 4.0f) { lo = 3.0f; hi = 4.0f; }
    else { lo = 4.0f; hi = 6.0f; }

    const float denom = hi - lo;
    if (denom <= 0.0f || x == lo) return quantize_fp4_e2m1_rn(s * lo);
    if (x == hi) return quantize_fp4_e2m1_rn(s * hi);

    float p = fminf(fmaxf((x - lo) / denom, 0.0f), 1.0f);
    const float chosen = (u <= p) ? hi : lo;
    return quantize_fp4_e2m1_rn(s * chosen);
}

/**
 * @brief Pack two FP4 values into a single byte.
 */
__device__ __forceinline__ uint8_t pack_fp4(uint8_t val0, uint8_t val1) {
    return (val1 << 4) | (val0 & 0xF);
}

} // anonymous namespace

// ============================================================================
// NVFP4 Block Quantization Kernels (CUTLASS-compatible layout)
// ============================================================================

/**
 * @brief Per-block NVFP4 quantization kernel with CUTLASS-compatible scale layout.
 *
 * Quantizes BF16 input to FP4 E2M1 with UE4M3 block scales in the interleaved
 * layout expected by CUTLASS matmul_cutlass_fp4().
 *
 * @tparam TILE_SIZE Tile size (default 128).
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output UE4M3 block scales in CUTLASS interleaved layout.
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param num_scale_cols Number of scale columns (ceil(K/16)).
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int num_scale_cols)
{
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, M);
    const int col_end = min(col_start + TILE_SIZE, K);

    // Process each 16-element block within this tile
    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block abs_max
        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Phase 2: Compute UE4M3 scale
        // Scale maps block_amax to FP4 max (6.0)
        // decode_scale = block_amax / kFP4Max
        // For CUTLASS, we store the decode scale directly as UE4M3
        float decode_scale = block_amax / kFP4Max;
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        // Store scale in CUTLASS interleaved layout
        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        // Phase 3: Quantize to FP4
        // quant_scale = kFP4Max / block_amax = 1 / decode_scale
        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float quant_scale = kFP4Max / fmaxf(actual_decode * kFP4Max, 1e-10f);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * quant_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * quant_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block NVFP4 weight quantization kernel with CUTLASS-compatible layout.
 *
 * For weight matrix (N, K) = (OC, C):
 * - Quantizes along the K dimension
 * - Scale layout: (N, ceil(K/16)) with CUTLASS interleaving
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_weight_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int num_scale_cols)
{
    const int tile_row = blockIdx.x;  // Along N dimension
    const int tile_col = blockIdx.y;  // Along K dimension

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, N);
    const int col_end = min(col_start + TILE_SIZE, K);

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;  // N index
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block abs_max
        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Phase 2: Compute and store UE4M3 scale
        float decode_scale = block_amax / kFP4Max;
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        // Phase 3: Quantize to FP4
        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float quant_scale = kFP4Max / fmaxf(actual_decode * kFP4Max, 1e-10f);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * quant_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * quant_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block NVFP4 quantization with stochastic rounding (for gradients).
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_stochastic_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
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

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = block_amax / kFP4Max;
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float quant_scale = kFP4Max / fmaxf(actual_decode * kFP4Max, 1e-10f);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            const float rand0 = curand_uniform(&rng_state);
            const float rand1 = curand_uniform(&rng_state);

            uint8_t fp4_0 = quantize_fp4_e2m1_sr(values[i] * quant_scale, rand0);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_sr(values[i + 1] * quant_scale, rand1) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

// ============================================================================
// Two-Level Scaling Kernels (with global amax)
// ============================================================================

/**
 * @brief Per-block NVFP4 quantization with two-level scaling (CUTLASS layout).
 *
 * Uses global amax to compute global encode scale, which is baked into block scales.
 * This matches the TransformerEngine NVFP4 recipe and allows alpha correction post-GEMM.
 *
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
 * @param[in] global_amax_in Pre-computed global amax (device pointer).
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param num_scale_cols Number of scale columns (ceil(K/16)).
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_cutlass_auto_kernel(
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

    // Load global scales once per block
    __shared__ float s_global_encode_scale;
    __shared__ float s_global_decode_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        const float enc = compute_global_encode_scale(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block abs_max
        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Phase 2: Compute UE4M3 scale with global scaling baked in
        float decode_scale = compute_decode_scale(block_amax, global_encode_scale);
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        // Store scale in CUTLASS interleaved layout
        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        // Phase 3: Quantize to FP4
        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float encode_scale = compute_encode_scale(actual_decode, global_decode_scale);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block NVFP4 stochastic quantization with two-level scaling.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_stochastic_cutlass_auto_kernel(
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

    __shared__ float s_global_encode_scale;
    __shared__ float s_global_decode_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        const float enc = compute_global_encode_scale(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = compute_decode_scale(block_amax, global_encode_scale);
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float encode_scale = compute_encode_scale(actual_decode, global_decode_scale);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            const float rand0 = curand_uniform(&rng_state);
            const float rand1 = curand_uniform(&rng_state);

            uint8_t fp4_0 = quantize_fp4_e2m1_sr(values[i] * encode_scale, rand0);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_sr(values[i + 1] * encode_scale, rand1) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block NVFP4 weight quantization with two-level scaling (CUTLASS layout).
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_weight_cutlass_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int num_scale_cols)
{
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, N);
    const int col_end = min(col_start + TILE_SIZE, K);

    __shared__ float s_global_encode_scale;
    __shared__ float s_global_decode_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        const float enc = compute_global_encode_scale(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int global_row = row_start + local_row;
        const int block_col_start = col_start + local_block * kNVFP4BlockSize;
        const int block_col_end = min(block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = block_col_end - block_col_start;

        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = compute_decode_scale(block_amax, global_encode_scale);
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(global_row, scale_col, num_scale_cols);
        block_scales[scale_offset] = ue4m3_scale;

        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float encode_scale = compute_encode_scale(actual_decode, global_decode_scale);

        const int out_col_start = block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block NVFP4 weight quantization producing a transposed output layout.
 *
 * Input weights are provided as row-major BF16 (N, K). The output FP4 matrix is written
 * as row-major (K, N) (packed as (K, N/2) bytes). This matches the layout needed for
 * backward dgrad GEMMs without an explicit BF16 transpose.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_nvfp4_weight_cutlass_transpose_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int N, int K,
    int num_scale_cols_out)
{
    // Output matrix shape: (K, N)
    const int tile_row = blockIdx.x;  // rows over K
    const int tile_col = blockIdx.y;  // cols over N

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, K);
    const int col_end = min(col_start + TILE_SIZE, N);

    __shared__ float s_global_encode_scale;
    __shared__ float s_global_decode_scale;
    if (threadIdx.x == 0) {
        const float ga = *global_amax_in;
        const float enc = compute_global_encode_scale(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_rows = row_end - row_start;
    const int num_blocks_per_row = div_ceil(col_end - col_start, kNVFP4BlockSize);
    const int total_blocks = num_rows * num_blocks_per_row;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int local_row = block_idx / num_blocks_per_row;
        const int local_block = block_idx % num_blocks_per_row;

        const int out_row = row_start + local_row;  // 0..K-1
        const int out_block_col_start = col_start + local_block * kNVFP4BlockSize;  // 0..N-1
        const int out_block_col_end = min(out_block_col_start + kNVFP4BlockSize, col_end);
        const int block_width = out_block_col_end - out_block_col_start;

        float block_amax = 0.0f;
        float values[kNVFP4BlockSize];

        // Read transposed: out(out_row, out_col) = in(out_col, out_row)
        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            const int in_row = out_block_col_start + i;  // 0..N-1
            const int in_col = out_row;                  // 0..K-1
            float val = (float)in[in_row * K + in_col];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = compute_decode_scale(block_amax, global_encode_scale);
        uint8_t ue4m3_scale = float_to_ue4m3(fmaxf(decode_scale, 1e-10f));

        // Store scale for output row/scale_col
        const int scale_col = (col_start / kNVFP4BlockSize) + local_block;
        const size_t scale_offset = nvfp4_cutlass_scale_offset(out_row, scale_col, num_scale_cols_out);
        block_scales[scale_offset] = ue4m3_scale;

        float actual_decode = ue4m3_to_float(ue4m3_scale);
        float encode_scale = compute_encode_scale(actual_decode, global_decode_scale);

        const int out_byte_col_start = out_block_col_start / kNVFP4ValuesPerByte;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            out_fp4[out_row * (N / 2) + out_byte_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

// Global namespace to match kernels.h declarations

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Compute the size of NVFP4 scale tensor for CUTLASS layout.
 *
 * @param rows Number of rows in the data matrix.
 * @param cols Number of columns in the data matrix.
 * @return Number of UE4M3 scale elements needed.
 */
size_t compute_nvfp4_cutlass_scale_size(int rows, int cols) {
    int num_scale_cols = div_ceil(cols, kNVFP4BlockSize);
    // Align to atom dimensions for efficient access
    int aligned_rows = div_ceil(rows, kNVFP4TileDim) * kNVFP4TileDim;
    int aligned_cols = div_ceil(num_scale_cols, 4) * 4;
    return static_cast<size_t>(aligned_rows) * aligned_cols;
}

void nvfp4_scales_linear_to_cutlass(
    uint8_t* out_cutlass,
    const uint8_t* in_linear,
    int rows,
    int cols,
    cudaStream_t stream)
{
    const int num_scale_cols = div_ceil(cols, kNVFP4BlockSize);
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(num_scale_cols);
    if (total == 0) return;

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    nvfp4_scales_linear_to_cutlass_kernel<<<blocks, kThreads, 0, stream>>>(
        out_cutlass, in_linear, rows, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

void nvfp4_scales_cutlass_to_linear(
    uint8_t* out_linear,
    const uint8_t* in_cutlass,
    int rows,
    int cols,
    cudaStream_t stream)
{
    const int num_scale_cols = div_ceil(cols, kNVFP4BlockSize);
    const size_t total = static_cast<size_t>(rows) * static_cast<size_t>(num_scale_cols);
    if (total == 0) return;

    constexpr int kThreads = 256;
    const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
    nvfp4_scales_cutlass_to_linear_kernel<<<blocks, kThreads, 0, stream>>>(
        out_linear, in_cutlass, rows, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 quantization with CUTLASS-compatible scale layout.
 *
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_nvfp4_cutlass(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const nv_bfloat16* in,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_cutlass_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, in, M, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 weight quantization with CUTLASS-compatible layout.
 *
 * @param[out] out_fp4 Output packed FP4 data (N, K/2 bytes).
 * @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
 * @param[in] in Input BF16 weight data (N, K) row-major.
 * @param N Number of rows (out_channels).
 * @param K Number of columns (in_channels).
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_nvfp4_weight_cutlass(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(N, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_weight_cutlass_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, in, N, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 quantization with stochastic rounding.
 */
void quantize_nvfp4_stochastic_cutlass(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const nv_bfloat16* in,
    int M, int K,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_stochastic_cutlass_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, in, M, K, num_scale_cols, seed);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Two-Level Scaling Host Launchers (with global amax)
// ============================================================================

/**
 * @brief Host launcher for NVFP4 quantization with two-level scaling.
 *
 * Computes global amax first, then quantizes with global scaling baked into block scales.
 * The global_amax output is needed for alpha correction after GEMM.
 *
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output UE4M3 block scales in CUTLASS layout.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_nvfp4_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax (needed for alpha scaling post-GEMM)
    abs_max(global_amax, in, (long)M * K, dp, stream);

    // 2) Quantize with on-device auto-scaling (CUDA graph capture safe)
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 quantization using a pre-computed global amax.
 *
 * This skips the abs_max reduction and directly uses `global_amax` for two-level scaling.
 * Intended to reuse amax computed in preceding fused ops (e.g., RMSNorm/SwiGLU) to
 * reduce extra memory traffic on fast GPUs.
 */
void quantize_nvfp4_cutlass_from_amax(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    (void)dp;
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 stochastic quantization with two-level scaling.
 */
void quantize_nvfp4_stochastic_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax
    abs_max(global_amax, in, (long)M * K, dp, stream);

    // 2) Quantize with stochastic rounding + on-device auto-scaling
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_stochastic_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 stochastic quantization using a pre-computed global amax.
 */
void quantize_nvfp4_stochastic_cutlass_from_amax(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    (void)dp;
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(M, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_stochastic_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 weight quantization with two-level scaling.
 */
void quantize_nvfp4_weight_cutlass_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax
    abs_max(global_amax, in, (long)N * K, dp, stream);

    // 2) Quantize with on-device auto-scaling
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(N, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_weight_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 weight quantization using a pre-computed global amax.
 */
void quantize_nvfp4_weight_cutlass_from_amax(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    const float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    (void)dp;
    const int num_scale_cols = div_ceil(K, kNVFP4BlockSize);

    dim3 grid(div_ceil(N, kNVFP4TileDim), div_ceil(K, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_weight_cutlass_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, num_scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for NVFP4 weight quantization producing a transposed (K, N) output layout.
 *
 * This avoids an explicit BF16 transpose when computing dgrad in backward passes.
 */
void quantize_nvfp4_weight_cutlass_transpose_auto_scale(
    uint8_t* out_fp4,
    uint8_t* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax (same for W and W^T)
    abs_max(global_amax, in, (long)N * K, dp, stream);

    // 2) Quantize-and-transpose in one pass
    const int out_cols = N;
    const int out_rows = K;
    const int num_scale_cols_out = div_ceil(out_cols, kNVFP4BlockSize);

    dim3 grid(div_ceil(out_rows, kNVFP4TileDim), div_ceil(out_cols, kNVFP4TileDim));
    const int threads_per_block = 256;

    quantize_nvfp4_weight_cutlass_transpose_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, num_scale_cols_out);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for NVFP4 quantization with CUTLASS layout.
 */
void quantize_nvfp4_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    const Tensor& in,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (block_scales.DType != ETensorDType::BYTE) {
        throw std::runtime_error("quantize_nvfp4_cutlass: block_scales must be BYTE (UE4M3)");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_cutlass: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_cutlass(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        in.get<nv_bfloat16>(),
        M, K, dp, stream);
}

/**
 * @brief Tensor-based wrapper for NVFP4 weight quantization with CUTLASS layout.
 */
void quantize_nvfp4_weight_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    const Tensor& in,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_weight_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_weight_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (block_scales.DType != ETensorDType::BYTE) {
        throw std::runtime_error("quantize_nvfp4_weight_cutlass: block_scales must be BYTE (UE4M3)");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_nvfp4_weight_cutlass: tensors must be 2D");
    }

    const int N = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_weight_cutlass(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        in.get<nv_bfloat16>(),
        N, K, dp, stream);
}

/**
 * @brief Tensor-based wrapper for NVFP4 stochastic quantization.
 */
void quantize_nvfp4_stochastic_cutlass(
    Tensor& out_fp4,
    Tensor& block_scales,
    const Tensor& in,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_nvfp4_stochastic_cutlass: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_nvfp4_stochastic_cutlass: output must be BYTE or FP4_E2M1");
    }
    if (block_scales.DType != ETensorDType::BYTE) {
        throw std::runtime_error("quantize_nvfp4_stochastic_cutlass: block_scales must be BYTE (UE4M3)");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_nvfp4_stochastic_cutlass(
        out_fp4.get<uint8_t>(),
        block_scales.get<uint8_t>(),
        in.get<nv_bfloat16>(),
        M, K, seed, dp, stream);
}

// end of global namespace functions
