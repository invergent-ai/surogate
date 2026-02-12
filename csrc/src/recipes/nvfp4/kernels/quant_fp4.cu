// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file quant_fp4.cu
 * @brief CUDA kernels for FP4 E2M1 quantization with two-level block scaling.
 *
 * FP4 E2M1 format:
 * - 2-bit exponent, 1-bit mantissa
 * - Representable values: +/-{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
 * - Max value: 6.0 (vs FP8 E4M3 max of 448.0)
 * - Storage: 2 values packed per uint8_t byte
 *
 * Two-level scaling structure:
 * - Level 1 (block-wise): FP8 E4M3 scale per 16 consecutive values
 * - Level 2 (per-tensor): FP32 global scale (amax_rowwise, amax_columnwise)
 *
 * Block scale shape: (round_to_multiple(M, 128), round_to_multiple(ceil(K/16), 4))
 * with F8_128x4 tensor reordering for cuBLAS compatibility.
 *
 * Requires: CUDA 12.8+, Blackwell GPU (SM100+) for native FP4 PTX instructions.
 */

#include "kernels/kernel_utils.cuh"
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

// FP4 E2M1 constants
constexpr float FP4_MAX = 6.0f;
constexpr float FP8_E4M3_MAX = 448.0f;
constexpr int FP4_BLOCK_SIZE = 16;      // Elements per block scale
constexpr int FP4_TILE_DIM = 128;       // Tile dimension for kernel
constexpr int FP4_VALUES_PER_BYTE = 2;  // 2 FP4 values packed per byte

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
 * @brief Convert 8 float32 values to 8 FP4 E2M1 values packed into a uint32_t.
 *
 * Uses PTX cvt.rn.satfinite.e2m1x2.f32 instruction which converts 2 floats to
 * 2 e2m1 values in a single byte. Four such conversions produce a uint32_t
 * with 8 packed FP4 values.
 *
 * This is significantly faster than scalar __nv_cvt_float_to_fp4 calls:
 * - Single instruction converts 2 values vs 2 separate conversions
 * - Direct byte assembly avoids intermediate storage
 * - Better instruction-level parallelism
 *
 * @param array Input array of 8 float values to quantize.
 * @return uint32_t with 8 packed FP4 E2M1 values.
 */
__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(float (&array)[8]) {
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
          "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7]));
    return val;
}

/**
 * @brief Compute global encode scale for FP4 quantization.
 *
 * The global encode scale maps the tensor's dynamic range to the FP4 range.
 * Formula: global_encode_scale = fp8_max * fp4_max / global_amax
 *
 * Uses rcp_approx_ftz for fast reciprocal computation.
 *
 * @param global_amax The absolute maximum value of the tensor.
 * @return The global encode scale.
 */
__device__ __forceinline__ float compute_global_encode_scale_fp4(float global_amax) {
    // Handle zero amax
    if (global_amax == 0.0f) {
        return 1.0f;
    }
    // Use fast approximate reciprocal: scale = FP8_MAX * FP4_MAX / amax
    float scale = FP8_E4M3_MAX * FP4_MAX * rcp_approx_ftz(fmaxf(global_amax, 1e-10f));
    // Clamp to avoid overflow
    scale = fminf(scale, 3.4e38f);
    if (scale == 0.0f) {
        return 1.0f;
    }
    return scale;
}

/**
 * @brief Compute per-block decode scale for FP4 quantization.
 *
 * Uses rcp_approx_ftz for fast reciprocal of FP4_MAX.
 *
 * @param block_amax The absolute maximum value of the block.
 * @param global_encode_scale The global encode scale.
 * @return The decode scale (stored as FP8 E4M3).
 */
__device__ __forceinline__ float compute_decode_scale_fp4(float block_amax, float global_encode_scale) {
    // decode_scale = block_amax / FP4_MAX * global_encode_scale
    // Use fast reciprocal: 1/6.0 ≈ 0.16667
    float decode_scale = block_amax * rcp_approx_ftz(FP4_MAX) * global_encode_scale;
    return fminf(decode_scale, 3.4e38f);
}

/**
 * @brief Compute encode scale from decode scale for quantization.
 *
 * Uses rcp_approx_ftz for fast reciprocal computation.
 *
 * @param decode_scale The decode scale.
 * @param global_decode_scale The global decode scale (1 / global_encode_scale).
 * @return The encode scale for quantization.
 */
__device__ __forceinline__ float compute_encode_scale_fp4(float decode_scale, float global_decode_scale) {
    // encode_scale = 1 / (decode_scale * global_decode_scale)
    float denom = decode_scale * global_decode_scale;
    return (denom != 0.0f) ? fminf(rcp_approx_ftz(denom), 3.4e38f) : 0.0f;
}

/**
 * @brief FP4 E2M1 quantization using CUDA's native FP4 conversion.
 *
 * Uses `__nv_cvt_float_to_fp4(..., __NV_E2M1, cudaRoundNearest)` to match the FP4 encoding
 * expected by cuDNN (CUDNN_DATA_FP4_E2M1). Returns the 4-bit storage value in the low nibble.
 */
__device__ __forceinline__ uint8_t quantize_fp4_e2m1_rn(float val) {
    const __nv_fp4_storage_t storage = __nv_cvt_float_to_fp4(val, __NV_E2M1, cudaRoundNearest);
    return static_cast<uint8_t>(storage & 0xF);
}

/**
 * @brief FP4 E2M1 stochastic rounding.
 *
 * Implements unbiased stochastic rounding between the two nearest representable FP4 values.
 * This is used for gradient quantization in the NVFP4 training recipe to reduce bias.
 *
 * @param val Value in the *scaled* FP4 domain (i.e. after applying encode_scale).
 * @param u   Uniform random sample in (0, 1] (curand_uniform output).
 * @return FP4 storage nibble in low 4 bits.
 */
__device__ __forceinline__ uint8_t quantize_fp4_e2m1_sr(float val, float u) {
    // Handle NaNs/Infs by saturating.
    if (!isfinite(val)) {
        const float s = signbit(val) ? -1.0f : 1.0f;
        return quantize_fp4_e2m1_rn(s * FP4_MAX);
    }

    const float s = (val < 0.0f) ? -1.0f : 1.0f;
    float x = fabsf(val);

    // Saturate outside representable range.
    if (x >= FP4_MAX) {
        return quantize_fp4_e2m1_rn(s * FP4_MAX);
    }

    // Positive representable magnitudes for FP4 E2M1.
    // (Matches CUDA's __NV_E2M1 interpretation.)
    float lo = 0.0f;
    float hi = 0.5f;
    if (x < 0.5f) {
        lo = 0.0f; hi = 0.5f;
    } else if (x < 1.0f) {
        lo = 0.5f; hi = 1.0f;
    } else if (x < 1.5f) {
        lo = 1.0f; hi = 1.5f;
    } else if (x < 2.0f) {
        lo = 1.5f; hi = 2.0f;
    } else if (x < 3.0f) {
        lo = 2.0f; hi = 3.0f;
    } else if (x < 4.0f) {
        lo = 3.0f; hi = 4.0f;
    } else {
        lo = 4.0f; hi = 6.0f;
    }

    // If x is exactly representable (or in a degenerate interval), keep it.
    const float denom = hi - lo;
    if (denom <= 0.0f || x == lo) {
        return quantize_fp4_e2m1_rn(s * lo);
    }
    if (x == hi) {
        return quantize_fp4_e2m1_rn(s * hi);
    }

    float p = (x - lo) / denom;  // probability of rounding up to hi
    p = fminf(fmaxf(p, 0.0f), 1.0f);

    // curand_uniform returns (0, 1]; use <= so p=1 maps to hi.
    const float chosen = (u <= p) ? hi : lo;
    return quantize_fp4_e2m1_rn(s * chosen);
}

/**
 * @brief Software FP4 E2M1 dequantization.
 *
 * @param code The 4-bit FP4 E2M1 value.
 * @return The dequantized float value.
 */
__device__ __forceinline__ float dequantize_fp4_e2m1(uint8_t code) {
    // Use CUDA's native FP4 decode to match cuDNN's FP4_E2M1 interpretation.
    return __half2float(__nv_cvt_fp4_to_halfraw(static_cast<__nv_fp4_storage_t>(code & 0xF), __NV_E2M1));
}

/**
 * @brief Pack two FP4 values into a single byte.
 *
 * @param val0 First FP4 value (goes to low nibble).
 * @param val1 Second FP4 value (goes to high nibble).
 * @return Packed byte.
 */
__device__ __forceinline__ uint8_t pack_fp4(uint8_t val0, uint8_t val1) {
    return (val1 << 4) | (val0 & 0xF);
}

/**
 * @brief Unpack two FP4 values from a single byte.
 *
 * @param packed The packed byte.
 * @param[out] val0 First FP4 value (from low nibble).
 * @param[out] val1 Second FP4 value (from high nibble).
 */
__device__ __forceinline__ void unpack_fp4(uint8_t packed, uint8_t& val0, uint8_t& val1) {
    val0 = packed & 0xF;
    val1 = (packed >> 4) & 0xF;
}

/**
 * @brief Compute scale swizzled offset for cuBLAS F8_128x4 format.
 *
 * cuBLAS requires scale factors in a specific swizzled layout:
 * - 512B base blocks (128 rows x 4 columns)
 * - Each base block divided into 4 column blocks (32 rows x 4 columns)
 *
 * @param row_idx Row index in scale matrix.
 * @param col_idx Column index in scale matrix.
 * @param col_length Total columns in scale matrix.
 * @return Swizzled offset.
 */
__device__ __forceinline__ size_t scale_swizzled_offset(size_t row_idx, size_t col_idx, uint32_t col_length) {
    constexpr uint32_t kTotalRowsPerBaseBlock = 128;
    constexpr uint32_t kRowsPerBaseBlockCol = 32;
    constexpr uint32_t kColsPerBaseBlockCol = 4;

    const size_t rb = row_idx / kTotalRowsPerBaseBlock;
    const size_t rem = row_idx % kTotalRowsPerBaseBlock;
    const size_t d4 = rem / kRowsPerBaseBlockCol;
    const size_t d3 = rem % kRowsPerBaseBlockCol;
    const size_t cbg = col_idx / kColsPerBaseBlockCol;
    const size_t d5 = col_idx % kColsPerBaseBlockCol;

    const size_t cbg_cnt = div_ceil(col_length, kColsPerBaseBlockCol);
    return ((rb * cbg_cnt + cbg) * kRowsPerBaseBlockCol + d3) * 16 + d4 * kColsPerBaseBlockCol + d5;
}

} // anonymous namespace

// ============================================================================
// Four Over Six (4/6) Adaptive Block Scaling Constants and Helpers
// ============================================================================

namespace {

// 4/6 tensor scale = 1536 (= 384 * 4) instead of standard 2688 (= 448 * 6)
constexpr float FP4_4O6_TENSOR_SCALE = 384.0f * 4.0f;  // 1536

/**
 * @brief Compute global encode scale for 4/6 quantization.
 * Uses tensor scale 1536 instead of 2688.
 */
__device__ __forceinline__ float compute_global_encode_scale_4o6(float global_amax) {
    if (global_amax == 0.0f) return 1.0f;
    return FP4_4O6_TENSOR_SCALE * rcp_approx_ftz(fmaxf(global_amax, 1e-10f));
}

/**
 * @brief Compute 4/6 decode scale for a block.
 *
 * For max=6: decode_scale = block_amax / 6 * tensor_scale / global_amax = block_amax * 256 / global_amax
 * For max=4: decode_scale = block_amax / 4 * tensor_scale / global_amax = block_amax * 384 / global_amax
 */
__device__ __forceinline__ float compute_decode_scale_4o6(float block_amax, float global_encode_scale, float selected_max) {
    return block_amax * rcp_approx_ftz(selected_max) * global_encode_scale;
}

/**
 * @brief Compute quantization error for a set of values.
 *
 * @param values Input values (16 elements)
 * @param encode_scale Scale to apply before quantization
 * @param decode_scale Scale to apply after dequantization (reconstructs original)
 * @param metric 0=MSE, 1=L1, 2=AbsMax
 * @return Error value
 */
__device__ __forceinline__ float compute_quant_error_4o6(
    const float* values, int count, float encode_scale, float decode_scale, int metric)
{
    float error = 0.0f;
    float max_error = 0.0f;

    for (int i = 0; i < count; ++i) {
        float v = values[i];
        float scaled = v * encode_scale;
        uint8_t fp4 = quantize_fp4_e2m1_rn(scaled);
        float dequant = dequantize_fp4_e2m1(fp4);
        float reconstructed = dequant * decode_scale;
        float diff = fabsf(v - reconstructed);

        if (metric == 0) {  // MSE
            error += diff * diff;
        } else if (metric == 1) {  // L1
            error += diff;
        } else {  // AbsMax
            max_error = fmaxf(max_error, diff);
        }
    }

    return (metric == 2) ? max_error : error;
}

} // anonymous namespace

// ============================================================================
// 4/6 Block Quantization Kernel
// ============================================================================

/**
 * @brief FP4 block quantization kernel with Four Over Six (4/6) adaptive block scaling.
 *
 * For each 16-element block, evaluates both max=4 and max=6 scaling and selects
 * the option with lower quantization error.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_block_4o6_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int scale_cols, int metric)
{
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
        const float enc = compute_global_encode_scale_4o6(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = (enc != 0.0f) ? rcp_approx_ftz(enc) : 1.0f;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int count = block_col_end - block_col_start;

        // Load values and compute block amax
        float values[16];
        float block_amax = 0.0f;
        for (int i = 0; i < 16; ++i) {
            if (i < count) {
                float val = (float)in[global_row * K + block_col_start + i];
                values[i] = val;
                block_amax = fmaxf(block_amax, fabsf(val));
            } else {
                values[i] = 0.0f;
            }
        }

        // Evaluate both max=6 and max=4 options
        float decode_scale_6 = compute_decode_scale_4o6(block_amax, global_encode_scale, 6.0f);
        float decode_scale_4 = compute_decode_scale_4o6(block_amax, global_encode_scale, 4.0f);

        float encode_scale_6 = (decode_scale_6 * global_decode_scale != 0.0f) ?
            rcp_approx_ftz(decode_scale_6 * global_decode_scale) : 0.0f;
        float encode_scale_4 = (decode_scale_4 * global_decode_scale != 0.0f) ?
            rcp_approx_ftz(decode_scale_4 * global_decode_scale) : 0.0f;

        // Reconstruct scales for error computation
        float recon_scale_6 = decode_scale_6 * global_decode_scale;
        float recon_scale_4 = decode_scale_4 * global_decode_scale;

        float error_6 = compute_quant_error_4o6(values, count, encode_scale_6, recon_scale_6, metric);
        float error_4 = compute_quant_error_4o6(values, count, encode_scale_4, recon_scale_4, metric);

        // Select the option with lower error
        bool use_max_4 = (error_4 < error_6);
        float selected_decode_scale = use_max_4 ? decode_scale_4 : decode_scale_6;
        float selected_encode_scale = use_max_4 ? encode_scale_4 : encode_scale_6;

        // Store scale as FP8 E4M3
        __nv_fp8_e4m3 scale_fp8;
        float clamped_scale = fminf(fmaxf(selected_decode_scale, 1.0f / 128.0f), 480.0f);
        scale_fp8.__x = __nv_cvt_float_to_fp8(clamped_scale, __nv_saturation_t::__NV_SATFINITE,
                                              __nv_fp8_interpretation_t::__NV_E4M3);

        const int scale_row = global_row;
        const int scale_col = (col_start / FP4_BLOCK_SIZE) + block_col;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);
        block_scales[scale_offset] = scale_fp8;

        // Quantize values
        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;
        for (int i = 0; i < count; i += 2) {
            float v0 = values[i] * selected_encode_scale;
            float v1 = (i + 1 < count) ? values[i + 1] * selected_encode_scale : 0.0f;
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(v0);
            uint8_t fp4_1 = (i + 1 < count) ? quantize_fp4_e2m1_rn(v1) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

// ============================================================================
// FP4 Block Quantization Kernels
// ============================================================================

/**
 * @brief Per-block FP4 quantization kernel.
 *
 * Each CUDA block processes one (BLOCK_SIZE x BLOCK_SIZE) tile.
 * Within each tile, computes per-16-element block scales and quantizes to FP4.
 *
 * @tparam TILE_SIZE Tile size (default 128).
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output FP8 E4M3 block scales.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param scale_cols Number of columns in scale tensor.
 * @param global_encode_scale Pre-computed global encode scale.
 * @param global_decode_scale Pre-computed global decode scale.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_block_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int scale_cols,
    float global_encode_scale, float global_decode_scale)
{
    // Each CUDA block handles one (TILE_SIZE x TILE_SIZE) tile
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    // Tile boundaries
    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, M);
    const int col_end = min(col_start + TILE_SIZE, K);

    // Shared memory for block-level amax tracking
    __shared__ float s_tile_amax;
    if (threadIdx.x == 0) {
        s_tile_amax = 0.0f;
    }
    __syncthreads();

    // Process each 16-element block within this tile
    // Each thread processes multiple blocks
    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    float thread_amax = 0.0f;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block amax
        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Track tile amax
        thread_amax = fmaxf(thread_amax, block_amax);

        // Phase 2: Compute and store block scale
        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        // Store scale as FP8 E4M3
        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);

        // Compute scale position (with swizzling for cuBLAS compatibility)
        const int scale_row = global_row;
        const int scale_col = (col_start / FP4_BLOCK_SIZE) + block_col;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);
        block_scales[scale_offset] = scale_fp8;

        // Phase 3: Quantize and pack FP4 values
        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            uint8_t packed = pack_fp4(fp4_0, fp4_1);
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = packed;
        }
    }

    // Reduce thread amax to tile amax
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
    }

    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_tile_amax), __float_as_uint(thread_amax));
    }
    __syncthreads();

    // Update global amax
    if (threadIdx.x == 0 && s_tile_amax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(s_tile_amax));
    }
}

/**
 * @brief Per-block FP4 quantization kernel with on-device auto scaling.
 *
 * Reads the true global amax (computed separately) and computes the global encode/decode
 * scales inside the kernel. This avoids host synchronization and is CUDA-graph capture safe.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_block_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int scale_cols)
{
    // Each CUDA block handles one (TILE_SIZE x TILE_SIZE) tile
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
        const float enc = compute_global_encode_scale_fp4(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block amax
        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Phase 2: Compute and store block scale
        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                              __nv_fp8_interpretation_t::__NV_E4M3);

        const int scale_row = global_row;
        const int scale_col = (col_start / FP4_BLOCK_SIZE) + block_col;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);
        block_scales[scale_offset] = scale_fp8;

        // Phase 3: Quantize and pack FP4 values
        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief Per-block FP4 dequantization kernel.
 *
 * @tparam TILE_SIZE Tile size used during quantization.
 * @param[out] out Output BF16 data (M, K).
 * @param[in] in_fp4 Input packed FP4 data (M, K/2 bytes).
 * @param[in] block_scales Input FP8 E4M3 block scales.
 * @param M Number of rows.
 * @param K Number of columns.
 * @param scale_cols Number of columns in scale tensor.
 * @param global_decode_scale Global decode scale.
 */
template<int TILE_SIZE = 128>
__global__ void dequantize_fp4_block_kernel(
    nv_bfloat16* __restrict__ out,
    const uint8_t* __restrict__ in_fp4,
    const __nv_fp8_e4m3* __restrict__ block_scales,
    int M, int K, int scale_cols,
    float global_decode_scale)
{
    const long total_elements = (long)M * K;

    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements; idx += gridDim.x * blockDim.x) {
        const int row = idx / K;
        const int col = idx % K;

        // Compute which block this element belongs to
        const int scale_row = row;
        const int scale_col = col / FP4_BLOCK_SIZE;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);

        // Load and convert block scale from FP8 to float
        const __nv_fp8_e4m3 scale_fp8 = block_scales[scale_offset];
        float decode_scale = __half2float(__nv_cvt_fp8_to_halfraw(scale_fp8.__x, __nv_fp8_interpretation_t::__NV_E4M3));
        decode_scale *= global_decode_scale;

        // Load and unpack FP4 value
        const int packed_idx = row * (K / 2) + col / 2;
        const uint8_t packed = in_fp4[packed_idx];
        uint8_t fp4_0, fp4_1;
        unpack_fp4(packed, fp4_0, fp4_1);

        // Select the correct FP4 value
        const uint8_t fp4_val = (col % 2 == 0) ? fp4_0 : fp4_1;

        // Dequantize
        float dequant_val = dequantize_fp4_e2m1(fp4_val) * decode_scale;
        out[idx] = (nv_bfloat16)dequant_val;
    }
}

/**
 * @brief FP4 quantization kernel with stochastic rounding (for backward pass).
 *
 * Uses Philox RNG for stochastic rounding to reduce quantization bias
 * during gradient computation.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_block_stochastic_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int scale_cols,
    float global_encode_scale, float global_decode_scale,
    unsigned int seed)
{
    // Initialize RNG state
    curandStatePhilox4_32_10_t rng_state;
    curand_init(seed, blockIdx.x * blockDim.x + threadIdx.x, 0, &rng_state);

    // Each CUDA block handles one (TILE_SIZE x TILE_SIZE) tile
    const int tile_row = blockIdx.x;
    const int tile_col = blockIdx.y;

    // Tile boundaries
    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, M);
    const int col_end = min(col_start + TILE_SIZE, K);

    __shared__ float s_tile_amax;
    if (threadIdx.x == 0) {
        s_tile_amax = 0.0f;
    }
    __syncthreads();

    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    float thread_amax = 0.0f;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block amax
        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        thread_amax = fmaxf(thread_amax, block_amax);

        // Phase 2: Compute and store block scale
        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);

        const int scale_row = global_row;
        const int scale_col = (col_start / FP4_BLOCK_SIZE) + block_col;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);
        block_scales[scale_offset] = scale_fp8;

        // Phase 3: Quantize with stochastic rounding
        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            const float rand0 = curand_uniform(&rng_state);
            const float rand1 = curand_uniform(&rng_state);

            const float scaled0 = values[i] * encode_scale;
            const float scaled1 = (i + 1 < block_width) ? (values[i + 1] * encode_scale) : 0.0f;

            const uint8_t fp4_0 = quantize_fp4_e2m1_sr(scaled0, rand0);
            const uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_sr(scaled1, rand1) : 0;
            uint8_t packed = pack_fp4(fp4_0, fp4_1);
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = packed;
        }
    }

    // Reduce thread amax
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
    }

    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_tile_amax), __float_as_uint(thread_amax));
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_tile_amax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(s_tile_amax));
    }
}

/**
 * @brief FP4 quantization kernel with stochastic rounding and on-device auto scaling.
 *
 * Like quantize_fp4_block_stochastic_kernel, but reads global amax from device memory and
 * computes global encode/decode scales inside the kernel (capture-safe).
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_block_stochastic_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int scale_cols,
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
        const float enc = compute_global_encode_scale_fp4(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                              __nv_fp8_interpretation_t::__NV_E4M3);

        const int scale_row = global_row;
        const int scale_col = (col_start / FP4_BLOCK_SIZE) + block_col;
        const size_t scale_offset = scale_swizzled_offset(scale_row, scale_col, scale_cols);
        block_scales[scale_offset] = scale_fp8;

        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            const float rand0 = curand_uniform(&rng_state);
            const float rand1 = curand_uniform(&rng_state);

            const float scaled0 = values[i] * encode_scale;
            const float scaled1 = (i + 1 < block_width) ? (values[i + 1] * encode_scale) : 0.0f;

            const uint8_t fp4_0 = quantize_fp4_e2m1_sr(scaled0, rand0);
            const uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_sr(scaled1, rand1) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

// ============================================================================
// Weight Quantization Kernel (Column-Major Scale Layout for cuDNN)
// ============================================================================

/**
 * @brief FP4 weight quantization kernel producing column-major scale layout.
 *
 * For weight matrix (N, K) = (OC, C) in row-major:
 * - cuDNN interprets it as (K, N) column-major
 * - cuDNN expects scale tensor of shape (K/16, N) with column-major swizzle
 *
 * This kernel quantizes along columns (K dimension) rather than rows,
 * producing scales in the transposed layout cuDNN expects.
 *
 * @tparam TILE_SIZE Tile size (default 128).
 * @param[out] out_fp4 Output packed FP4 data (N, K/2 bytes) - same as input layout.
 * @param[out] block_scales Output FP8 E4M3 block scales in (K/16, N) column-major swizzled.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 data (N, K) row-major = (OC, C).
 * @param N Number of rows (OC).
 * @param K Number of columns (C).
 * @param scale_rows Number of rows in scale tensor (K/16 rounded).
 * @param scale_cols Number of cols in scale tensor (N rounded).
 * @param global_encode_scale Pre-computed global encode scale.
 * @param global_decode_scale Pre-computed global decode scale.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_weight_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int scale_rows, int scale_cols,
    float global_encode_scale, float global_decode_scale)
{
    // Each CUDA block handles one (TILE_SIZE x TILE_SIZE) tile
    // Tile indices are in terms of the weight matrix (N, K)
    const int tile_row = blockIdx.x;  // Along N dimension
    const int tile_col = blockIdx.y;  // Along K dimension

    // Tile boundaries in weight matrix
    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, N);
    const int col_end = min(col_start + TILE_SIZE, K);

    // Shared memory for tile amax tracking
    __shared__ float s_tile_amax;
    if (threadIdx.x == 0) {
        s_tile_amax = 0.0f;
    }
    __syncthreads();

    // For weight quantization, we compute scales along the K dimension (columns)
    // Each block of 16 consecutive K values (for a given N) shares one scale
    // Scale layout: (K/16, N) column-major for cuDNN
    const int num_block_rows = (row_end - row_start);  // N direction
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);  // K/16 direction
    const int total_blocks = num_block_rows * num_block_cols;

    float thread_amax = 0.0f;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;  // N index within tile
        const int block_col = block_idx % num_block_cols;  // K/16 index within tile

        const int global_row = row_start + block_row;  // N index
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;  // K start
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        // Phase 1: Compute block amax (along K direction for this N row)
        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        // Track tile amax
        thread_amax = fmaxf(thread_amax, block_amax);

        // Phase 2: Compute and store block scale
        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        // Store scale as FP8 E4M3
        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);

        // Compute scale position for cuDNN's B operand scale tensor.
        // cuDNN expects the B scale tensor to be provided in F8_128x4 reordering. Empirically, treating
        // N as the 128-blocked (row) dimension and K/16 as the 4-blocked (col) dimension matches the
        // required layout.
        const int k_block_idx = (col_start / FP4_BLOCK_SIZE) + block_col;  // K/16 index
        const int n_idx = global_row;                                      // N index
        const size_t scale_offset = scale_swizzled_offset((size_t)n_idx, (size_t)k_block_idx, (uint32_t)scale_rows);
        block_scales[scale_offset] = scale_fp8;

        // Phase 3: Quantize and pack FP4 values (same as regular quantization)
        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;

        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            uint8_t packed = pack_fp4(fp4_0, fp4_1);
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = packed;
        }
    }

    // Reduce thread amax to tile amax
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
    }

    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_tile_amax), __float_as_uint(thread_amax));
    }
    __syncthreads();

    // Update global amax
    if (threadIdx.x == 0 && s_tile_amax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(s_tile_amax));
    }
}

/**
 * @brief FP4 weight quantization kernel with on-device auto scaling (1D block scaling).
 *
 * Reads the true global amax (computed separately) and computes the global encode/decode
 * scales inside the kernel. Does not write to global_amax_in.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_weight_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int scale_rows, int scale_cols)
{
    (void)scale_cols;  // Layout size is implied by swizzle; only total allocation matters.

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
        const float enc = compute_global_encode_scale_fp4(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_block_rows = (row_end - row_start);
    const int num_block_cols = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_block_rows * num_block_cols;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int block_row = block_idx / num_block_cols;
        const int block_col = block_idx % num_block_cols;

        const int global_row = row_start + block_row;
        const int block_col_start = col_start + block_col * FP4_BLOCK_SIZE;
        const int block_col_end = min(block_col_start + FP4_BLOCK_SIZE, col_end);
        const int block_width = block_col_end - block_col_start;

        float block_amax = 0.0f;
        float values[FP4_BLOCK_SIZE];

        #pragma unroll
        for (int i = 0; i < block_width; ++i) {
            float val = (float)in[global_row * K + block_col_start + i];
            values[i] = val;
            block_amax = fmaxf(block_amax, fabsf(val));
        }

        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                              __nv_fp8_interpretation_t::__NV_E4M3);

        const int k_block_idx = (col_start / FP4_BLOCK_SIZE) + block_col;
        const int n_idx = global_row;
        const size_t scale_offset = scale_swizzled_offset((size_t)n_idx, (size_t)k_block_idx, (uint32_t)scale_rows);
        block_scales[scale_offset] = scale_fp8;

        const int out_col_start = block_col_start / FP4_VALUES_PER_BYTE;
        #pragma unroll
        for (int i = 0; i < block_width; i += 2) {
            uint8_t fp4_0 = quantize_fp4_e2m1_rn(values[i] * encode_scale);
            uint8_t fp4_1 = (i + 1 < block_width) ? quantize_fp4_e2m1_rn(values[i + 1] * encode_scale) : 0;
            out_fp4[global_row * (K / 2) + out_col_start + i / 2] = pack_fp4(fp4_0, fp4_1);
        }
    }
}

/**
 * @brief FP4 weight quantization kernel using 16x16 block scaling (TransformerEngine NVFP4 recipe).
 *
 * We compute one scale per 16x16 block (N-block x K-block) and replicate it across the 16 columns
 * (N dimension) that cuDNN expects individual scales for.
 *
 * Scale tensor layout matches cuDNN's B operand expectation: (K/16, N) with F8_128x4 swizzle.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_weight_2d_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int scale_rows, int scale_cols,
    float global_encode_scale, float global_decode_scale)
{
    const int tile_row = blockIdx.x;  // along N
    const int tile_col = blockIdx.y;  // along K

    const int row_start = tile_row * TILE_SIZE;
    const int col_start = tile_col * TILE_SIZE;
    const int row_end = min(row_start + TILE_SIZE, N);
    const int col_end = min(col_start + TILE_SIZE, K);

    __shared__ float s_tile_amax;
    if (threadIdx.x == 0) {
        s_tile_amax = 0.0f;
    }
    __syncthreads();

    const int num_n_blocks = div_ceil(row_end - row_start, FP4_BLOCK_SIZE);  // 16 rows per block
    const int num_k_blocks = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);  // 16 cols per block
    const int total_blocks = num_n_blocks * num_k_blocks;

    float thread_amax = 0.0f;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int nb = block_idx / num_k_blocks;
        const int kb = block_idx % num_k_blocks;

        const int n0 = row_start + nb * FP4_BLOCK_SIZE;
        const int k0 = col_start + kb * FP4_BLOCK_SIZE;
        const int n1 = min(n0 + FP4_BLOCK_SIZE, row_end);
        const int k1 = min(k0 + FP4_BLOCK_SIZE, col_end);

        // Phase 1: compute 16x16 block amax.
        float block_amax = 0.0f;
        for (int n = n0; n < n1; ++n) {
            const long row_base = (long)n * K;
            #pragma unroll
            for (int k = k0; k < k1; ++k) {
                float v = (float)in[row_base + k];
                block_amax = fmaxf(block_amax, fabsf(v));
            }
        }
        thread_amax = fmaxf(thread_amax, block_amax);

        // Phase 2: compute scales (shared for this 16x16 block).
        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE, __nv_fp8_interpretation_t::__NV_E4M3);

        const int k_block_idx = (k0 / FP4_BLOCK_SIZE);  // K/16 index in the full matrix

        // Write (replicated) scales for all N entries in this 16-row block.
        for (int n = n0; n < n1; ++n) {
            const int n_idx = n;
            const size_t scale_offset = scale_swizzled_offset((size_t)n_idx, (size_t)k_block_idx, (uint32_t)scale_rows);
            block_scales[scale_offset] = scale_fp8;
        }

        // Phase 3: quantize values using the shared scale.
        const int out_col_start = k0 / FP4_VALUES_PER_BYTE;
        for (int n = n0; n < n1; ++n) {
            const long row_base = (long)n * K;
            for (int k = k0; k < k1; k += 2) {
                float v0 = (float)in[row_base + k] * encode_scale;
                float v1 = (k + 1 < k1) ? (float)in[row_base + (k + 1)] * encode_scale : 0.0f;
                uint8_t fp4_0 = quantize_fp4_e2m1_rn(v0);
                uint8_t fp4_1 = (k + 1 < k1) ? quantize_fp4_e2m1_rn(v1) : 0;
                out_fp4[n * (K / 2) + out_col_start + (k - k0) / 2] = pack_fp4(fp4_0, fp4_1);
            }
        }
    }

    // Reduce thread amax to tile amax
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFF, thread_amax, offset));
    }

    if (threadIdx.x % 32 == 0) {
        atomicMax(reinterpret_cast<unsigned int*>(&s_tile_amax), __float_as_uint(thread_amax));
    }
    __syncthreads();

    if (threadIdx.x == 0 && s_tile_amax > 0.0f) {
        atomicMax(reinterpret_cast<unsigned int*>(global_amax), __float_as_uint(s_tile_amax));
    }
}

/**
 * @brief FP4 weight quantization kernel with 16x16 block scaling and on-device auto scaling.
 *
 * Reads the true global amax (computed separately) and computes the global encode/decode
 * scales inside the kernel. Does not write to global_amax_in.
 */
template<int TILE_SIZE = 128>
__global__ void quantize_fp4_weight_2d_auto_kernel(
    uint8_t* __restrict__ out_fp4,
    __nv_fp8_e4m3* __restrict__ block_scales,
    const float* __restrict__ global_amax_in,
    const nv_bfloat16* __restrict__ in,
    int N, int K, int scale_rows, int scale_cols)
{
    (void)scale_cols;

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
        const float enc = compute_global_encode_scale_fp4(ga);
        s_global_encode_scale = enc;
        s_global_decode_scale = 1.0f / enc;
    }
    __syncthreads();

    const float global_encode_scale = s_global_encode_scale;
    const float global_decode_scale = s_global_decode_scale;

    const int num_n_blocks = div_ceil(row_end - row_start, FP4_BLOCK_SIZE);
    const int num_k_blocks = div_ceil(col_end - col_start, FP4_BLOCK_SIZE);
    const int total_blocks = num_n_blocks * num_k_blocks;

    for (int block_idx = threadIdx.x; block_idx < total_blocks; block_idx += blockDim.x) {
        const int nb = block_idx / num_k_blocks;
        const int kb = block_idx % num_k_blocks;

        const int n0 = row_start + nb * FP4_BLOCK_SIZE;
        const int k0 = col_start + kb * FP4_BLOCK_SIZE;
        const int n1 = min(n0 + FP4_BLOCK_SIZE, row_end);
        const int k1 = min(k0 + FP4_BLOCK_SIZE, col_end);

        float block_amax = 0.0f;
        for (int n = n0; n < n1; ++n) {
            const long row_base = (long)n * K;
            #pragma unroll
            for (int k = k0; k < k1; ++k) {
                float v = (float)in[row_base + k];
                block_amax = fmaxf(block_amax, fabsf(v));
            }
        }

        float decode_scale = compute_decode_scale_fp4(block_amax, global_encode_scale);
        float encode_scale = compute_encode_scale_fp4(decode_scale, global_decode_scale);

        __nv_fp8_e4m3 scale_fp8;
        scale_fp8.__x = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                              __nv_fp8_interpretation_t::__NV_E4M3);

        const int k_block_idx = (k0 / FP4_BLOCK_SIZE);
        for (int n = n0; n < n1; ++n) {
            const size_t scale_offset = scale_swizzled_offset((size_t)n, (size_t)k_block_idx, (uint32_t)scale_rows);
            block_scales[scale_offset] = scale_fp8;
        }

        const int out_col_start = k0 / FP4_VALUES_PER_BYTE;
        for (int n = n0; n < n1; ++n) {
            const long row_base = (long)n * K;
            for (int k = k0; k < k1; k += 2) {
                float v0 = (float)in[row_base + k] * encode_scale;
                float v1 = (k + 1 < k1) ? (float)in[row_base + (k + 1)] * encode_scale : 0.0f;
                uint8_t fp4_0 = quantize_fp4_e2m1_rn(v0);
                uint8_t fp4_1 = (k + 1 < k1) ? quantize_fp4_e2m1_rn(v1) : 0;
                out_fp4[n * (K / 2) + out_col_start + (k - k0) / 2] = pack_fp4(fp4_0, fp4_1);
            }
        }
    }
}

// ============================================================================
// Host Launcher Functions
// ============================================================================

/**
 * @brief Host launcher for FP4 block quantization.
 *
 * Quantizes a BF16 tensor to FP4 E2M1 with two-level block scaling.
 *
 * @param[out] out_fp4 Output packed FP4 data (M, K/2 bytes).
 * @param[out] block_scales Output FP8 E4M3 block scales.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 data (M, K).
 * @param M Number of rows.
 * @param K Number of columns.
 * @param global_encode_scale Pre-computed global encode scale.
 * @param global_decode_scale Pre-computed global decode scale.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_fp4_block(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    float global_encode_scale,
    float global_decode_scale,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // Calculate scale tensor dimensions
    const int scale_rows = div_ceil(M, FP4_TILE_DIM);
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;  // Align to 4

    // Launch one CUDA block per tile
    dim3 grid(scale_rows, div_ceil(K, FP4_TILE_DIM));
    const int threads_per_block = 256;

    // Initialize global amax
    CUDA_CHECK(cudaMemsetAsync(global_amax, 0, sizeof(float), stream));

    quantize_fp4_block_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, scale_cols,
        global_encode_scale, global_decode_scale);
    CUDA_CHECK(cudaGetLastError());
}

// Forward declaration for abs_max (from quant.cu)
void abs_max(float* scale, const nv_bfloat16* in, long N, const cudaDeviceProp& dp, cudaStream_t stream);

/**
 * @brief Host launcher for FP4 block quantization with proper two-stage scaling.
 *
 * Implements TransformerEngine's NVFP4 scaling approach:
 * 1. First pass: Compute global amax of the entire tensor
 * 2. Compute global_encode_scale = FP8_MAX * FP4_MAX / global_amax
 * 3. Second pass: Quantize with block scales that incorporate global_encode_scale
 *
 * This ensures block decode scales (stored as FP8 E4M3) are in a good numerical range.
 * After cuDNN matmul, apply alpha correction: alpha = (amax_a * amax_b) / (FP4_MAX^2 * FP8_MAX^2)
 *
 * IMPORTANT: The global_amax output contains the TRUE per-tensor amax (computed in step 1),
 * which is needed for alpha scaling after matmul. The quantization kernel computes
 * a separate tile-reduced amax internally but does NOT overwrite global_amax.
 */
void quantize_fp4_block_auto_scale(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax (needed for alpha scaling post-matmul).
    abs_max(global_amax, in, (long)M * K, dp, stream);

    // 2) Quantize using on-device auto-scaling (capture-safe; no host sync).
    const int scale_rows = div_ceil(M, FP4_TILE_DIM);
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;

    dim3 grid(scale_rows, div_ceil(K, FP4_TILE_DIM));
    constexpr int threads_per_block = 256;

    quantize_fp4_block_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for FP4 block quantization with Four Over Six (4/6) adaptive block scaling.
 *
 * Uses tensor scale 1536 (= 384 * 4) instead of standard 2688 (= 448 * 6).
 * For dequantization, use global_decode_scale = amax / 1536.
 */
void quantize_fp4_block_4o6_auto_scale(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    int metric,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) Compute true global amax
    abs_max(global_amax, in, (long)M * K, dp, stream);

    // 2) Quantize using 4/6 adaptive scaling
    const int scale_rows = div_ceil(M, FP4_TILE_DIM);
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;

    dim3 grid(scale_rows, div_ceil(K, FP4_TILE_DIM));
    constexpr int threads_per_block = 256;

    quantize_fp4_block_4o6_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, scale_cols, metric);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for FP4 block dequantization.
 */
void dequantize_fp4_block(
    nv_bfloat16* out,
    const uint8_t* in_fp4,
    const __nv_fp8_e4m3* block_scales,
    float global_decode_scale,
    int M, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;

    const int threads_per_block = 256;
    const int num_blocks = std::max(2 * dp.multiProcessorCount,
                                     (int)((M * (long)K + threads_per_block - 1) / threads_per_block));

    dequantize_fp4_block_kernel<128><<<num_blocks, threads_per_block, 0, stream>>>(
        out, in_fp4, block_scales, M, K, scale_cols, global_decode_scale);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Kernel: scatter row-major FP8 scales into F8_128x4 swizzled layout.
 */
__global__ void swizzle_fp8_scales_kernel(
    __nv_fp8_e4m3* __restrict__ out_swizzled,
    const __nv_fp8_e4m3* __restrict__ in_rowmajor,
    int scale_rows, int scale_cols)
{
    const long total = (long)scale_rows * scale_cols;
    for (long idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (long)gridDim.x * blockDim.x) {
        const int row = idx / scale_cols;
        const int col = idx % scale_cols;
        const size_t swizzled = scale_swizzled_offset(row, col, scale_cols);
        out_swizzled[swizzled] = in_rowmajor[idx];
    }
}

/**
 * @brief Swizzle FP8 block scales from row-major to F8_128x4 layout (in-place).
 */
void swizzle_fp8_scales_rowmajor_to_f8_128x4(
    __nv_fp8_e4m3* scales,
    int scale_rows,
    int scale_cols,
    cudaStream_t stream)
{
    const long total = (long)scale_rows * scale_cols;
    if (total == 0) return;

    // Allocate temp buffer for the row-major source
    const size_t bytes = total * sizeof(__nv_fp8_e4m3);
    __nv_fp8_e4m3* temp = nullptr;
    CUDA_CHECK(cudaMalloc(&temp, bytes));
    CUDA_CHECK(cudaMemcpyAsync(temp, scales, bytes, cudaMemcpyDeviceToDevice, stream));

    const int threads = 256;
    const int blocks = std::min((int)((total + threads - 1) / threads), 1024);
    swizzle_fp8_scales_kernel<<<blocks, threads, 0, stream>>>(scales, temp, scale_rows, scale_cols);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize before freeing temp
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(temp));
}

/**
 * @brief Host launcher for FP4 block quantization with stochastic rounding.
 */
void quantize_fp4_block_stochastic(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    float global_encode_scale,
    float global_decode_scale,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int scale_rows = div_ceil(M, FP4_TILE_DIM);
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;

    dim3 grid(scale_rows, div_ceil(K, FP4_TILE_DIM));
    const int threads_per_block = 256;

    CUDA_CHECK(cudaMemsetAsync(global_amax, 0, sizeof(float), stream));

    quantize_fp4_block_stochastic_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, scale_cols,
        global_encode_scale, global_decode_scale, seed);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_fp4_block_stochastic_auto_scale(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int M, int K,
    unsigned int seed,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) True global amax for alpha scaling.
    abs_max(global_amax, in, (long)M * K, dp, stream);

    // 2) Quantize with stochastic rounding + on-device auto scaling (capture-safe).
    const int scale_rows = div_ceil(M, FP4_TILE_DIM);
    const int scale_cols = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;
    dim3 grid(scale_rows, div_ceil(K, FP4_TILE_DIM));
    constexpr int threads_per_block = 256;

    quantize_fp4_block_stochastic_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, scale_cols, seed);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for FP4 weight quantization with column-major scale layout.
 *
 * Quantizes a weight matrix to FP4 E2M1 with scales in the transposed layout
 * that cuDNN expects for the B operand in FP4 matmul.
 *
 * For weight matrix (N, K) = (OC, C):
 * - cuDNN interprets it as (K, N) column-major
 * - Scale tensor shape: (K/16, N) with F8_128x4 swizzle
 *
 * @param[out] out_fp4 Output packed FP4 data (N, K/2 bytes).
 * @param[out] block_scales Output FP8 E4M3 block scales in (K/16, N) column-major swizzled.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 weight data (N, K) row-major.
 * @param N Number of rows (out_channels).
 * @param K Number of columns (in_channels).
 * @param global_encode_scale Pre-computed global encode scale.
 * @param global_decode_scale Pre-computed global decode scale.
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_fp4_weight(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    float global_encode_scale,
    float global_decode_scale,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // Scale tensor dimensions: (K/16, N) for cuDNN column-major layout
    // K/16 is the row dimension (along K blocks), N is the column dimension
    const int scale_rows = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;  // K/16 aligned to 4
    const int scale_cols = div_ceil(N, FP4_TILE_DIM) * FP4_TILE_DIM;      // N aligned to 128

    // Launch one CUDA block per (N, K) tile
    dim3 grid(div_ceil(N, FP4_TILE_DIM), div_ceil(K, FP4_TILE_DIM));
    const int threads_per_block = 256;

    // Initialize global amax
    CUDA_CHECK(cudaMemsetAsync(global_amax, 0, sizeof(float), stream));

    quantize_fp4_weight_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, scale_rows, scale_cols,
        global_encode_scale, global_decode_scale);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_fp4_weight_2d(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    float global_encode_scale,
    float global_decode_scale,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int scale_rows = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;  // K/16 aligned to 4
    const int scale_cols = div_ceil(N, FP4_TILE_DIM) * FP4_TILE_DIM;      // N aligned to 128

    dim3 grid(div_ceil(N, FP4_TILE_DIM), div_ceil(K, FP4_TILE_DIM));
    const int threads_per_block = 256;

    CUDA_CHECK(cudaMemsetAsync(global_amax, 0, sizeof(float), stream));

    quantize_fp4_weight_2d_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, scale_rows, scale_cols,
        global_encode_scale, global_decode_scale);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for FP4 weight quantization with proper two-stage scaling.
 *
 * Like quantize_fp4_block_auto_scale, but produces scales in the transposed
 * layout cuDNN expects for the B operand.
 *
 * @param[out] out_fp4 Output packed FP4 data (N, K/2 bytes).
 * @param[out] block_scales Output FP8 E4M3 block scales in (K/16, N) swizzled.
 * @param[out] global_amax Output per-tensor absolute maximum.
 * @param[in] in Input BF16 weight data (N, K) row-major.
 * @param N Number of rows (out_channels).
 * @param K Number of columns (in_channels).
 * @param dp CUDA device properties.
 * @param stream CUDA stream.
 */
void quantize_fp4_weight_auto_scale(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) True global amax for alpha scaling.
    abs_max(global_amax, in, (long)N * K, dp, stream);

    // 2) Quantize using on-device auto scaling (capture-safe; no host sync).
    const int scale_rows = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;  // K/16 aligned to 4
    const int scale_cols = div_ceil(N, FP4_TILE_DIM) * FP4_TILE_DIM;      // N aligned to 128

    dim3 grid(div_ceil(N, FP4_TILE_DIM), div_ceil(K, FP4_TILE_DIM));
    constexpr int threads_per_block = 256;

    quantize_fp4_weight_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, scale_rows, scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

void quantize_fp4_weight_2d_auto_scale(
    uint8_t* out_fp4,
    __nv_fp8_e4m3* block_scales,
    float* global_amax,
    const nv_bfloat16* in,
    int N, int K,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    // 1) True global amax for alpha scaling.
    abs_max(global_amax, in, (long)N * K, dp, stream);

    // 2) Quantize using 16x16 weight scaling + on-device auto scaling (capture-safe).
    const int scale_rows = div_ceil(div_ceil(K, FP4_BLOCK_SIZE), 4) * 4;
    const int scale_cols = div_ceil(N, FP4_TILE_DIM) * FP4_TILE_DIM;

    dim3 grid(div_ceil(N, FP4_TILE_DIM), div_ceil(K, FP4_TILE_DIM));
    constexpr int threads_per_block = 256;

    quantize_fp4_weight_2d_auto_kernel<128><<<grid, threads_per_block, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, N, K, scale_rows, scale_cols);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Tensor-based Wrapper Functions
// ============================================================================

/**
 * @brief Tensor-based wrapper for FP4 block quantization.
 */
void quantize_fp4_block(
    Tensor& out_fp4,
    Tensor& block_scales,
    Tensor& global_amax,
    const Tensor& in,
    float global_encode_scale,
    float global_decode_scale,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (in.DType != ETensorDType::BF16) {
        throw std::runtime_error("quantize_fp4_block: input must be BF16");
    }
    if (out_fp4.DType != ETensorDType::BYTE && out_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("quantize_fp4_block: output must be BYTE or FP4_E2M1");
    }
    if (block_scales.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("quantize_fp4_block: block_scales must be FP8_E4M3");
    }
    if (in.Rank != 2 || out_fp4.Rank != 2) {
        throw std::runtime_error("quantize_fp4_block: tensors must be 2D");
    }

    const int M = in.Sizes[0];
    const int K = in.Sizes[1];

    quantize_fp4_block(
        out_fp4.get<uint8_t>(),
        block_scales.get<__nv_fp8_e4m3>(),
        global_amax.get<float>(),
        in.get<nv_bfloat16>(),
        M, K,
        global_encode_scale, global_decode_scale,
        dp, stream);
}

/**
 * @brief Tensor-based wrapper for FP4 block dequantization.
 */
void dequantize_fp4_block(
    Tensor& out,
    const Tensor& in_fp4,
    const Tensor& block_scales,
    float global_decode_scale,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    if (out.DType != ETensorDType::BF16) {
        throw std::runtime_error("dequantize_fp4_block: output must be BF16");
    }
    if (in_fp4.DType != ETensorDType::BYTE && in_fp4.DType != ETensorDType::FP4_E2M1) {
        throw std::runtime_error("dequantize_fp4_block: input must be BYTE or FP4_E2M1");
    }
    if (block_scales.DType != ETensorDType::FP8_E4M3) {
        throw std::runtime_error("dequantize_fp4_block: block_scales must be FP8_E4M3");
    }

    const int M = out.Sizes[0];
    const int K = out.Sizes[1];

    dequantize_fp4_block(
        out.get<nv_bfloat16>(),
        in_fp4.get<uint8_t>(),
        block_scales.get<__nv_fp8_e4m3>(),
        global_decode_scale,
        M, K,
        dp, stream);
}

// ============================================================================
// FP4 Alpha Scaling Kernel (for post-matmul correction)
// ============================================================================

/**
 * @brief Kernel to scale output tensor by alpha factor computed from global amax values.
 *
 * Applies: output *= (global_amax_a * global_amax_b) / (FP4_MAX^2 * FP8_MAX^2)
 *
 * This correction is needed because:
 * - Block scales store: block_amax / FP4_MAX * global_encode_scale
 * - global_encode_scale = FP8_MAX * FP4_MAX / global_amax
 * - After matmul with block scale dequant, we need to multiply by
 *   (amax_a * amax_b) / (FP4_MAX^2 * FP8_MAX^2) to get correct result.
 *
 * @tparam T Output tensor element type (nv_bfloat16 or float)
 */
template<typename T>
__global__ void fp4_alpha_scale_kernel(
    T* __restrict__ out,
    const float* __restrict__ global_amax_a,
    const float* __restrict__ global_amax_b,
    long N)
{
    // FP4_MAX = 6.0, FP8_E4M3_MAX = 448.0
    // factor = 6.0 * 6.0 * 448.0 * 448.0 = 7,225,344.0
    constexpr float factor = 6.0f * 6.0f * 448.0f * 448.0f;

    // Read global amax values (scalar reads, done once per block)
    __shared__ float s_alpha;
    if (threadIdx.x == 0) {
        float amax_a = *global_amax_a;
        float amax_b = *global_amax_b;
        s_alpha = (amax_a * amax_b) / factor;
    }
    __syncthreads();

    float alpha = s_alpha;

    // Scale output elements
    for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float val = (float)out[i];
        out[i] = (T)(val * alpha);
    }
}

/**
 * @brief Host launcher for FP4 alpha scaling.
 *
 * Scales output tensor by alpha = (global_amax_a * global_amax_b) / (6^2 * 448^2)
 *
 * @param out Output tensor to scale in-place (BF16 or FP32)
 * @param global_amax_a Global amax of tensor A (device pointer)
 * @param global_amax_b Global amax of tensor B (device pointer)
 * @param N Number of elements
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void fp4_alpha_scale(
    nv_bfloat16* out,
    const float* global_amax_a,
    const float* global_amax_b,
    long N,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int threads_per_block = 256;
    const int num_blocks = std::min(
        (int)((N + threads_per_block - 1) / threads_per_block),
        2 * dp.multiProcessorCount);

    fp4_alpha_scale_kernel<nv_bfloat16><<<num_blocks, threads_per_block, 0, stream>>>(
        out, global_amax_a, global_amax_b, N);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Host launcher for FP4 alpha scaling (FP32 version).
 */
void fp4_alpha_scale(
    float* out,
    const float* global_amax_a,
    const float* global_amax_b,
    long N,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int threads_per_block = 256;
    const int num_blocks = std::min(
        (int)((N + threads_per_block - 1) / threads_per_block),
        2 * dp.multiProcessorCount);

    fp4_alpha_scale_kernel<float><<<num_blocks, threads_per_block, 0, stream>>>(
        out, global_amax_a, global_amax_b, N);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Tensor-based wrapper for FP4 alpha scaling.
 */
void fp4_alpha_scale(
    Tensor& out,
    const Tensor& global_amax_a,
    const Tensor& global_amax_b,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    long N = (long)out.nelem();

    if (out.DType == ETensorDType::BF16) {
        fp4_alpha_scale(
            out.get<nv_bfloat16>(),
            global_amax_a.get<float>(),
            global_amax_b.get<float>(),
            N, dp, stream);
    } else if (out.DType == ETensorDType::FP32) {
        fp4_alpha_scale(
            out.get<float>(),
            global_amax_a.get<float>(),
            global_amax_b.get<float>(),
            N, dp, stream);
    } else {
        throw std::runtime_error("fp4_alpha_scale: unsupported output dtype (must be BF16 or FP32)");
    }
}

// ============================================================================
// Fused FP4 Alpha Scale + Type Conversion (for B200/datacenter optimization)
// ============================================================================

/**
 * @brief Fused kernel: alpha scale FP32 input and convert to BF16 in one pass.
 *
 * Combines fp4_alpha_scale() + convert_dtype() into a single kernel to:
 * - Eliminate intermediate FP32 storage after matmul
 * - Reduce global memory traffic by 2x (read FP32 once, write BF16 once)
 * - Save one kernel launch overhead
 *
 * This is particularly beneficial on datacenter GPUs (B200/H100) where
 * kernel launch overhead and memory bandwidth are critical.
 *
 * @param[out] out_bf16 Output BF16 tensor
 * @param[in] in_f32 Input FP32 tensor (from matmul)
 * @param[in] global_amax_a Global amax of tensor A (device pointer)
 * @param[in] global_amax_b Global amax of tensor B (device pointer)
 * @param N Number of elements
 */
__global__ void fp4_alpha_scale_convert_kernel(
    nv_bfloat16* __restrict__ out_bf16,
    const float* __restrict__ in_f32,
    const float* __restrict__ global_amax_a,
    const float* __restrict__ global_amax_b,
    long N)
{
    constexpr float factor = 6.0f * 6.0f * 448.0f * 448.0f;

    __shared__ float s_alpha;
    if (threadIdx.x == 0) {
        float amax_a = *global_amax_a;
        float amax_b = *global_amax_b;
        s_alpha = (amax_a * amax_b) / factor;
    }
    __syncthreads();

    const float alpha = s_alpha;

    // Process 4 elements per thread for better memory coalescing
    const long tid = blockIdx.x * blockDim.x + threadIdx.x;
    const long stride = gridDim.x * blockDim.x;

    // Vectorized path: process 4 floats at a time
    const long N4 = N / 4 * 4;
    for (long i = tid * 4; i < N4; i += stride * 4) {
        // Load 4 floats
        float4 vals = *reinterpret_cast<const float4*>(&in_f32[i]);

        // Scale and convert to BF16
        nv_bfloat16 out0 = (nv_bfloat16)(vals.x * alpha);
        nv_bfloat16 out1 = (nv_bfloat16)(vals.y * alpha);
        nv_bfloat16 out2 = (nv_bfloat16)(vals.z * alpha);
        nv_bfloat16 out3 = (nv_bfloat16)(vals.w * alpha);

        // Store as 2 bfloat162 (4 BF16 values)
        *reinterpret_cast<nv_bfloat162*>(&out_bf16[i]) = {out0, out1};
        *reinterpret_cast<nv_bfloat162*>(&out_bf16[i + 2]) = {out2, out3};
    }

    // Handle remainder
    for (long i = N4 + tid; i < N; i += stride) {
        out_bf16[i] = (nv_bfloat16)(in_f32[i] * alpha);
    }
}

/**
 * @brief Fused alpha scale + BF16 conversion for FP4 matmul output.
 *
 * Combines alpha scaling and type conversion into a single kernel,
 * eliminating intermediate storage and reducing memory traffic.
 *
 * @param out_bf16 Output BF16 tensor
 * @param in_f32 Input FP32 tensor (from FP4 matmul)
 * @param global_amax_a Global amax of tensor A (device pointer)
 * @param global_amax_b Global amax of tensor B (device pointer)
 * @param N Number of elements
 * @param dp CUDA device properties
 * @param stream CUDA stream
 */
void fp4_alpha_scale_convert(
    nv_bfloat16* out_bf16,
    const float* in_f32,
    const float* global_amax_a,
    const float* global_amax_b,
    long N,
    const cudaDeviceProp& dp,
    cudaStream_t stream)
{
    const int threads_per_block = 256;
    // Use more blocks on datacenter GPUs for better occupancy
    const int num_blocks = std::min(
        (int)((N / 4 + threads_per_block - 1) / threads_per_block),
        4 * dp.multiProcessorCount);

    fp4_alpha_scale_convert_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        out_bf16, in_f32, global_amax_a, global_amax_b, N);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Kernel to compute FP4 alpha from two global amax values.
 *
 * Computes alpha = (global_amax_a * global_amax_b) / (FP4_MAX^2 * FP8_MAX^2)
 * and stores it to a device buffer for use with CUTLASS epilogue fusion.
 */
__global__ void compute_fp4_alpha_kernel(
    float* __restrict__ alpha_out,
    const float* __restrict__ global_amax_a,
    const float* __restrict__ global_amax_b)
{
    // FP4_MAX = 6.0, FP8_E4M3_MAX = 448.0
    // factor = 6.0 * 6.0 * 448.0 * 448.0 = 7,225,344.0
    constexpr float factor = 6.0f * 6.0f * 448.0f * 448.0f;

    float amax_a = *global_amax_a;
    float amax_b = *global_amax_b;
    *alpha_out = (amax_a * amax_b) / factor;
}

/**
 * @brief Compute FP4 alpha scaling factor from two global amax values.
 *
 * Computes alpha = (global_amax_a * global_amax_b) / (6^2 * 448^2)
 * and stores it to a device buffer. Use this with matmul_cutlass_fp4_alpha().
 *
 * @param alpha_out Output alpha value (device pointer to single float)
 * @param global_amax_a Global amax of tensor A (device pointer)
 * @param global_amax_b Global amax of tensor B (device pointer)
 * @param stream CUDA stream
 */
void compute_fp4_alpha(
    float* alpha_out,
    const float* global_amax_a,
    const float* global_amax_b,
    cudaStream_t stream)
{
    compute_fp4_alpha_kernel<<<1, 1, 0, stream>>>(
        alpha_out, global_amax_a, global_amax_b);
    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Kernel to compute FP4 alpha for Four Over Six (4/6) quantization.
 *
 * For 4/6 quantization, the tensor scale factor is 1536 instead of 2688:
 * alpha = (global_amax_a * global_amax_b) / (1536 * 1536)
 */
__global__ void compute_fp4_alpha_4o6_kernel(
    float* __restrict__ alpha_out,
    const float* __restrict__ global_amax_a,
    const float* __restrict__ global_amax_b)
{
    // 4/6 tensor scale factor = 384 * 4 = 1536
    // factor = 1536^2 = 2,359,296
    constexpr float factor = 1536.0f * 1536.0f;

    float amax_a = *global_amax_a;
    float amax_b = *global_amax_b;
    *alpha_out = (amax_a * amax_b) / factor;
}

/**
 * @brief Compute FP4 alpha for Four Over Six (4/6) quantization.
 *
 * For 4/6 quantization, the tensor scale is 1536 instead of 2688.
 * alpha = (global_amax_a * global_amax_b) / (1536^2)
 */
void compute_fp4_alpha_4o6(
    float* alpha_out,
    const float* global_amax_a,
    const float* global_amax_b,
    cudaStream_t stream)
{
    compute_fp4_alpha_4o6_kernel<<<1, 1, 0, stream>>>(
        alpha_out, global_amax_a, global_amax_b);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Quartet-II style tensor-scale helpers
// ============================================================================

__global__ void compute_fp4_tensor_scale_kernel(
    float* __restrict__ tensor_scale_out,
    const float* __restrict__ global_amax,
    float fp4_max,
    float fp8_max)
{
    const float denom = fp4_max * fp8_max;
    const float amax = *global_amax;
    *tensor_scale_out = (denom != 0.0f) ? (amax / denom) : 0.0f;
}

void compute_fp4_tensor_scale(
    float* tensor_scale_out,
    const float* global_amax,
    float fp4_max,
    float fp8_max,
    cudaStream_t stream)
{
    compute_fp4_tensor_scale_kernel<<<1, 1, 0, stream>>>(
        tensor_scale_out, global_amax, fp4_max, fp8_max);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void compute_fp4_alpha_from_tensor_scale_kernel(
    float* __restrict__ alpha_out,
    const float* __restrict__ tensor_scale_a,
    const float* __restrict__ tensor_scale_b)
{
    *alpha_out = (*tensor_scale_a) * (*tensor_scale_b);
}

void compute_fp4_alpha_from_tensor_scale(
    float* alpha_out,
    const float* tensor_scale_a,
    const float* tensor_scale_b,
    cudaStream_t stream)
{
    compute_fp4_alpha_from_tensor_scale_kernel<<<1, 1, 0, stream>>>(
        alpha_out, tensor_scale_a, tensor_scale_b);
    CUDA_CHECK(cudaGetLastError());
}
