// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Based on "Four Over Six: More Accurate NVFP4 Quantization with Adaptive Block Scaling"
// arXiv:2512.02010

/**
 * @file quant_nvfp4_4o6_sm100.cu
 * @brief SM100+ (Blackwell) specific kernels for Four Over Six (4/6) NVFP4 quantization.
 *
 * This file contains the SM100+ specific kernels that use native FP4 PTX instructions.
 * The PTX instructions cvt.rn.satfinite.e2m1x2.f32 and cvt.rn.f16x2.e2m1x2 are only
 * available on SM100+ (Blackwell) architecture.
 *
 * The 4/6 algorithm evaluates both scaling to max=6.0 (standard) and max=4.0
 * for each block, selecting the option with lower quantization error.
 */

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Include CUTLASS arch config to get CUTLASS_ARCH_MMA_SM100_SUPPORTED defined
#include "cutlass/arch/config.h"

// Architecture guard - only compile SM100+ code when supported
#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

#include "kernels/kernel_utils.cuh"

// ============================================================================
// Implementation namespace (not anonymous to avoid template issues)
// ============================================================================

namespace nvfp4_4o6_impl {

// Constants
constexpr int kBlockSize = 16;
constexpr float kFP4Max = 6.0f;
constexpr float kFP8E4M3Max = 448.0f;

// For 4/6 adaptive quantization, we use tensor scale = 1536 (= 384 * 4).
// This is different from standard NVFP4 which uses 2688 (= 448 * 6).
//
// Why 1536? From arXiv:2512.02010:
// - For max=6 blocks: decode_scale = block_amax * 256 / global_amax (256 = 1536/6)
// - For max=4 blocks: decode_scale = block_amax * 384 / global_amax (384 = 1536/4)
// Both 256 and 384 fit in UE4M3 (max ~480), avoiding clamping issues.
//
// If we used 2688:
// - For max=4: decode_scale = block_amax * 672 / global_amax (672 > 480 = UE4M3 max!)
//   This would cause clamping and incorrect reconstruction.
//
// The alpha formula for 4/6 matmul is: (amax_A * amax_B) / (1536^2)
constexpr float k4o6TensorScaleFactor = 384.0f * 4.0f;  // = 1536 for adaptive 4/6

// Error metric enum
enum class ErrorMetric { MSE, L1, AbsMax };

// ============================================================================
// PTX Intrinsics for SM100+ FP4 Quantization
// ============================================================================

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

__device__ __forceinline__ void fp32x8_to_e2m1x8_with_dequant(
    float (&scaled)[8],
    uint32_t& out_fp4,
    float (&out_dequant)[8])
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t dq1, dq2, dq3, dq4;

    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
        "}"
        : "=r"(out_fp4), "=r"(dq1), "=r"(dq2), "=r"(dq3), "=r"(dq4)
        : "f"(scaled[0]), "f"(scaled[1]), "f"(scaled[2]), "f"(scaled[3]),
          "f"(scaled[4]), "f"(scaled[5]), "f"(scaled[6]), "f"(scaled[7])
    );

    out_dequant[0] = __half2float(__ushort_as_half(dq1 & 0xFFFF));
    out_dequant[1] = __half2float(__ushort_as_half((dq1 >> 16) & 0xFFFF));
    out_dequant[2] = __half2float(__ushort_as_half(dq2 & 0xFFFF));
    out_dequant[3] = __half2float(__ushort_as_half((dq2 >> 16) & 0xFFFF));
    out_dequant[4] = __half2float(__ushort_as_half(dq3 & 0xFFFF));
    out_dequant[5] = __half2float(__ushort_as_half((dq3 >> 16) & 0xFFFF));
    out_dequant[6] = __half2float(__ushort_as_half(dq4 & 0xFFFF));
    out_dequant[7] = __half2float(__ushort_as_half((dq4 >> 16) & 0xFFFF));
#else
    // Fallback for non-SM100 device compilation (should never be called at runtime)
    out_fp4 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) out_dequant[i] = 0.0f;
#endif
}

__device__ __forceinline__ uint32_t fp32x8_to_e2m1x8(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
    uint32_t val;
    asm volatile(
        "{\n"
        ".reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "}"
        : "=r"(val)
        : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]),
          "f"(array[4]), "f"(array[5]), "f"(array[6]), "f"(array[7])
    );
    return val;
#else
    // Fallback for non-SM100 device compilation (should never be called at runtime)
    return 0;
#endif
}

// ============================================================================
// Error Metric Computation
// ============================================================================

template <ErrorMetric Metric>
__device__ __forceinline__ float compute_error(
    float (&original)[8],
    float (&dequant)[8],
    float scale)
{
    float err = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float reconstructed = dequant[i] * scale;
        float diff = reconstructed - original[i];

        if constexpr (Metric == ErrorMetric::MSE) {
            err += diff * diff;
        } else if constexpr (Metric == ErrorMetric::L1) {
            err += fabsf(diff);
        } else if constexpr (Metric == ErrorMetric::AbsMax) {
            err = fmaxf(err, fabsf(diff));
        }
    }
    return err;
}

// ============================================================================
// 4/6 Block Quantization
// ============================================================================

template <ErrorMetric Metric>
__device__ __forceinline__ void quantize_block_4o6(
    float (&vals)[16],
    float global_encode_scale,
    uint8_t (&out_fp4)[8],
    uint8_t& out_block_scale)
{
    // Find block amax
    float block_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    if (block_amax == 0.0f) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_fp4[i] = 0;
        out_block_scale = 0;
        return;
    }

    // Compute encode scales for both max=6.0 and max=4.0 options
    // encode_scale = FP4_MAX / block_amax (scales values to [-FP4_MAX, FP4_MAX] range)
    float encode_scale_6 = 6.0f * rcp_approx_ftz(fmaxf(block_amax, 1e-12f));
    float encode_scale_4 = 4.0f * rcp_approx_ftz(fmaxf(block_amax, 1e-12f));

    // Block decode scales for reconstructing original values from dequant (which is in [-6, 6])
    // recon_scale = block_amax / FP4_MAX
    float recon_scale_6 = block_amax * rcp_approx_ftz(6.0f);
    float recon_scale_4 = block_amax * rcp_approx_ftz(4.0f);

    float err_6 = 0.0f, err_4 = 0.0f;
    uint32_t fp4_6[2], fp4_4[2];

    #pragma unroll
    for (int g = 0; g < 2; ++g) {
        float group[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            group[i] = vals[g * 8 + i];
        }

        // Quantize with max=6.0
        float scaled_6[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) scaled_6[i] = group[i] * encode_scale_6;

        float dequant_6[8];
        fp32x8_to_e2m1x8_with_dequant(scaled_6, fp4_6[g], dequant_6);
        // dequant_6 is in [-6, 6] range, multiply by recon_scale_6 to get original scale
        err_6 += compute_error<Metric>(group, dequant_6, recon_scale_6);

        // Quantize with max=4.0
        float scaled_4[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) scaled_4[i] = group[i] * encode_scale_4;

        float dequant_4[8];
        fp32x8_to_e2m1x8_with_dequant(scaled_4, fp4_4[g], dequant_4);
        // dequant_4 is in [-4, 4] range (clipped to representable FP4 values), multiply by recon_scale_4
        err_4 += compute_error<Metric>(group, dequant_4, recon_scale_4);
    }

    bool use_scale_4 = (err_4 < err_6);
    uint32_t* selected_fp4 = use_scale_4 ? fp4_4 : fp4_6;
    float selected_recon_scale = use_scale_4 ? recon_scale_4 : recon_scale_6;

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out_fp4[i] = (selected_fp4[0] >> (i * 8)) & 0xFF;
        out_fp4[i + 4] = (selected_fp4[1] >> (i * 8)) & 0xFF;
    }

    // Compute the CUTLASS-compatible decode scale.
    // Uses tensor scale 1536 (= 384 * 4) for 4/6 adaptive quantization.
    //
    // For max=6: decode_scale = block_amax / 6 * (1536 / global_amax) = block_amax * 256 / global_amax
    // For max=4: decode_scale = block_amax / 4 * (1536 / global_amax) = block_amax * 384 / global_amax
    //
    // Both 256 and 384 fit in UE4M3 (max ~480), avoiding clamping.
    //
    // CUTLASS dequant: fp4 * decode_scale * alpha
    //   where alpha = (amax_A * amax_B) / 1536^2
    //
    // For max=4: fp4 * (block_amax * 384 / amax) * (amax_A * amax_B) / 1536^2
    //          = fp4 * block_amax / 4 ✓
    // For max=6: fp4 * (block_amax * 256 / amax) * (amax_A * amax_B) / 1536^2
    //          = fp4 * block_amax / 6 ✓
    float decode_scale = selected_recon_scale * global_encode_scale;

    // Convert to UE4M3 using the same method as standard quantization
    // Clamp to representable range and use hardware conversion
    decode_scale = fminf(fmaxf(decode_scale, 1.0f / 128.0f), 480.0f);
    out_block_scale = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                            __nv_fp8_interpretation_t::__NV_E4M3);
}

template <ErrorMetric Metric>
__device__ __forceinline__ void quantize_block_4o6_stochastic(
    float (&vals)[16],
    float global_encode_scale,
    uint32_t (&random)[4],
    uint8_t (&out_fp4)[8],
    uint8_t& out_block_scale)
{
    // Find block amax
    float block_amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        block_amax = fmaxf(block_amax, fabsf(vals[i]));
    }

    if (block_amax == 0.0f) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) out_fp4[i] = 0;
        out_block_scale = 0;
        return;
    }

    // Compute encode scales for both max=6.0 and max=4.0 options
    float encode_scale_6 = 6.0f * rcp_approx_ftz(fmaxf(block_amax, 1e-12f));
    float encode_scale_4 = 4.0f * rcp_approx_ftz(fmaxf(block_amax, 1e-12f));

    // Block decode scales for reconstructing original values from dequant (which is in [-6, 6])
    float recon_scale_6 = block_amax * rcp_approx_ftz(6.0f);
    float recon_scale_4 = block_amax * rcp_approx_ftz(4.0f);

    float err_6 = 0.0f, err_4 = 0.0f;
    uint32_t fp4_6[2], fp4_4[2];

    #pragma unroll
    for (int g = 0; g < 2; ++g) {
        float group[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            group[i] = vals[g * 8 + i];
        }

        float scaled_6[8], scaled_4[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            scaled_6[i] = group[i] * encode_scale_6;
            scaled_4[i] = group[i] * encode_scale_4;
        }

        float dequant_6[8], dequant_4[8];
        fp32x8_to_e2m1x8_with_dequant(scaled_6, fp4_6[g], dequant_6);
        fp32x8_to_e2m1x8_with_dequant(scaled_4, fp4_4[g], dequant_4);

        // Use recon_scale (block_amax / FP4_MAX) for error computation
        err_6 += compute_error<Metric>(group, dequant_6, recon_scale_6);
        err_4 += compute_error<Metric>(group, dequant_4, recon_scale_4);
    }

    bool use_scale_4 = (err_4 < err_6);
    float selected_encode_scale = use_scale_4 ? encode_scale_4 : encode_scale_6;
    float selected_recon_scale = use_scale_4 ? recon_scale_4 : recon_scale_6;

    // Apply stochastic rounding with the selected scale
    uint32_t fp4_stochastic[2];

    #pragma unroll
    for (int g = 0; g < 2; ++g) {
        float group[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float scaled = vals[g * 8 + i] * selected_encode_scale;
            uint32_t rand_bits = (random[g * 2 + (i / 4)] >> ((i % 4) * 8)) & 0xFF;
            float noise = (float(rand_bits) / 256.0f - 0.5f) * 0.5f;
            group[i] = scaled + noise;
        }
        fp4_stochastic[g] = fp32x8_to_e2m1x8(group);
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        out_fp4[i] = (fp4_stochastic[0] >> (i * 8)) & 0xFF;
        out_fp4[i + 4] = (fp4_stochastic[1] >> (i * 8)) & 0xFF;
    }

    // Compute the CUTLASS-compatible decode scale (same formula as non-stochastic version).
    // Uses tensor scale 1536 for 4/6 adaptive quantization.
    float decode_scale = selected_recon_scale * global_encode_scale;

    // Convert to UE4M3 using the same method as standard quantization
    decode_scale = fminf(fmaxf(decode_scale, 1.0f / 128.0f), 480.0f);
    out_block_scale = __nv_cvt_float_to_fp8(decode_scale, __nv_saturation_t::__NV_SATFINITE,
                                            __nv_fp8_interpretation_t::__NV_E4M3);
}

// ============================================================================
// CUTLASS Layout Scale Index (Sm1xxBlkScaledConfig SfKMajorAtom layout)
// ============================================================================

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
 */
__device__ __forceinline__ int cutlass_scale_index(int row, int scale_col, int num_scale_cols) {
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
    int atom_offset = (row_atom * atoms_per_row + col_atom) * kAtomSize;

    // Within-atom offset using SfKMajorAtom stride pattern: ((16,4), (0,1))
    // Decompose row_in_atom into (row_in_32, row_blk32) where row_blk32 = row_in_atom / 32
    int row_in_32 = row_in_atom % 32;
    int row_blk32 = row_in_atom / 32;  // 0-3

    // SfKMajorAtom mapping: stride<0,0> = 16, stride<0,1> = 4, stride<1,1> = 1
    // Physical offset = row_in_32 * 16 + row_blk32 * 4 + col_in_atom
    int intra_offset = row_in_32 * 16 + row_blk32 * 4 + col_in_atom;

    return atom_offset + intra_offset;
}

// ============================================================================
// Main Quantization Kernels
// ============================================================================

template <ErrorMetric Metric>
__global__ void quantize_4o6_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int num_scale_cols)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = M * num_scale_cols;

    if (block_idx >= total_blocks) return;

    int row = block_idx / num_scale_cols;
    int scale_col = block_idx % num_scale_cols;
    int col_start = scale_col * kBlockSize;

    // Compute global encode scale for 4/6 quantization.
    // We use tensor scale 1536 (= 384 * 4) for adaptive 4/6 mode.
    // - For max=6: decode_scale = block_amax / 6 * 1536 / global_amax = block_amax * 256 / global_amax
    // - For max=4: decode_scale = block_amax / 4 * 1536 / global_amax = block_amax * 384 / global_amax
    // Both 256 and 384 fit in UE4M3 (max ~480).
    float g_amax = *global_amax;
    float global_encode_scale = (g_amax > 0.0f) ?
        k4o6TensorScaleFactor * rcp_approx_ftz(fmaxf(g_amax, 1e-12f)) : 1.0f;

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int col = col_start + i;
        if (col < K) {
            vals[i] = __bfloat162float(in[row * K + col]);
        } else {
            vals[i] = 0.0f;
        }
    }

    uint8_t fp4_out[8];
    uint8_t block_scale;
    quantize_block_4o6<Metric>(vals, global_encode_scale, fp4_out, block_scale);

    int fp4_col_start = col_start / 2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int fp4_col = fp4_col_start + i;
        if (fp4_col < K / 2) {
            out_fp4[row * (K / 2) + fp4_col] = fp4_out[i];
        }
    }

    int scale_idx = cutlass_scale_index(row, scale_col, num_scale_cols);
    block_scales[scale_idx] = block_scale;
}

template <ErrorMetric Metric>
__global__ void quantize_4o6_stochastic_cutlass_kernel(
    uint8_t* __restrict__ out_fp4,
    uint8_t* __restrict__ block_scales,
    const float* __restrict__ global_amax,
    const nv_bfloat16* __restrict__ in,
    int M, int K, int num_scale_cols,
    unsigned int seed)
{
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_blocks = M * num_scale_cols;

    if (block_idx >= total_blocks) return;

    int row = block_idx / num_scale_cols;
    int scale_col = block_idx % num_scale_cols;
    int col_start = scale_col * kBlockSize;

    // Compute global encode scale for 4/6 quantization (same as non-stochastic version).
    float g_amax = *global_amax;
    float global_encode_scale = (g_amax > 0.0f) ?
        k4o6TensorScaleFactor * rcp_approx_ftz(fmaxf(g_amax, 1e-12f)) : 1.0f;

    float vals[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int col = col_start + i;
        if (col < K) {
            vals[i] = __bfloat162float(in[row * K + col]);
        } else {
            vals[i] = 0.0f;
        }
    }

    uint32_t random[4];
    uint32_t state = seed ^ (block_idx * 1103515245 + 12345);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        state = state * 1103515245 + 12345;
        random[i] = state;
    }

    uint8_t fp4_out[8];
    uint8_t block_scale;
    quantize_block_4o6_stochastic<Metric>(vals, global_encode_scale, random, fp4_out, block_scale);

    int fp4_col_start = col_start / 2;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int fp4_col = fp4_col_start + i;
        if (fp4_col < K / 2) {
            out_fp4[row * (K / 2) + fp4_col] = fp4_out[i];
        }
    }

    int scale_idx = cutlass_scale_index(row, scale_col, num_scale_cols);
    block_scales[scale_idx] = block_scale;
}

}  // namespace nvfp4_4o6_impl

// ============================================================================
// Public API (nvfp4_4o6_sm100 namespace)
// ============================================================================

namespace nvfp4_4o6_sm100 {

using nvfp4_4o6_impl::ErrorMetric;
using nvfp4_4o6_impl::quantize_4o6_cutlass_kernel;
using nvfp4_4o6_impl::quantize_4o6_stochastic_cutlass_kernel;

void quantize_4o6_cutlass_mse(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_cutlass_kernel<ErrorMetric::MSE><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
}

void quantize_4o6_cutlass_l1(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_cutlass_kernel<ErrorMetric::L1><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
}

void quantize_4o6_cutlass_absmax(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_cutlass_kernel<ErrorMetric::AbsMax><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols);
}

void quantize_4o6_stochastic_cutlass_mse(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_stochastic_cutlass_kernel<ErrorMetric::MSE><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
}

void quantize_4o6_stochastic_cutlass_l1(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_stochastic_cutlass_kernel<ErrorMetric::L1><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
}

void quantize_4o6_stochastic_cutlass_absmax(
    uint8_t* out_fp4, uint8_t* block_scales, const float* global_amax,
    const nv_bfloat16* in, int M, int K, int num_scale_cols,
    unsigned int seed, cudaStream_t stream)
{
    int total_blocks = M * num_scale_cols;
    int threads = 256;
    int blocks = (total_blocks + threads - 1) / threads;

    quantize_4o6_stochastic_cutlass_kernel<ErrorMetric::AbsMax><<<blocks, threads, 0, stream>>>(
        out_fp4, block_scales, global_amax, in, M, K, num_scale_cols, seed);
}

bool is_supported() {
    return true;
}

}  // namespace nvfp4_4o6_sm100

#else  // !SM100+

// Stub implementations for non-SM100 builds
namespace nvfp4_4o6_sm100 {

void quantize_4o6_cutlass_mse(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, cudaStream_t) {}

void quantize_4o6_cutlass_l1(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, cudaStream_t) {}

void quantize_4o6_cutlass_absmax(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, cudaStream_t) {}

void quantize_4o6_stochastic_cutlass_mse(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, unsigned int, cudaStream_t) {}

void quantize_4o6_stochastic_cutlass_l1(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, unsigned int, cudaStream_t) {}

void quantize_4o6_stochastic_cutlass_absmax(
    uint8_t*, uint8_t*, const float*, const nv_bfloat16*,
    int, int, int, unsigned int, cudaStream_t) {}

bool is_supported() {
    return false;
}

}  // namespace nvfp4_4o6_sm100

#endif  // defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
