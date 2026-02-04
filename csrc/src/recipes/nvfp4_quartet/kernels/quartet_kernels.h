// Copyright (c) 2025-2026, IST Austria, developed by Erik Schultheis
// Adapted for Surogate by the Surogate team
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

namespace quartet {

// Constants
constexpr int HADAMARD_DIM = 128;
constexpr int QUANT_GROUP_SIZE = 16;
constexpr float FP4_MAX = 6.0f;
constexpr float FP8_MAX_EDEN = 255.99f;
constexpr float BACKWARD_SCALE_OVERRIDE = (17.0f / 16.0f) * 0.93f;  // ~1.054

// ============================================================================
// Hadamard Matrix Initialization
// ============================================================================

/**
 * Initialize 128x128 Hadamard matrix with random column sign flips
 *
 * Generates a normalized Hadamard matrix H (128x128) where columns are
 * randomly sign-flipped based on the seed. This provides per-backward-pass
 * re-randomization for gradient unbiasedness.
 *
 * @param H Output Hadamard matrix (128x128, bfloat16)
 * @param seed Random seed for column sign generation
 * @param stream CUDA stream
 */
void initialize_hadamard_128(
    nv_bfloat16* H,
    unsigned int seed,
    cudaStream_t stream = nullptr);

// ============================================================================
// Hadamard Transform (128x128 groups)
// ============================================================================

/**
 * Apply 128x128 Hadamard transform: Y = X @ H^T
 *
 * @param y Output tensor (same shape as x)
 * @param H Hadamard matrix (128x128, bfloat16)
 * @param x Input tensor (M x N, must be divisible by 128)
 * @param M Number of rows
 * @param N Number of columns
 * @param transpose If true, computes X^T @ H^T
 * @param stream CUDA stream
 */
void group_transform_128(
    nv_bfloat16* y,
    const nv_bfloat16* H,
    const nv_bfloat16* x,
    int M, int N,
    bool transpose,
    cudaStream_t stream = nullptr);

// ============================================================================
// EDEN Quantization with Fused Hadamard Transform
// ============================================================================

/**
 * Fused Hadamard transform + EDEN FP4 quantization
 *
 * Computes: Y_fp4 = EDEN_Quantize(X @ H^T)
 *
 * Key features:
 * - RTN (round-to-nearest) for FP4 values
 * - EDEN correction factor per 128-element group: correction = sum(x^2) / sum(x*q)
 * - Stochastic rounding applied to E4M3 scales only
 *
 * @param y Output FP4 data (packed, M*N/2 bytes)
 * @param scales_fp8 Output FP8 E4M3 block scales (M*N/16 elements)
 * @param global_scale_ptr Output global FP32 scale
 * @param scratch_scales Scratch buffer for intermediate BF16 scales (M*N/16 elements)
 * @param max_scale Scratch buffer for max scale (single unsigned int)
 * @param h Hadamard matrix (128x128, bfloat16)
 * @param x Input tensor (M x N, bfloat16)
 * @param seed RNG seed for stochastic rounding
 * @param fp4_max Maximum FP4 value (typically 6.0)
 * @param fp8_max Maximum FP8 scale value (typically 255.99 for EDEN)
 * @param M Number of rows
 * @param N Number of columns
 * @param transposeX If true, input is interpreted as transposed
 * @param stream CUDA stream
 */
void group_transform_128_eden(
    __nv_fp4x2_storage_t* y,
    __nv_fp8_e4m3* scales_fp8,
    float* global_scale_ptr,
    nv_bfloat16* scratch_scales,
    unsigned* max_scale,
    const nv_bfloat16* h,
    const nv_bfloat16* x,
    long seed,
    float fp4_max,
    float fp8_max,
    int M, int N,
    bool transposeX,
    cudaStream_t stream = nullptr);

// ============================================================================
// Standalone EDEN FP4 Quantization
// ============================================================================

/**
 * EDEN FP4 quantization without Hadamard transform
 *
 * @param y_ptr Output FP4 data
 * @param scale_ptr Output FP8 E4M3 scales
 * @param x_ptr Input tensor (bfloat16)
 * @param amax_ptr Pre-computed abs-max
 * @param scale_override Scale override factor (1.0 for forward, ~1.054 for backward)
 * @param seed RNG seed
 * @param nelem Number of elements
 * @param stream CUDA stream
 */
void eden_fp4(
    __nv_fp4x4_e2m1* y_ptr,
    __nv_fp8_e4m3* scale_ptr,
    const nv_bfloat16* x_ptr,
    const float* amax_ptr,
    float scale_override,
    long seed,
    long nelem,
    cudaStream_t stream = nullptr);

// ============================================================================
// Dequantize -> Transpose -> Hadamard -> Re-quantize Pipeline
// ============================================================================

/**
 * Dequantize FP4 -> Apply transposed Hadamard -> Re-quantize with EDEN
 *
 * Used for re-quantizing W^T in backward pass with different randomization
 * (per-step Hadamard re-randomization for gradient unbiasedness)
 *
 * @param y Output FP4 data
 * @param scales_fp8 Output FP8 E4M3 scales
 * @param global_scale_ptr Output global scale
 * @param scratch_scales Scratch buffer for intermediate scales
 * @param max_scale Scratch buffer for max scale
 * @param h Hadamard matrix (with new random signs for this backward pass)
 * @param x Input FP4 data
 * @param x_scales Input FP8 scales
 * @param x_global_scale Input global scale
 * @param seed RNG seed
 * @param fp4_max Maximum FP4 value
 * @param fp8_max Maximum FP8 scale value
 * @param M Number of rows
 * @param N Number of columns
 * @param stream CUDA stream
 */
void dequant_tp_had_quant(
    __nv_fp4x2_storage_t* y,
    __nv_fp8_e4m3* scales_fp8,
    float* global_scale_ptr,
    nv_bfloat16* scratch_scales,
    unsigned* max_scale,
    const nv_bfloat16* h,
    const __nv_fp4x2_storage_t* x,
    const __nv_fp8_e4m3* x_scales,
    const float* x_global_scale,
    long seed,
    float fp4_max,
    float fp8_max,
    int M, int N,
    cudaStream_t stream = nullptr);

// ============================================================================
// Scale Conversion with Stochastic Rounding
// ============================================================================

/**
 * Convert BF16 scales to FP8 E4M3 with stochastic rounding
 *
 * @param scales_fp8 Output FP8 scales
 * @param global_scale_ptr Output global scale
 * @param scales_bf16 Input BF16 scales
 * @param max_scale_ptr Input max scale (for global scale computation)
 * @param seed RNG seed
 * @param groups Number of scale groups
 * @param inv_fp8_max Inverse of FP8 max value (1/255.99)
 * @param stream CUDA stream
 */
void launch_eden_convert_scales_kernel(
    __nv_fp8_e4m3* scales_fp8,
    float* global_scale_ptr,
    const nv_bfloat16* scales_bf16,
    const unsigned* max_scale_ptr,
    long seed,
    int groups,
    float inv_fp8_max,
    cudaStream_t stream = nullptr);

}  // namespace quartet
