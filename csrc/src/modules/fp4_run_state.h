// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FP4_RUN_STATE_H
#define SUROGATE_SRC_MODULES_FP4_RUN_STATE_H

#include <algorithm>
#include "utilities/tensor.h"
#include "utilities/allocator.h"

namespace modules {

/**
 * @brief FP4 forward quantization buffers (when enable_fp4_forward is set).
 *
 * FP4 uses E2M1 format with two-level block scaling:
 * - Level 1: FP8 E4M3 scale per 16 consecutive values
 * - Level 2: FP32 global per-tensor scale (amax)
 *
 * These buffers are transient and shared across layers since FP4 data
 * is consumed immediately by the matmul and not needed for backward.
 * Backward pass uses BF16 cached activations for stability (or stochastic
 * rounding when enable_fp4_backward is set).
 *
 * Data layout: (M, K/2) bytes - 2 FP4 values packed per byte
 * Scale layout: (ceil(M/128), ceil(ceil(K/16)/4)*4) for F8_128x4 swizzling
 */
struct FP4ForwardQuantActivations {
    // LN1 -> QKV projection input
    Tensor ln1_data;          ///< (B*T, C/2) packed FP4 E2M1
    Tensor ln1_scales;        ///< FP8 E4M3 block scales
    float* ln1_global_amax;   ///< Device pointer to global amax

    // LN2 -> MLP up projection input
    Tensor ln2_data;          ///< (B*T, C/2) packed FP4 E2M1
    Tensor ln2_scales;        ///< FP8 E4M3 block scales
    float* ln2_global_amax;

    // Att -> output projection input
    Tensor att_data;          ///< (B*T, AttC/2) packed FP4 E2M1
    Tensor att_scales;        ///< FP8 E4M3 block scales
    float* att_global_amax;

    // Swiglu -> MLP down projection input
    Tensor swiglu_data;       ///< (B*T, D/2) packed FP4 E2M1
    Tensor swiglu_scales;     ///< FP8 E4M3 block scales
    float* swiglu_global_amax;

    // Hadamard transform workspace (reused across all activations)
    Tensor hadamard_workspace;  ///< (B*T, max_dim) BF16 temporary

    // Global amax storage (4 floats for ln1, ln2, att, swiglu)
    Tensor global_amax_buffer;
};

/**
 * @brief Compute scale tensor dimensions for FP4 block quantization
 *
 * Scale tensor uses F8_128x4 swizzled layout:
 * - Each row of input has K/16 block scales (one per 16 elements)
 * - Rows are grouped into 128-row base blocks
 * - Columns are grouped into 4-column groups
 * The swizzled layout requires: rows aligned to 128, cols aligned to 4
 */
inline std::pair<long, long> compute_fp4_scale_shape(long rows, long cols) {
    const long scale_rows = ((rows + 127) / 128) * 128;  // Align to 128-row base blocks
    const long scale_cols_raw = (cols + 15) / 16;  // 16-element blocks
    const long scale_cols = ((scale_cols_raw + 3) / 4) * 4;  // 4-column alignment
    return {scale_rows, scale_cols};
}

/**
 * @brief Helper to allocate FP4 forward buffers
 */
inline void allocate_fp4_forward_buffers(
    FP4ForwardQuantActivations& quants,
    TensorAllocator& allocator,
    long B, long T, long C, long D, long AttC,
    ETensorDType activation_dtype)
{
    const auto fp4_dtype = ETensorDType::BYTE;  // Packed FP4 (2 values per byte)
    const auto scale_dtype = ETensorDType::FP8_E4M3;
    const long M = B * T;  // Batch dimension flattened

    // Global amax buffer: 4 floats for ln1, ln2, att, swiglu
    quants.global_amax_buffer = allocator.allocate(
        ETensorDType::FP32, "fp4_global_amax", EAllocationType::ON_DEVICE, {4L});
    float* amax_ptr = quants.global_amax_buffer.template get<float>();
    quants.ln1_global_amax = amax_ptr + 0;
    quants.ln2_global_amax = amax_ptr + 1;
    quants.att_global_amax = amax_ptr + 2;
    quants.swiglu_global_amax = amax_ptr + 3;

    // LN1 -> QKV projection input: (B*T, C)
    {
        auto [sr, sc] = compute_fp4_scale_shape(M, C);
        quants.ln1_data = allocator.allocate(
            fp4_dtype, "fp4_fwd_ln1_data", EAllocationType::ON_DEVICE, {M, C / 2});
        quants.ln1_scales = allocator.allocate(
            scale_dtype, "fp4_fwd_ln1_scales", EAllocationType::ON_DEVICE, {sr, sc});
    }

    // LN2 -> MLP up projection input: (B*T, C)
    {
        auto [sr, sc] = compute_fp4_scale_shape(M, C);
        quants.ln2_data = allocator.allocate(
            fp4_dtype, "fp4_fwd_ln2_data", EAllocationType::ON_DEVICE, {M, C / 2});
        quants.ln2_scales = allocator.allocate(
            scale_dtype, "fp4_fwd_ln2_scales", EAllocationType::ON_DEVICE, {sr, sc});
    }

    // Att -> output projection input: (B*T, AttC)
    {
        auto [sr, sc] = compute_fp4_scale_shape(M, AttC);
        quants.att_data = allocator.allocate(
            fp4_dtype, "fp4_fwd_att_data", EAllocationType::ON_DEVICE, {M, AttC / 2});
        quants.att_scales = allocator.allocate(
            scale_dtype, "fp4_fwd_att_scales", EAllocationType::ON_DEVICE, {sr, sc});
    }

    // Swiglu -> MLP down projection input: (B*T, D)
    {
        auto [sr, sc] = compute_fp4_scale_shape(M, D);
        quants.swiglu_data = allocator.allocate(
            fp4_dtype, "fp4_fwd_swiglu_data", EAllocationType::ON_DEVICE, {M, D / 2});
        quants.swiglu_scales = allocator.allocate(
            scale_dtype, "fp4_fwd_swiglu_scales", EAllocationType::ON_DEVICE, {sr, sc});
    }

    // Hadamard transform workspace: largest dimension among C, AttC, D
    const long max_dim = std::max({C, AttC, D});
    quants.hadamard_workspace = allocator.allocate(
        activation_dtype, "fp4_hadamard_ws", EAllocationType::ON_DEVICE, {M, max_dim});
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FP4_RUN_STATE_H
