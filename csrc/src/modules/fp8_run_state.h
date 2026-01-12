// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_FP8_RUN_STATE_H
#define SUROGATE_SRC_MODULES_FP8_RUN_STATE_H

#include "utilities/tensor.h"
#include "utilities/allocator.h"

namespace modules {

/**
 * @brief FP8 forward-only activation buffers (when enable_fp8_forward is set).
 *
 * These buffers are used for FP8 quantization during forward pass only.
 * They are transient and can be shared across layers since the FP8 data
 * is consumed immediately by the matmul and not needed for backward.
 * Backward pass uses BF16 cached activations for stability.
 */
struct FP8ForwardQuantActivations {
    Tensor ln1;      ///< (B, T, C) in FP8 E4M3 - input to QKV projection
    Tensor ln2;      ///< (B, T, C) in FP8 E4M3 - input to MLP up projection
    Tensor att;      ///< (B, T, Hq*Hs) in FP8 E4M3 - input to output projection
    Tensor swiglu;   ///< (B, T, D) in FP8 E4M3 - input to MLP down projection
};

/**
 * @brief Helper to allocate FP8 forward buffers
 */
inline void allocate_fp8_forward_buffers(
    FP8ForwardQuantActivations& quants,
    Tensor& stats_buffer,
    TensorAllocator& allocator,
    long B, long T, long C, long D, long AttC,
    ETensorDType fp8_dtype)
{
    // Allocate stats buffer: 4 pairs (abs_max, scale) = 8 floats
    stats_buffer = allocator.allocate(ETensorDType::FP32, "fp8_fwd_stats",
                                      EAllocationType::ON_DEVICE, {8L});
    float* fp8_stats = stats_buffer.get<float>();

    // LN1 -> QKV projection input
    quants.ln1 = allocator.allocate(fp8_dtype, "fp8_fwd_ln1",
                                    EAllocationType::ON_DEVICE, {B, T, C});
    quants.ln1.Stats = fp8_stats + 0;

    // LN2 -> MLP up projection input
    quants.ln2 = allocator.allocate(fp8_dtype, "fp8_fwd_ln2",
                                    EAllocationType::ON_DEVICE, {B, T, C});
    quants.ln2.Stats = fp8_stats + 2;

    // Att -> output projection input
    quants.att = allocator.allocate(fp8_dtype, "fp8_fwd_att",
                                    EAllocationType::ON_DEVICE, {B, T, AttC});
    quants.att.Stats = fp8_stats + 4;

    // SwiGLU -> MLP down projection input
    quants.swiglu = allocator.allocate(fp8_dtype, "fp8_fwd_swiglu",
                                       EAllocationType::ON_DEVICE, {B, T, D});
    quants.swiglu.Stats = fp8_stats + 6;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_FP8_RUN_STATE_H
