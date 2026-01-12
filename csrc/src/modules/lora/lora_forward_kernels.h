// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_FORWARD_KERNELS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_FORWARD_KERNELS_H

#include "kernels/kernels.h"
#include "lora_weights.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {
namespace detail {

inline void apply_lora_contribution(
    Tensor& output,
    int output_offset,
    const Tensor& input,
    const LoRALayerWeights<Tensor>& lora,
    Tensor& intermediate,
    Tensor& slice_buffer,
    float scaling,
    int BT,
    int in_features,
    int out_features,
    int rank,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (!lora.has_value()) return;
    if (out_features <= 0 || BT <= 0) return;

    // intermediate = input @ A^T  (BT x rank)
    matmul(intermediate, lora.A, input, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

    // Scale intermediate so we can use GEMM accumulate for B @ intermediate^T.
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    const long total_out_features = output.Sizes[output.Rank - 1];
    if (output_offset < 0 || output_offset + out_features > total_out_features) {
        throw std::logic_error("apply_lora_contribution: output_offset out of bounds");
    }

    // Packed destination: accumulate directly.
    if (output_offset == 0 && out_features == total_out_features) {
        matmul(output, lora.B, intermediate, std::nullopt, nullptr, nullptr,
               handle, workspace, out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/true, stream);
        return;
    }

    // Fused projections: prefer direct strided accumulate when aligned, else fall back to packed delta + add.
    Tensor output_slice = output;
    output_slice.Data = output.Data + (std::size_t)output_offset * get_dtype_size(output.DType);
    bool aligned = ((uintptr_t)output_slice.Data % 16) == 0;
    if (aligned) {
        matmul_strided_c(output_slice, lora.B, intermediate, std::nullopt, nullptr, nullptr,
                         handle, workspace,
                         out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/true,
                         (int)total_out_features, stream);
        return;
    }

    Tensor packed_delta = slice_buffer;
    packed_delta.DType = output.DType;
    packed_delta.Rank = 2;
    packed_delta.Sizes[0] = BT;
    packed_delta.Sizes[1] = out_features;
    for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed_delta.Sizes[i] = 1;

    matmul(packed_delta, lora.B, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/false, stream);
    add_2d_slice(output, packed_delta, BT, total_out_features, out_features, output_offset, stream);
}

/**
 * @brief Apply LoRA contributions for a single MoE expert
 *
 * This function applies LoRA to an expert's gate_up and down projections.
 * It's called during the MoE expert forward pass for each expert that has
 * tokens routed to it.
 *
 * @param gate_up Output of the expert's gate+up projection (N, 2*D) - modified in place
 * @param down_output Output of the expert's down projection (N, C) - modified in place
 * @param expert_input Input to the expert (N, C) - used for gate/up LoRA
 * @param activated Activated output (N, D) - used for down LoRA
 * @param expert_lora The LoRA weights for this expert
 * @param intermediate Scratch tensor for LoRA computation (N, rank)
 * @param slice_buffer Scratch tensor for slicing
 * @param scaling LoRA scaling factor
 * @param N Number of tokens routed to this expert
 * @param C Hidden size
 * @param D Expert intermediate size
 * @param rank LoRA rank
 * @param handle cuBLAS handle
 * @param workspace cuBLAS workspace
 * @param stream CUDA stream
 */
inline void apply_expert_lora(
    Tensor& gate_up,          // (N, 2*D) - gate+up projection output
    Tensor& down_output,      // (N, C) - down projection output
    const Tensor& expert_input,  // (N, C) - input to expert
    const Tensor& activated,     // (N, D) - activated value (after SwiGLU)
    const LoRAExpertWeights<Tensor>& expert_lora,
    Tensor& intermediate,
    Tensor& slice_buffer,
    float scaling,
    int N,            // Number of tokens for this expert
    int C,            // Hidden size
    int D,            // Expert intermediate size
    int rank,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (N <= 0) return;

    // Apply LoRA to gate projection (first half of gate_up)
    if (expert_lora.gate.has_value() && expert_lora.gate->has_value()) {
        apply_lora_contribution(gate_up, D, expert_input, *expert_lora.gate,
                                intermediate, slice_buffer,
                                scaling, N, C, D, rank,
                                handle, workspace, stream);
    }

    // Apply LoRA to up projection (second half of gate_up)
    // Note: In the fused gate_up layout, up is at offset 0 and gate is at offset D
    if (expert_lora.up.has_value() && expert_lora.up->has_value()) {
        apply_lora_contribution(gate_up, 0, expert_input, *expert_lora.up,
                                intermediate, slice_buffer,
                                scaling, N, C, D, rank,
                                handle, workspace, stream);
    }

    // Apply LoRA to down projection
    if (expert_lora.down.has_value() && expert_lora.down->has_value()) {
        apply_lora_contribution(down_output, 0, activated, *expert_lora.down,
                                intermediate, slice_buffer,
                                scaling, N, D, C, rank,
                                handle, workspace, stream);
    }
}

} // namespace detail
} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_FORWARD_KERNELS_H
