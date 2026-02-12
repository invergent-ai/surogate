// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_MODEL_UTILS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_MODEL_UTILS_H

#include <vector>
#include <string>
#include <map>
#include <optional>
#include "lora_config.h"
#include "lora_weights.h"
#include "lora_utils.h"
#include "runtime/training/runtime_options.h"
#include "utilities/comm.h"
#include "kernels/kernels.h"

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
    float dropout_prob,
    unsigned int dropout_seed,
    bool is_training,
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

    // Apply dropout to intermediate activations (inverted dropout)
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(intermediate, dropout_prob, dropout_seed, stream);
    }

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
 * @brief Apply LoRA contribution to FP32 output (e.g., router logits)
 *
 * Similar to apply_lora_contribution but the output is always FP32.
 * This is used for router LoRA where logits are computed in FP32 for numerical stability.
 * The LoRA computation is done in the work dtype (BF16) and then we use FP32 matmul
 * accumulation since router outputs are small (BT x num_experts, typically <128 experts).
 */
inline void apply_lora_contribution_fp32(
    Tensor& output,
    int output_offset,
    const Tensor& input,
    const LoRALayerWeights<Tensor>& lora,
    Tensor& intermediate,
    Tensor& slice_buffer,
    float scaling,
    float dropout_prob,
    unsigned int dropout_seed,
    bool is_training,
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
    // Compute in work dtype (typically BF16)
    matmul(intermediate, lora.A, input, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

    // Apply dropout to intermediate activations (inverted dropout)
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(intermediate, dropout_prob, dropout_seed, stream);
    }

    // Scale intermediate so the final B projection includes the LoRA scaling
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // For router logits (small number of outputs), we compute the delta in BF16,
    // convert to FP32, then accumulate. This is cleaner than mixed-precision matmul.
    // Use slice_buffer as temporary for the BF16 delta
    Tensor lora_delta = slice_buffer;
    lora_delta.DType = lora.B.DType;  // Work dtype (typically BF16)
    lora_delta.Rank = 2;
    lora_delta.Sizes[0] = BT;
    lora_delta.Sizes[1] = out_features;
    for (int i = 2; i < MAX_TENSOR_DIM; ++i) lora_delta.Sizes[i] = 1;

    matmul(lora_delta, lora.B, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, out_features, BT, rank, EMMTranspose::TN, /*accumulate=*/false, stream);

    const long total_out_features = output.Sizes[output.Rank - 1];
    if (output_offset < 0 || output_offset + out_features > total_out_features) {
        throw std::logic_error("apply_lora_contribution_fp32: output_offset out of bounds");
    }

    // Simple case: full output (no offset) - use fused BF16->FP32 accumulate
    // We convert the BF16 delta to FP32 in-place using the output buffer end as temp,
    // then add to output. For small router outputs, this overhead is minimal.
    if (output_offset == 0 && out_features == total_out_features) {
        // Accumulate BF16 delta into FP32 output element-wise
        // output[i] += bf16_to_float(lora_delta[i])
        fused_bf16_accum_to_fp32(output.get<float>(), lora_delta.get<nv_bfloat16>(),
                                  BT * out_features, stream);
        return;
    }

    // With offset: accumulate strided
    float* output_ptr = output.get<float>() + output_offset;
    fused_bf16_accum_to_fp32_strided(output_ptr, lora_delta.get<nv_bfloat16>(),
                                      BT, out_features, (int)total_out_features, stream);
}

/**
 * @brief Backward pass for router LoRA (weight gradients only, no input gradient)
 *
 * Computes gradients for router LoRA weights without propagating gradients to input.
 * This is used for router LoRA where the input gradient is computed separately by the
 * base model (d_ln2 is computed in the MoE backward pass independently).
 *
 * The forward pass computes: logits += scaling * (input @ A^T @ B^T)
 * The backward computes:
 *   intermediate = input @ A^T  (BT x rank)
 *   dB = intermediate^T @ d_logits   (rank x E)
 *   dA = (B @ d_logits^T)^T @ input  (rank x C) -> transposed storage: (C x rank)?
 *
 * Note: d_logits is in FP32, input is in BF16, LoRA weights are in BF16.
 * We use dtype conversion for mixed-precision matmuls.
 *
 * @param dA Gradient w.r.t lora_A (rank, in_features)
 * @param dB Gradient w.r.t lora_B (out_features, rank)
 * @param d_logits Gradient w.r.t output logits (BT, out_features) - FP32
 * @param input Input to router (BT, in_features) - BF16
 * @param A lora_A weights (rank, in_features) - BF16
 * @param B lora_B weights (out_features, rank) - BF16
 * @param scaling LoRA scaling factor
 * @param intermediate Scratch buffer (BT, rank)
 * @param intermediate2 Scratch buffer (BT, out_features) for FP32->BF16 conversion
 * @param slice_buffer Additional scratch buffer
 * @param BT Batch * sequence length
 * @param in_features Hidden size (C)
 * @param out_features Number of experts (E)
 * @param rank LoRA rank
 * @param accumulate Whether to accumulate gradients
 * @param handle cuBLAS handle
 * @param workspace cuBLAS workspace
 * @param stream CUDA stream
 */
inline void backward_lora_router(
    Tensor& dA,
    Tensor& dB,
    const Tensor& d_logits,   // FP32
    const Tensor& input,      // BF16
    const Tensor& A,
    const Tensor& B,
    float scaling,
    float dropout_prob,
    unsigned int dropout_seed,
    bool is_training,
    Tensor& intermediate,
    Tensor& intermediate2,
    Tensor& slice_buffer,
    int BT,
    int in_features,
    int out_features,
    int rank,
    bool accumulate,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (!A.Data || !B.Data) return;
    if (BT <= 0 || out_features <= 0) return;

    // Convert d_logits from FP32 to BF16 for matmul (router outputs are small, so this is OK)
    // Use slice_buffer as temp for the BF16 version of d_logits
    Tensor d_logits_bf16 = slice_buffer;
    d_logits_bf16.DType = ETensorDType::BF16;
    d_logits_bf16.Rank = 2;
    d_logits_bf16.Sizes[0] = BT;
    d_logits_bf16.Sizes[1] = out_features;
    for (int i = 2; i < MAX_TENSOR_DIM; ++i) d_logits_bf16.Sizes[i] = 1;

    convert_dtype(d_logits_bf16.get<nv_bfloat16>(), d_logits.get<float>(),
                  BT * out_features, stream);

    // intermediate = input @ A^T  (BT x rank)
    matmul(intermediate, A, input, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

    // Apply dropout with same mask as forward pass
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(intermediate, dropout_prob, dropout_seed, stream);
    }

    // Apply scaling to intermediate
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dB = intermediate^T @ d_logits_bf16  => (rank, E)
    // In column-major: dB^T(E, rank) = d_logits_bf16^T(E, BT) @ intermediate(BT, rank)
    matmul(dB, intermediate, d_logits_bf16, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // For dA, we need: dA = (B @ d_logits^T)^T @ input
    // First: temp = d_logits_bf16 @ B^T  => (BT x rank)
    // Then: dA = temp^T @ input  => (rank x C)
    Tensor temp = intermediate2;
    if (temp.nelem() < (long)(BT * rank)) {
        // Use a portion of workspace or reuse intermediate (they're the same size)
        temp = intermediate;  // Recompute intermediate after using it
    }
    temp.DType = intermediate.DType;
    temp.Rank = 2;
    temp.Sizes[0] = BT;
    temp.Sizes[1] = rank;

    // temp = d_logits_bf16 @ B^T  => (BT, rank)
    // In column-major: temp^T(rank, BT) = B(rank, E) @ d_logits_bf16^T(E, BT)
    matmul(temp, B, d_logits_bf16, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);

    // Apply dropout with same mask as forward pass
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(temp, dropout_prob, dropout_seed, stream);
    }

    // Apply scaling
    if (scaling != 1.0f) {
        vector_add_sr(temp, temp, temp, 0.5f * scaling, temp.nelem(), /*seed=*/0, stream);
    }

    // dA = temp^T @ input  => (rank, C)
    // In column-major: dA^T(C, rank) = input^T(C, BT) @ temp(BT, rank)
    matmul(dA, input, temp, std::nullopt, nullptr, nullptr,
           handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);
}

/**
 * @brief Apply LoRA contributions for a single MoE expert
 *
 * This function applies LoRA to an expert's gate_up (fused) or gate/up and down projections.
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
    float dropout_prob,
    unsigned int dropout_base_seed,
    bool is_training,
    int expert_idx,   // Expert index for unique dropout seed
    int N,            // Number of tokens for this expert
    int C,            // Hidden size
    int D,            // Expert intermediate size
    int rank,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (N <= 0) return;

    // Helper to compute unique dropout seed per projection
    auto get_dropout_seed = [&](int proj_type) -> unsigned int {
        // seed = base_seed + expert_idx * 10000 + proj_type * 1000
        return dropout_base_seed + expert_idx * 10000 + proj_type * 1000;
    };

    // Apply LoRA to fused gate_up (preferred if present)
    if (expert_lora.gate_up.has_value() && expert_lora.gate_up->has_value()) {
        apply_lora_contribution(gate_up, 0, expert_input, *expert_lora.gate_up,
                                intermediate, slice_buffer,
                                scaling, dropout_prob, get_dropout_seed(3), is_training,
                                N, C, 2 * D, rank,
                                handle, workspace, stream);
    } else {
        // Apply LoRA to gate projection (first half of gate_up)
        if (expert_lora.gate.has_value() && expert_lora.gate->has_value()) {
            apply_lora_contribution(gate_up, D, expert_input, *expert_lora.gate,
                                    intermediate, slice_buffer,
                                    scaling, dropout_prob, get_dropout_seed(0), is_training,
                                    N, C, D, rank,
                                    handle, workspace, stream);
        }

        // Apply LoRA to up projection (second half of gate_up)
        // Note: In the fused gate_up layout, up is at offset 0 and gate is at offset D
        if (expert_lora.up.has_value() && expert_lora.up->has_value()) {
            apply_lora_contribution(gate_up, 0, expert_input, *expert_lora.up,
                                    intermediate, slice_buffer,
                                    scaling, dropout_prob, get_dropout_seed(1), is_training,
                                    N, C, D, rank,
                                    handle, workspace, stream);
        }
    }

    // Apply LoRA to down projection
    if (expert_lora.down.has_value() && expert_lora.down->has_value()) {
        apply_lora_contribution(down_output, 0, activated, *expert_lora.down,
                                intermediate, slice_buffer,
                                scaling, dropout_prob, get_dropout_seed(2), is_training,
                                N, D, C, rank,
                                handle, workspace, stream);
    }
}

inline void backward_lora_layer(
    Tensor& dA,
    Tensor& dB,
    Tensor& dx,
    const Tensor& dL_dy,
    int dL_dy_offset,
    const Tensor& x,
    const Tensor& A,
    const Tensor& B,
    float scaling,
    float dropout_prob,
    unsigned int dropout_seed,
    bool is_training,
    Tensor& intermediate,
    Tensor& slice_buffer,
    int BT,
    int in_features,
    int out_features,
    int rank,
    bool accumulate,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (!A.Data || !B.Data) return;

    Tensor dL_dy_slice = dL_dy;
    const long full_out_features = dL_dy.Sizes[dL_dy.Rank - 1];
    if (dL_dy_offset < 0 || dL_dy_offset + out_features > full_out_features) {
        throw std::logic_error("backward_lora_layer: dL_dy_offset out of bounds");
    }

    // Pack fused slice into a contiguous buffer.
    if (dL_dy_offset != 0 || out_features != full_out_features) {
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = out_features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_out_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)out_features * elem_size;
        const std::size_t width = (std::size_t)out_features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)dL_dy_offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_ptr, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));

        dL_dy_slice = packed;
        dL_dy_offset = 0;
    }

    // intermediate = x @ A^T (BT x rank) - recompute forward for dB
    matmul(intermediate, A, x, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

    // Apply same dropout mask as forward (same seed produces identical mask)
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(intermediate, dropout_prob, dropout_seed, stream);
    }

    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dB = (x @ A^T)^T @ dL_dy (uses dropout-ed intermediate)
    matmul(dB, intermediate, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // intermediate = B @ dL_dy^T  => (BT x rank) view - gradient w.r.t. dropped activations
    matmul(intermediate, B, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);

    // Apply same dropout mask to gradient (backprop through dropout)
    if (is_training && dropout_prob > 0.0f) {
        lora_dropout_scale(intermediate, dropout_prob, dropout_seed, stream);
    }

    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dA = x^T @ (dL_dy @ B)
    matmul(dA, x, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // dx += (dL_dy @ B) @ A
    matmul(dx, A, intermediate, std::nullopt, nullptr, nullptr,
           handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
}

/**
 * @brief Backward pass for a single MoE expert's LoRA contributions
 *
 * Computes gradients for expert-specific LoRA weights (gate, up, down).
 * This function is called during the backward pass for each expert that had
 * tokens routed to it during forward.
 *
 * The backward follows the chain rule through:
 * - down_lora: dL/dA_down, dL/dB_down from d_res_ffn (gradient of MLP output)
 * - up_lora: dL/dA_up, dL/dB_up from d_mlp_up (gradient of gate_up output)
 * - gate_lora: dL/dA_gate, dL/dB_gate from d_mlp_up
 *
 * @param expert_lora_grads Per-expert LoRA gradients (output)
 * @param expert_lora_weights Per-expert LoRA weights (input)
 * @param expert_input Input to the expert (N, C)
 * @param d_gate_up Gradient w.r.t. gate_up output (N, 2*D)
 * @param activated Activated value from forward (N, D) - for down backward
 * @param d_down_output Gradient w.r.t. down projection output (N, C)
 * @param d_expert_input Gradient w.r.t. expert input (accumulated) (N, C)
 * @param d_activated Gradient w.r.t. activated value (accumulated) (N, D)
 * @param intermediate Scratch buffer (N, rank)
 * @param slice_buffer Scratch buffer for slicing
 * @param scaling LoRA scaling factor
 * @param N Number of tokens for this expert
 * @param C Hidden size
 * @param D Expert intermediate size
 * @param rank LoRA rank
 * @param accumulate Whether to accumulate into gradient tensors
 * @param handle cuBLAS handle
 * @param workspace cuBLAS workspace
 * @param stream CUDA stream
 */
inline void backward_lora_expert(
    LoRAExpertWeights<Tensor>& expert_lora_grads,
    const LoRAExpertWeights<Tensor>& expert_lora_weights,
    const Tensor& expert_input,       // (N, C) - input to expert
    const Tensor& d_gate_up,          // (N, 2*D) - gradient of gate_up
    const Tensor& activated,          // (N, D) - activated value from forward
    const Tensor& d_down_output,      // (N, C) - gradient of down proj output
    Tensor& d_expert_input,           // (N, C) - gradient w.r.t. expert input (accumulated)
    Tensor& d_activated,              // (N, D) - gradient w.r.t. activated (accumulated)
    Tensor& intermediate,
    Tensor& slice_buffer,
    float scaling,
    float dropout_prob,
    unsigned int dropout_base_seed,
    bool is_training,
    int expert_idx,   // Expert index for unique dropout seed
    int N,            // Number of tokens for this expert
    int C,            // Hidden size
    int D,            // Expert intermediate size
    int rank,
    bool accumulate,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (N <= 0) return;

    // Helper to compute unique dropout seed per projection
    auto get_dropout_seed = [&](int proj_type) -> unsigned int {
        // seed = base_seed + expert_idx * 10000 + proj_type * 1000
        return dropout_base_seed + expert_idx * 10000 + proj_type * 1000;
    };

    // Backward through down projection LoRA
    // y_down += scaling * B_down @ (A_down @ activated^T)^T
    // dA_down = activated^T @ (dL_dy_down @ B_down^T) * scaling
    // dB_down = (activated @ A_down^T)^T @ dL_dy_down * scaling
    // d_activated += (dL_dy_down @ B_down^T) @ A_down * scaling
    if (expert_lora_weights.down.has_value() && expert_lora_weights.down->has_value() &&
        expert_lora_grads.down.has_value()) {
        backward_lora_layer(
            expert_lora_grads.down->A,
            expert_lora_grads.down->B,
            d_activated,
            d_down_output, 0,  // offset 0 since down output is not packed
            activated,
            expert_lora_weights.down->A,
            expert_lora_weights.down->B,
            scaling,
            dropout_prob, get_dropout_seed(2), is_training,  // proj_type=2 for down
            intermediate, slice_buffer,
            N, D, C, rank, accumulate,
            handle, workspace, stream);
    }

    // Backward through fused gate_up LoRA (preferred if present)
    if (expert_lora_weights.gate_up.has_value() && expert_lora_weights.gate_up->has_value() &&
        expert_lora_grads.gate_up.has_value()) {
        backward_lora_layer(
            expert_lora_grads.gate_up->A,
            expert_lora_grads.gate_up->B,
            d_expert_input,
            d_gate_up, 0,  // fused gate_up uses full output
            expert_input,
            expert_lora_weights.gate_up->A,
            expert_lora_weights.gate_up->B,
            scaling,
            dropout_prob, get_dropout_seed(3), is_training,  // proj_type=3 for gate_up
            intermediate, slice_buffer,
            N, C, 2 * D, rank, accumulate,
            handle, workspace, stream);
    } else {
        // Backward through gate projection LoRA (second half of gate_up at offset D)
        // gate is at offset D in the fused gate_up tensor
        if (expert_lora_weights.gate.has_value() && expert_lora_weights.gate->has_value() &&
            expert_lora_grads.gate.has_value()) {
            backward_lora_layer(
                expert_lora_grads.gate->A,
                expert_lora_grads.gate->B,
                d_expert_input,
                d_gate_up, D,  // gate is at offset D
                expert_input,
                expert_lora_weights.gate->A,
                expert_lora_weights.gate->B,
                scaling,
                dropout_prob, get_dropout_seed(0), is_training,  // proj_type=0 for gate
                intermediate, slice_buffer,
                N, C, D, rank, accumulate,
                handle, workspace, stream);
        }

        // Backward through up projection LoRA (first half of gate_up at offset 0)
        if (expert_lora_weights.up.has_value() && expert_lora_weights.up->has_value() &&
            expert_lora_grads.up.has_value()) {
            backward_lora_layer(
                expert_lora_grads.up->A,
                expert_lora_grads.up->B,
                d_expert_input,
                d_gate_up, 0,  // up is at offset 0
                expert_input,
                expert_lora_weights.up->A,
                expert_lora_weights.up->B,
                scaling,
                dropout_prob, get_dropout_seed(1), is_training,  // proj_type=1 for up
                intermediate, slice_buffer,
                N, C, D, rank, accumulate,
                handle, workspace, stream);
        }
    }
}

/**
 * @brief Fused backward pass for QKV LoRA projections
 *
 * Optimizes QKV backward by:
 * 1. Computing dL_dy @ B^T for all projections, then batching the dx accumulation
 * 2. Reusing x (ln1) across all three projections instead of redundant loads
 * 3. Reducing kernel launch overhead from 15 matmuls to 12 matmuls
 *
 * Mathematical formulation (for each projection p in {q,k,v}):
 *   dA_p = x^T @ (dL_dy_p @ B_p^T) * scaling
 *   dB_p = (x @ A_p^T)^T @ dL_dy_p * scaling
 *   dx += (dL_dy_p @ B_p^T) @ A_p * scaling
 *
 * Fusion strategy:
 * - Phase 1: Compute x @ A^T for all projections (reuses x)
 * - Phase 2: Compute dB for all projections
 * - Phase 3: Compute dL_dy @ B^T and dA for all projections
 * - Phase 4: Accumulate dx contributions
 */
inline void backward_lora_qkv_fused(
    // Gradient outputs for Q
    Tensor& dA_q, Tensor& dB_q,
    // Gradient outputs for K
    Tensor& dA_k, Tensor& dB_k,
    // Gradient outputs for V
    Tensor& dA_v, Tensor& dB_v,
    // Input gradient accumulator
    Tensor& dx,
    // Upstream gradient (packed QKV)
    const Tensor& dL_dy,
    // Forward input (shared across Q, K, V)
    const Tensor& x,
    // LoRA weights
    const LoRALayerWeights<Tensor>& lora_q,
    const LoRALayerWeights<Tensor>& lora_k,
    const LoRALayerWeights<Tensor>& lora_v,
    // Dimensions
    float scaling,
    float dropout_prob,
    unsigned int dropout_seed_q,
    unsigned int dropout_seed_k,
    unsigned int dropout_seed_v,
    bool is_training,
    int BT,
    int in_features,    // C (hidden size)
    int q_out_features, // Hq * Hs
    int kv_out_features,// Hkv * Hs
    int rank,
    bool accumulate,
    // Intermediates (must be pre-allocated)
    Tensor& intermediate1,  // (BT, rank) for x @ A^T
    Tensor& intermediate2,  // (BT, rank) for dL_dy @ B^T
    Tensor& slice_buffer,   // For slicing packed QKV gradients
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    const bool has_q = lora_q.has_value() && dA_q.Data;
    const bool has_k = lora_k.has_value() && dA_k.Data;
    const bool has_v = lora_v.has_value() && dA_v.Data;

    if (!has_q && !has_k && !has_v) return;

    (void)intermediate2;

    // Offsets into packed QKV gradient tensor
    const int q_offset = 0;
    const int k_offset = q_out_features;
    const int v_offset = q_out_features + kv_out_features;

    auto backward_proj = [&](bool enabled,
                             Tensor& dA, Tensor& dB,
                             const LoRALayerWeights<Tensor>& lora,
                             int offset, int out_features,
                             unsigned int seed) {
        if (!enabled) return;
        backward_lora_layer(
            dA, dB,
            dx,
            dL_dy, offset,
            x,
            lora.A, lora.B,
            scaling,
            dropout_prob, seed, is_training,
            intermediate1, slice_buffer,
            BT, in_features, out_features, rank, accumulate,
            handle, workspace, stream);
    };

    backward_proj(has_q, dA_q, dB_q, lora_q, q_offset, q_out_features, dropout_seed_q);
    backward_proj(has_k, dA_k, dB_k, lora_k, k_offset, kv_out_features, dropout_seed_k);
    backward_proj(has_v, dA_v, dB_v, lora_v, v_offset, kv_out_features, dropout_seed_v);
}

/**
 * @brief Fused backward pass for MLP LoRA projections (gate, up, down)
 *
 * Optimizes MLP backward by processing gate and up together (shared input x=ln2)
 * and down separately (input x=swiglu).
 *
 * For gate/up (shared x = ln2, shared dL_dy = d_mlp_up):
 *   dA_gate = x^T @ (dL_dy[D:] @ B_gate^T) * scaling
 *   dA_up   = x^T @ (dL_dy[:D] @ B_up^T) * scaling
 *   dx_ln2 += contributions from both
 *
 * For down (x = swiglu, dL_dy = d_res_ffn):
 *   dA_down = swiglu^T @ (dL_dy @ B_down^T) * scaling
 *   dx_swiglu += contribution
 */
inline void backward_lora_mlp_up_gate_fused(
    // Gradient outputs for up
    Tensor& dA_up, Tensor& dB_up,
    // Gradient outputs for gate
    Tensor& dA_gate, Tensor& dB_gate,
    // Input gradient accumulator (d_ln2)
    Tensor& dx,
    // Upstream gradient (packed up+gate from SwiGLU backward)
    const Tensor& dL_dy,
    // Forward input (ln2 output, shared across up and gate)
    const Tensor& x,
    // LoRA weights
    const LoRALayerWeights<Tensor>& lora_up,
    const LoRALayerWeights<Tensor>& lora_gate,
    // Dimensions
    float scaling,
    float dropout_prob,
    unsigned int dropout_seed_up,
    unsigned int dropout_seed_gate,
    bool is_training,
    int BT,
    int in_features,    // C (hidden size)
    int out_features,   // D (intermediate size)
    int rank,
    bool accumulate,
    // Intermediates
    Tensor& intermediate1,
    Tensor& intermediate2,
    Tensor& slice_buffer,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    const bool has_up = lora_up.has_value() && dA_up.Data;
    const bool has_gate = lora_gate.has_value() && dA_gate.Data;

    if (!has_up && !has_gate) return;

    // dL_dy is packed as [d_up (D), d_gate (D)] with full_features columns per row.
    // Instead of copying slices via cudaMemcpy2D, we create view tensors and use
    // strided matmul (ldb = full_features) to read directly from the packed layout.
    const long full_features = dL_dy.Sizes[dL_dy.Rank - 1];
    const std::size_t elem_size = get_dtype_size(dL_dy.DType);
    const int ldb_stride = (int)full_features;  // leading dim override for strided dL_dy reads

    // Zero-copy dL_dy views (just pointer arithmetic, no memory copy)
    auto make_dL_dy_view = [&](int offset) -> Tensor {
        Tensor view = dL_dy;
        view.Data = dL_dy.Data + (std::size_t)offset * elem_size;
        view.Sizes[view.Rank - 1] = out_features;
        return view;
    };

    // Helper: process one projection's backward pass using strided dL_dy reads.
    // Computes h, dB, g, dA into the provided outputs; leaves g in intermediate2.
    auto process_projection = [&](
        const LoRALayerWeights<Tensor>& lora,
        Tensor& dA, Tensor& dB,
        const Tensor& dL_dy_view,
        unsigned int dropout_seed)
    {
        // intermediate1 = x @ A^T  (recompute forward activation)
        matmul(intermediate1, lora.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);

        if (is_training && dropout_prob > 0.0f) {
            lora_dropout_scale(intermediate1, dropout_prob, dropout_seed, stream);
        }
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB = intermediate1^T @ dL_dy  (strided read from packed gradient)
        matmul_strided(dB, intermediate1, dL_dy_view, std::nullopt, nullptr, nullptr,
                       handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate,
                       /*lda=*/-1, /*ldb=*/ldb_stride, /*ldc=*/-1, stream);

        // intermediate2 = dL_dy @ B^T  (strided read from packed gradient)
        matmul_strided(intermediate2, lora.B, dL_dy_view, std::nullopt, nullptr, nullptr,
                       handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false,
                       /*lda=*/-1, /*ldb=*/ldb_stride, /*ldc=*/-1, stream);

        if (is_training && dropout_prob > 0.0f) {
            lora_dropout_scale(intermediate2, dropout_prob, dropout_seed, stream);
        }
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA = x^T @ intermediate2
        matmul(dA, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);
    };

    // When both projections are present, fuse the dx accumulation:
    // dx += g_up @ A_up + g_gate @ A_gate  →  dx += g_cat @ A_cat  (single GEMM)
    if (has_up && has_gate) {
        // Carve g_cat (BT, 2*rank) and A_cat (2*rank, C) from slice_buffer.
        const std::size_t g_cat_elems = (std::size_t)BT * 2 * rank;
        const std::size_t a_cat_elems = (std::size_t)2 * rank * in_features;

        Tensor g_cat;
        g_cat.DType = dL_dy.DType;
        g_cat.Rank = 2;
        g_cat.Sizes[0] = BT;
        g_cat.Sizes[1] = 2 * rank;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) g_cat.Sizes[i] = 1;
        g_cat.Data = slice_buffer.Data;

        Tensor A_cat;
        A_cat.DType = lora_up.A.DType;
        A_cat.Rank = 2;
        A_cat.Sizes[0] = 2 * rank;
        A_cat.Sizes[1] = in_features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) A_cat.Sizes[i] = 1;
        A_cat.Data = slice_buffer.Data + g_cat_elems * elem_size;

        // --- Phase 1: up projection ---
        Tensor dL_dy_up = make_dL_dy_view(0);
        process_projection(lora_up, dA_up, dB_up, dL_dy_up, dropout_seed_up);

        // Pack g_up (contiguous BT x rank in intermediate2) into g_cat[:, 0:r]
        CUDA_CHECK(cudaMemcpy2DAsync(
            g_cat.Data,                                        // dst
            (std::size_t)(2 * rank) * elem_size,               // dst pitch
            intermediate2.Data,                                // src
            (std::size_t)rank * elem_size,                     // src pitch
            (std::size_t)rank * elem_size,                     // width
            (std::size_t)BT,                                   // height
            cudaMemcpyDeviceToDevice, stream));

        // --- Phase 2: gate projection ---
        Tensor dL_dy_gate = make_dL_dy_view(out_features);
        process_projection(lora_gate, dA_gate, dB_gate, dL_dy_gate, dropout_seed_gate);

        // Pack g_gate (contiguous BT x rank in intermediate2) into g_cat[:, r:2r]
        CUDA_CHECK(cudaMemcpy2DAsync(
            g_cat.Data + (std::size_t)rank * elem_size,        // dst (offset to second half)
            (std::size_t)(2 * rank) * elem_size,               // dst pitch
            intermediate2.Data,                                // src
            (std::size_t)rank * elem_size,                     // src pitch
            (std::size_t)rank * elem_size,                     // width
            (std::size_t)BT,                                   // height
            cudaMemcpyDeviceToDevice, stream));

        // --- Phase 3: fused dx accumulation ---
        // Concatenate A_up and A_gate into A_cat = [A_up; A_gate] (contiguous rows)
        CUDA_CHECK(cudaMemcpyAsync(
            A_cat.Data,
            lora_up.A.Data,
            (std::size_t)rank * in_features * elem_size,
            cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            A_cat.Data + (std::size_t)rank * in_features * elem_size,
            lora_gate.A.Data,
            (std::size_t)rank * in_features * elem_size,
            cudaMemcpyDeviceToDevice, stream));

        // dx += g_cat @ A_cat  (single GEMM: M=C, N=BT, K=2r)
        matmul(dx, A_cat, g_cat, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, 2 * rank, EMMTranspose::NN, /*accumulate=*/true, stream);

    } else if (has_up) {
        // Single projection: strided dL_dy + standard dx accumulation
        Tensor dL_dy_up = make_dL_dy_view(0);
        process_projection(lora_up, dA_up, dB_up, dL_dy_up, dropout_seed_up);

        matmul(dx, lora_up.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);

    } else { // has_gate only
        Tensor dL_dy_gate = make_dL_dy_view(out_features);
        process_projection(lora_gate, dA_gate, dB_gate, dL_dy_gate, dropout_seed_gate);

        matmul(dx, lora_gate.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }
}

} // namespace detail
} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_MODEL_UTILS_H
