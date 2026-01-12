// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_LORA_BACKWARD_KERNELS_H
#define SUROGATE_SRC_MODULES_LORA_LORA_BACKWARD_KERNELS_H

#include <stdexcept>
#include "kernels/kernels.h"
#include "lora_weights.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {
namespace detail {

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

    // intermediate = x @ A^T (BT x rank)
    matmul(intermediate, A, x, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
    if (scaling != 1.0f) {
        vector_add_sr(intermediate, intermediate, intermediate, 0.5f * scaling, intermediate.nelem(), /*seed=*/0, stream);
    }

    // dB = (x @ A^T)^T @ dL_dy
    matmul(dB, intermediate, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

    // intermediate = B @ dL_dy^T  => (BT x rank) view
    matmul(intermediate, B, dL_dy_slice, std::nullopt, nullptr, nullptr,
           handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
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
    int N,            // Number of tokens for this expert
    int C,            // Hidden size
    int D,            // Expert intermediate size
    int rank,
    bool accumulate,
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    if (N <= 0) return;

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
            intermediate, slice_buffer,
            N, D, C, rank, accumulate,
            handle, workspace, stream);
    }

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
            intermediate, slice_buffer,
            N, C, D, rank, accumulate,
            handle, workspace, stream);
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

    // Offsets into packed QKV gradient tensor
    const int q_offset = 0;
    const int k_offset = q_out_features;
    const int v_offset = q_out_features + kv_out_features;
    const long full_qkv_features = dL_dy.Sizes[dL_dy.Rank - 1];

    // Helper to extract a slice from packed QKV gradient
    auto extract_slice = [&](int offset, int features) -> Tensor {
        if (offset == 0 && features == full_qkv_features) {
            return dL_dy;
        }
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_qkv_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)features * elem_size;
        const std::size_t width = (std::size_t)features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_ptr, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));
        return packed;
    };

    // =======================================================================
    // Phase 1 & 2: For each projection, compute x @ A^T, then dB
    // This reuses x across all projections
    // =======================================================================

    if (has_q) {
        Tensor dL_dy_q = extract_slice(q_offset, q_out_features);

        // intermediate1 = x @ A_q^T (BT x rank)
        matmul(intermediate1, lora_q.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_q = intermediate1^T @ dL_dy_q
        matmul(dB_q, intermediate1, dL_dy_q, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, q_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_q @ dL_dy_q^T (for dA_q and dx)
        matmul(intermediate2, lora_q.B, dL_dy_q, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, q_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_q = x^T @ intermediate2
        matmul(dA_q, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_q
        matmul(dx, lora_q.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    if (has_k) {
        Tensor dL_dy_k = extract_slice(k_offset, kv_out_features);

        // intermediate1 = x @ A_k^T (BT x rank)
        matmul(intermediate1, lora_k.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_k = intermediate1^T @ dL_dy_k
        matmul(dB_k, intermediate1, dL_dy_k, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, kv_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_k @ dL_dy_k^T
        matmul(intermediate2, lora_k.B, dL_dy_k, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, kv_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_k = x^T @ intermediate2
        matmul(dA_k, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_k
        matmul(dx, lora_k.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    if (has_v) {
        Tensor dL_dy_v = extract_slice(v_offset, kv_out_features);

        // intermediate1 = x @ A_v^T (BT x rank)
        matmul(intermediate1, lora_v.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_v = intermediate1^T @ dL_dy_v
        matmul(dB_v, intermediate1, dL_dy_v, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, kv_out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_v @ dL_dy_v^T
        matmul(intermediate2, lora_v.B, dL_dy_v, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, kv_out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_v = x^T @ intermediate2
        matmul(dA_v, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_v
        matmul(dx, lora_v.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }
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

    // dL_dy is packed as [d_up (D), d_gate (D)]
    const int up_offset = 0;
    const int gate_offset = out_features;
    const long full_features = dL_dy.Sizes[dL_dy.Rank - 1];

    auto extract_slice = [&](int offset, int features) -> Tensor {
        if (offset == 0 && features == full_features) {
            return dL_dy;
        }
        Tensor packed = slice_buffer;
        packed.DType = dL_dy.DType;
        packed.Rank = 2;
        packed.Sizes[0] = BT;
        packed.Sizes[1] = features;
        for (int i = 2; i < MAX_TENSOR_DIM; ++i) packed.Sizes[i] = 1;

        const std::size_t elem_size = get_dtype_size(dL_dy.DType);
        const std::size_t src_pitch = (std::size_t)full_features * elem_size;
        const std::size_t dst_pitch = (std::size_t)features * elem_size;
        const std::size_t width = (std::size_t)features * elem_size;
        const std::byte* src_ptr = dL_dy.Data + (std::size_t)offset * elem_size;
        CUDA_CHECK(cudaMemcpy2DAsync(packed.Data, dst_pitch, src_pitch, src_pitch, width, (std::size_t)BT,
                                     cudaMemcpyDeviceToDevice, stream));
        return packed;
    };

    // Process up projection
    if (has_up) {
        Tensor dL_dy_up = extract_slice(up_offset, out_features);

        // intermediate1 = x @ A_up^T
        matmul(intermediate1, lora_up.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_up = intermediate1^T @ dL_dy_up
        matmul(dB_up, intermediate1, dL_dy_up, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_up @ dL_dy_up^T
        matmul(intermediate2, lora_up.B, dL_dy_up, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_up = x^T @ intermediate2
        matmul(dA_up, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_up
        matmul(dx, lora_up.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }

    // Process gate projection
    if (has_gate) {
        Tensor dL_dy_gate = extract_slice(gate_offset, out_features);

        // intermediate1 = x @ A_gate^T
        matmul(intermediate1, lora_gate.A, x, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, in_features, EMMTranspose::TN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate1, intermediate1, intermediate1, 0.5f * scaling, intermediate1.nelem(), /*seed=*/0, stream);
        }

        // dB_gate = intermediate1^T @ dL_dy_gate
        matmul(dB_gate, intermediate1, dL_dy_gate, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, out_features, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // intermediate2 = B_gate @ dL_dy_gate^T
        matmul(intermediate2, lora_gate.B, dL_dy_gate, std::nullopt, nullptr, nullptr,
               handle, workspace, rank, BT, out_features, EMMTranspose::NN, /*accumulate=*/false, stream);
        if (scaling != 1.0f) {
            vector_add_sr(intermediate2, intermediate2, intermediate2, 0.5f * scaling, intermediate2.nelem(), /*seed=*/0, stream);
        }

        // dA_gate = x^T @ intermediate2
        matmul(dA_gate, x, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, rank, BT, EMMTranspose::NT, /*accumulate=*/accumulate, stream);

        // dx += intermediate2 @ A_gate
        matmul(dx, lora_gate.A, intermediate2, std::nullopt, nullptr, nullptr,
               handle, workspace, in_features, BT, rank, EMMTranspose::NN, /*accumulate=*/true, stream);
    }
}

} // namespace detail
} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_LORA_BACKWARD_KERNELS_H
