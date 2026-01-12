// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H
#define SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H

/**
 * @file fast_expert_lora.h
 * @brief Fast LoRA fusion for MoE experts.
 */

#include "lora_types.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {
namespace detail {

/**
 * @brief State saved during fast expert LoRA forward pass. (Legacy)
 */
struct FastExpertLoRAState {
    Tensor e;           ///< Gate output before SiLU (N, D) - becomes de after backward
    Tensor g;           ///< Up output before SiLU (N, D) - becomes dg after backward
    Tensor input;       ///< Expert input (N, C) - needed for LoRA gradient computation
    int N = 0;          ///< Number of tokens for this expert
    int C = 0;          ///< Hidden size
    int D = 0;          ///< Intermediate size
};

/**
 * @brief Forward pass for a SINGLE expert using cuBLASLt.
 */
inline void fast_expert_lora_forward(
    Tensor& output,                   ///< (N, C)
    const Tensor& input,              ///< (N, C)
    const Tensor& gate_up_proj,       ///< (2*D, C)
    const Tensor& down_proj,          ///< (C, D)
    const LoRAExpertWeights<Tensor>& lora,
    FastExpertLoRAState& state,
    float scaling,
    int N, int C, int D, int lora_rank,
    Tensor& lora_intermediate,         ///< (N, rank)
    Tensor& h_buffer,                  ///< (N, D)
    Tensor& gate_up_buffer,            ///< (N, 2*D)
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    state.N = N; state.C = C; state.D = D;
    state.input = input;

    // 1. Base gate+up projection: gate_up = x @ W_gate_up^T
    matmul(gate_up_buffer, gate_up_proj, input, std::nullopt, nullptr, nullptr, handle, workspace, 2 * D, N, C, EMMTranspose::TN, false, stream);

    // 2. Split gate_up into e and g
    if (state.e.is_null()) {
        state.e = Tensor::empty(h_buffer.DType, {(long)N, (long)D});
        state.g = Tensor::empty(h_buffer.DType, {(long)N, (long)D});
    }
    split_gate_up(gate_up_buffer, state.g, state.e, N, D, stream);

    // 3. Apply LoRA for Gate and Up
    if (lora.gate.has_value() && lora.gate->has_value()) {
        matmul(lora_intermediate, lora.gate->A, input, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate, lora_intermediate, lora_intermediate, 0.5f * scaling, lora_intermediate.nelem(), 0, stream);
        matmul(state.e, lora.gate->B, lora_intermediate, std::nullopt, nullptr, nullptr, handle, workspace, D, N, lora_rank, EMMTranspose::TN, true, stream);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        matmul(lora_intermediate, lora.up->A, input, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate, lora_intermediate, lora_intermediate, 0.5f * scaling, lora_intermediate.nelem(), 0, stream);
        matmul(state.g, lora.up->B, lora_intermediate, std::nullopt, nullptr, nullptr, handle, workspace, D, N, lora_rank, EMMTranspose::TN, true, stream);
    }

    // 4. h = silu(e) * g
    silu_mul_forward(h_buffer, state.e, state.g, N, D, stream);

    // 5. Down projection: y = h @ W_down^T + lora_down(h)
    matmul(output, down_proj, h_buffer, std::nullopt, nullptr, nullptr, handle, workspace, C, N, D, EMMTranspose::TN, false, stream);

    if (lora.down.has_value() && lora.down->has_value()) {
        matmul(lora_intermediate, lora.down->A, h_buffer, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate, lora_intermediate, lora_intermediate, 0.5f * scaling, lora_intermediate.nelem(), 0, stream);
        matmul(output, lora.down->B, lora_intermediate, std::nullopt, nullptr, nullptr, handle, workspace, C, N, lora_rank, EMMTranspose::TN, true, stream);
    }
}

/**
 * @brief Backward pass for a SINGLE expert using cuBLASLt.
 */
inline void fast_expert_lora_backward(
    LoRAExpertWeights<Tensor>& lora_grads,
    Tensor& dx,                       ///< (N, C)
    const Tensor& dy,                 ///< (N, C)
    const Tensor& gate_up_proj,       ///< (2*D, C)
    const Tensor& down_proj,          ///< (C, D)
    const LoRAExpertWeights<Tensor>& lora,
    FastExpertLoRAState& state,
    float scaling,
    int lora_rank,
    bool accumulate,
    Tensor& lora_intermediate1,        ///< (N, rank)
    Tensor& lora_intermediate2,        ///< (N, D)
    Tensor& d_gate_up_buffer,          ///< (N, 2*D)
    cublasLtHandle_t handle,
    Tensor& workspace,
    cudaStream_t stream) {

    const int N = state.N;
    const int C = state.C;
    const int D = state.D;

    // 1. dh = dy @ W_down
    Tensor dh = lora_intermediate2;
    matmul(dh, down_proj, dy, std::nullopt, nullptr, nullptr, handle, workspace, D, N, C, EMMTranspose::NN, false, stream);

    if (lora.down.has_value() && lora.down->has_value()) {
        matmul(lora_intermediate1, lora.down->B, dy, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(dh, lora.down->A, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, D, N, lora_rank, EMMTranspose::NN, true, stream);
    }

    // 2. In-place SiLU backward: e->de, g->dg, reconstruct h
    Tensor h;
    h.Data = d_gate_up_buffer.Data;
    h.DType = state.e.DType;
    h.Sizes = {N, D};
    silu_mul_backward_inplace(state.e, state.g, dh, &h, N, D, stream);

    // 3. Down LoRA gradients
    if (lora.down.has_value() && lora.down->has_value() && lora_grads.down.has_value()) {
        matmul(lora_intermediate1, h, lora.down->A, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.down->B, dy, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, C, lora_rank, N, EMMTranspose::TN, accumulate, stream);

        matmul(lora_intermediate1, dy, lora.down->B, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.down->A, lora_intermediate1, h, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, D, N, EMMTranspose::TN, accumulate, stream);
    }

    // 4. Gate/Up LoRA gradients
    if (lora.gate.has_value() && lora.gate->has_value() && lora_grads.gate.has_value()) {
        matmul(lora_intermediate1, state.input, lora.gate->A, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.gate->B, state.e, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, D, lora_rank, N, EMMTranspose::TN, accumulate, stream);

        matmul(lora_intermediate1, state.e, lora.gate->B, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.gate->A, lora_intermediate1, state.input, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, C, N, EMMTranspose::TN, accumulate, stream);
    }

    if (lora.up.has_value() && lora.up->has_value() && lora_grads.up.has_value()) {
        matmul(lora_intermediate1, state.input, lora.up->A, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, C, EMMTranspose::TN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.up->B, state.g, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, D, lora_rank, N, EMMTranspose::TN, accumulate, stream);

        matmul(lora_intermediate1, state.g, lora.up->B, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(lora_grads.up->A, lora_intermediate1, state.input, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, C, N, EMMTranspose::TN, accumulate, stream);
    }

    // 5. dx = [dg | de] @ W_gate_up + LoRA contribs
    concat_d_gate_up(state.g, state.e, d_gate_up_buffer, N, D, stream);
    matmul(dx, gate_up_proj, d_gate_up_buffer, std::nullopt, nullptr, nullptr, handle, workspace, C, N, 2 * D, EMMTranspose::NN, false, stream);

    if (lora.gate.has_value() && lora.gate->has_value()) {
        matmul(lora_intermediate1, state.e, lora.gate->B, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(dx, lora.gate->A, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, C, N, lora_rank, EMMTranspose::NN, true, stream);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        matmul(lora_intermediate1, state.g, lora.up->B, std::nullopt, nullptr, nullptr, handle, workspace, lora_rank, N, D, EMMTranspose::NN, false, stream);
        if (scaling != 1.0f) vector_add_sr(lora_intermediate1, lora_intermediate1, lora_intermediate1, 0.5f * scaling, lora_intermediate1.nelem(), 0, stream);
        matmul(dx, lora.up->A, lora_intermediate1, std::nullopt, nullptr, nullptr, handle, workspace, C, N, lora_rank, EMMTranspose::NN, true, stream);
    }
}

/**
 * @brief Forward pass for ALL experts using Grouped GEMM and Batched LoRA.
 */
inline void grouped_fast_expert_lora_forward(
    Tensor& total_output,             ///< (total_tokens, C)
    const Tensor& permuted_input,      ///< (total_tokens, C)
    const Tensor& batched_gate_up_proj, ///< (num_experts, 2*D, C)
    const Tensor& batched_down_proj,    ///< (num_experts, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_e,                   ///< (total_tokens, D) - Saved state (gate output)
    Tensor& total_g,                   ///< (total_tokens, D) - Saved state (up output)
    const Tensor& expert_offsets,     ///< (num_experts + 1)
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    Tensor& lora_intermediate,         ///< (total_tokens, rank)
    Tensor& h_buffer,                  ///< (total_tokens, D)
    Tensor& gate_up_buffer,            ///< (total_tokens, 2*D)
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr) {

    int total_tokens = gate_up_buffer.Sizes[0];

    // DEBUG: Log expert token distribution
    static int call_count = 0;
    if (call_count < 10) {  // Only log first 10 calls
        std::vector<int> h_offsets(num_experts + 1);
        if (host_offsets) {
            std::memcpy(h_offsets.data(), host_offsets, (num_experts + 1) * sizeof(int));
        } else {
            cudaMemcpyAsync(h_offsets.data(), expert_offsets.get<int>(),
                          (num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        fprintf(stderr, "[MoE LoRA FWD Call %d] Total tokens: %d, Num experts: %d\n",
                call_count, total_tokens, num_experts);
        fprintf(stderr, "[MoE LoRA FWD] Expert distribution: ");
        int experts_with_tokens = 0;
        for (int e = 0; e < num_experts; ++e) {
            int tokens_e = h_offsets[e + 1] - h_offsets[e];
            if (tokens_e > 0) experts_with_tokens++;
            fprintf(stderr, "E%d:%d ", e, tokens_e);
        }
        fprintf(stderr, "\n[MoE LoRA FWD] Experts with tokens: %d/%d\n", experts_with_tokens, num_experts);
        call_count++;
    }

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight, int M, int K, float alpha, float beta, EMMTranspose mode) {
        // Check for dtype consistency - all tensors must match for grouped GEMM
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error("MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                     "Set lora_dtype='bf16' in your config to match activation dtype.");
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode);
        } else {
            moe_grouped_gemm(out.get<float>(), in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode);
        }
    };

    // Check for prior CUDA errors
    cudaError_t prior_err = cudaGetLastError();
    if (prior_err != cudaSuccess) {
        fprintf(stderr, "[FAST LORA DEBUG] Prior CUDA error: %s\n", cudaGetErrorString(prior_err));
    }

    // 1. Base gate+up projection: gate_up = x @ W_gate_up^T
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(
            gate_up_buffer.get<nv_bfloat16>(),
            permuted_input.get<nv_bfloat16>(),
            batched_gate_up_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    } else {
        moe_grouped_gemm_gate_up(
            gate_up_buffer.get<float>(),
            permuted_input.get<float>(),
            batched_gate_up_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    }

    // DEBUG: Check for errors after gate_up GEMM
    cudaError_t gemm_err = cudaStreamSynchronize(stream);
    if (gemm_err != cudaSuccess) {
        fprintf(stderr, "[FAST LORA DEBUG] Error after moe_grouped_gemm_gate_up: %s\n",
                cudaGetErrorString(gemm_err));
    }

    // 2. Split gate_up_buffer into contiguous total_g (up) and total_e (gate) buffers
    // gate_up_buffer layout: [up (D cols) | gate (D cols)] per row (row-major)
    // Note: split_gate_up expects output: first param is up, second is gate
    // Sync before kernel to catch any previous async errors
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[FAST LORA DEBUG] CUDA sync error before split_gate_up: %s\n",
                cudaGetErrorString(sync_err));
    }

    split_gate_up(gate_up_buffer, total_g, total_e, total_tokens, D, stream);

    // Sync after kernel to see if error is in this kernel specifically
    sync_err = cudaStreamSynchronize(stream);
    if (sync_err != cudaSuccess) {
        fprintf(stderr, "[FAST LORA DEBUG] CUDA sync error after split_gate_up: %s\n",
                cudaGetErrorString(sync_err));
    }

    // 3. Apply Grouped LoRA for Gate and Up (modifies total_e and total_g in-place)
    // NOTE: For experts with zero tokens, the LoRA weights will NOT be updated in this forward pass.
    // The gradients for these experts will remain zero (or accumulated from previous micro-steps).
    if (lora.gate.has_value() && lora.gate->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, permuted_input, lora.gate->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_grouped_gemm(total_e, lora_intermediate, lora.gate->B, D, lora_rank, scaling, 1.0f, EMMTranspose::TN);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, permuted_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_grouped_gemm(total_g, lora_intermediate, lora.up->B, D, lora_rank, scaling, 1.0f, EMMTranspose::TN);
    }

    // 4. h = silu(e) * g (use contiguous total_e and total_g buffers)
    silu_mul_forward(h_buffer, total_e, total_g, total_tokens, D, stream);

    // 5. Down projection: y = h @ W_down^T + lora_down(h)
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(
            total_output.get<nv_bfloat16>(),
            h_buffer.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    } else {
        moe_grouped_gemm_down(
            total_output.get<float>(),
            h_buffer.get<float>(),
            batched_down_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, h_buffer, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_grouped_gemm(total_output, lora_intermediate, lora.down->B, C, lora_rank, scaling, 1.0f, EMMTranspose::TN);
    }
}

/**
 * @brief Backward pass for ALL experts using Grouped GEMM and Batched LoRA.
 */
inline void grouped_fast_expert_lora_backward(
    LoRAGroupedExpertWeights<Tensor>& lora_grads,
    Tensor& total_dx,                 ///< (total_tokens, C)
    const Tensor& total_dy,            ///< (total_tokens, C)
    const Tensor& batched_gate_up_proj, ///< (num_experts, 2*D, C)
    const Tensor& batched_down_proj,    ///< (num_experts, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_e,                   ///< (total_tokens, D) - will become de
    Tensor& total_g,                   ///< (total_tokens, D) - will become dg
    const Tensor& total_input,         ///< (total_tokens, C)
    const Tensor& expert_offsets,     ///< (num_experts + 1)
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    bool accumulate,
    Tensor& lora_intermediate1,        ///< (total_tokens, rank)
    Tensor& lora_intermediate2,        ///< (total_tokens, D)
    Tensor& d_gate_up_buffer,          ///< (total_tokens, 2*D)
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr) {

    const int total_tokens = total_dy.Sizes[0];

    // DEBUG: Log expert token distribution in backward
    static int bwd_call_count = 0;
    if (bwd_call_count < 10) {  // Only log first 10 calls
        std::vector<int> h_offsets(num_experts + 1);
        if (host_offsets) {
            std::memcpy(h_offsets.data(), host_offsets, (num_experts + 1) * sizeof(int));
        } else {
            cudaMemcpyAsync(h_offsets.data(), expert_offsets.get<int>(),
                          (num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }

        fprintf(stderr, "[MoE LoRA BWD Call %d] Total tokens: %d, Num experts: %d, Accumulate: %d\n",
                bwd_call_count, total_tokens, num_experts, accumulate);
        fprintf(stderr, "[MoE LoRA BWD] Expert distribution: ");
        int experts_with_tokens = 0;
        for (int e = 0; e < num_experts; ++e) {
            int tokens_e = h_offsets[e + 1] - h_offsets[e];
            if (tokens_e > 0) experts_with_tokens++;
            fprintf(stderr, "E%d:%d ", e, tokens_e);
        }
        fprintf(stderr, "\n[MoE LoRA BWD] Experts with tokens: %d/%d\n", experts_with_tokens, num_experts);
        bwd_call_count++;
    }

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight, int M, int K, float alpha, float beta, EMMTranspose mode) {
        // Check for dtype consistency - all tensors must match for grouped GEMM
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error("MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                     "Set lora_dtype='bf16' in your config to match activation dtype.");
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>(), in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode);
        } else {
            moe_grouped_gemm(out.get<float>(), in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode);
        }
    };

    auto dispatch_weight_grad = [&](Tensor& d_weight, const Tensor& dy, const Tensor& in, int M, int N, float alpha, float beta) {
        // Activations (dy, in) use activation dtype (typically BF16)
        // Gradients (d_weight) may use a different dtype (e.g., FP32 for precision)
        // The GEMM must use the activation dtype, then we accumulate into gradient dtype
        if (dy.DType == ETensorDType::BF16) {
            // Activations are BF16 - compute GEMM in BF16
            if (d_weight.DType == ETensorDType::BF16) {
                // Both BF16 - direct accumulation
                moe_grouped_gemm_weight_grad(d_weight.get<nv_bfloat16>(), dy.get<nv_bfloat16>(), in.get<nv_bfloat16>(),
                                             expert_offsets.get<int>(), num_experts, M, N, handle, stream, host_offsets,
                                             alpha, beta);
            } else {
                // Activations BF16, gradients FP32 - compute in BF16 then accumulate to FP32
                // cuBLAS supports mixed precision accumulation with CUBLAS_COMPUTE_32F
                // For now, compute in BF16 directly into the BF16 portion then cast
                // TODO: Add proper mixed-precision support with FP32 accumulator
                // Workaround: use the same dtype for computation (FP32 grads require FP32 activations)
                // This is a limitation - for now, throw a clear error
                throw std::runtime_error("MoE LoRA backward: lora_dtype=fp32 with bf16 activations not yet supported. "
                                         "Set lora_dtype='bf16' in your config.");
            }
        } else {
            // Activations are FP32
            moe_grouped_gemm_weight_grad(d_weight.get<float>(), dy.get<float>(), in.get<float>(),
                                         expert_offsets.get<int>(), num_experts, M, N, handle, stream, host_offsets,
                                         alpha, beta);
        }
    };

    // 1. dh = dy @ W_down (no transpose on weight if we use moe_grouped_gemm_down_backward)
    Tensor total_dh = lora_intermediate2;
    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down_backward(
            total_dh.get<nv_bfloat16>(), total_dy.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    } else {
        moe_grouped_gemm_down_backward(
            total_dh.get<float>(), total_dy.get<float>(),
            batched_down_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_grouped_gemm(total_dh, lora_intermediate1, lora.down->A, D, lora_rank, scaling, 1.0f, EMMTranspose::NN);
    }

    // 2. In-place SiLU backward: e->de, g->dg, reconstruct h
    Tensor total_h;
    total_h.Data = d_gate_up_buffer.Data;
    total_h.DType = total_e.DType;
    total_h.Sizes = {total_tokens, D};
    silu_mul_backward_inplace(total_e, total_g, total_dh, &total_h, total_tokens, D, stream);

    // 3. Down LoRA gradients
    if (lora.down.has_value() && lora.down->has_value() && lora_grads.down.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_h, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_weight_grad(lora_grads.down->B, total_dy, lora_intermediate1, C, lora_rank, scaling, accumulate ? 1.0f : 0.0f);

        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_weight_grad(lora_grads.down->A, lora_intermediate1, total_h, lora_rank, D, scaling, accumulate ? 1.0f : 0.0f);
    }

    // 4. Gate/Up LoRA gradients
    if (lora.gate.has_value() && lora.gate->has_value() && lora_grads.gate.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_input, lora.gate->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_weight_grad(lora_grads.gate->B, total_e, lora_intermediate1, D, lora_rank, scaling, accumulate ? 1.0f : 0.0f);

        dispatch_grouped_gemm(lora_intermediate1, total_e, lora.gate->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_weight_grad(lora_grads.gate->A, lora_intermediate1, total_input, lora_rank, C, scaling, accumulate ? 1.0f : 0.0f);
    }

    if (lora.up.has_value() && lora.up->has_value() && lora_grads.up.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN);
        dispatch_weight_grad(lora_grads.up->B, total_g, lora_intermediate1, D, lora_rank, scaling, accumulate ? 1.0f : 0.0f);

        dispatch_grouped_gemm(lora_intermediate1, total_g, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_weight_grad(lora_grads.up->A, lora_intermediate1, total_input, lora_rank, C, scaling, accumulate ? 1.0f : 0.0f);
    }

    // 5. dx = [dg | de] @ W_gate_up + LoRA contribs
    concat_d_gate_up(total_g, total_e, d_gate_up_buffer, total_tokens, D, stream);

    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up_backward(
            total_dx.get<nv_bfloat16>(), d_gate_up_buffer.get<nv_bfloat16>(),
            batched_gate_up_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    } else {
        moe_grouped_gemm_gate_up_backward(
            total_dx.get<float>(), d_gate_up_buffer.get<float>(),
            batched_gate_up_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets
        );
    }

    if (lora.gate.has_value() && lora.gate->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_e, lora.gate->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_grouped_gemm(total_dx, lora_intermediate1, lora.gate->A, C, lora_rank, scaling, 1.0f, EMMTranspose::NN);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_g, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN);
        dispatch_grouped_gemm(total_dx, lora_intermediate1, lora.up->A, C, lora_rank, scaling, 1.0f, EMMTranspose::NN);
    }
}

} // namespace detail
} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H
