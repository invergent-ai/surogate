// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H
#define SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H

/**
 * @file fast_expert_lora.h
 * @brief Fast LoRA fusion for MoE experts.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <fmt/core.h>

#include "lora_types.h"
#include "kernels/kernels.h"
#include "modules/moe/moe_types.h"
#include "modules/model_config.h"
#include "utilities/dtype.h"
#include "utilities/utils.h"

namespace modules {
namespace detail {

/**
 * @brief State saved during fast expert LoRA forward pass. (Legacy)
 */
struct FastExpertLoRAState {
    Tensor e;           ///< Gate output before SiLU (N, D) - becomes d_gate after backward
    Tensor g;           ///< Up output before SiLU (N, D) - becomes d_up after backward
    Tensor input;       ///< Expert input (N, C) - needed for LoRA gradient computation
    int N = 0;          ///< Number of tokens for this expert
    int C = 0;          ///< Hidden size
    int D = 0;          ///< Intermediate size
};

inline bool sample_tensor_to_f32(const Tensor& t,
                                 std::size_t elem_offset,
                                 std::size_t count,
                                 std::vector<float>& out,
                                 cudaStream_t stream) {
    if (!t.Data) {
        return false;
    }
    const std::size_t total = static_cast<std::size_t>(t.nelem());
    if (total == 0 || elem_offset >= total) {
        return false;
    }
    const std::size_t n = std::min(count, total - elem_offset);
    out.assign(n, 0.0f);
    if (t.DType == ETensorDType::BF16) {
        std::vector<nv_bfloat16> tmp(n);
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(),
                                   reinterpret_cast<const nv_bfloat16*>(t.Data) + elem_offset,
                                   n * sizeof(nv_bfloat16),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = __bfloat162float(tmp[i]);
        }
        return true;
    }
    if (t.DType == ETensorDType::FP16) {
        std::vector<half> tmp(n);
        CUDA_CHECK(cudaMemcpyAsync(tmp.data(),
                                   reinterpret_cast<const half*>(t.Data) + elem_offset,
                                   n * sizeof(half),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = __half2float(tmp[i]);
        }
        return true;
    }
    if (t.DType == ETensorDType::FP32) {
        CUDA_CHECK(cudaMemcpyAsync(out.data(),
                                   reinterpret_cast<const float*>(t.Data) + elem_offset,
                                   n * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return true;
    }
    return false;
}

inline void log_sample_stats(const char* tag,
                             int layer_idx,
                             const std::vector<float>& vals) {
    if (vals.empty()) {
        return;
    }
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (float v : vals) {
        const float av = std::fabs(v);
        sum_abs += av;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
        if (av > max_abs) {
            max_abs = av;
        }
    }
    const float mean_abs = static_cast<float>(sum_abs / vals.size());
    const float l2 = static_cast<float>(std::sqrt(sum_sq));
    fprintf(stderr,
            "[MOE_LORA_SPLIT] layer=%d tag=%s n=%zu l2=%.6f max_abs=%.6f mean_abs=%.6f\n",
            layer_idx,
            tag ? tag : "<none>",
            vals.size(),
            l2,
            max_abs,
            mean_abs);
}

inline void log_sample_delta(const char* tag,
                             int layer_idx,
                             const std::vector<float>& base,
                             const std::vector<float>& post) {
    if (base.empty() || post.empty()) {
        return;
    }
    const std::size_t n = std::min(base.size(), post.size());
    double sum_abs = 0.0;
    double sum_sq = 0.0;
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float d = post[i] - base[i];
        const float ad = std::fabs(d);
        sum_abs += ad;
        sum_sq += static_cast<double>(d) * static_cast<double>(d);
        if (ad > max_abs) {
            max_abs = ad;
        }
    }
    const float mean_abs = static_cast<float>(sum_abs / n);
    const float l2 = static_cast<float>(std::sqrt(sum_sq));
    fprintf(stderr,
            "[MOE_LORA_SPLIT] layer=%d tag=%s n=%zu l2=%.6f max_abs=%.6f mean_abs=%.6f\n",
            layer_idx,
            tag ? tag : "<none>",
            n,
            l2,
            max_abs,
            mean_abs);
}

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
 *
 * @param selection_info If non-null, selective expert dequantization is enabled.
 *        The batched weight tensors contain only num_active experts in compact order.
 *        expert_offsets still uses global expert indices and needs remapping.
 */
inline void grouped_fast_expert_lora_forward(
    Tensor& total_output,             ///< (total_tokens, C)
    const Tensor& permuted_input,      ///< (total_tokens, C)
    const Tensor& batched_gate_up_proj, ///< (num_experts or num_active, 2*D, C) - BnB layout [up|gate]
    const Tensor& batched_down_proj,    ///< (num_experts or num_active, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_gate,                ///< (total_tokens, D) - Saved gate output (before SiLU)
    Tensor& total_up,                  ///< (total_tokens, D) - Saved up output
    const Tensor& expert_offsets,     ///< (num_experts + 1) - global expert indices
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    Tensor& lora_intermediate,         ///< (total_tokens, rank)
    Tensor& h_buffer,                  ///< (total_tokens, D)
    Tensor& gate_up_buffer,            ///< (total_tokens, 2*D)
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr,
    const SelectiveExpertInfo* selection_info = nullptr,
    int layer_idx = -1) {

    int total_tokens = gate_up_buffer.Sizes[0];

    // Handle selective expert dequantization
    // When enabled, batched_gate_up_proj/batched_down_proj contain only num_active experts
    const bool use_selective = selection_info && selection_info->enabled;
    const int gemm_num_experts = use_selective ? selection_info->num_active : num_experts;
    const int* active_indices = use_selective ? selection_info->active_experts.data() : nullptr;

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight, int M, int K, float alpha, float beta, EMMTranspose mode, bool weight_is_compact = true, int out_offset = 0) {
        // Check for dtype consistency - all tensors must match for grouped GEMM
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error("MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                     "Set lora_dtype='bf16' in your config to match activation dtype.");
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>() + out_offset, in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        } else {
            moe_grouped_gemm(out.get<float>() + out_offset, in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    // 1. Base gate+up projection: gate_up = x @ W_gate_up^T
    // When using selective dequant, batched weights have compact indexing
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up(
            gate_up_buffer.get<nv_bfloat16>(),
            permuted_input.get<nv_bfloat16>(),
            batched_gate_up_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    } else {
        moe_grouped_gemm_gate_up(
            gate_up_buffer.get<float>(),
            permuted_input.get<float>(),
            batched_gate_up_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    }

    // 2. Split gate_up_buffer into up and gate components
    // BnB weight loading (bnb_weights.cpp) stores MoE expert weights as [up | gate]:
    //   - up_proj in first D columns (offset 0)
    //   - gate_proj in second D columns (offset D)
    // split_gate_up(input, out_up, out_gate, ...) extracts both halves

    split_gate_up(gate_up_buffer, total_up, total_gate, total_tokens, D, stream);

    const bool moe_lora_split_trace = (std::getenv("SUROGATE_MOE_LORA_SPLIT_TRACE") != nullptr);
    static int moe_lora_split_count = 0;
    const bool do_trace = moe_lora_split_trace && (moe_lora_split_count < 4);
    const std::size_t sample_count = 1024;
    std::vector<float> base_gate_sample;
    std::vector<float> base_up_sample;
    std::vector<float> base_out_sample;
    if (do_trace) {
        if (sample_tensor_to_f32(total_gate, 0, sample_count, base_gate_sample, stream)) {
            log_sample_stats("GATE_BASE", layer_idx, base_gate_sample);
        }
        if (sample_tensor_to_f32(total_up, 0, sample_count, base_up_sample, stream)) {
            log_sample_stats("UP_BASE", layer_idx, base_up_sample);
        }
    }

    // 3. Apply Grouped LoRA for Gate and Up projections
    if (lora.gate.has_value() && lora.gate->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, permuted_input, lora.gate->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_grouped_gemm(total_gate, lora_intermediate, lora.gate->B, D, lora_rank, scaling, 1.0f, EMMTranspose::TN, false);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, permuted_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_grouped_gemm(total_up, lora_intermediate, lora.up->B, D, lora_rank, scaling, 1.0f, EMMTranspose::TN, false);
    }

    if (do_trace) {
        std::vector<float> post_gate_sample;
        std::vector<float> post_up_sample;
        if (sample_tensor_to_f32(total_gate, 0, sample_count, post_gate_sample, stream)) {
            log_sample_stats("GATE_POST", layer_idx, post_gate_sample);
            log_sample_delta("GATE_DELTA", layer_idx, base_gate_sample, post_gate_sample);
        }
        if (sample_tensor_to_f32(total_up, 0, sample_count, post_up_sample, stream)) {
            log_sample_stats("UP_POST", layer_idx, post_up_sample);
            log_sample_delta("UP_DELTA", layer_idx, base_up_sample, post_up_sample);
        }
    }

    // 4. Compute h = silu(gate) * up
    // silu_mul_forward(out, gate_in, up_in, ...) computes out = silu(gate_in) * up_in
    silu_mul_forward(h_buffer, total_gate, total_up, total_tokens, D, stream);

    // 5. Down projection: y = h @ W_down^T + lora_down(h)
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(
            total_output.get<nv_bfloat16>(),
            h_buffer.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    } else {
        moe_grouped_gemm_down(
            total_output.get<float>(),
            h_buffer.get<float>(),
            batched_down_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    }

    if (do_trace) {
        if (sample_tensor_to_f32(total_output, 0, sample_count, base_out_sample, stream)) {
            log_sample_stats("DOWN_BASE", layer_idx, base_out_sample);
        }
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, h_buffer, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_grouped_gemm(total_output, lora_intermediate, lora.down->B, C, lora_rank, scaling, 1.0f, EMMTranspose::TN, false);
    }

    if (do_trace) {
        std::vector<float> post_out_sample;
        if (sample_tensor_to_f32(total_output, 0, sample_count, post_out_sample, stream)) {
            log_sample_stats("DOWN_POST", layer_idx, post_out_sample);
            log_sample_delta("DOWN_DELTA", layer_idx, base_out_sample, post_out_sample);
        }
        moe_lora_split_count++;
    }
}

/**
 * @brief Backward pass for ALL experts using Grouped GEMM and Batched LoRA.
 *
 * @param selection_info If non-null, selective expert dequantization is enabled.
 *        The batched weight tensors contain only num_active experts in compact order.
 *        expert_offsets still uses global expert indices and needs remapping.
 */
inline void grouped_fast_expert_lora_backward(
    LoRAGroupedExpertWeights<Tensor>& lora_grads,
    Tensor& total_dx,                 ///< (total_tokens, C)
    const Tensor& total_dy,            ///< (total_tokens, C)
    const Tensor& batched_gate_up_proj, ///< (num_experts or num_active, 2*D, C) - BnB layout [up|gate]
    const Tensor& batched_down_proj,    ///< (num_experts or num_active, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_gate,                ///< (total_tokens, D) - gate from forward, becomes d_gate
    Tensor& total_up,                  ///< (total_tokens, D) - up from forward, becomes d_up
    const Tensor& total_input,         ///< (total_tokens, C)
    const Tensor& expert_offsets,     ///< (num_experts + 1) - global expert indices
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    bool accumulate,
    Tensor& lora_intermediate1,        ///< (total_tokens, rank)
    Tensor& lora_intermediate2,        ///< (total_tokens, D)
    Tensor& d_gate_up_buffer,          ///< (total_tokens, 2*D)
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr,
    const SelectiveExpertInfo* selection_info = nullptr) {

    const int total_tokens = total_dy.Sizes[0];

    // Handle selective expert dequantization
    // When enabled, batched_gate_up_proj/batched_down_proj contain only num_active experts
    const bool use_selective = selection_info && selection_info->enabled;
    const int gemm_num_experts = use_selective ? selection_info->num_active : num_experts;
    const int* active_indices = use_selective ? selection_info->active_experts.data() : nullptr;

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight, int M, int K, float alpha, float beta, EMMTranspose mode, bool weight_is_compact = true, int out_offset = 0) {
        // Check for dtype consistency - all tensors must match for grouped GEMM
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error(fmt::format(
                "MoE LoRA backward: dtype mismatch in grouped GEMM. in={}, weight={}, out={}. "
                "Set lora_dtype='bf16' in your config to match activation dtype.",
                dtype_to_str(in.DType), dtype_to_str(weight.DType), dtype_to_str(out.DType)));
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>() + out_offset, in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        } else {
            moe_grouped_gemm(out.get<float>() + out_offset, in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    auto dispatch_weight_grad = [&](Tensor& d_weight, const Tensor& dy, const Tensor& in, int M, int N, float alpha, float beta, bool weight_is_compact = true) {
        // Activations (dy, in) use activation dtype (typically BF16)
        // Gradients (d_weight) may use a different dtype (e.g., FP32 for precision)
        // The GEMM must use the activation dtype, then we accumulate into gradient dtype
        if (dy.DType == ETensorDType::BF16) {
            // Activations are BF16 - compute GEMM in BF16
            if (d_weight.DType == ETensorDType::BF16) {
                // Both BF16 - direct accumulation
                moe_grouped_gemm_weight_grad(d_weight.get<nv_bfloat16>(), dy.get<nv_bfloat16>(), in.get<nv_bfloat16>(),
                                             expert_offsets.get<int>(), num_experts, M, N, handle, stream, host_offsets,
                                             alpha, beta, active_indices, weight_is_compact, gemm_num_experts);
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
                                         alpha, beta, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    // 1. dh = dy @ W_down (no transpose on weight if we use moe_grouped_gemm_down_backward)
    Tensor total_dh = lora_intermediate2;
    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down_backward(
            total_dh.get<nv_bfloat16>(), total_dy.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    } else {
        moe_grouped_gemm_down_backward(
            total_dh.get<float>(), total_dy.get<float>(),
            batched_down_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_grouped_gemm(total_dh, lora_intermediate1, lora.down->A, D, lora_rank, scaling, 1.0f, EMMTranspose::NN, false);
    }

    // 2. In-place SiLU backward: convert gate/up activations to gradients
    // Forward was: h = silu(gate) * up
    // silu_mul_backward_inplace(gate, up, dh, h_out) computes:
    //   - gate -> d_gate (gradient w.r.t. gate input)
    //   - up -> d_up (gradient w.r.t. up input)
    //   - h_out (optional): reconstructed h for LoRA backward
    Tensor total_h;
    total_h.Data = d_gate_up_buffer.Data;
    total_h.DType = total_gate.DType;
    total_h.Sizes = {total_tokens, D};
    silu_mul_backward_inplace(total_gate, total_up, total_dh, &total_h, total_tokens, D, stream);
    // After this: total_gate = d_gate, total_up = d_up

    // 3. Down LoRA gradients
    if (lora.down.has_value() && lora.down->has_value() && lora_grads.down.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_h, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_weight_grad(lora_grads.down->B, total_dy, lora_intermediate1, C, lora_rank, scaling, accumulate ? 1.0f : 0.0f, false);

        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_weight_grad(lora_grads.down->A, lora_intermediate1, total_h, lora_rank, D, scaling, accumulate ? 1.0f : 0.0f, false);
    }

    // 4. Gate/Up LoRA gradients
    if (lora.gate.has_value() && lora.gate->has_value() && lora_grads.gate.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_input, lora.gate->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_weight_grad(lora_grads.gate->B, total_gate, lora_intermediate1, D, lora_rank, scaling, accumulate ? 1.0f : 0.0f, false);

        dispatch_grouped_gemm(lora_intermediate1, total_gate, lora.gate->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_weight_grad(lora_grads.gate->A, lora_intermediate1, total_input, lora_rank, C, scaling, accumulate ? 1.0f : 0.0f, false);
    }

    if (lora.up.has_value() && lora.up->has_value() && lora_grads.up.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_weight_grad(lora_grads.up->B, total_up, lora_intermediate1, D, lora_rank, scaling, accumulate ? 1.0f : 0.0f, false);

        dispatch_grouped_gemm(lora_intermediate1, total_up, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_weight_grad(lora_grads.up->A, lora_intermediate1, total_input, lora_rank, C, scaling, accumulate ? 1.0f : 0.0f, false);
    }

    // 5. Compute dx = [d_up | d_gate] @ W_gate_up + LoRA contributions
    // BnB weight layout is [up | gate], so gradient layout is [d_up | d_gate]
    // concat_d_gate_up(first, second, out) creates out = [first | second]
    concat_d_gate_up(total_up, total_gate, d_gate_up_buffer, total_tokens, D, stream);

    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_gate_up_backward(
            total_dx.get<nv_bfloat16>(), d_gate_up_buffer.get<nv_bfloat16>(),
            batched_gate_up_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    } else {
        moe_grouped_gemm_gate_up_backward(
            total_dx.get<float>(), d_gate_up_buffer.get<float>(),
            batched_gate_up_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, // Base weights are compact
            gemm_num_experts
        );
    }

    // Add LoRA contributions to dx
    if (lora.gate.has_value() && lora.gate->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_gate, lora.gate->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_grouped_gemm(total_dx, lora_intermediate1, lora.gate->A, C, lora_rank, scaling, 1.0f, EMMTranspose::NN, false);
    }

    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_up, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_grouped_gemm(total_dx, lora_intermediate1, lora.up->A, C, lora_rank, scaling, 1.0f, EMMTranspose::NN, false);
    }
}

/**
 * @brief Forward pass for ALL experts (non-gated MLP) using Grouped GEMM and Batched LoRA.
 *
 * This variant supports non-gated activations (SiLU/ReLU2).
 */
inline void grouped_fast_expert_lora_forward_nongated(
    Tensor& total_output,             ///< (total_tokens, C)
    const Tensor& permuted_input,      ///< (total_tokens, C)
    const Tensor& batched_up_proj,     ///< (num_experts or num_active, D, C)
    const Tensor& batched_down_proj,   ///< (num_experts or num_active, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_up,                 ///< (total_tokens, D) - saved up pre-activation
    const Tensor& expert_offsets,     ///< (num_experts + 1) - global expert indices
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    Tensor& lora_intermediate,         ///< (total_tokens, rank)
    Tensor& h_buffer,                  ///< (total_tokens, D) - activation output
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr,
    const SelectiveExpertInfo* selection_info = nullptr,
    ActivationType activation_type = ActivationType::SiLU) {

    const int total_tokens = total_up.Sizes[0];

    const bool use_selective = selection_info && selection_info->enabled;
    const int gemm_num_experts = use_selective ? selection_info->num_active : num_experts;
    const int* active_indices = use_selective ? selection_info->active_experts.data() : nullptr;

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight,
                                     int M, int K, float alpha, float beta, EMMTranspose mode,
                                     bool weight_is_compact = true, int out_offset = 0) {
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error("MoE LoRA: dtype mismatch between activation and LoRA weights. "
                                     "Set lora_dtype='bf16' in your config to match activation dtype.");
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>() + out_offset, in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        } else {
            moe_grouped_gemm(out.get<float>() + out_offset, in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    // 1. Base up projection
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm(
            total_up.get<nv_bfloat16>(),
            permuted_input.get<nv_bfloat16>(),
            batched_up_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, D, C, handle, stream, host_offsets,
            /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
            active_indices, true, gemm_num_experts);
    } else {
        moe_grouped_gemm(
            total_up.get<float>(),
            permuted_input.get<float>(),
            batched_up_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, D, C, handle, stream, host_offsets,
            /*alpha=*/1.0f, /*beta=*/0.0f, EMMTranspose::TN,
            active_indices, true, gemm_num_experts);
    }

    // 2. LoRA up contribution
    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, permuted_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_grouped_gemm(total_up, lora_intermediate, lora.up->B, D, lora_rank, scaling, 1.0f, EMMTranspose::TN, false);
    }

    // 3. Activation
    const long N = static_cast<long>(total_tokens) * static_cast<long>(D);
    switch (activation_type) {
        case ActivationType::SiLU:
            silu_forward(h_buffer, total_up, N, stream);
            break;
        case ActivationType::ReLU2:
            relu2_forward(h_buffer, total_up, N, stream);
            break;
        default:
            throw std::logic_error("grouped_fast_expert_lora_forward_nongated: unsupported activation");
    }

    // 4. Down projection
    if (permuted_input.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down(
            total_output.get<nv_bfloat16>(),
            h_buffer.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    } else {
        moe_grouped_gemm_down(
            total_output.get<float>(),
            h_buffer.get<float>(),
            batched_down_proj.get<float>(),
            expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate, h_buffer, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_grouped_gemm(total_output, lora_intermediate, lora.down->B, C, lora_rank, scaling, 1.0f, EMMTranspose::TN, false);
    }
}

/**
 * @brief Backward pass for ALL experts (non-gated MLP) using Grouped GEMM and Batched LoRA.
 */
inline void grouped_fast_expert_lora_backward_nongated(
    LoRAGroupedExpertWeights<Tensor>& lora_grads,
    Tensor& total_dx,                 ///< (total_tokens, C)
    const Tensor& total_dy,            ///< (total_tokens, C)
    const Tensor& batched_up_proj,     ///< (num_experts or num_active, D, C)
    const Tensor& batched_down_proj,   ///< (num_experts or num_active, C, D)
    const LoRAGroupedExpertWeights<Tensor>& lora,
    Tensor& total_up,                 ///< (total_tokens, D) - up pre-activation (overwritten with d_up)
    const Tensor& total_input,         ///< (total_tokens, C)
    const Tensor& expert_offsets,     ///< (num_experts + 1) - global expert indices
    float scaling,
    int num_experts, int C, int D, int lora_rank,
    bool accumulate,
    Tensor& lora_intermediate1,        ///< (total_tokens, rank)
    Tensor& lora_intermediate2,        ///< (total_tokens, D)
    Tensor& h_buffer,                  ///< (total_tokens, D)
    cublasHandle_t handle,
    cudaStream_t stream,
    const int* host_offsets = nullptr,
    const SelectiveExpertInfo* selection_info = nullptr,
    ActivationType activation_type = ActivationType::SiLU) {

    const int total_tokens = total_dy.Sizes[0];
    const bool use_selective = selection_info && selection_info->enabled;
    const int gemm_num_experts = use_selective ? selection_info->num_active : num_experts;
    const int* active_indices = use_selective ? selection_info->active_experts.data() : nullptr;

    auto dispatch_grouped_gemm = [&](Tensor& out, const Tensor& in, const Tensor& weight,
                                     int M, int K, float alpha, float beta, EMMTranspose mode,
                                     bool weight_is_compact = true, int out_offset = 0) {
        if (in.DType != weight.DType || in.DType != out.DType) {
            throw std::runtime_error(fmt::format(
                "MoE LoRA backward: dtype mismatch in grouped GEMM. in={}, weight={}, out={}. "
                "Set lora_dtype='bf16' in your config to match activation dtype.",
                dtype_to_str(in.DType), dtype_to_str(weight.DType), dtype_to_str(out.DType)));
        }
        if (in.DType == ETensorDType::BF16) {
            moe_grouped_gemm(out.get<nv_bfloat16>() + out_offset, in.get<nv_bfloat16>(), weight.get<nv_bfloat16>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        } else {
            moe_grouped_gemm(out.get<float>() + out_offset, in.get<float>(), weight.get<float>(),
                             expert_offsets.get<int>(), num_experts, M, K, handle, stream, host_offsets,
                             alpha, beta, mode, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    auto dispatch_weight_grad = [&](Tensor& d_weight, const Tensor& dy, const Tensor& in,
                                    int M, int N, float alpha, float beta, bool weight_is_compact = true) {
        if (dy.DType == ETensorDType::BF16) {
            if (d_weight.DType == ETensorDType::BF16) {
                moe_grouped_gemm_weight_grad(d_weight.get<nv_bfloat16>(), dy.get<nv_bfloat16>(), in.get<nv_bfloat16>(),
                                             expert_offsets.get<int>(), num_experts, M, N, handle, stream, host_offsets,
                                             alpha, beta, active_indices, weight_is_compact, gemm_num_experts);
            } else {
                throw std::runtime_error("MoE LoRA backward: lora_dtype=fp32 with bf16 activations not yet supported. "
                                         "Set lora_dtype='bf16' in your config.");
            }
        } else {
            moe_grouped_gemm_weight_grad(d_weight.get<float>(), dy.get<float>(), in.get<float>(),
                                         expert_offsets.get<int>(), num_experts, M, N, handle, stream, host_offsets,
                                         alpha, beta, active_indices, weight_is_compact, gemm_num_experts);
        }
    };

    // 1. dh = dy @ W_down
    Tensor total_dh = lora_intermediate2;
    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_down_backward(
            total_dh.get<nv_bfloat16>(), total_dy.get<nv_bfloat16>(),
            batched_down_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    } else {
        moe_grouped_gemm_down_backward(
            total_dh.get<float>(), total_dy.get<float>(),
            batched_down_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    }

    if (lora.down.has_value() && lora.down->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_grouped_gemm(total_dh, lora_intermediate1, lora.down->A, D, lora_rank, scaling, 1.0f, EMMTranspose::NN, false);
    }

    // 2. Activation forward (for LoRA down grads) and backward (compute d_up)
    const long N = static_cast<long>(total_tokens) * static_cast<long>(D);
    switch (activation_type) {
        case ActivationType::SiLU:
            silu_forward(h_buffer, total_up, N, stream);
            silu_backward(total_up, total_up, total_dh, N, stream);
            break;
        case ActivationType::ReLU2:
            relu2_forward(h_buffer, total_up, N, stream);
            relu2_backward(total_up, total_up, total_dh, N, stream);
            break;
        default:
            throw std::logic_error("grouped_fast_expert_lora_backward_nongated: unsupported activation");
    }

    // 3. Down LoRA gradients
    if (lora.down.has_value() && lora.down->has_value() && lora_grads.down.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, h_buffer, lora.down->A, lora_rank, D, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_weight_grad(lora_grads.down->B, total_dy, lora_intermediate1, C, lora_rank, scaling, accumulate ? 1.0f : 0.0f, false);

        dispatch_grouped_gemm(lora_intermediate1, total_dy, lora.down->B, lora_rank, C, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_weight_grad(lora_grads.down->A, lora_intermediate1, h_buffer, lora_rank, D, scaling, accumulate ? 1.0f : 0.0f, false);
    }

    // 4. Up LoRA gradients
    if (lora.up.has_value() && lora.up->has_value() && lora_grads.up.has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_input, lora.up->A, lora_rank, C, 1.0f, 0.0f, EMMTranspose::TN, false);
        dispatch_weight_grad(lora_grads.up->B, total_up, lora_intermediate1, D, lora_rank, scaling, accumulate ? 1.0f : 0.0f, false);

        dispatch_grouped_gemm(lora_intermediate1, total_up, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_weight_grad(lora_grads.up->A, lora_intermediate1, total_input, lora_rank, C, scaling, accumulate ? 1.0f : 0.0f, false);
    }

    // 5. dx = d_up @ W_up
    if (total_dy.DType == ETensorDType::BF16) {
        moe_grouped_gemm_up_backward(
            total_dx.get<nv_bfloat16>(), total_up.get<nv_bfloat16>(),
            batched_up_proj.get<nv_bfloat16>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    } else {
        moe_grouped_gemm_up_backward(
            total_dx.get<float>(), total_up.get<float>(),
            batched_up_proj.get<float>(), expert_offsets.get<int>(),
            num_experts, C, D, handle, stream, host_offsets, active_indices,
            true, gemm_num_experts);
    }

    // Add LoRA contributions to dx
    if (lora.up.has_value() && lora.up->has_value()) {
        dispatch_grouped_gemm(lora_intermediate1, total_up, lora.up->B, lora_rank, D, 1.0f, 0.0f, EMMTranspose::NN, false);
        dispatch_grouped_gemm(total_dx, lora_intermediate1, lora.up->A, C, lora_rank, scaling, 1.0f, EMMTranspose::NN, false);
    }
}

} // namespace detail
} // namespace modules

#endif // SUROGATE_SRC_MODULES_LORA_FAST_EXPERT_LORA_H
