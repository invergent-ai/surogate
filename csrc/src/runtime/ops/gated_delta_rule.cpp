// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 gated delta rule operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <cmath>
#include <string>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"

namespace dsl {
namespace {

bool tensor_shape_matches(const Tensor& t, long n0, long n1, long n2, long n3) {
    return t.Rank == 4 &&
           t.Sizes[0] == n0 &&
           t.Sizes[1] == n1 &&
           t.Sizes[2] == n2 &&
           t.Sizes[3] == n3;
}
}  // namespace

void CompiledExecutor::dispatch_gated_delta_rule_common(const CompiledOp& op,
                                                        const char* op_name) {
    if (op.inputs.size() < 5) {
        throw std::runtime_error(std::string(op_name) + ": expected at least 5 inputs");
    }

    Tensor& q = resolve_tensor(op.inputs[0]);
    Tensor& k = resolve_tensor(op.inputs[1]);
    Tensor& v = resolve_tensor(op.inputs[2]);
    Tensor& g = resolve_tensor(op.inputs[3]);
    Tensor& beta = resolve_tensor(op.inputs[4]);

    if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4 || g.Rank != 3 || beta.Rank != 3) {
        throw std::runtime_error(
            std::string(op_name) + ": expected q/k/v rank 4 and g/beta rank 3");
    }

    const long B = q.Sizes[0];
    const long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long Kdim = q.Sizes[3];
    const long Vdim = v.Sizes[3];

    if (!tensor_shape_matches(k, B, T, H, Kdim)) {
        throw std::runtime_error(std::string(op_name) + ": k shape must match q");
    }
    if (v.Sizes[0] != B || v.Sizes[1] != T || v.Sizes[2] != H) {
        throw std::runtime_error(std::string(op_name) + ": v must share B/T/H with q");
    }
    if (g.Sizes[0] != B || g.Sizes[1] != T || g.Sizes[2] != H) {
        throw std::runtime_error(std::string(op_name) + ": g shape must be [B,T,H]");
    }
    if (beta.Sizes[0] != B || beta.Sizes[1] != T || beta.Sizes[2] != H) {
        throw std::runtime_error(std::string(op_name) + ": beta shape must be [B,T,H]");
    }

    Tensor* initial_state = nullptr;
    if (op.inputs.size() > 5 && !op.inputs[5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[5]);
        if (initial_state->DType != ETensorDType::FP32) {
            throw std::runtime_error(
                std::string(op_name) + ": initial_state must be FP32");
        }
        if (!tensor_shape_matches(*initial_state, B, H, Kdim, Vdim)) {
            throw std::runtime_error(
                std::string(op_name) + ": initial_state must be [B,H,K,V]");
        }
    }

    Tensor* out_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.DType == v.DType && tensor_shape_matches(out_ref, B, T, H, Vdim)) {
            out_ptr = &out_ref;
        }
    }
    if (!out_ptr) {
        Tensor out_t = mRunState.temp_alloc(v.DType, {B, T, H, Vdim});
        mTemps.push_back(out_t);
        out_ptr = &mTemps.back();
    }

    Tensor* final_state_ptr = nullptr;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        Tensor& state_ref = ensure_output_tensor(op.outputs[1]);
        if (state_ref.DType == ETensorDType::FP32 &&
            tensor_shape_matches(state_ref, B, H, Kdim, Vdim)) {
            final_state_ptr = &state_ref;
        }
    }
    if (!final_state_ptr) {
        Tensor state_t = mRunState.temp_alloc(ETensorDType::FP32, {B, H, Kdim, Vdim});
        mTemps.push_back(state_t);
        final_state_ptr = &mTemps.back();
    }

    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(Kdim));
    }

    int chunk_size = op.attrs.chunk_size > 0 ? op.attrs.chunk_size : 64;
    const int num_chunks = static_cast<int>((T + chunk_size - 1) / chunk_size);

    Tensor state_scratch = mRunState.temp_alloc(ETensorDType::FP32, {B, H, Kdim, Vdim});
    mTemps.push_back(state_scratch);

    // Allocate checkpoints to save during forward (avoids recomputation in backward)
    Tensor fwd_checkpoints = mRunState.temp_alloc(
        ETensorDType::FP32, {B, H, static_cast<long>(num_chunks + 1), Kdim, Vdim});
    mTemps.push_back(fwd_checkpoints);

    gated_delta_rule_chunk_forward_v2(
        *out_ptr,
        *final_state_ptr,
        state_scratch,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale,
        chunk_size,
        op.attrs.use_qk_l2norm_in_kernel,
        &fwd_checkpoints,
        mRunState.MainStream);

    // Save checkpoints for backward (avoid re-running checkpoint kernel)
    if (mSaved) {
        (*mSaved)["gdr_checkpoints"] = fwd_checkpoints;
    }

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], *out_ptr);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        store_tensor(op.outputs[1], *final_state_ptr);
    }
}

void CompiledExecutor::dispatch_chunk_gated_delta_rule(const CompiledOp& op) {
    dispatch_gated_delta_rule_common(op, "chunk_gated_delta_rule");
}

void CompiledExecutor::dispatch_chunk_gated_delta_rule_backward(const CompiledOp& op) {
    // Inputs:
    //   d_out, d_final_state(optional), q, k, v, g, beta, initial_state(optional)
    // Outputs:
    //   d_q, d_k, d_v, d_g, d_beta, d_initial_state(optional)
    if (op.inputs.size() < 7) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: expected at least 7 inputs");
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);

    Tensor* d_final_state = nullptr;
    std::size_t tensor_input_offset = 1;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        d_final_state = &resolve_tensor(op.inputs[1]);
    }
    tensor_input_offset = 2;

    Tensor& q = resolve_tensor(op.inputs[tensor_input_offset + 0]);
    Tensor& k = resolve_tensor(op.inputs[tensor_input_offset + 1]);
    Tensor& v = resolve_tensor(op.inputs[tensor_input_offset + 2]);
    Tensor& g = resolve_tensor(op.inputs[tensor_input_offset + 3]);
    Tensor& beta = resolve_tensor(op.inputs[tensor_input_offset + 4]);

    Tensor* initial_state = nullptr;
    if (op.inputs.size() > tensor_input_offset + 5 &&
        !op.inputs[tensor_input_offset + 5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[tensor_input_offset + 5]);
    }

    const long B = q.Sizes[0];
    const long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long Kdim = q.Sizes[3];
    const long Vdim = v.Sizes[3];

    auto ensure_or_temp = [&](std::size_t out_idx,
                              ETensorDType dtype,
                              const std::vector<long>& shape) -> Tensor* {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            if (out_ref.DType == dtype && tensor_shape_matches(
                    out_ref, shape[0], shape[1], shape[2], shape[3])) {
                return &out_ref;
            }
        }
        Tensor temp = mRunState.temp_alloc(dtype, shape);
        mTemps.push_back(temp);
        return &mTemps.back();
    };

    Tensor* d_q = ensure_or_temp(0, q.DType, {B, T, H, Kdim});
    Tensor* d_k = ensure_or_temp(1, k.DType, {B, T, H, Kdim});
    Tensor* d_v = ensure_or_temp(2, v.DType, {B, T, H, Vdim});

    Tensor* d_g = nullptr;
    if (op.outputs.size() > 3 && !op.outputs[3].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[3]);
        if (out_ref.DType == g.DType &&
            out_ref.Rank == 3 &&
            out_ref.Sizes[0] == B &&
            out_ref.Sizes[1] == T &&
            out_ref.Sizes[2] == H) {
            d_g = &out_ref;
        }
    }
    if (!d_g) {
        Tensor temp = mRunState.temp_alloc(g.DType, {B, T, H});
        mTemps.push_back(temp);
        d_g = &mTemps.back();
    }

    Tensor* d_beta = nullptr;
    if (op.outputs.size() > 4 && !op.outputs[4].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[4]);
        if (out_ref.DType == beta.DType &&
            out_ref.Rank == 3 &&
            out_ref.Sizes[0] == B &&
            out_ref.Sizes[1] == T &&
            out_ref.Sizes[2] == H) {
            d_beta = &out_ref;
        }
    }
    if (!d_beta) {
        Tensor temp = mRunState.temp_alloc(beta.DType, {B, T, H});
        mTemps.push_back(temp);
        d_beta = &mTemps.back();
    }

    Tensor* d_initial = nullptr;
    if (op.outputs.size() > 5 && !op.outputs[5].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[5]);
        if (out_ref.DType == ETensorDType::FP32 &&
            tensor_shape_matches(out_ref, B, H, Kdim, Vdim)) {
            d_initial = &out_ref;
        }
    }
    if (!d_initial) {
        Tensor temp = mRunState.temp_alloc(ETensorDType::FP32, {B, H, Kdim, Vdim});
        mTemps.push_back(temp);
        d_initial = &mTemps.back();
    }

    int chunk_size = op.attrs.chunk_size > 0 ? op.attrs.chunk_size : 64;
    const int num_chunks = static_cast<int>((T + chunk_size - 1) / chunk_size);

    // Check for saved checkpoints from forward pass
    bool skip_checkpoint = false;
    Tensor checkpoints;
    if (mSaved) {
        auto it = mSaved->find("gdr_checkpoints");
        if (it != mSaved->end()) {
            checkpoints = it->second;
            skip_checkpoint = true;
        }
    }
    if (!skip_checkpoint) {
        checkpoints = mRunState.temp_alloc(
            ETensorDType::FP32, {B, H, static_cast<long>(num_chunks + 1), Kdim, Vdim});
        mTemps.push_back(checkpoints);
    }

    const int Lp = 64;
    constexpr long kWsAlignFloats = 128 / sizeof(float);
    auto align_up = [](long x, long align) { return ((x + align - 1) / align) * align; };
    const long c_storage_floats = static_cast<long>(Kdim) * Kdim;
    long chunk_ws_stride = 0;
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Lp;   // M
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Lp;   // A
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim; // W
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Vdim; // VNEW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Vdim; // DU
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim; // DW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim; // DQ
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim; // DK
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);        // DG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);        // DB
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Kdim) * Vdim; // DHT1
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + c_storage_floats;              // C fp32
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + 1;                             // EG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats);                                 // total stride
    const long dh_storage_per_chunk = static_cast<long>(Kdim) * Vdim;
    const long workspace_size = static_cast<long>(num_chunks) * chunk_ws_stride
        + static_cast<long>(num_chunks) * dh_storage_per_chunk;
    Tensor state_scratch = mRunState.temp_alloc(ETensorDType::FP32, {B, H, workspace_size});
    mTemps.push_back(state_scratch);

    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(Kdim));
    }

    gated_delta_rule_chunk_backward_v2(
        *d_q,
        *d_k,
        *d_v,
        *d_g,
        *d_beta,
        *d_initial,
        d_out,
        d_final_state,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale,
        chunk_size,
        op.attrs.use_qk_l2norm_in_kernel,
        checkpoints,
        state_scratch,
        skip_checkpoint,
        mRunState.MainStream);

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], *d_q);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], *d_k);
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) store_tensor(op.outputs[2], *d_v);
    if (op.outputs.size() > 3 && !op.outputs[3].name.empty()) store_tensor(op.outputs[3], *d_g);
    if (op.outputs.size() > 4 && !op.outputs[4].name.empty()) store_tensor(op.outputs[4], *d_beta);
    if (op.outputs.size() > 5 && !op.outputs[5].name.empty()) store_tensor(op.outputs[5], *d_initial);
}

}  // namespace dsl
