// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 gated delta rule operation dispatch.

#include "runtime/dsl/compiled_ops.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

bool tensor_shape_matches(const Tensor& t, long n0, long n1, long n2, long n3) {
    return t.Rank == 4 &&
           t.Sizes[0] == n0 &&
           t.Sizes[1] == n1 &&
           t.Sizes[2] == n2 &&
           t.Sizes[3] == n3;
}

bool is_supported_gdr_dtype(ETensorDType dtype) {
    return dtype == ETensorDType::FP16 ||
           dtype == ETensorDType::BF16 ||
           dtype == ETensorDType::FP32;
}

bool supports_gdr_v2(const Tensor& q, const Tensor& v) {
    const long kdim = q.Sizes[3];
    const long vdim = v.Sizes[3];
    return kdim <= 128 && vdim <= 128;
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
    if (!is_supported_gdr_dtype(q.DType) ||
        !is_supported_gdr_dtype(g.DType) ||
        !is_supported_gdr_dtype(beta.DType)) {
        throw std::runtime_error(
            std::string(op_name) + ": v2 supports q/g/beta dtypes in {FP16,BF16,FP32}");
    }
    if (!supports_gdr_v2(q, v)) {
        throw std::runtime_error(
            std::string(op_name) + ": v2 supports K,V <= 128; got K="
            + std::to_string(Kdim) + " V=" + std::to_string(Vdim));
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
    const long checkpoints_elems =
        B * H * static_cast<long>(num_chunks + 1) * Kdim * Vdim;
    constexpr long kFwdWsAlignFloats = 128 / sizeof(float);
    auto align_up = [](long x, long align) { return ((x + align - 1) / align) * align; };
    const long fwd_lp = 64;
    long fwd_ws_stride = 0;
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats) + fwd_lp * Vdim;  // u
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats) + fwd_lp * Kdim;  // w
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats) + fwd_lp * Kdim;  // k
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats) + fwd_lp * Vdim;  // vnew_pre
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats) + fwd_lp;         // gcum
    fwd_ws_stride = align_up(fwd_ws_stride, kFwdWsAlignFloats);
    const long fwd_workspace_elems =
        B * H * static_cast<long>(num_chunks) * fwd_ws_stride;

    Tensor state_scratch = mRunState.temp_alloc(ETensorDType::FP32, {B, H, Kdim, Vdim});
    mTemps.push_back(state_scratch);

    const bool use_v2 = true;
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        std::cerr << "[GDR fwd] B=" << B << " T=" << T << " H=" << H
                  << " K=" << Kdim << " V=" << Vdim
                  << " q_dtype=" << static_cast<int>(q.DType)
                  << " g_dtype=" << static_cast<int>(g.DType)
                  << " beta_dtype=" << static_cast<int>(beta.DType)
                  << " use_v2=" << (use_v2 ? 1 : 0) << std::endl;
    }
    // Prefer persistent per-op checkpoint slots so backward can reuse forward
    // checkpoints and skip expensive checkpoint recomputation.
    Tensor fwd_checkpoints;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
         capture_status != cudaStreamCaptureStatusNone);
    auto& scratch = mRunState.scratch();
    const std::size_t slot = scratch.gdr_fwd_write_count++;
    if (scratch.gdr_fwd_checkpoints.size() <= slot) {
        if (in_capture) {
            throw std::runtime_error(
                "chunk_gated_delta_rule: missing preallocated forward checkpoint slot during CUDA graph capture");
        }
        scratch.gdr_fwd_checkpoints.resize(slot + 1);
    }
    Tensor& slot_tensor = scratch.gdr_fwd_checkpoints[slot];
    if (!slot_tensor.Data ||
        slot_tensor.DType != ETensorDType::FP32 ||
        slot_tensor.nelem() < checkpoints_elems) {
        if (in_capture) {
            throw std::runtime_error(
                "chunk_gated_delta_rule: preallocated forward checkpoint slot is too small during CUDA graph capture");
        }
        const std::string name = "gdr_fwd_checkpoints_" + std::to_string(slot);
        slot_tensor = mRunState.Allocator->allocate(
            ETensorDType::FP32, name.c_str(), EAllocationType::ON_DEVICE, {checkpoints_elems});
    }
    fwd_checkpoints = slot_tensor;
    fwd_checkpoints.Rank = 5;
    fwd_checkpoints.Sizes[0] = B;
    fwd_checkpoints.Sizes[1] = H;
    fwd_checkpoints.Sizes[2] = static_cast<long>(num_chunks + 1);
    fwd_checkpoints.Sizes[3] = Kdim;
    fwd_checkpoints.Sizes[4] = Vdim;
    for (int i = 5; i < MAX_TENSOR_DIM; ++i) fwd_checkpoints.Sizes[i] = 1;

    Tensor fwd_workspace;
    Tensor& persistent_fwd_workspace = scratch.gdr_fwd_workspace;
    if (persistent_fwd_workspace.Data &&
        persistent_fwd_workspace.DType == ETensorDType::FP32 &&
        persistent_fwd_workspace.nelem() >= fwd_workspace_elems) {
        fwd_workspace = persistent_fwd_workspace;
    } else if (in_capture) {
        throw std::runtime_error(
            "chunk_gated_delta_rule: missing preallocated forward workspace during CUDA graph capture");
    } else {
        if (!persistent_fwd_workspace.Data ||
            persistent_fwd_workspace.DType != ETensorDType::FP32 ||
            persistent_fwd_workspace.nelem() < fwd_workspace_elems) {
            persistent_fwd_workspace = mRunState.Allocator->allocate(
                ETensorDType::FP32,
                "gdr_fwd_workspace",
                EAllocationType::ON_DEVICE,
                {fwd_workspace_elems});
        }
        fwd_workspace = persistent_fwd_workspace;
    }

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
        &fwd_workspace,
        mRunState.MainStream);
    CUDA_CHECK(cudaGetLastError());

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
    if (!is_supported_gdr_dtype(q.DType) ||
        !is_supported_gdr_dtype(g.DType) ||
        !is_supported_gdr_dtype(beta.DType)) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: v2 supports q/g/beta dtypes in {FP16,BF16,FP32}");
    }
    if (!supports_gdr_v2(q, v)) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: v2 supports K,V <= 128; got K="
            + std::to_string(Kdim) + " V=" + std::to_string(Vdim));
    }

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

    const bool use_v2 = true;
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        std::cerr << "[GDR bwd] B=" << B << " T=" << T << " H=" << H
                  << " K=" << Kdim << " V=" << Vdim
                  << " q_dtype=" << static_cast<int>(q.DType)
                  << " g_dtype=" << static_cast<int>(g.DType)
                  << " beta_dtype=" << static_cast<int>(beta.DType)
                  << " use_v2=" << (use_v2 ? 1 : 0) << std::endl;
    }

    const long checkpoints_elems =
        B * H * static_cast<long>(num_chunks + 1) * Kdim * Vdim;
    bool skip_checkpoint = false;
    Tensor checkpoints;
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool in_capture =
        (cudaStreamIsCapturing(mRunState.MainStream, &capture_status) == cudaSuccess &&
         capture_status != cudaStreamCaptureStatusNone);
    auto& scratch = mRunState.scratch();
    if (scratch.gdr_fwd_write_count > scratch.gdr_fwd_read_count &&
        scratch.gdr_fwd_write_count <= scratch.gdr_fwd_checkpoints.size()) {
        const std::size_t slot =
            scratch.gdr_fwd_write_count - 1 - scratch.gdr_fwd_read_count;
        Tensor& saved_cp = scratch.gdr_fwd_checkpoints[slot];
        if (saved_cp.Data &&
            saved_cp.DType == ETensorDType::FP32 &&
            saved_cp.nelem() >= checkpoints_elems) {
            checkpoints = saved_cp;
            skip_checkpoint = true;
            scratch.gdr_fwd_read_count++;
        }
    }
    if (!skip_checkpoint) {
        // Fallback: materialize checkpoints inside backward.
        Tensor& gdr_cp = scratch.gdr_bwd_checkpoints;
        if (gdr_cp.Data && gdr_cp.DType == ETensorDType::FP32 && gdr_cp.nelem() >= checkpoints_elems) {
            checkpoints = gdr_cp;
        } else if (in_capture) {
            throw std::runtime_error(
                "chunk_gated_delta_rule_backward: missing preallocated backward checkpoints during CUDA graph capture");
        } else {
            if (!gdr_cp.Data || gdr_cp.DType != ETensorDType::FP32 || gdr_cp.nelem() < checkpoints_elems) {
                gdr_cp = mRunState.Allocator->allocate(
                    ETensorDType::FP32, "gdr_bwd_checkpoints", EAllocationType::ON_DEVICE, {checkpoints_elems});
            }
            checkpoints = gdr_cp;
        }
    }
    checkpoints.Rank = 5;
    checkpoints.Sizes[0] = B;
    checkpoints.Sizes[1] = H;
    checkpoints.Sizes[2] = static_cast<long>(num_chunks + 1);
    checkpoints.Sizes[3] = Kdim;
    checkpoints.Sizes[4] = Vdim;
    for (int i = 5; i < MAX_TENSOR_DIM; ++i) checkpoints.Sizes[i] = 1;
    if (std::getenv("SUROGATE_DEBUG_GDR")) {
        std::cerr << "[GDR bwd cp] reuse_fwd_checkpoints=" << (skip_checkpoint ? 1 : 0) << std::endl;
    }

    const int Lp = 64;
    // Keep runtime workspace sizing compatible with v2 multi-kernel routing.
    // The launcher may choose the multi-kernel path for K/V up to 128, even when
    // older host-side "can_wmma" checks would say false.
    // Allocate a superset layout that is valid for both multi-kernel and scalar v2.
    constexpr long kWsAlignFloats = 128 / sizeof(float);
    auto align_up = [](long x, long align) { return ((x + align - 1) / align) * align; };
    const long c_storage_floats = static_cast<long>(Kdim) * Kdim;
    long chunk_ws_stride = 0;
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Lp;    // M
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim;  // W
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Vdim;  // VNEW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Vdim;  // DU
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim;  // DW
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim;  // DQ
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim;  // DK
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);          // DG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);          // DB
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Kdim) * Vdim; // DHT1
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + c_storage_floats;                // C fp32
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + 1;                                // EG
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp) * Kdim;    // K_NORM
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);            // INVQ
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats) + static_cast<long>(Lp);            // INVK
    chunk_ws_stride = align_up(chunk_ws_stride, kWsAlignFloats);                                    // total stride
    const long dh_storage_per_chunk = static_cast<long>(Kdim) * Vdim;
    const long multikernel_ws_size = static_cast<long>(num_chunks) * chunk_ws_stride
        + static_cast<long>(num_chunks) * dh_storage_per_chunk;

    // v2 scalar fallback only needs one chunk-local scratch per (B,H).
    // Extra 2*Lp*Lp is used by scalar internals for temporary matmul buffers.
    const long scalar_ws_stride =
        static_cast<long>(chunk_size) * (4 * Kdim + 2 * Vdim + 2) +
        static_cast<long>(2 * Lp * Lp);

    const long workspace_size = std::max(multikernel_ws_size, scalar_ws_stride);
    const long workspace_elems = B * H * workspace_size;

    // This workspace can be very large on Qwen3.5 (hundreds of MB). Keeping it on the
    // temp stack collides with recompute/checkpoint stack usage and causes Stack OOM.
    // Use a persistent scratch allocation and reuse it across steps.
    Tensor gdr_ws;
    Tensor& persistent_ws = mRunState.scratch().gdr_bwd_workspace;
    if (persistent_ws.Data && persistent_ws.DType == ETensorDType::FP32 && persistent_ws.nelem() >= workspace_elems) {
        gdr_ws = persistent_ws;
    } else if (in_capture) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: missing preallocated backward workspace during CUDA graph capture");
    } else {
        if (!persistent_ws.Data || persistent_ws.DType != ETensorDType::FP32 || persistent_ws.nelem() < workspace_elems) {
            persistent_ws = mRunState.Allocator->allocate(
                ETensorDType::FP32, "gdr_bwd_workspace", EAllocationType::ON_DEVICE, {workspace_elems});
        }
        gdr_ws = persistent_ws;
    }

    Tensor state_scratch = gdr_ws;
    state_scratch.Rank = 3;
    state_scratch.Sizes[0] = B;
    state_scratch.Sizes[1] = H;
    state_scratch.Sizes[2] = workspace_size;
    for (int i = 3; i < MAX_TENSOR_DIM; ++i) state_scratch.Sizes[i] = 1;

    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(Kdim));
    }

    const bool debug_nan = (std::getenv("SUROGATE_DEBUG_GDR_NAN") != nullptr);

    auto count_nonfinite = [&](const char* name, const Tensor& t) {
        if (!debug_nan) return;
        if (!t.Data) {
            std::cerr << "[GDR bwd nan] " << name << ": <null>\n";
            return;
        }
        if (t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32) {
            std::cerr << "[GDR bwd nan] " << name << ": skipped dtype=" << static_cast<int>(t.DType) << "\n";
            return;
        }
        Tensor cnt = mRunState.temp_alloc(ETensorDType::INT32, {1});
        CUDA_CHECK(cudaMemsetAsync(cnt.Data, 0, sizeof(int), mRunState.MainStream));
        count_non_finite(cnt, t, mRunState.MainStream);
        int host_cnt = 0;
        CUDA_CHECK(cudaMemcpyAsync(&host_cnt, cnt.get<int>(), sizeof(int), cudaMemcpyDeviceToHost, mRunState.MainStream));
        CUDA_CHECK(cudaStreamSynchronize(mRunState.MainStream));
        mRunState.temp_free(cnt);
        std::cerr << "[GDR bwd nan] " << name << ": count=" << host_cnt << "\n";
    };

    if (debug_nan) {
        count_nonfinite("d_out", d_out);
        if (d_final_state) count_nonfinite("d_final_state", *d_final_state);
        count_nonfinite("q", q);
        count_nonfinite("k", k);
        count_nonfinite("v", v);
        count_nonfinite("g", g);
        count_nonfinite("beta", beta);
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
    CUDA_CHECK(cudaGetLastError());

    if (debug_nan) {
        count_nonfinite("d_q", *d_q);
        count_nonfinite("d_k", *d_k);
        count_nonfinite("d_v", *d_v);
        count_nonfinite("d_g", *d_g);
        count_nonfinite("d_beta", *d_beta);
    }
    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], *d_q);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], *d_k);
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) store_tensor(op.outputs[2], *d_v);
    if (op.outputs.size() > 3 && !op.outputs[3].name.empty()) store_tensor(op.outputs[3], *d_g);
    if (op.outputs.size() > 4 && !op.outputs[4].name.empty()) store_tensor(op.outputs[4], *d_beta);
    if (op.outputs.size() > 5 && !op.outputs[5].name.empty()) store_tensor(op.outputs[5], *d_initial);
}

}  // namespace dsl
