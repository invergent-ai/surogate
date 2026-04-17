// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 gated delta rule operation dispatch using JIT-compiled Triton kernels.

#include "runtime/executor/compiled_ops.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "utilities/comm.h"
#include "utilities/utils.h"

namespace dsl {
namespace {

Tensor make_persistent_tensor(DslRunState& run_state,
                              std::unordered_map<std::string, void*>& buffers,
                              std::unordered_map<std::string, size_t>& sizes,
                              const std::string& key,
                              ETensorDType dtype,
                              const std::vector<long>& shape) {
    const size_t elem_sz = get_dtype_size(dtype);
    size_t nelem = 1;
    for (long d : shape) {
        nelem *= static_cast<size_t>(d);
    }
    const size_t bytes = nelem * elem_sz;
    if (bytes == 0) {
        Tensor t;
        t.DType = dtype;
        t.Rank = static_cast<int>(shape.size());
        for (int i = 0; i < t.Rank; ++i) t.Sizes[i] = shape[static_cast<size_t>(i)];
        return t;
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool capturing =
        (cudaStreamIsCapturing(run_state.MainStream, &capture_status) == cudaSuccess &&
         capture_status != cudaStreamCaptureStatusNone);

    auto it = buffers.find(key);
    if (it == buffers.end() || sizes[key] < bytes) {
        if (capturing) {
            throw std::runtime_error(
                "gated_delta_rule: missing preallocated persistent buffer for '" + key +
                "' during CUDA graph capture");
        }
        if (it != buffers.end() && it->second != nullptr) {
            CUDA_CHECK(cudaFree(it->second));
        }
        void* buf = nullptr;
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        buffers[key] = buf;
        sizes[key] = bytes;
    }

    Tensor t;
    t.DType = dtype;
    t.Rank = static_cast<int>(shape.size());
    for (int i = 0; i < t.Rank; ++i) t.Sizes[i] = shape[static_cast<size_t>(i)];
    t.Data = static_cast<std::byte*>(buffers[key]);
    return t;
}


bool tensor_shape_matches(const Tensor& t, long n0, long n1, long n2, long n3) {
    return t.Rank == 4 &&
           t.Sizes[0] == n0 &&
           t.Sizes[1] == n1 &&
           t.Sizes[2] == n2 &&
           t.Sizes[3] == n3;
}

inline int cdiv(int a, int b) { return (a + b - 1) / b; }
inline int next_power_of_2(int v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4; v |= v >> 8; v |= v >> 16;
    return v + 1;
}

}  // namespace

void CompiledExecutor::dispatch_gated_delta_rule_common(const CompiledOp& op,
                                                         const char* op_name) {
    if (!mGdrKernels.is_ready()) {
        throw std::runtime_error(
            std::string(op_name) + ": JIT Triton kernels not loaded. "
            "Ensure compile_jit_kernels() ran in Python and manifests were passed via RuntimeOptions.");
    }
    if (op.inputs.size() < 5) {
        throw std::runtime_error(std::string(op_name) + ": expected at least 5 inputs");
    }

    Tensor& q = resolve_tensor(op.inputs[0]);
    Tensor& k = resolve_tensor(op.inputs[1]);
    Tensor& v = resolve_tensor(op.inputs[2]);
    Tensor& g_input = resolve_tensor(op.inputs[3]);
    Tensor& beta = resolve_tensor(op.inputs[4]);

    if (q.Rank != 4 || k.Rank != 4 || v.Rank != 4 || g_input.Rank != 3 || beta.Rank != 3) {
        throw std::runtime_error(
            std::string(op_name) + ": expected q/k/v rank 4 and g/beta rank 3");
    }

    const long B = q.Sizes[0];
    const long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long K = q.Sizes[3];
    const long V = v.Sizes[3];

    if (!tensor_shape_matches(k, B, T, H, K)) {
        throw std::runtime_error(std::string(op_name) + ": k shape must match q");
    }
    if (v.Sizes[0] != B || v.Sizes[1] != T || v.Sizes[2] != H) {
        throw std::runtime_error(std::string(op_name) + ": v must share B/T/H with q");
    }

    Tensor* initial_state = nullptr;
    if (op.inputs.size() > 5 && !op.inputs[5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[5]);
    }

    // Allocate outputs.
    // IMPORTANT: Do NOT store pointers into mTemps — subsequent push_back() calls
    // can reallocate the vector, invalidating any prior pointers. Instead, allocate
    // all temp tensors first, then take stable pointers after all push_backs.
    Tensor out_val;
    bool out_is_ref = false;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.DType == v.DType && tensor_shape_matches(out_ref, B, T, H, V)) {
            out_val = out_ref;
            out_is_ref = true;
        }
    }
    if (!out_is_ref) {
        out_val = make_persistent_tensor(mRunState, mMoeSavedBuffers, mMoeSavedSizes,
                                         op.op_id + ".out_fallback", v.DType, {B, T, H, V});
    }

    Tensor state_val;
    bool state_is_ref = false;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        Tensor& state_ref = ensure_output_tensor(op.outputs[1]);
        if (state_ref.DType == ETensorDType::FP32 &&
            tensor_shape_matches(state_ref, B, H, K, V)) {
            state_val = state_ref;
            state_is_ref = true;
        }
    }
    if (!state_is_ref) {
        state_val = make_persistent_tensor(mRunState, mMoeSavedBuffers, mMoeSavedSizes,
                                           op.op_id + ".state_fallback", ETensorDType::FP32,
                                           {B, H, K, V});
    }
    Tensor* out_ptr = &out_val;
    Tensor* final_state_ptr = &state_val;

    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }

    const int BT = 64;
    const int NT = cdiv(static_cast<int>(T), BT);
    const int BK_kkt = std::min(std::max(next_power_of_2(static_cast<int>(K)), 16), 64);
    const int BV_h = (K > 64) ? 32 : std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
    const int BK_o = std::min(std::max(next_power_of_2(static_cast<int>(K)), 16), 64);
    const int BV_o = std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
    const int BH = static_cast<int>(B * H);
    cudaStream_t stream = mRunState.MainStream;

    // Optional L2 normalization of q and k (Qwen3.5 requires this)
    const bool use_l2norm = op.attrs.use_qk_l2norm_in_kernel;
    void* q_eff = q.Data;
    void* k_eff = k.Data;
    if (use_l2norm) {
        Tensor q_norm = mRunState.temp_alloc(q.DType, {B, T, H, K}, "gated_delta_rule_q_norm");
        mTemps.push_back(q_norm);
        Tensor q_rstd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_q_rstd");
        mTemps.push_back(q_rstd);
        Tensor k_norm = mRunState.temp_alloc(k.DType, {B, T, H, K}, "gated_delta_rule_k_norm");
        mTemps.push_back(k_norm);
        Tensor k_rstd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_k_rstd");
        mTemps.push_back(k_rstd);

        // L2 norm Q: (x, y, rstd, T)
        {
            void* x = q.Data; void* y = q_norm.Data; void* r = q_rstd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        // L2 norm K: same kernel (D=K)
        {
            void* x = k.Data; void* y = k_norm.Data; void* r = k_rstd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        q_eff = q_norm.Data;
        k_eff = k_norm.Data;
    }

    // Allocate intermediates
    Tensor g_cum = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_g_cum");
    mTemps.push_back(g_cum);
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H, BT}, "gated_delta_rule_A");
    mTemps.push_back(A);
    Tensor Ai = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, BT}, "gated_delta_rule_Ai");
    mTemps.push_back(Ai);
    CUDA_CHECK(cudaMemsetAsync(Ai.Data, 0, Ai.nelem() * 2, stream));
    Tensor w = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, K}, "gated_delta_rule_w");
    mTemps.push_back(w);
    Tensor u = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, V}, "gated_delta_rule_u");
    mTemps.push_back(u);
    Tensor h = mRunState.temp_alloc(ETensorDType::BF16, {B, static_cast<long>(NT), H, K, V}, "gated_delta_rule_h");
    mTemps.push_back(h);
    Tensor v_new = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, V}, "gated_delta_rule_v_new");
    mTemps.push_back(v_new);

    // If no initial_state, allocate a zeroed one.
    // Note: the AOT kernel expects h0 as BF16 (see gdr_fwd_h manifest signature).
    Tensor h0_buf;
    void* h0_ptr;
    if (initial_state) {
        h0_ptr = initial_state->Data;
    } else {
        h0_buf = mRunState.temp_alloc(ETensorDType::BF16, {B, H, K, V}, "gated_delta_rule_h0_buf");
        mTemps.push_back(h0_buf);
        CUDA_CHECK(cudaMemsetAsync(h0_buf.Data, 0, h0_buf.nelem() * 2, stream));
        h0_ptr = h0_buf.Data;
    }

    // ---- Forward pipeline: launch JIT Triton kernels ----

    // 1. cumsum_fwd: (g_input, g_cum, T) grid=(NT, B*H)
    {
        void* g_in_ptr = g_input.Data;
        void* g_out_ptr = g_cum.Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &g_in_ptr, &g_out_ptr, &T_val };
        mGdrKernels.cumsum_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                               args, 3, stream);
    }

    // 2. kkt_fwd: (k, g_cum, beta, A, T) grid=(NT, B*H)
    {
        void* g_ptr = g_cum.Data;
        void* beta_ptr = beta.Data;
        void* A_ptr = A.Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &g_ptr, &beta_ptr, &A_ptr, &T_val };
        mGdrKernels.kkt_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                            args, 5, stream);
    }

    // 3. solve_tril: (A, Ai, T) grid=(NT, B*H)
    {
        void* A_ptr = A.Data;
        void* Ai_ptr = Ai.Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &A_ptr, &Ai_ptr, &T_val };
        mGdrKernels.solve_tril({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                               args, 3, stream);
    }

    // 4. wy_fwd: (k, v, beta, w, u, Ai, g_cum, T) grid=(NT, B*H)
    {
        void* v_ptr = v.Data;
        void* beta_ptr = beta.Data;
        void* w_ptr = w.Data;
        void* u_ptr = u.Data;
        void* Ai_ptr = Ai.Data;
        void* g_ptr = g_cum.Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &v_ptr, &beta_ptr, &w_ptr, &u_ptr, &Ai_ptr, &g_ptr, &T_val };
        mGdrKernels.wy_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                           args, 8, stream);
    }

    // 5. fwd_h: (k, u, w, v_new, g_cum, h, h0, ht, T) grid=(cdiv(V,BV_h), B*H)
    {
        void* u_ptr = u.Data;
        void* w_ptr = w.Data;
        void* vn_ptr = v_new.Data;
        void* g_ptr = g_cum.Data;
        void* h_ptr = h.Data;
        void* ht_ptr = final_state_ptr->Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &u_ptr, &w_ptr, &vn_ptr, &g_ptr, &h_ptr, &h0_ptr, &ht_ptr, &T_val };
        mGdrKernels.fwd_h({static_cast<unsigned>(cdiv(static_cast<int>(V), BV_h)),
                           static_cast<unsigned>(BH), 1},
                          args, 9, stream);
    }

    // 6. fwd_o: (q, k, v_new, h, g_cum, o, scale, T) grid=(cdiv(V,BV_o), NT, B*H)
    {
        void* vn_ptr = v_new.Data;
        void* h_ptr = h.Data;
        void* g_ptr = g_cum.Data;
        void* o_ptr = out_ptr->Data;
        float scale_val = scale;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &q_eff, &k_eff, &vn_ptr, &h_ptr, &g_ptr, &o_ptr, &scale_val, &T_val };
        mGdrKernels.fwd_o({static_cast<unsigned>(cdiv(static_cast<int>(V), BV_o)),
                           static_cast<unsigned>(NT),
                           static_cast<unsigned>(BH)},
                          args, 8, stream);
    }

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
    static const bool debug_replay = std::getenv("SUROGATE_DEBUG_REPLAY") != nullptr;
    if (debug_replay) {
        fprintf(stderr, "[GDR_BWD] inputs=%zu outputs=%zu\n", op.inputs.size(), op.outputs.size());
        for (std::size_t i = 0; i < op.inputs.size(); ++i) {
            fprintf(stderr, "[GDR_BWD]   input[%zu] name='%s' slot=%d layer=%d tid=%d\n",
                    i, op.inputs[i].name.c_str(), static_cast<int>(op.inputs[i].slot),
                    op.inputs[i].layer_idx, op.inputs[i].tensor_id);
        }
    }
    if (!mGdrKernels.is_ready()) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: JIT Triton kernels not loaded.");
    }
    if (op.inputs.size() < 7) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: expected at least 7 inputs");
    }

    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving d_out...\n");
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving d_final_state...\n");
    Tensor* d_final_state = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        d_final_state = &resolve_tensor(op.inputs[1]);
    }
    const std::size_t offs = 2;

    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving q...\n");
    Tensor& q = resolve_tensor(op.inputs[offs + 0]);
    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving k...\n");
    Tensor& k = resolve_tensor(op.inputs[offs + 1]);
    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving v...\n");
    Tensor& v = resolve_tensor(op.inputs[offs + 2]);
    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving g_input...\n");
    Tensor& g_input = resolve_tensor(op.inputs[offs + 3]);
    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving beta...\n");
    Tensor& beta = resolve_tensor(op.inputs[offs + 4]);

    if (debug_replay) fprintf(stderr, "[GDR_BWD] resolving initial_state...\n");
    Tensor* initial_state = nullptr;
    if (op.inputs.size() > offs + 5 && !op.inputs[offs + 5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[offs + 5]);
    }
    if (debug_replay) fprintf(stderr, "[GDR_BWD] all inputs resolved, q.Data=%p\n", q.Data);

    const long B = q.Sizes[0];
    const long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long K = q.Sizes[3];
    const long V = v.Sizes[3];

    // Allocate gradient outputs.
    // IMPORTANT: Do NOT store pointers into mTemps — subsequent push_back() calls
    // can reallocate the vector, invalidating any prior pointers. Store as values.
    auto ensure_or_temp_val = [&](std::size_t out_idx,
                                  ETensorDType dtype,
                                  const std::vector<long>& shape) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            bool ok = true;
            if (out_ref.DType != dtype) ok = false;
            if (shape.size() == 4 && !tensor_shape_matches(out_ref, shape[0], shape[1], shape[2], shape[3])) ok = false;
            if (shape.size() == 3 && (out_ref.Rank != 3 || out_ref.Sizes[0] != shape[0] ||
                out_ref.Sizes[1] != shape[1] || out_ref.Sizes[2] != shape[2])) ok = false;
            if (ok) return out_ref;
        }
        return make_persistent_tensor(mRunState, mMoeSavedBuffers, mMoeSavedSizes,
                                      op.op_id + ".bwd_out" + std::to_string(out_idx) + ".fallback",
                                      dtype, shape);
    };

    if (debug_replay) fprintf(stderr, "[GDR_BWD] allocating outputs B=%ld T=%ld H=%ld K=%ld V=%ld\n", B, T, H, K, V);
    Tensor d_q_val = ensure_or_temp_val(0, q.DType, {B, T, H, K});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_q done\n");
    Tensor d_k_val = ensure_or_temp_val(1, k.DType, {B, T, H, K});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_k done\n");
    Tensor d_v_val = ensure_or_temp_val(2, v.DType, {B, T, H, V});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_v done\n");
    Tensor d_g_val = ensure_or_temp_val(3, g_input.DType, {B, T, H});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_g done\n");
    Tensor d_beta_val = ensure_or_temp_val(4, beta.DType, {B, T, H});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_beta done\n");
    Tensor d_initial_val = ensure_or_temp_val(5, ETensorDType::FP32, {B, H, K, V});
    if (debug_replay) fprintf(stderr, "[GDR_BWD] d_initial done, about to launch kernels\n");
    Tensor* d_q = &d_q_val;
    Tensor* d_k = &d_k_val;
    Tensor* d_v = &d_v_val;
    Tensor* d_g = &d_g_val;
    Tensor* d_beta = &d_beta_val;
    Tensor* d_initial = &d_initial_val;

    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }

    const int BT = 64;
    const int NT = cdiv(static_cast<int>(T), BT);
    const int BV_h = (K > 64) ? 32 : std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
    const int BK_bwd = std::min(std::max(next_power_of_2(static_cast<int>(K)), 16), 64);
    const int BV_bwd = std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
    const int NK = cdiv(static_cast<int>(K), BK_bwd);
    const int BH = static_cast<int>(B * H);
    cudaStream_t stream = mRunState.MainStream;

    // Optional L2 normalization recompute (mirrors forward)
    if (debug_replay) fprintf(stderr, "[GDR_BWD] use_l2norm=%d\n", op.attrs.use_qk_l2norm_in_kernel);
    const bool use_l2norm = op.attrs.use_qk_l2norm_in_kernel;
    void* q_eff = q.Data;
    void* k_eff = k.Data;
    void* dq_data = d_q->Data;
    void* dk_data = d_k->Data;
    Tensor q_norm_bwd, k_norm_bwd, q_rstd_bwd, k_rstd_bwd;
    Tensor dq_norm_buf, dk_norm_buf;
    const Tensor* q_eff_tensor = &q;
    const Tensor* k_eff_tensor = &k;
    if (use_l2norm) {
        if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: allocating temps...\n");
        q_norm_bwd = mRunState.temp_alloc(q.DType, {B, T, H, K}, "gated_delta_rule_q_norm_bwd");
        mTemps.push_back(q_norm_bwd);
        q_rstd_bwd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_q_rstd_bwd");
        mTemps.push_back(q_rstd_bwd);
        k_norm_bwd = mRunState.temp_alloc(k.DType, {B, T, H, K}, "gated_delta_rule_k_norm_bwd");
        mTemps.push_back(k_norm_bwd);
        k_rstd_bwd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_k_rstd_bwd");
        mTemps.push_back(k_rstd_bwd);
        if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: launching q_norm kernel...\n");
        {
            void* x = q.Data; void* y = q_norm_bwd.Data; void* r = q_rstd_bwd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: launching k_norm kernel...\n");
        {
            void* x = k.Data; void* y = k_norm_bwd.Data; void* r = k_rstd_bwd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: done, allocating dq/dk norm buffers...\n");
        q_eff = q_norm_bwd.Data;
        k_eff = k_norm_bwd.Data;
        q_eff_tensor = &q_norm_bwd;
        k_eff_tensor = &k_norm_bwd;
        // Backward pipeline writes dq_norm/dk_norm to temp buffers
        try {
            if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: temp_alloc dq_norm q.DType=%d\n", static_cast<int>(q.DType));
            dq_norm_buf = mRunState.temp_alloc(q.DType, {B, T, H, K}, "gated_delta_rule_dq_norm_buf");
            if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: dq_norm_buf.Data=%p\n", dq_norm_buf.Data);
            mTemps.push_back(dq_norm_buf);
            dk_norm_buf = mRunState.temp_alloc(k.DType, {B, T, H, K}, "gated_delta_rule_dk_norm_buf");
            if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm: dk_norm_buf.Data=%p\n", dk_norm_buf.Data);
            mTemps.push_back(dk_norm_buf);
        } catch (const std::exception& e) {
            fprintf(stderr, "[GDR_BWD] l2norm temp_alloc FAILED: %s\n", e.what());
            throw;
        }
        dq_data = dq_norm_buf.Data;
        dk_data = dk_norm_buf.Data;
        if (debug_replay) fprintf(stderr, "[GDR_BWD] l2norm path complete\n");
    }

    // ---- Recompute forward intermediates ----
    if (debug_replay) fprintf(stderr, "[GDR_BWD] recomputing forward intermediates...\n");
    // g_cum
    Tensor g_cum = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_g_cum");
    mTemps.push_back(g_cum);
    {
        void* g_in = g_input.Data; void* g_out = g_cum.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &g_in, &g_out, &Tv };
        mGdrKernels.cumsum_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1}, args, 3, stream);
    }
    // A, Ai
    Tensor A = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H, BT}, "gated_delta_rule_A");
    mTemps.push_back(A);
    {
        void* gp = g_cum.Data; void* bp = beta.Data; void* ap = A.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &gp, &bp, &ap, &Tv };
        mGdrKernels.kkt_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1}, args, 5, stream);
    }
    Tensor Ai = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, BT}, "gated_delta_rule_Ai");
    mTemps.push_back(Ai);
    CUDA_CHECK(cudaMemsetAsync(Ai.Data, 0, Ai.nelem() * 2, stream));
    {
        void* ap = A.Data; void* aip = Ai.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &ap, &aip, &Tv };
        mGdrKernels.solve_tril({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1}, args, 3, stream);
    }
    // w, u
    Tensor w = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, K}, "gated_delta_rule_w");
    mTemps.push_back(w);
    Tensor u_buf = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, V}, "gated_delta_rule_u_buf");
    mTemps.push_back(u_buf);
    {
        void* vp = v.Data; void* bp = beta.Data;
        void* wp = w.Data; void* up = u_buf.Data; void* aip = Ai.Data; void* gp = g_cum.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &vp, &bp, &wp, &up, &aip, &gp, &Tv };
        mGdrKernels.wy_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1}, args, 8, stream);
    }
    // h, v_new
    Tensor h = mRunState.temp_alloc(ETensorDType::BF16, {B, static_cast<long>(NT), H, K, V}, "gated_delta_rule_h");
    mTemps.push_back(h);
    Tensor ht_dummy = mRunState.temp_alloc(ETensorDType::FP32, {B, H, K, V}, "gated_delta_rule_ht_dummy");
    mTemps.push_back(ht_dummy);
    Tensor v_new = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, V}, "gated_delta_rule_v_new");
    mTemps.push_back(v_new);

    void* h0_ptr;
    Tensor h0_buf;
    if (initial_state) {
        h0_ptr = initial_state->Data;
    } else {
        h0_buf = mRunState.temp_alloc(ETensorDType::BF16, {B, H, K, V}, "gated_delta_rule_h0_buf");
        mTemps.push_back(h0_buf);
        CUDA_CHECK(cudaMemsetAsync(h0_buf.Data, 0, h0_buf.nelem() * 2, stream));
        h0_ptr = h0_buf.Data;
    }
    {
        void* up = u_buf.Data; void* wp = w.Data;
        void* vnp = v_new.Data; void* gp = g_cum.Data; void* hp = h.Data;
        void* htp = ht_dummy.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &up, &wp, &vnp, &gp, &hp, &h0_ptr, &htp, &Tv };
        mGdrKernels.fwd_h({static_cast<unsigned>(cdiv(static_cast<int>(V), BV_h)),
                           static_cast<unsigned>(BH), 1}, args, 9, stream);
    }

    // ---- Backward pipeline ----
    if (debug_replay) fprintf(stderr, "[GDR_BWD] starting backward pipeline...\n");

    // bwd_dv_local
    {
        void* gp = g_cum.Data;
        void* dop = d_out.Data; void* dvp = d_v->Data;
        float sv = scale; int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &q_eff, &k_eff, &gp, &dop, &dvp, &sv, &Tv };
        mGdrKernels.bwd_dv_local({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                 args, 7, stream);
    }

    // bwd_dhu
    Tensor dh = mRunState.temp_alloc(ETensorDType::BF16, {B, static_cast<long>(NT), H, K, V}, "gated_delta_rule_dh");
    mTemps.push_back(dh);
    Tensor dv2 = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, V}, "gated_delta_rule_dv2");
    mTemps.push_back(dv2);
    {
        void* wp = w.Data; void* gp = g_cum.Data;
        void* dhtp = d_final_state ? d_final_state->Data : nullptr;
        Tensor dht_zero;
        if (!dhtp) {
            dht_zero = mRunState.temp_alloc(ETensorDType::FP32, {B, H, K, V}, "gated_delta_rule_dht_zero");
            mTemps.push_back(dht_zero);
            CUDA_CHECK(cudaMemsetAsync(dht_zero.Data, 0, dht_zero.nelem() * sizeof(float), stream));
            dhtp = dht_zero.Data;
        }
        void* dh0p = d_initial->Data;
        void* dop = d_out.Data; void* dhp = dh.Data;
        void* dvp = d_v->Data; void* dv2p = dv2.Data;
        float sv = scale; int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &q_eff, &k_eff, &wp, &gp, &dhtp, &dh0p, &dop, &dhp, &dvp, &dv2p, &sv, &Tv };
        mGdrKernels.bwd_dhu({static_cast<unsigned>(cdiv(static_cast<int>(V), BV_h)),
                             static_cast<unsigned>(BH), 1},
                            args, 12, stream);
    }

    // bwd_dqkwg
    Tensor dw = mRunState.temp_alloc(ETensorDType::BF16, {B, T, H, K}, "gated_delta_rule_dw");
    mTemps.push_back(dw);
    Tensor dg_nk = mRunState.temp_alloc(ETensorDType::FP32, {static_cast<long>(NK), B, T, H}, "gated_delta_rule_dg_nk");
    mTemps.push_back(dg_nk);
    {
        void* vnp = v_new.Data;
        void* gp = g_cum.Data; void* hp = h.Data; void* dop = d_out.Data;
        void* dhp = dh.Data;
        void* dwp = dw.Data; void* dv2p = dv2.Data; void* dgnkp = dg_nk.Data;
        float sv = scale;
        int32_t Bv = static_cast<int32_t>(B);
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &q_eff, &k_eff, &vnp, &gp, &hp, &dop, &dhp, &dq_data, &dk_data, &dwp, &dv2p, &dgnkp, &sv, &Bv, &Tv };
        mGdrKernels.bwd_dqkwg({static_cast<unsigned>(NK),
                                static_cast<unsigned>(NT),
                                static_cast<unsigned>(BH)},
                               args, 15, stream);
    }
    const long dg_bth = B * T * H;

    // TODO: dg_nk reduction across NK dimension needs a small kernel or cumsum approach.
    // For now, the dg reduction is deferred to the cumsum_rev step below, which
    // expects a pre-reduced dg tensor. We handle the NK reduction via a simple
    // device-side sum using the host-side temp approach.
    // Sum dg_nk[NK, B, T, H] -> dg[B, T, H]
    // dg is d_g output. We sum in-place.
    {
        // Simple approach: use first slice as accumulator, add remaining slices.
        // For NK=2 (typical), this is just one addition.
        float* dg_base = dg_nk.get<float>();
        float* dg_out_ptr = d_g->get<float>();
        // Copy first slice
        CUDA_CHECK(cudaMemcpyAsync(dg_out_ptr, dg_base, dg_bth * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
        for (int nk = 1; nk < NK; ++nk) {
            // dg_out += dg_nk[nk]
            // Use a simple element-wise add via cublas or a tiny kernel.
            // For correctness, use cuBLAS saxpy: y = alpha*x + y
            float alpha = 1.0f;
            cublasHandle_t handle = mRunState.cublas_handle();
            cublasSetStream(handle, stream);
            cublasSaxpy(handle, static_cast<int>(dg_bth), &alpha,
                        dg_base + nk * dg_bth, 1, dg_out_ptr, 1);
        }
    }

    // bwd_wy: (k, v, beta, g, Ai, dw, dv2, dk, dv, db, dg_wy, T)
    Tensor dg_wy = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_dg_wy");
    mTemps.push_back(dg_wy);
    {
        void* vp = v.Data; void* bp = beta.Data;
        void* gp = g_cum.Data; void* aip = Ai.Data; void* dwp = dw.Data;
        void* dv2p = dv2.Data; void* dvp = d_v->Data;
        void* dbp = d_beta->Data; void* dgwyp = dg_wy.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &k_eff, &vp, &bp, &gp, &aip, &dwp, &dv2p, &dk_data, &dvp, &dbp, &dgwyp, &Tv };
        mGdrKernels.bwd_wy({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                           args, 12, stream);
    }

    // dg += dg_wy
    {
        float alpha = 1.0f;
        cublasHandle_t handle = mRunState.cublas_handle();
        cublasSetStream(handle, stream);
        cublasSaxpy(handle, static_cast<int>(B * T * H), &alpha,
                    dg_wy.get<float>(), 1, d_g->get<float>(), 1);
    }

    // cumsum_rev: reverse cumulative sum for dg
    Tensor dg_out = mRunState.temp_alloc(d_g->DType, {B, T, H}, "gated_delta_rule_dg_out");
    mTemps.push_back(dg_out);
    {
        void* dg_in = d_g->Data; void* dg_outp = dg_out.Data;
        int32_t Tv = static_cast<int32_t>(T);
        void* args[] = { &dg_in, &dg_outp, &Tv };
        mGdrKernels.cumsum_rev({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                               args, 3, stream);
    }
    // Copy result back to d_g output
    CUDA_CHECK(cudaMemcpyAsync(d_g->Data, dg_out.Data,
                                d_g->nelem() * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));

    // L2 norm backward: map dq_norm/dk_norm -> dq/dk
    if (use_l2norm) {
        // l2norm_bwd(x_norm, rstd, dout, dx, T)
        {
            void* xn = q_norm_bwd.Data; void* r = q_rstd_bwd.Data;
            void* dout = dq_norm_buf.Data; void* dx = d_q->Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &xn, &r, &dout, &dx, &T_val };
            mGdrKernels.l2norm_bwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 5, stream);
        }
        {
            void* xn = k_norm_bwd.Data; void* r = k_rstd_bwd.Data;
            void* dout = dk_norm_buf.Data; void* dx = d_k->Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &xn, &r, &dout, &dx, &T_val };
            mGdrKernels.l2norm_bwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 5, stream);
        }
    }

    if (mComm && mComm->ep_enabled() && (mComm->dp_size() == 1)) {
        mComm->all_reduce_avg(*d_g, stream);
    }

    CUDA_CHECK(cudaGetLastError());

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], *d_q);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], *d_k);
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) store_tensor(op.outputs[2], *d_v);
    if (op.outputs.size() > 3 && !op.outputs[3].name.empty()) store_tensor(op.outputs[3], *d_g);
    if (op.outputs.size() > 4 && !op.outputs[4].name.empty()) store_tensor(op.outputs[4], *d_beta);
    if (op.outputs.size() > 5 && !op.outputs[5].name.empty()) store_tensor(op.outputs[5], *d_initial);
}

}  // namespace dsl
