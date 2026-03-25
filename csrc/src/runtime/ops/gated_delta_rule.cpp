// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Qwen3.5 gated delta rule operation dispatch using JIT-compiled Triton kernels.

#include "runtime/dsl/compiled_ops.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <algorithm>

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

    // In flat-token mode: tensor shape is [1, total_tokens, H, K].
    // We need to reshape to [flat_batch_size, max_q_len, H, K] for the
    // recurrent/chunk kernels to process each sequence independently.
    const bool flat_mode = mDecodeState && mDecodeState->flat_token_mode
                        && mDecodeState->flat_batch_size > 1;

    long B = q.Sizes[0];
    long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long K = q.Sizes[3];
    const long V_dim = v.Sizes[3];

    // Early extraction of op attributes and initial state needed by the
    // mixed-batch split dispatch (before B/T are modified by flat reshape).
    const int BT = 64;
    const bool use_l2norm = op.attrs.use_qk_l2norm_in_kernel;
    float scale = op.attrs.delta_rule_scale;
    if (!(scale > 0.0f)) {
        scale = 1.0f / std::sqrt(static_cast<float>(K));
    }
    const bool decode_recurrent_mode = mDecodeState && mDecodeState->recurrent_states;
    Tensor* initial_state = nullptr;
    if (op.inputs.size() > 5 && !op.inputs[5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[5]);
    }
    // Decode mode: inject saved recurrent state as initial_state.
    int layer_idx_for_state = -1;
    Tensor injected_state;
    if (decode_recurrent_mode) {
        std::string field;
        for (const auto& inp : op.inputs) {
            if (parse_block_param(inp.name, layer_idx_for_state, field) && layer_idx_for_state >= 0)
                break;
        }
        if (layer_idx_for_state >= 0) {
            auto it = mDecodeState->recurrent_states->find(layer_idx_for_state);
            if (it != mDecodeState->recurrent_states->end() && it->second != nullptr) {
                injected_state.Data = static_cast<std::byte*>(it->second);
                injected_state.DType = ETensorDType::BF16;
                injected_state.Rank = 4;
                injected_state.Sizes[0] = mDecodeState->flat_token_mode ? mDecodeState->flat_batch_size : B;
                injected_state.Sizes[1] = H;
                injected_state.Sizes[2] = K;
                injected_state.Sizes[3] = V_dim;
                initial_state = &injected_state;
            }
        }
    }

    // Flat-token reshape: scatter [1, total_tokens, ...] → [B_real, max_q_len, ...]
    // and allocate padded buffers for the kernel.
    long B_real = B;
    long T_padded = T;
    Tensor q_padded, k_padded, v_padded, g_padded, beta_padded;
    bool did_flat_reshape = false;

    if (flat_mode) {
        B_real = mDecodeState->flat_batch_size;
        T_padded = mDecodeState->flat_max_q_len;
        const auto* indptr = mDecodeState->q_indptr_host;
        cudaStream_t stream = mRunState.MainStream;

        // Classify slots: decode (q_len<=1) vs prefill (q_len>1).
        int num_decode = 0, num_prefill = 0;
        long max_prefill_q = 0;
        for (long i = 0; i < B_real; ++i) {
            const int ql = indptr[i + 1] - indptr[i];
            if (ql <= 1) { ++num_decode; }
            else { ++num_prefill; max_prefill_q = std::max(max_prefill_q, static_cast<long>(ql)); }
        }

        if (num_decode == B_real) {
            // ALL decode: reshape to [B, 1, ...] → recurrent path.
            B = B_real; T = 1;
            q.Sizes[0] = B; q.Sizes[1] = 1;
            k.Sizes[0] = B; k.Sizes[1] = 1;
            v.Sizes[0] = B; v.Sizes[1] = 1;
            g_input.Sizes[0] = B; g_input.Sizes[1] = 1;
            beta.Sizes[0] = B; beta.Sizes[1] = 1;
            // Fall through to T<=1 recurrent path below.
        } else if (num_decode == 0) {
            // ALL prefill: scatter to [B, T_padded, ...] → chunk path.
            T_padded = max_prefill_q;
            q_padded = mRunState.temp_alloc(q.DType, {B_real, T_padded, H, K}, "gdr_q_pad");
            mTemps.push_back(q_padded);
            k_padded = mRunState.temp_alloc(k.DType, {B_real, T_padded, H, K}, "gdr_k_pad");
            mTemps.push_back(k_padded);
            v_padded = mRunState.temp_alloc(v.DType, {B_real, T_padded, H, V_dim}, "gdr_v_pad");
            mTemps.push_back(v_padded);
            g_padded = mRunState.temp_alloc(g_input.DType, {B_real, T_padded, H}, "gdr_g_pad");
            mTemps.push_back(g_padded);
            beta_padded = mRunState.temp_alloc(beta.DType, {B_real, T_padded, H}, "gdr_beta_pad");
            mTemps.push_back(beta_padded);
            CUDA_CHECK(cudaMemsetAsync(q_padded.Data, 0, q_padded.nelem() * 2, stream));
            CUDA_CHECK(cudaMemsetAsync(k_padded.Data, 0, k_padded.nelem() * 2, stream));
            CUDA_CHECK(cudaMemsetAsync(v_padded.Data, 0, v_padded.nelem() * 2, stream));
            CUDA_CHECK(cudaMemsetAsync(g_padded.Data, 0, g_padded.nelem() * 2, stream));
            CUDA_CHECK(cudaMemsetAsync(beta_padded.Data, 0, beta_padded.nelem() * 2, stream));

            const std::size_t e4d = static_cast<std::size_t>(H * K) * 2;
            const std::size_t ev = static_cast<std::size_t>(H * V_dim) * 2;
            const std::size_t e3d = static_cast<std::size_t>(H) * 2;
            for (long i = 0; i < B_real; ++i) {
                const int qs = indptr[i], ql = indptr[i+1] - indptr[i];
                if (ql <= 0) continue;
                const auto so4 = std::size_t(qs)*e4d, do4 = std::size_t(i)*T_padded*e4d, c4 = std::size_t(ql)*e4d;
                CUDA_CHECK(cudaMemcpyAsync((char*)q_padded.Data+do4, (const char*)q.Data+so4, c4, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync((char*)k_padded.Data+do4, (const char*)k.Data+so4, c4, cudaMemcpyDeviceToDevice, stream));
                const auto sov = std::size_t(qs)*ev, dov = std::size_t(i)*T_padded*ev;
                CUDA_CHECK(cudaMemcpyAsync((char*)v_padded.Data+dov, (const char*)v.Data+sov, std::size_t(ql)*ev, cudaMemcpyDeviceToDevice, stream));
                const auto so3 = std::size_t(qs)*e3d, do3 = std::size_t(i)*T_padded*e3d, c3 = std::size_t(ql)*e3d;
                CUDA_CHECK(cudaMemcpyAsync((char*)g_padded.Data+do3, (const char*)g_input.Data+so3, c3, cudaMemcpyDeviceToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync((char*)beta_padded.Data+do3, (const char*)beta.Data+so3, c3, cudaMemcpyDeviceToDevice, stream));
            }
            B = B_real; T = T_padded;
            did_flat_reshape = true;
            // Fall through to chunk path below.
        } else {
            // ---------------------------------------------------------------
            // MIXED batch: split into decode + prefill sub-batches.
            // Decode slots → recurrent kernel (exact, no padding).
            // Prefill slots → chunk kernel (padded to max_prefill_q).
            // This avoids zero-padding corruption of Mamba recurrent state
            // that occurs when decode slots are processed through the chunk
            // kernel with T_padded >> 1.
            // ---------------------------------------------------------------
            // Build index arrays for each sub-batch.
            std::vector<int> dec_idx, pre_idx;
            dec_idx.reserve(num_decode);
            pre_idx.reserve(num_prefill);
            for (int i = 0; i < static_cast<int>(B_real); ++i) {
                if (indptr[i+1] - indptr[i] <= 1) dec_idx.push_back(i);
                else pre_idx.push_back(i);
            }

            const std::size_t e4d = static_cast<std::size_t>(H * K) * 2;
            const std::size_t ev = static_cast<std::size_t>(H * V_dim) * 2;
            const std::size_t e3d = static_cast<std::size_t>(H) * 2;
            const std::size_t eout = static_cast<std::size_t>(H * V_dim) * 2;
            const std::size_t state_per_seq_bf16 = static_cast<std::size_t>(H * K * V_dim) * 2;
            const std::size_t state_per_seq_fp32 = static_cast<std::size_t>(H * K * V_dim) * 4;

            // Allocate flat output for the entire batch.
            const int total_tokens = mDecodeState->flat_total_tokens;
            Tensor flat_out = mRunState.temp_alloc(q.DType,
                {1, static_cast<long>(total_tokens), H, V_dim}, "gdr_split_flat_out");
            mTemps.push_back(flat_out);

            // Allocate combined final_state [B_real, H, K, V] in FP32.
            Tensor combined_final_state = mRunState.temp_alloc(ETensorDType::FP32,
                {B_real, H, K, V_dim}, "gdr_split_final_state");
            mTemps.push_back(combined_final_state);

            // ---- DECODE sub-batch (recurrent kernel, T=1) ----
            if (!dec_idx.empty() && mGdrKernels.has_recurrent_fwd()) {
                const long Bd = static_cast<long>(dec_idx.size());

                // Extract decode tokens: [Bd, 1, H, K] etc from flat array.
                Tensor dq = mRunState.temp_alloc(q.DType, {Bd, 1, H, K}, "gdr_dec_q");
                mTemps.push_back(dq);
                Tensor dk = mRunState.temp_alloc(k.DType, {Bd, 1, H, K}, "gdr_dec_k");
                mTemps.push_back(dk);
                Tensor dv = mRunState.temp_alloc(v.DType, {Bd, 1, H, V_dim}, "gdr_dec_v");
                mTemps.push_back(dv);
                Tensor dg = mRunState.temp_alloc(g_input.DType, {Bd, 1, H}, "gdr_dec_g");
                mTemps.push_back(dg);
                Tensor dbeta = mRunState.temp_alloc(beta.DType, {Bd, 1, H}, "gdr_dec_beta");
                mTemps.push_back(dbeta);
                Tensor dout = mRunState.temp_alloc(q.DType, {Bd, 1, H, V_dim}, "gdr_dec_out");
                mTemps.push_back(dout);
                Tensor dht = mRunState.temp_alloc(ETensorDType::FP32, {Bd, H, K, V_dim}, "gdr_dec_ht");
                mTemps.push_back(dht);

                // Gather decode tokens from flat array.
                for (long d = 0; d < Bd; ++d) {
                    const int i = dec_idx[d];
                    const int qs = indptr[i];
                    CUDA_CHECK(cudaMemcpyAsync((char*)dq.Data + d*e4d, (const char*)q.Data + std::size_t(qs)*e4d, e4d, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dk.Data + d*e4d, (const char*)k.Data + std::size_t(qs)*e4d, e4d, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dv.Data + d*ev, (const char*)v.Data + std::size_t(qs)*ev, ev, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dg.Data + d*e3d, (const char*)g_input.Data + std::size_t(qs)*e3d, e3d, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)dbeta.Data + d*e3d, (const char*)beta.Data + std::size_t(qs)*e3d, e3d, cudaMemcpyDeviceToDevice, stream));
                }

                // Gather decode initial states from full [B_real, H, K, V] compact buffer.
                Tensor dh0 = mRunState.temp_alloc(ETensorDType::BF16, {Bd, H, K, V_dim}, "gdr_dec_h0");
                mTemps.push_back(dh0);
                if (initial_state) {
                    for (long d = 0; d < Bd; ++d) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            (char*)dh0.Data + d * state_per_seq_bf16,
                            (const char*)initial_state->Data + dec_idx[d] * state_per_seq_bf16,
                            state_per_seq_bf16, cudaMemcpyDeviceToDevice, stream));
                    }
                } else {
                    CUDA_CHECK(cudaMemsetAsync(dh0.Data, 0, dh0.bytes(), stream));
                }

                // L2 norm for decode sub-batch.
                void* dq_eff = dq.Data;
                void* dk_eff = dk.Data;
                if (use_l2norm) {
                    const int BH_d = static_cast<int>(Bd * H);
                    Tensor dqn = mRunState.temp_alloc(dq.DType, {Bd, 1, H, K}, "gdr_dec_qn");
                    mTemps.push_back(dqn);
                    Tensor dqr = mRunState.temp_alloc(ETensorDType::FP32, {Bd, 1, H}, "gdr_dec_qr");
                    mTemps.push_back(dqr);
                    Tensor dkn = mRunState.temp_alloc(dk.DType, {Bd, 1, H, K}, "gdr_dec_kn");
                    mTemps.push_back(dkn);
                    Tensor dkr = mRunState.temp_alloc(ETensorDType::FP32, {Bd, 1, H}, "gdr_dec_kr");
                    mTemps.push_back(dkr);
                    {
                        void* x = dq.Data; void* y = dqn.Data; void* r = dqr.Data;
                        int32_t T1 = 1;
                        void* a[] = {&x, &y, &r, &T1};
                        mGdrKernels.l2norm_fwd_q({1, static_cast<unsigned>(BH_d), 1}, a, 4, stream);
                    }
                    {
                        void* x = dk.Data; void* y = dkn.Data; void* r = dkr.Data;
                        int32_t T1 = 1;
                        void* a[] = {&x, &y, &r, &T1};
                        mGdrKernels.l2norm_fwd_q({1, static_cast<unsigned>(BH_d), 1}, a, 4, stream);
                    }
                    dq_eff = dqn.Data;
                    dk_eff = dkn.Data;
                }

                // Recurrent kernel for decode sub-batch.
                {
                    const int BV_rec = std::min(std::max(next_power_of_2(static_cast<int>(V_dim)), 16), 64);
                    const int NV_rec = cdiv(static_cast<int>(V_dim), BV_rec);
                    const int BH_d = static_cast<int>(Bd * H);
                    void* h0p = dh0.Data;
                    void* htp = dht.Data;
                    void* vd = dv.Data;
                    void* gd = dg.Data;
                    void* bd = dbeta.Data;
                    void* od = dout.Data;
                    int32_t T1 = 1;
                    void* args[] = {&dq_eff, &dk_eff, &vd, &gd, &bd, &od, &h0p, &htp, &scale, &T1};
                    mGdrKernels.recurrent_fwd(
                        {static_cast<unsigned>(NV_rec), static_cast<unsigned>(BH_d), 1},
                        args, 10, stream);
                }

                // Scatter decode outputs back to flat array.
                for (long d = 0; d < Bd; ++d) {
                    const int i = dec_idx[d];
                    const int qs = indptr[i];
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)flat_out.Data + std::size_t(qs)*eout,
                        (const char*)dout.Data + d*eout,
                        eout, cudaMemcpyDeviceToDevice, stream));
                }
                // Scatter decode final states into combined buffer.
                for (long d = 0; d < Bd; ++d) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)combined_final_state.Data + dec_idx[d] * state_per_seq_fp32,
                        (const char*)dht.Data + d * state_per_seq_fp32,
                        state_per_seq_fp32, cudaMemcpyDeviceToDevice, stream));
                }
            }

            // ---- PREFILL sub-batch (chunk kernel) ----
            if (!pre_idx.empty()) {
                const long Bp = static_cast<long>(pre_idx.size());
                const long Tp = max_prefill_q;

                Tensor pq = mRunState.temp_alloc(q.DType, {Bp, Tp, H, K}, "gdr_pre_q");
                mTemps.push_back(pq);
                Tensor pk = mRunState.temp_alloc(k.DType, {Bp, Tp, H, K}, "gdr_pre_k");
                mTemps.push_back(pk);
                Tensor pv = mRunState.temp_alloc(v.DType, {Bp, Tp, H, V_dim}, "gdr_pre_v");
                mTemps.push_back(pv);
                Tensor pg = mRunState.temp_alloc(g_input.DType, {Bp, Tp, H}, "gdr_pre_g");
                mTemps.push_back(pg);
                Tensor pbeta = mRunState.temp_alloc(beta.DType, {Bp, Tp, H}, "gdr_pre_beta");
                mTemps.push_back(pbeta);
                CUDA_CHECK(cudaMemsetAsync(pq.Data, 0, pq.bytes(), stream));
                CUDA_CHECK(cudaMemsetAsync(pk.Data, 0, pk.bytes(), stream));
                CUDA_CHECK(cudaMemsetAsync(pv.Data, 0, pv.bytes(), stream));
                CUDA_CHECK(cudaMemsetAsync(pg.Data, 0, pg.bytes(), stream));
                CUDA_CHECK(cudaMemsetAsync(pbeta.Data, 0, pbeta.bytes(), stream));

                // Scatter prefill tokens.
                for (long p = 0; p < Bp; ++p) {
                    const int i = pre_idx[p];
                    const int qs = indptr[i], ql = indptr[i+1] - indptr[i];
                    if (ql <= 0) continue;
                    const auto so4 = std::size_t(qs)*e4d, do4 = std::size_t(p)*Tp*e4d, c4 = std::size_t(ql)*e4d;
                    CUDA_CHECK(cudaMemcpyAsync((char*)pq.Data+do4, (const char*)q.Data+so4, c4, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)pk.Data+do4, (const char*)k.Data+so4, c4, cudaMemcpyDeviceToDevice, stream));
                    const auto sov = std::size_t(qs)*ev, dov = std::size_t(p)*Tp*ev;
                    CUDA_CHECK(cudaMemcpyAsync((char*)pv.Data+dov, (const char*)v.Data+sov, std::size_t(ql)*ev, cudaMemcpyDeviceToDevice, stream));
                    const auto so3 = std::size_t(qs)*e3d, do3 = std::size_t(p)*Tp*e3d, c3 = std::size_t(ql)*e3d;
                    CUDA_CHECK(cudaMemcpyAsync((char*)pg.Data+do3, (const char*)g_input.Data+so3, c3, cudaMemcpyDeviceToDevice, stream));
                    CUDA_CHECK(cudaMemcpyAsync((char*)pbeta.Data+do3, (const char*)beta.Data+so3, c3, cudaMemcpyDeviceToDevice, stream));
                }

                // Gather prefill initial states.
                Tensor ph0 = mRunState.temp_alloc(ETensorDType::BF16, {Bp, H, K, V_dim}, "gdr_pre_h0");
                mTemps.push_back(ph0);
                if (initial_state) {
                    for (long p = 0; p < Bp; ++p) {
                        CUDA_CHECK(cudaMemcpyAsync(
                            (char*)ph0.Data + p * state_per_seq_bf16,
                            (const char*)initial_state->Data + pre_idx[p] * state_per_seq_bf16,
                            state_per_seq_bf16, cudaMemcpyDeviceToDevice, stream));
                    }
                } else {
                    CUDA_CHECK(cudaMemsetAsync(ph0.Data, 0, ph0.bytes(), stream));
                }

                // L2 norm for prefill sub-batch.
                void* pq_eff = pq.Data;
                void* pk_eff = pk.Data;
                if (use_l2norm) {
                    const int BH_p = static_cast<int>(Bp * H);
                    const int NT_p = cdiv(static_cast<int>(Tp), BT);
                    Tensor pqn = mRunState.temp_alloc(pq.DType, {Bp, Tp, H, K}, "gdr_pre_qn");
                    mTemps.push_back(pqn);
                    Tensor pqr = mRunState.temp_alloc(ETensorDType::FP32, {Bp, Tp, H}, "gdr_pre_qr");
                    mTemps.push_back(pqr);
                    Tensor pkn = mRunState.temp_alloc(pk.DType, {Bp, Tp, H, K}, "gdr_pre_kn");
                    mTemps.push_back(pkn);
                    Tensor pkr = mRunState.temp_alloc(ETensorDType::FP32, {Bp, Tp, H}, "gdr_pre_kr");
                    mTemps.push_back(pkr);
                    {
                        void* x = pq.Data; void* y = pqn.Data; void* r = pqr.Data;
                        int32_t Tv = static_cast<int32_t>(Tp);
                        void* a[] = {&x, &y, &r, &Tv};
                        mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT_p), static_cast<unsigned>(BH_p), 1}, a, 4, stream);
                    }
                    {
                        void* x = pk.Data; void* y = pkn.Data; void* r = pkr.Data;
                        int32_t Tv = static_cast<int32_t>(Tp);
                        void* a[] = {&x, &y, &r, &Tv};
                        mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT_p), static_cast<unsigned>(BH_p), 1}, a, 4, stream);
                    }
                    pq_eff = pqn.Data;
                    pk_eff = pkn.Data;
                }

                // Chunk kernel pipeline for prefill sub-batch.
                const int BH_p = static_cast<int>(Bp * H);
                const int NT_p = cdiv(static_cast<int>(Tp), BT);
                const int BV_h = (K > 64) ? 32 : std::min(std::max(next_power_of_2(static_cast<int>(V_dim)), 16), 64);
                const int BV_o = std::min(std::max(next_power_of_2(static_cast<int>(V_dim)), 16), 64);

                Tensor pg_cum = mRunState.temp_alloc(ETensorDType::FP32, {Bp, Tp, H}, "gdr_pre_gcum");
                mTemps.push_back(pg_cum);
                Tensor pA = mRunState.temp_alloc(ETensorDType::FP32, {Bp, Tp, H, static_cast<long>(BT)}, "gdr_pre_A");
                mTemps.push_back(pA);
                Tensor pAi = mRunState.temp_alloc(ETensorDType::BF16, {Bp, Tp, H, static_cast<long>(BT)}, "gdr_pre_Ai");
                mTemps.push_back(pAi);
                CUDA_CHECK(cudaMemsetAsync(pAi.Data, 0, pAi.bytes(), stream));
                Tensor pw = mRunState.temp_alloc(ETensorDType::BF16, {Bp, Tp, H, K}, "gdr_pre_w");
                mTemps.push_back(pw);
                Tensor pu = mRunState.temp_alloc(ETensorDType::BF16, {Bp, Tp, H, V_dim}, "gdr_pre_u");
                mTemps.push_back(pu);
                Tensor ph = mRunState.temp_alloc(ETensorDType::BF16, {Bp, static_cast<long>(NT_p), H, K, V_dim}, "gdr_pre_h");
                mTemps.push_back(ph);
                Tensor pvn = mRunState.temp_alloc(ETensorDType::BF16, {Bp, Tp, H, V_dim}, "gdr_pre_vn");
                mTemps.push_back(pvn);
                Tensor pout = mRunState.temp_alloc(pq.DType, {Bp, Tp, H, V_dim}, "gdr_pre_out");
                mTemps.push_back(pout);
                Tensor pht = mRunState.temp_alloc(ETensorDType::FP32, {Bp, H, K, V_dim}, "gdr_pre_ht");
                mTemps.push_back(pht);

                // 1. cumsum
                { void* gi = pg.Data; void* go = pg_cum.Data; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&gi, &go, &Tv};
                  mGdrKernels.cumsum_fwd({(unsigned)NT_p, (unsigned)BH_p, 1}, a, 3, stream); }
                // Zero g_cum at padding positions to prevent gate decay corruption.
                // Prefill slots with q_len < Tp have zero-padded positions where
                // cumsum carries the last real gate value forward.
                {
                    const std::size_t h_fp32 = static_cast<std::size_t>(H) * sizeof(float);
                    for (long p = 0; p < Bp; ++p) {
                        const int i = pre_idx[p];
                        const int ql = indptr[i+1] - indptr[i];
                        if (ql >= Tp) continue;
                        const std::size_t off = (static_cast<std::size_t>(p) * Tp + ql) * h_fp32;
                        const std::size_t cnt = static_cast<std::size_t>(Tp - ql) * h_fp32;
                        CUDA_CHECK(cudaMemsetAsync(
                            reinterpret_cast<char*>(pg_cum.Data) + off, 0, cnt, stream));
                    }
                }
                // 2. kkt
                { void* gp = pg_cum.Data; void* bp = pbeta.Data; void* ap = pA.Data; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&pk_eff, &gp, &bp, &ap, &Tv};
                  mGdrKernels.kkt_fwd({(unsigned)NT_p, (unsigned)BH_p, 1}, a, 5, stream); }
                // 3. solve_tril
                { void* ap = pA.Data; void* ai = pAi.Data; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&ap, &ai, &Tv};
                  mGdrKernels.solve_tril({(unsigned)NT_p, (unsigned)BH_p, 1}, a, 3, stream); }
                // 4. wy
                { void* vp = pv.Data; void* bp = pbeta.Data; void* wp = pw.Data; void* up = pu.Data;
                  void* ai = pAi.Data; void* gp = pg_cum.Data; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&pk_eff, &vp, &bp, &wp, &up, &ai, &gp, &Tv};
                  mGdrKernels.wy_fwd({(unsigned)NT_p, (unsigned)BH_p, 1}, a, 8, stream); }
                // 5. fwd_h
                { void* up = pu.Data; void* wp = pw.Data; void* vp = pvn.Data; void* gp = pg_cum.Data;
                  void* hp = ph.Data; void* h0p = ph0.Data; void* htp = pht.Data; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&pk_eff, &up, &wp, &vp, &gp, &hp, &h0p, &htp, &Tv};
                  mGdrKernels.fwd_h({(unsigned)cdiv((int)V_dim, BV_h), (unsigned)BH_p, 1}, a, 9, stream); }
                // 6. fwd_o
                { void* vp = pvn.Data; void* hp = ph.Data; void* gp = pg_cum.Data; void* op = pout.Data;
                  float sv = scale; int32_t Tv = (int32_t)Tp;
                  void* a[] = {&pq_eff, &pk_eff, &vp, &hp, &gp, &op, &sv, &Tv};
                  mGdrKernels.fwd_o({(unsigned)cdiv((int)V_dim, BV_o), (unsigned)NT_p, (unsigned)BH_p}, a, 8, stream); }

                CUDA_CHECK(cudaGetLastError());

                // Gather prefill outputs to flat array.
                for (long p = 0; p < Bp; ++p) {
                    const int i = pre_idx[p];
                    const int qs = indptr[i], ql = indptr[i+1] - indptr[i];
                    if (ql <= 0) continue;
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)flat_out.Data + std::size_t(qs)*eout,
                        (const char*)pout.Data + std::size_t(p)*Tp*eout,
                        std::size_t(ql)*eout, cudaMemcpyDeviceToDevice, stream));
                }
                // Scatter prefill final states into combined buffer.
                for (long p = 0; p < Bp; ++p) {
                    CUDA_CHECK(cudaMemcpyAsync(
                        (char*)combined_final_state.Data + pre_idx[p] * state_per_seq_fp32,
                        (const char*)pht.Data + p * state_per_seq_fp32,
                        state_per_seq_fp32, cudaMemcpyDeviceToDevice, stream));
                }
            }

            // Store outputs.
            if (!op.outputs.empty() && !op.outputs[0].name.empty())
                store_tensor(op.outputs[0], flat_out);
            if (op.outputs.size() > 1 && !op.outputs[1].name.empty())
                store_tensor(op.outputs[1], combined_final_state);

            // Save recurrent state for next step.
            const bool drm = mDecodeState && mDecodeState->recurrent_states;
            const bool srs = drm && mDecodeState->strict_state_buffers;
            int layer_idx = -1;
            if (drm) {
                std::string field;
                for (const auto& inp : op.inputs) {
                    if (parse_block_param(inp.name, layer_idx, field) && layer_idx >= 0) break;
                }
            }
            if (drm && layer_idx >= 0) {
                auto& states = *mDecodeState->recurrent_states;
                auto it_state = states.find(layer_idx);
                void* saved = (it_state != states.end()) ? it_state->second : nullptr;
                const std::size_t total_state_bf16 = B_real * state_per_seq_bf16;
                if (!saved) {
                    if (srs) {
                        throw std::runtime_error(std::string(op_name)
                            + ": strict decode recurrent state missing for layer "
                            + std::to_string(layer_idx));
                    }
                    CUDA_CHECK(cudaMalloc(&saved, total_state_bf16));
                    states[layer_idx] = saved;
                }
                if (mDecodeState->recurrent_state_bytes) {
                    auto& sbm = *mDecodeState->recurrent_state_bytes;
                    if (!srs) sbm[layer_idx] = state_per_seq_bf16;
                }
                convert_dtype(reinterpret_cast<nv_bfloat16*>(saved),
                              combined_final_state.get<float>(),
                              static_cast<long>(B_real * H * K * V_dim), stream);
            }
            return;  // Mixed batch fully handled.
        }
    }

    // Use padded tensors if we reshaped, otherwise use originals.
    Tensor& q_eff_tensor = did_flat_reshape ? q_padded : q;
    Tensor& k_eff_tensor = did_flat_reshape ? k_padded : k;
    Tensor& v_eff_tensor = did_flat_reshape ? v_padded : v;
    Tensor& g_eff_tensor = did_flat_reshape ? g_padded : g_input;
    Tensor& beta_eff_tensor = did_flat_reshape ? beta_padded : beta;

    const long V = V_dim;  // alias for the rest of the function
    const std::size_t recurrent_state_bytes =
        static_cast<std::size_t>(B * H * K * V) * sizeof(nv_bfloat16);
    const std::size_t recurrent_state_per_seq_bytes =
        recurrent_state_bytes / static_cast<std::size_t>(std::max<long>(1, B));

    if (!did_flat_reshape) {
        if (!tensor_shape_matches(k, B, T, H, K)) {
            throw std::runtime_error(std::string(op_name) + ": k shape must match q");
        }
        if (v.Sizes[0] != B || v.Sizes[1] != T || v.Sizes[2] != H) {
            throw std::runtime_error(std::string(op_name) + ": v must share B/T/H with q");
        }
    }
    // For non-flat or single-batch path: re-inject initial_state with
    // updated B (may have changed from flat reshape above).
    const bool strict_recurrent_state =
        decode_recurrent_mode && mDecodeState->strict_state_buffers;
    if (decode_recurrent_mode && layer_idx_for_state >= 0 && !flat_mode) {
        auto it = mDecodeState->recurrent_states->find(layer_idx_for_state);
        if (it != mDecodeState->recurrent_states->end() && it->second != nullptr) {
            injected_state.Data = static_cast<std::byte*>(it->second);
            injected_state.DType = ETensorDType::BF16;
            injected_state.Rank = 4;
            injected_state.Sizes[0] = B;
            injected_state.Sizes[1] = H;
            injected_state.Sizes[2] = K;
            injected_state.Sizes[3] = V_dim;
            initial_state = &injected_state;
        } else if (strict_recurrent_state) {
            throw std::runtime_error(
                std::string(op_name) + ": strict decode recurrent state missing for layer "
                + std::to_string(layer_idx_for_state));
        }
    }
    // For flat_mode all-decode path: B was changed, re-inject with correct B.
    if (flat_mode && decode_recurrent_mode && layer_idx_for_state >= 0 && injected_state.Data) {
        injected_state.Sizes[0] = B;
    }
    if (strict_recurrent_state && layer_idx_for_state >= 0) {
        if (!mDecodeState->recurrent_state_bytes) {
            throw std::runtime_error(
                std::string(op_name) + ": strict decode recurrent state requires recurrent_state_bytes map");
        }
        auto it_bytes = mDecodeState->recurrent_state_bytes->find(layer_idx_for_state);
        if (it_bytes == mDecodeState->recurrent_state_bytes->end()
            || it_bytes->second != recurrent_state_per_seq_bytes) {
            throw std::runtime_error(
                std::string(op_name) + ": strict decode recurrent state byte-size mismatch for layer "
                + std::to_string(layer_idx_for_state));
        }
    }

    // Allocate outputs
    Tensor* out_ptr = nullptr;
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
        if (out_ref.DType == v.DType && tensor_shape_matches(out_ref, B, T, H, V)) {
            out_ptr = &out_ref;
        }
    }
    if (!out_ptr) {
        Tensor out_t = mRunState.temp_alloc(v.DType, {B, T, H, V}, "gated_delta_rule_output");
        mTemps.push_back(out_t);
        out_ptr = &mTemps.back();
    }

    Tensor* final_state_ptr = nullptr;
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        Tensor& state_ref = ensure_output_tensor(op.outputs[1]);
        if (state_ref.DType == ETensorDType::FP32 &&
            tensor_shape_matches(state_ref, B, H, K, V)) {
            final_state_ptr = &state_ref;
        }
    }
    if (!final_state_ptr) {
        Tensor state_t = mRunState.temp_alloc(ETensorDType::FP32, {B, H, K, V}, "gated_delta_rule_final_state");
        mTemps.push_back(state_t);
        final_state_ptr = &mTemps.back();
    }

    const int BH = static_cast<int>(B * H);
    cudaStream_t stream = mRunState.MainStream;
    void* q_eff = q_eff_tensor.Data;
    void* k_eff = k_eff_tensor.Data;
    if (use_l2norm) {
        const int NT_l2 = cdiv(static_cast<int>(T), BT);
        Tensor q_norm = mRunState.temp_alloc(q_eff_tensor.DType, {B, T, H, K}, "gated_delta_rule_q_norm");
        mTemps.push_back(q_norm);
        Tensor q_rstd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_q_rstd");
        mTemps.push_back(q_rstd);
        Tensor k_norm = mRunState.temp_alloc(k_eff_tensor.DType, {B, T, H, K}, "gated_delta_rule_k_norm");
        mTemps.push_back(k_norm);
        Tensor k_rstd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_k_rstd");
        mTemps.push_back(k_rstd);

        // L2 norm Q: (x, y, rstd, T)
        {
            void* x = q_eff_tensor.Data; void* y = q_norm.Data; void* r = q_rstd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT_l2), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        // L2 norm K: same kernel (D=K)
        {
            void* x = k_eff_tensor.Data; void* y = k_norm.Data; void* r = k_rstd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT_l2), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        q_eff = q_norm.Data;
        k_eff = k_norm.Data;
    }

    // -----------------------------------------------------------------------
    // Recurrent path: for T<=1 (decode), use the fused recurrent kernel
    // -----------------------------------------------------------------------
    if (T <= 1 && mGdrKernels.has_recurrent_fwd()) {
        const int BV_rec = std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
        const int NV_rec = cdiv(static_cast<int>(V), BV_rec);

        Tensor h0_buf;
        void* h0_ptr = nullptr;
        if (initial_state) {
            h0_ptr = initial_state->Data;
        } else {
            // For empty/length-1 prompts there is no prefill-produced recurrent
            // state; decode should start from zero state.
            h0_buf = mRunState.temp_alloc(ETensorDType::BF16, {B, H, K, V}, "gated_delta_rule_h0_rec");
            mTemps.push_back(h0_buf);
            CUDA_CHECK(cudaMemsetAsync(h0_buf.Data, 0, h0_buf.nelem() * 2, stream));
            h0_ptr = h0_buf.Data;
        }
        void* ht_ptr = final_state_ptr->Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* v_data = v_eff_tensor.Data;
        void* g_data = g_eff_tensor.Data;
        void* beta_data = beta_eff_tensor.Data;
        void* args[] = {
            &q_eff, &k_eff, &v_data, &g_data, &beta_data,
            &out_ptr->Data, &h0_ptr, &ht_ptr, &scale, &T_val
        };
        mGdrKernels.recurrent_fwd(
            {static_cast<unsigned>(NV_rec), static_cast<unsigned>(BH), 1},
            args, 10, stream);
        CUDA_CHECK(cudaGetLastError());

        if (!op.outputs.empty() && !op.outputs[0].name.empty())
            store_tensor(op.outputs[0], *out_ptr);
        if (op.outputs.size() > 1 && !op.outputs[1].name.empty())
            store_tensor(op.outputs[1], *final_state_ptr);

        // Save recurrent state for next step
        if (decode_recurrent_mode && layer_idx_for_state >= 0) {
            auto& states = *mDecodeState->recurrent_states;
            auto it_state = states.find(layer_idx_for_state);
            void* saved = (it_state != states.end()) ? it_state->second : nullptr;
            if (!saved) {
                if (strict_recurrent_state) {
                    throw std::runtime_error(
                        std::string(op_name)
                        + ": strict decode recurrent state missing for layer "
                        + std::to_string(layer_idx_for_state));
                }
                CUDA_CHECK(cudaMalloc(&saved, recurrent_state_bytes));
                states[layer_idx_for_state] = saved;
            }
            if (mDecodeState->recurrent_state_bytes) {
                auto& state_bytes_map = *mDecodeState->recurrent_state_bytes;
                if (strict_recurrent_state) {
                    auto it_bytes = state_bytes_map.find(layer_idx_for_state);
                    if (it_bytes == state_bytes_map.end()
                        || it_bytes->second != recurrent_state_per_seq_bytes) {
                        throw std::runtime_error(
                            std::string(op_name)
                            + ": strict decode recurrent state byte-size mismatch for layer "
                            + std::to_string(layer_idx_for_state));
                    }
                } else {
                    state_bytes_map[layer_idx_for_state] = recurrent_state_per_seq_bytes;
                }
            } else if (strict_recurrent_state) {
                throw std::runtime_error(
                    std::string(op_name)
                    + ": strict decode recurrent state requires recurrent_state_bytes map");
            }
            if (final_state_ptr->DType == ETensorDType::FP32) {
                convert_dtype(reinterpret_cast<nv_bfloat16*>(saved),
                              final_state_ptr->get<float>(),
                              static_cast<long>(B * H * K * V), stream);
            } else {
                CUDA_CHECK(cudaMemcpyAsync(saved, final_state_ptr->Data, recurrent_state_bytes,
                                           cudaMemcpyDeviceToDevice, stream));
            }
        }
        return;
    }

    const int NT = cdiv(static_cast<int>(T), BT);
    const int BK_kkt = std::min(std::max(next_power_of_2(static_cast<int>(K)), 16), 64);
    const int BV_h = (K > 64) ? 32 : std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);
    const int BK_o = std::min(std::max(next_power_of_2(static_cast<int>(K)), 16), 64);
    const int BV_o = std::min(std::max(next_power_of_2(static_cast<int>(V)), 16), 64);

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
        void* g_in_ptr = g_eff_tensor.Data;
        void* g_out_ptr = g_cum.Data;
        int32_t T_val = static_cast<int32_t>(T);
        void* args[] = { &g_in_ptr, &g_out_ptr, &T_val };
        mGdrKernels.cumsum_fwd({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                               args, 3, stream);
    }

    // 2. kkt_fwd: (k, g_cum, beta, A, T) grid=(NT, B*H)
    {
        void* g_ptr = g_cum.Data;
        void* beta_ptr = beta_eff_tensor.Data;
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
        void* v_ptr = v_eff_tensor.Data;
        void* beta_ptr = beta_eff_tensor.Data;
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

    // Flat-token mode: gather output from padded [B, T_padded, H, V] back to
    // flat [1, total_tokens, H, V] for the rest of the graph.
    if (did_flat_reshape) {
        const auto* indptr = mDecodeState->q_indptr_host;
        const int total_tokens = mDecodeState->flat_total_tokens;
        Tensor flat_out = mRunState.temp_alloc(out_ptr->DType,
            {1, static_cast<long>(total_tokens), H, V}, "gdr_flat_out");
        mTemps.push_back(flat_out);
        const std::size_t elem_bytes = static_cast<std::size_t>(H * V) * 2;
        for (long i = 0; i < B_real; ++i) {
            const int q_start = indptr[i];
            const int q_len = indptr[i + 1] - indptr[i];
            if (q_len <= 0) continue;
            const auto src_off = static_cast<std::size_t>(i) * static_cast<std::size_t>(T_padded) * elem_bytes;
            const auto dst_off = static_cast<std::size_t>(q_start) * elem_bytes;
            CUDA_CHECK(cudaMemcpyAsync(
                reinterpret_cast<char*>(flat_out.Data) + dst_off,
                reinterpret_cast<const char*>(out_ptr->Data) + src_off,
                static_cast<std::size_t>(q_len) * elem_bytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        out_ptr = &mTemps.back();
    }

    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], *out_ptr);
    }
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        store_tensor(op.outputs[1], *final_state_ptr);
    }

    // Decode mode: save the final recurrent state for the next decode step.
    // The state buffer must persist across steps (NOT stack-allocated).
    if (decode_recurrent_mode && layer_idx_for_state >= 0) {
        auto& states = *mDecodeState->recurrent_states;
        auto it_state = states.find(layer_idx_for_state);
        void* saved = (it_state != states.end()) ? it_state->second : nullptr;

        // Reallocate if the buffer is too small for the current batch.
        // The saved buffer was sized for a previous B; if B grew we must reallocate.
        if (saved) {
            // Check current buffer capacity vs required size.
            // The saved buffer holds B_old * per_seq_bytes. We need B * per_seq_bytes.
            // Track capacity via recurrent_state_bytes map (stores per_seq not total).
            // We need to compare total: saved_capacity vs recurrent_state_bytes.
            // Use a simple heuristic: if the compact capacity is tracked, verify it.
            // Otherwise, always reallocate to be safe.
            auto& cap_map = *mDecodeState->recurrent_state_bytes;
            auto it_cap = cap_map.find(layer_idx_for_state);
            // The map stores per-seq bytes but we need B * per_seq.
            // We must also track B. Since we can't, just reallocate every time B > 1
            // and the buffer might be undersized. A more robust approach: use
            // a capacity map that stores total bytes.
            // For now, free+realloc when not strict (acceptable overhead).
            if (!strict_recurrent_state) {
                CUDA_CHECK(cudaFree(saved));
                saved = nullptr;
                states[layer_idx_for_state] = nullptr;
            }
        }

        if (!saved) {
            if (strict_recurrent_state) {
                throw std::runtime_error(
                    std::string(op_name)
                    + ": strict decode recurrent state missing for layer "
                    + std::to_string(layer_idx_for_state));
            }
            // Allocate persistent GPU buffer for B sequences.
            CUDA_CHECK(cudaMalloc(&saved, recurrent_state_bytes));
            states[layer_idx_for_state] = saved;
        }
        if (mDecodeState->recurrent_state_bytes) {
            auto& state_bytes_map = *mDecodeState->recurrent_state_bytes;
            if (strict_recurrent_state) {
                auto it_bytes = state_bytes_map.find(layer_idx_for_state);
                if (it_bytes == state_bytes_map.end()
                    || it_bytes->second != recurrent_state_per_seq_bytes) {
                    throw std::runtime_error(
                        std::string(op_name)
                        + ": strict decode recurrent state byte-size mismatch for layer "
                        + std::to_string(layer_idx_for_state));
                }
            } else {
                state_bytes_map[layer_idx_for_state] = recurrent_state_per_seq_bytes;
            }
        } else if (strict_recurrent_state) {
            throw std::runtime_error(
                std::string(op_name)
                + ": strict decode recurrent state requires recurrent_state_bytes map");
        }

        // The final_state is in FP32 but the kernel expects BF16 initial_state.
        // Convert FP32→BF16 into the saved buffer.
        if (final_state_ptr->DType == ETensorDType::FP32) {
            convert_dtype(reinterpret_cast<nv_bfloat16*>(saved),
                          final_state_ptr->get<float>(),
                          static_cast<long>(B * H * K * V),
                          mRunState.MainStream);
        } else {
            CUDA_CHECK(cudaMemcpyAsync(saved, final_state_ptr->Data, recurrent_state_bytes,
                                       cudaMemcpyDeviceToDevice, mRunState.MainStream));
        }
    }
}

void CompiledExecutor::dispatch_chunk_gated_delta_rule(const CompiledOp& op) {
    dispatch_gated_delta_rule_common(op, "chunk_gated_delta_rule");
}

void CompiledExecutor::dispatch_chunk_gated_delta_rule_backward(const CompiledOp& op) {
    if (!mGdrKernels.is_ready()) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: JIT Triton kernels not loaded.");
    }
    if (op.inputs.size() < 7) {
        throw std::runtime_error(
            "chunk_gated_delta_rule_backward: expected at least 7 inputs");
    }

    Tensor& d_out = resolve_tensor(op.inputs[0]);

    Tensor* d_final_state = nullptr;
    if (op.inputs.size() > 1 && !op.inputs[1].name.empty()) {
        d_final_state = &resolve_tensor(op.inputs[1]);
    }
    const std::size_t offs = 2;

    Tensor& q = resolve_tensor(op.inputs[offs + 0]);
    Tensor& k = resolve_tensor(op.inputs[offs + 1]);
    Tensor& v = resolve_tensor(op.inputs[offs + 2]);
    Tensor& g_input = resolve_tensor(op.inputs[offs + 3]);
    Tensor& beta = resolve_tensor(op.inputs[offs + 4]);

    Tensor* initial_state = nullptr;
    if (op.inputs.size() > offs + 5 && !op.inputs[offs + 5].name.empty()) {
        initial_state = &resolve_tensor(op.inputs[offs + 5]);
    }

    const long B = q.Sizes[0];
    const long T = q.Sizes[1];
    const long H = q.Sizes[2];
    const long K = q.Sizes[3];
    const long V = v.Sizes[3];

    // Allocate gradient outputs
    auto ensure_or_temp = [&](std::size_t out_idx,
                              ETensorDType dtype,
                              const std::vector<long>& shape) -> Tensor* {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            bool ok = true;
            if (out_ref.DType != dtype) ok = false;
            if (shape.size() == 4 && !tensor_shape_matches(out_ref, shape[0], shape[1], shape[2], shape[3])) ok = false;
            if (shape.size() == 3 && (out_ref.Rank != 3 || out_ref.Sizes[0] != shape[0] ||
                out_ref.Sizes[1] != shape[1] || out_ref.Sizes[2] != shape[2])) ok = false;
            if (ok) return &out_ref;
        }
        Tensor temp = mRunState.temp_alloc(dtype, shape);
        mTemps.push_back(temp);
        return &mTemps.back();
    };

    Tensor* d_q = ensure_or_temp(0, q.DType, {B, T, H, K});
    Tensor* d_k = ensure_or_temp(1, k.DType, {B, T, H, K});
    Tensor* d_v = ensure_or_temp(2, v.DType, {B, T, H, V});
    Tensor* d_g = ensure_or_temp(3, g_input.DType, {B, T, H});
    Tensor* d_beta = ensure_or_temp(4, beta.DType, {B, T, H});
    Tensor* d_initial = ensure_or_temp(5, ETensorDType::FP32, {B, H, K, V});

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
    const bool use_l2norm = op.attrs.use_qk_l2norm_in_kernel;
    void* q_eff = q.Data;
    void* k_eff = k.Data;
    void* dq_data = d_q->Data;
    void* dk_data = d_k->Data;
    Tensor q_norm_bwd, k_norm_bwd, q_rstd_bwd, k_rstd_bwd;
    Tensor dq_norm_buf, dk_norm_buf;
    if (use_l2norm) {
        q_norm_bwd = mRunState.temp_alloc(q.DType, {B, T, H, K}, "gated_delta_rule_q_norm_bwd");
        mTemps.push_back(q_norm_bwd);
        q_rstd_bwd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_q_rstd_bwd");
        mTemps.push_back(q_rstd_bwd);
        k_norm_bwd = mRunState.temp_alloc(k.DType, {B, T, H, K}, "gated_delta_rule_k_norm_bwd");
        mTemps.push_back(k_norm_bwd);
        k_rstd_bwd = mRunState.temp_alloc(ETensorDType::FP32, {B, T, H}, "gated_delta_rule_k_rstd_bwd");
        mTemps.push_back(k_rstd_bwd);
        {
            void* x = q.Data; void* y = q_norm_bwd.Data; void* r = q_rstd_bwd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        {
            void* x = k.Data; void* y = k_norm_bwd.Data; void* r = k_rstd_bwd.Data;
            int32_t T_val = static_cast<int32_t>(T);
            void* args[] = { &x, &y, &r, &T_val };
            mGdrKernels.l2norm_fwd_q({static_cast<unsigned>(NT), static_cast<unsigned>(BH), 1},
                                     args, 4, stream);
        }
        q_eff = q_norm_bwd.Data;
        k_eff = k_norm_bwd.Data;
        // Backward pipeline writes dq_norm/dk_norm to temp buffers
        try {
            dq_norm_buf = mRunState.temp_alloc(q.DType, {B, T, H, K}, "gated_delta_rule_dq_norm_buf");
            mTemps.push_back(dq_norm_buf);
            dk_norm_buf = mRunState.temp_alloc(k.DType, {B, T, H, K}, "gated_delta_rule_dk_norm_buf");
            mTemps.push_back(dk_norm_buf);
        } catch (const std::exception& e) {
            throw;
        }
        dq_data = dq_norm_buf.Data;
        dk_data = dk_norm_buf.Data;
    }

    // ---- Recompute forward intermediates ----
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

    // TODO: dg_nk reduction across NK dimension needs a small kernel or cumsum approach.
    // For now, the dg reduction is deferred to the cumsum_rev step below, which
    // expects a pre-reduced dg tensor. We handle the NK reduction via a simple
    // device-side sum using the host-side temp approach.
    // Sum dg_nk[NK, B, T, H] -> dg[B, T, H]
    // dg is d_g output. We sum in-place.
    {
        // Simple approach: use first slice as accumulator, add remaining slices.
        // For NK=2 (typical), this is just one addition.
        const long bth = B * T * H;
        float* dg_base = dg_nk.get<float>();
        float* dg_out_ptr = d_g->get<float>();
        // Copy first slice
        CUDA_CHECK(cudaMemcpyAsync(dg_out_ptr, dg_base, bth * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream));
        for (int nk = 1; nk < NK; ++nk) {
            // dg_out += dg_nk[nk]
            // Use a simple element-wise add via cublas or a tiny kernel.
            // For correctness, use cuBLAS saxpy: y = alpha*x + y
            float alpha = 1.0f;
            cublasHandle_t handle = mRunState.cublas_handle();
            cublasSetStream(handle, stream);
            cublasSaxpy(handle, static_cast<int>(bth), &alpha,
                        dg_base + nk * bth, 1, dg_out_ptr, 1);
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

    CUDA_CHECK(cudaGetLastError());

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], *d_q);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], *d_k);
    if (op.outputs.size() > 2 && !op.outputs[2].name.empty()) store_tensor(op.outputs[2], *d_v);
    if (op.outputs.size() > 3 && !op.outputs[3].name.empty()) store_tensor(op.outputs[3], *d_g);
    if (op.outputs.size() > 4 && !op.outputs[4].name.empty()) store_tensor(op.outputs[4], *d_beta);
    if (op.outputs.size() > 5 && !op.outputs[5].name.empty()) store_tensor(op.outputs[5], *d_initial);
}

}  // namespace dsl
