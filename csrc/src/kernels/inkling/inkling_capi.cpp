// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels (lane g2): a flat C entry point per kernel, for the GPU op tests only
// (tests/test_inkling2_kernels_gpu.py loads libsurogate-common.so with ctypes and passes torch device pointers).
// Every call runs on the given stream (0 = legacy default), returns 0 on success and 1 on error (message:
// inkling_capi_last_error). No production code calls these.

#include <cstring>
#include <exception>
#include <string>

#include "kernels/inkling/flash_rel/inkling_attention.h"
#include "kernels/inkling/inkling.h"
#include "kernels/inkling/moe_backward.h"
#include "kernels/inkling/visit_backward.h"

#define INKLING_CAPI extern "C" __attribute__((visibility("default")))

namespace {
thread_local std::string g_err;
template <typename F>
int guard(F&& f) {
    try {
        f();
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            g_err = cudaGetErrorString(e);
            return 1;
        }
        return 0;
    } catch (const std::exception& ex) {
        g_err = ex.what();
        return 1;
    }
}
using inkling::bf16;
cudaStream_t st(void* s) {
    return static_cast<cudaStream_t>(s);
}
}  // namespace

INKLING_CAPI const char* inkling_capi_last_error() {
    return g_err.c_str();
}

INKLING_CAPI int inkling_capi_log_tau(float* tau, const int* pos, int T, float n_floor, float alpha, void* s) {
    return guard([&] { inkling::log_tau(tau, pos, T, n_floor, alpha, st(s)); });
}

INKLING_CAPI int inkling_capi_qkv_norm_fwd(void* y, const void* x, int mode, const void* scale, const float* tau,
                                           float threshold, float eps, int kv_fp8, int T, int H, int d, void* s) {
    return guard([&] {
        inkling::QkvNormArgs a;
        a.mode = mode ? inkling::NormMode::Rms : inkling::NormMode::Clip;
        a.scale = static_cast<const bf16*>(scale);
        a.tau = tau;
        a.threshold = threshold;
        a.eps = eps;
        a.kv_fp8 = kv_fp8 != 0;
        a.T = T;
        a.H = H;
        a.d = d;
        inkling::qkv_norm_fwd(static_cast<bf16*>(y), static_cast<const bf16*>(x), a, st(s));
    });
}

INKLING_CAPI int inkling_capi_qkv_norm_bwd(float* dx, const void* dy, int dy_fp32, const void* x, int mode,
                                           const void* scale, const float* tau, float threshold, float eps, int T,
                                           int H, int d, void* s) {
    return guard([&] {
        inkling::QkvNormArgs a;
        a.mode = mode ? inkling::NormMode::Rms : inkling::NormMode::Clip;
        a.scale = static_cast<const bf16*>(scale);
        a.tau = tau;
        a.threshold = threshold;
        a.eps = eps;
        a.T = T;
        a.H = H;
        a.d = d;
        inkling::qkv_norm_bwd(dx, dy, dy_fp32 != 0, static_cast<const bf16*>(x), a, st(s));
    });
}

namespace {
inkling::ConvArgs conv_args(const int* pos, const void* carry, float scale, int T, int C, int head_dim) {
    inkling::ConvArgs a;
    a.positions = pos;
    a.carry = static_cast<const bf16*>(carry);
    a.scale = scale;
    a.T = T;
    a.C = C;
    a.head_dim = head_dim;
    return a;
}
}  // namespace

INKLING_CAPI int inkling_capi_short_conv_fwd(void* out, const void* x, const void* w, const void* addend,
                                             const void* residual, const int* pos, const void* carry, float scale,
                                             int T, int C, void* s) {
    return guard([&] {
        inkling::short_conv_fwd(static_cast<bf16*>(out), static_cast<const bf16*>(x), static_cast<const bf16*>(w),
                                static_cast<const bf16*>(addend), static_cast<const bf16*>(residual),
                                conv_args(pos, carry, scale, T, C, 128), st(s));
    });
}

INKLING_CAPI int inkling_capi_short_conv_bwd(float* dx, float* d_carry, const void* dy, int dy_fp32, const void* w,
                                             const float* d_carry_next, const int* pos, const void* carry,
                                             float scale, int T, int C, void* s) {
    return guard([&] {
        inkling::short_conv_bwd(dx, d_carry, dy, dy_fp32 != 0, static_cast<const bf16*>(w), d_carry_next,
                                conv_args(pos, carry, scale, T, C, 128), st(s));
    });
}

INKLING_CAPI int inkling_capi_dynamic_conv_fwd(void* out, const void* x, const void* static_w, const void* dyn,
                                               const int* pos, const void* carry, float scale, int T, int C,
                                               int head_dim, void* s) {
    return guard([&] {
        inkling::dynamic_conv_fwd(static_cast<bf16*>(out), static_cast<const bf16*>(x),
                                  static_cast<const bf16*>(static_w), static_cast<const bf16*>(dyn),
                                  conv_args(pos, carry, scale, T, C, head_dim), st(s));
    });
}

INKLING_CAPI int inkling_capi_dynamic_conv_bwd(float* dx, float* d_dyn, float* d_carry, const void* dy, int dy_fp32,
                                               const void* x, const void* static_w, const void* dyn,
                                               const float* d_carry_next, const int* pos, const void* carry,
                                               float scale, int T, int C, int head_dim, void* s) {
    return guard([&] {
        inkling::dynamic_conv_bwd(dx, d_dyn, d_carry, dy, dy_fp32 != 0, static_cast<const bf16*>(x),
                                  static_cast<const bf16*>(static_w), static_cast<const bf16*>(dyn), d_carry_next,
                                  conv_args(pos, carry, scale, T, C, head_dim), st(s));
    });
}

INKLING_CAPI int inkling_capi_attn_merge_fwd(void* result, float* out_s, float* lse_s, const float* o_n,
                                             const float* l_n, const float* o_f, const float* l_f, const void* sink,
                                             const void* gate, int T, int H, void* s) {
    return guard([&] {
        inkling::attn_merge_fwd(static_cast<bf16*>(result), out_s, lse_s, o_n, l_n, o_f, l_f,
                                static_cast<const bf16*>(sink), static_cast<const bf16*>(gate), T, H, st(s));
    });
}

INKLING_CAPI int inkling_capi_attn_merge_bwd_union(void* dout, float* delta, float* d_sink, float* d_gate,
                                                   const void* d_result, int d_result_fp32, const float* out_s,
                                                   const float* lse_s, const void* sink, const void* gate, int T,
                                                   int H, void* s) {
    return guard([&] {
        inkling::attn_merge_bwd_union(static_cast<bf16*>(dout), delta, d_sink, d_gate, d_result, d_result_fp32 != 0,
                                      out_s, lse_s, static_cast<const bf16*>(sink), static_cast<const bf16*>(gate), T,
                                      H, st(s));
    });
}

namespace {
inkling::AttnPartArgs part_args(const void* q, const void* k, const void* v, const int* cu_q, const int* cu_k,
                                int* seqused, int num_segs, int max_q, int max_k, int total_q, int Hq, int Hkv,
                                float scale, int window, const void* rel, int rel_extent, float* out, float* lse) {
    inkling::AttnPartArgs a;
    a.q = static_cast<const bf16*>(q);
    a.k = static_cast<const bf16*>(k);
    a.v = static_cast<const bf16*>(v);
    a.cu_q = cu_q;
    a.cu_k = cu_k;
    a.seqused_k = seqused;
    a.num_segs = num_segs;
    a.max_q = max_q;
    a.max_k = max_k;
    a.total_q = total_q;
    a.Hq = Hq;
    a.Hkv = Hkv;
    a.scale = scale;
    a.window = window;
    a.rel = static_cast<const bf16*>(rel);
    a.rel_extent = rel_extent;
    a.out = out;
    a.lse = lse;
    return a;
}
}  // namespace

/// part 0 = NEAR (rel required), 1 = FAR (seqused scratch [num_segs] required).
INKLING_CAPI int inkling_capi_attention_part_fwd(int part, const void* q, const void* k, const void* v,
                                                 const int* cu_q, const int* cu_k, int* seqused, int num_segs,
                                                 int max_q, int max_k, int total_q, int Hq, int Hkv, float scale,
                                                 int window, const void* rel, int rel_extent, float* out, float* lse,
                                                 void* s) {
    return guard([&] {
        auto a = part_args(q, k, v, cu_q, cu_k, seqused, num_segs, max_q, max_k, total_q, Hq, Hkv, scale, window, rel,
                           rel_extent, out, lse);
        if (part == 0)
            inkling::attention_near_fwd(a, st(s));
        else
            inkling::attention_far_fwd(a, st(s));
    });
}

INKLING_CAPI long inkling_capi_attention_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq) {
    return static_cast<long>(inkling::attention_bwd_workspace_bytes(total_q, max_q, num_segs, Hq));
}

INKLING_CAPI int inkling_capi_attention_part_bwd(int part, const void* q, const void* k, const void* v,
                                                 const int* cu_q, const int* cu_k, int num_segs, int max_q, int max_k,
                                                 int total_q, int total_k, int Hq, int Hkv, float scale, int window,
                                                 const void* rel, int rel_extent, const void* dout, const float* lse,
                                                 const float* delta, float* dq, float* dk, float* dv, float* d_rel,
                                                 void* workspace, void* s) {
    return guard([&] {
        inkling::AttnPartBwdArgs b;
        b.fwd = part_args(q, k, v, cu_q, cu_k, nullptr, num_segs, max_q, max_k, total_q, Hq, Hkv, scale, window, rel,
                          rel_extent, nullptr, nullptr);
        b.total_k = total_k;
        b.dout = static_cast<const bf16*>(dout);
        b.lse = lse;
        b.delta = delta;
        b.dq = dq;
        b.dk = dk;
        b.dv = dv;
        b.d_rel = d_rel;
        b.workspace = workspace;
        if (part == 0)
            inkling::attention_near_bwd(b, st(s));
        else
            inkling::attention_far_bwd(b, st(s));
    });
}

namespace {
inkling::AttnResArgs res_args(const void* const* src, const int* src_fp32, int N, const void* addend, int addend_fp32,
                              const float* query, float query_scale, const void* norm_w, float eps, int Q, int T,
                              int D) {
    inkling::AttnResArgs a;
    for (int n = 0; n < N && n < inkling::kAttnResMaxSources; ++n) {
        a.src[n] = src[n];
        a.src_fp32[n] = src_fp32[n] != 0;
    }
    a.N = N;
    a.addend = addend;
    a.addend_fp32 = addend_fp32 != 0;
    a.query = query;
    a.query_scale = query_scale;
    a.norm_w = static_cast<const bf16*>(norm_w);
    a.eps = eps;
    a.norm_eps = eps;
    a.Q = Q;
    a.T = T;
    a.D = D;
    return a;
}
}  // namespace

INKLING_CAPI int inkling_capi_attn_res_fwd(void* out, float* probs, float* rstd, float* rstd_out,
                                           const void* const* src, const int* src_fp32, int N, const void* addend,
                                           int addend_fp32, const float* query, float query_scale, const void* norm_w,
                                           float eps, int Q, int T, int D, void* s) {
    return guard([&] {
        inkling::attn_res_fwd(out, probs, rstd, rstd_out,
                              res_args(src, src_fp32, N, addend, addend_fp32, query, query_scale, norm_w, eps, Q, T, D),
                              st(s));
    });
}

INKLING_CAPI int inkling_capi_attn_res_bwd(float* const* d_src, float* d_addend, const void* d_out, int d_out_fp32,
                                           const float* probs, const float* rstd, const float* rstd_out,
                                           const void* const* src, const int* src_fp32, int N, const void* addend,
                                           int addend_fp32, const float* query, float query_scale, const void* norm_w,
                                           float eps, int Q, int T, int D, void* s) {
    return guard([&] {
        inkling::attn_res_bwd(d_src, d_addend, d_out, d_out_fp32 != 0, probs, rstd, rstd_out,
                              res_args(src, src_fp32, N, addend, addend_fp32, query, query_scale, norm_w, eps, Q, T, D),
                              st(s));
    });
}

INKLING_CAPI int inkling_capi_loop_blend_pair_fwd(void* pair, const float* dest, long dest_stride_s, const float* msg,
                                                  int S, int T, int D, void* s) {
    return guard([&] { inkling::loop_blend_pair_fwd(static_cast<bf16*>(pair), dest, dest_stride_s, msg, S, T, D, st(s)); });
}

INKLING_CAPI int inkling_capi_loop_blend_out_fwd(void* out, float* rstd, const float* dest, long dest_stride_s,
                                                 const float* msg, const float* gf, const void* norm_w,
                                                 long norm_stride_s, float eps, int S, int T, int D, void* s) {
    return guard([&] {
        inkling::loop_blend_out_fwd(static_cast<bf16*>(out), rstd, dest, dest_stride_s, msg, gf,
                                    static_cast<const bf16*>(norm_w), norm_stride_s, eps, S, T, D, st(s));
    });
}

INKLING_CAPI int inkling_capi_loop_blend_out_bwd(float* d_dest_s, float* d_msg, float* d_gf, const void* d_out,
                                                 int d_out_fp32, const float* dest, long dest_stride_s,
                                                 const float* msg, const float* gf, const float* rstd,
                                                 const void* norm_w, long norm_stride_s, int S, int T, int D,
                                                 void* s) {
    return guard([&] {
        inkling::loop_blend_out_bwd(d_dest_s, d_msg, d_gf, d_out, d_out_fp32 != 0, dest, dest_stride_s, msg, gf, rstd,
                                    static_cast<const bf16*>(norm_w), norm_stride_s, S, T, D, st(s));
    });
}

INKLING_CAPI int inkling_capi_loop_blend_pair_bwd(float* d_dest_s, float* d_msg, const float* d_pair,
                                                  const float* dest, long dest_stride_s, const float* msg, int S,
                                                  int T, int D, void* s) {
    return guard([&] { inkling::loop_blend_pair_bwd(d_dest_s, d_msg, d_pair, dest, dest_stride_s, msg, S, T, D, st(s)); });
}

INKLING_CAPI int inkling_capi_loop_blend_reduce_dest(float* d_dest, const float* d_dest_s, int broadcast,
                                                     int accumulate, int S, int T, int D, void* s) {
    return guard(
        [&] { inkling::loop_blend_reduce_dest(d_dest, d_dest_s, broadcast != 0, accumulate != 0, S, T, D, st(s)); });
}

INKLING_CAPI int inkling_capi_softcap_swiglu_fwd(void* y, const void* gate_up, const float* weights,
                                                 const int* scatter, int num_assign, int I, float gate_cap,
                                                 float up_cap, void* s) {
    return guard([&] {
        inkling::softcap_swiglu_fwd(static_cast<bf16*>(y), static_cast<const bf16*>(gate_up), weights, scatter,
                                    num_assign, I, gate_cap, up_cap, st(s));
    });
}

INKLING_CAPI int inkling_capi_softcap_swiglu_bwd(void* d_gate_up, float* d_weights, const void* dy, int dy_fp32,
                                                 const void* gate_up, const float* weights, const int* scatter,
                                                 int num_assign, int I, float gate_cap, float up_cap, void* s) {
    return guard([&] {
        inkling::softcap_swiglu_bwd(static_cast<bf16*>(d_gate_up), d_weights, dy, dy_fp32 != 0,
                                    static_cast<const bf16*>(gate_up), weights, scatter, num_assign, I, gate_cap,
                                    up_cap, st(s));
    });
}

INKLING_CAPI int inkling_capi_moe_combine_fwd(void* out, const void* rows, const int* scatter, int T, int top_k, int D,
                                              void* s) {
    return guard([&] {
        inkling::moe_combine_fwd(static_cast<bf16*>(out), static_cast<const bf16*>(rows), scatter, T, top_k, D, st(s));
    });
}

INKLING_CAPI int inkling_capi_moe_combine_bwd(void* d_rows, const void* d_out, int d_out_fp32, const int* scatter,
                                              int T, int top_k, int D, void* s) {
    return guard([&] {
        inkling::moe_combine_bwd(static_cast<bf16*>(d_rows), d_out, d_out_fp32 != 0, scatter, T, top_k, D, st(s));
    });
}

INKLING_CAPI int inkling_capi_round_e4m3(void* y, const void* x, long n, void* s) {
    return guard([&] { inkling::round_e4m3(static_cast<bf16*>(y), static_cast<const bf16*>(x), n, st(s)); });
}

// ---- backward-only entry points of the engine-op IR (visit_backward.h) ----
INKLING_CAPI long inkling_capi_attention_visit_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq, int E) {
    return static_cast<long>(inkling::attention_visit_bwd_workspace_bytes(total_q, max_q, num_segs, Hq, E));
}

INKLING_CAPI int inkling_capi_attention_visit_bwd(int is_global, const void* q, const void* k, const void* v,
                                                  const int* cu_q, const int* cu_k, int num_segs, int max_q, int max_k,
                                                  int total_q, int total_k, int Hq, int Hkv, float scale, int window,
                                                  int rel_extent, const void* r, const void* rel_proj, const float* o,
                                                  const float* l, const void* gate, long gate_ld, const void* sink,
                                                  long sink_ld, const void* d_result, int d_result_fp32, float* dq,
                                                  float* dk, float* dv, float* d_r, float* d_sink, float* d_gate,
                                                  void* workspace, void* s) {
    return guard([&] {
        inkling::AttnVisitBwdArgs a;
        a.geo = part_args(q, k, v, cu_q, cu_k, nullptr, num_segs, max_q, max_k, total_q, Hq, Hkv, scale, window, nullptr,
                          rel_extent, nullptr, nullptr);
        a.total_k = total_k;
        a.is_global = is_global != 0;
        a.r = static_cast<const bf16*>(r);
        a.rel_proj = static_cast<const bf16*>(rel_proj);
        a.o = o;
        a.l = l;
        a.gate = static_cast<const bf16*>(gate);
        a.gate_ld = gate_ld;
        a.sink = static_cast<const bf16*>(sink);
        a.sink_ld = sink_ld;
        a.d_result = d_result;
        a.d_result_fp32 = d_result_fp32 != 0;
        a.dq = dq;
        a.dk = dk;
        a.dv = dv;
        a.d_r = d_r;
        a.d_sink = d_sink;
        a.d_gate = d_gate;
        a.workspace = workspace;
        inkling::attention_visit_bwd(a, st(s));
    });
}

INKLING_CAPI long inkling_capi_attention_prologue_bwd_workspace_bytes(int T, int Hq, int Hkv, int d_rel) {
    return static_cast<long>(inkling::attention_prologue_bwd_workspace_bytes(T, Hq, Hkv, d_rel));
}

/// ptrs (17, in order): positions, qside, q_scale, r_scale, tau, k_raw, v_raw, k_conv_w, v_conv_w, k_dyn, v_dyn,
/// k_carry, v_carry, d_k_carry_next, d_v_carry_next, (unused), (unused);
/// grads (6): dq, d_r, dk, dv, d_gate, d_sink; outs (8): d_qside, d_k_raw, d_v_raw, d_k_dyn, d_v_dyn, d_k_carry,
/// d_v_carry, workspace.
INKLING_CAPI int inkling_capi_attention_prologue_bwd(int T, int Hq, int Hkv, int d_rel, int is_global, long qside_ld,
                                                     long d_qside_ld, float q_clip, float r_clip, float k_clip,
                                                     float eps, void* const* ptrs, void* const* grads, void* const* outs,
                                                     void* s) {
    return guard([&] {
        inkling::AttnPrologueBwdArgs a;
        a.T = T;
        a.Hq = Hq;
        a.Hkv = Hkv;
        a.d_rel = d_rel;
        a.is_global = is_global != 0;
        a.qside_ld = qside_ld;
        a.d_qside_ld = d_qside_ld;
        a.q_clip = q_clip;
        a.r_clip = r_clip;
        a.k_clip = k_clip;
        a.eps = eps;
        a.positions = static_cast<const int*>(ptrs[0]);
        a.qside = static_cast<const bf16*>(ptrs[1]);
        a.q_scale = static_cast<const bf16*>(ptrs[2]);
        a.r_scale = static_cast<const bf16*>(ptrs[3]);
        a.tau = static_cast<const float*>(ptrs[4]);
        a.k_raw = static_cast<const bf16*>(ptrs[5]);
        a.v_raw = static_cast<const bf16*>(ptrs[6]);
        a.k_conv_w = static_cast<const bf16*>(ptrs[7]);
        a.v_conv_w = static_cast<const bf16*>(ptrs[8]);
        a.k_dyn = static_cast<const bf16*>(ptrs[9]);
        a.v_dyn = static_cast<const bf16*>(ptrs[10]);
        a.k_carry = static_cast<const bf16*>(ptrs[11]);
        a.v_carry = static_cast<const bf16*>(ptrs[12]);
        a.d_k_carry_next = static_cast<const float*>(ptrs[13]);
        a.d_v_carry_next = static_cast<const float*>(ptrs[14]);
        a.dq = static_cast<const float*>(grads[0]);
        a.d_r = static_cast<const float*>(grads[1]);
        a.dk = static_cast<const float*>(grads[2]);
        a.dv = static_cast<const float*>(grads[3]);
        a.d_gate = static_cast<const float*>(grads[4]);
        a.d_sink = static_cast<const float*>(grads[5]);
        a.d_qside = static_cast<float*>(outs[0]);
        a.d_k_raw = static_cast<float*>(outs[1]);
        a.d_v_raw = static_cast<float*>(outs[2]);
        a.d_k_dyn = static_cast<float*>(outs[3]);
        a.d_v_dyn = static_cast<float*>(outs[4]);
        a.d_k_carry = static_cast<float*>(outs[5]);
        a.d_v_carry = static_cast<float*>(outs[6]);
        a.workspace = outs[7];
        inkling::attention_prologue_bwd(a, st(s));
    });
}

INKLING_CAPI int inkling_capi_kv_plane_gather_bf16(void* dst, const void* plane, const int* block_table, long pos0,
                                                   long rows, int Hkv, void* s) {
    return guard([&] {
        inkling::kv_plane_gather_bf16(static_cast<bf16*>(dst), static_cast<const std::uint8_t*>(plane), block_table, pos0,
                                      rows, Hkv, st(s));
    });
}

// The routed experts' backward for one group of experts (moe_backward.h). ints: T, D, I, ng, rank, max_rows.
// ptrs: x, g, tok, gam, u13, u2, A13, B13, A2, B2 (the four LoRA pointers all null = no adapter), dx, dgam,
// dA13, dB13, dA2, dB2 (each null = not needed), workspace. seg [ng + 1], f13 [ng], f2 [ng]: host arrays.
INKLING_CAPI long inkling_capi_moe_group_bwd_workspace_bytes(int rows, int D, int I, int rank) {
    return static_cast<long>(inkling::moe_group_bwd_workspace_bytes(rows, D, I, rank));
}

INKLING_CAPI int inkling_capi_moe_group_bwd(const int* ints, void* const* ptrs, const int* seg, const float* f13,
                                            const float* f2, float scale, long workspace_bytes, void* s) {
    return guard([&] {
        inkling::MoeGroupBwdArgs a;
        a.T = ints[0];
        a.D = ints[1];
        a.I = ints[2];
        a.ng = ints[3];
        a.rank = ints[4];
        a.max_rows = ints[5];
        a.x = static_cast<const bf16*>(ptrs[0]);
        a.g = static_cast<const float*>(ptrs[1]);
        a.tok = static_cast<const int*>(ptrs[2]);
        a.gam = static_cast<const float*>(ptrs[3]);
        a.u13 = static_cast<const bf16*>(ptrs[4]);
        a.u2 = static_cast<const bf16*>(ptrs[5]);
        a.A13 = static_cast<const float*>(ptrs[6]);
        a.B13 = static_cast<const float*>(ptrs[7]);
        a.A2 = static_cast<const float*>(ptrs[8]);
        a.B2 = static_cast<const float*>(ptrs[9]);
        a.dx = static_cast<float*>(ptrs[10]);
        a.dgam = static_cast<float*>(ptrs[11]);
        a.dA13 = static_cast<float*>(ptrs[12]);
        a.dB13 = static_cast<float*>(ptrs[13]);
        a.dA2 = static_cast<float*>(ptrs[14]);
        a.dB2 = static_cast<float*>(ptrs[15]);
        a.workspace = ptrs[16];
        a.workspace_bytes = static_cast<std::size_t>(workspace_bytes);
        a.seg = seg;
        a.f13 = f13;
        a.f2 = f2;
        a.scale = scale;
        inkling::moe_group_bwd(a, st(s));
    });
}
