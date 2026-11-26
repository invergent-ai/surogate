// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels: forward + backward of the visit's ops. Semantics and the backward formulas: the fp64
// references tests/inkling_ref/inkling_kernels_ref.py (same function names, minus `_fwd` / `_bwd`).
// Layouts: token-major; T = tokens of the chunk; attention lse [H, T]; positions int32 = position within the document
// (a chunk's documents are told apart by positions alone in the convolutions). All gradients fp32 unless stated.
// No kernel here allocates; every reduction has a fixed order (bitwise reproducible).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace inkling {

using bf16 = nv_bfloat16;

// 1. tau = 1 + alpha * ln(max((pos + 1) / n_floor, 1)), fp32 [T].
void log_tau(float* tau, const int* positions, int T, float n_floor, float alpha, cudaStream_t stream);

// 2. q / r / k scaled L2 clip and v RMS ("value norm"); rows [T, H, d], d <= 1024.
enum class NormMode : int { Clip = 0, Rms = 1 };
struct QkvNormArgs {
    NormMode mode = NormMode::Clip;
    const bf16* scale = nullptr;  // [H, d] or null
    const float* tau = nullptr;   // [T] or null (global q / r)
    float threshold = 22.627416997969522f;
    float eps = 1e-6f;
    bool kv_fp8 = false;  // k / v: store the fp32 value rounded to e4m3 (unit scale), as the engine's FP8 KV
    int T = 0, H = 0, d = 0;
};
void qkv_norm_fwd(bf16* y, const bf16* x, const QkvNormArgs& a, cudaStream_t stream);
/// dx [T, H, d] fp32 (written); dy fp32 or bf16 (dy_fp32 selects).
void qkv_norm_bwd(float* dx, const void* dy, bool dy_fp32, const bf16* x, const QkvNormArgs& a, cudaStream_t stream);

// 3. static short conv, kernel 4 (tap 3 = current token), conv = (sum_k w[c,k] x[t-3+k] + x[t]) * scale -> bf16;
//    with addend: out = bf16((residual + addend) + bf16(conv)). carry [3, C] = the previous chunk's last 3 inputs
//    (oldest first) or null. Requires T >= 3 for the carries.
struct ConvArgs {
    const int* positions = nullptr;  // [T]
    const bf16* carry = nullptr;     // [3, C] or null
    float scale = 0.4472135954999579f;
    int T = 0, C = 0;
    int head_dim = 128;  // dynamic conv only
};
void short_conv_fwd(bf16* out, const bf16* x, const bf16* w, const bf16* addend, const bf16* residual,
                    const ConvArgs& a, cudaStream_t stream);
/// dx [T, C] (written) incl. d_carry_next [3, C] (next chunk's gradient on our last 3 inputs; may be null);
/// d_carry [3, C] (written; may be null). d_addend = d_residual = dy: the caller routes dy (no kernel).
void short_conv_bwd(float* dx, float* d_carry, const void* dy, bool dy_fp32, const bf16* w, const float* d_carry_next,
                    const ConvArgs& a, cudaStream_t stream);

// 4. dynamic conv (global K / V): tap(t,c,L) = static_w[L,c] + dyn[t, (c/hd)*4 + L], lag 0 first, scale 9^-0.5.
void dynamic_conv_fwd(bf16* out, const bf16* x, const bf16* static_w, const bf16* dyn, const ConvArgs& a,
                      cudaStream_t stream);
/// dx [T, C], d_dyn [T, H*4] (written), d_carry [3, C] (written; may be null).
void dynamic_conv_bwd(float* dx, float* d_dyn, float* d_carry, const void* dy, bool dy_fp32, const bf16* x,
                      const bf16* static_w, const bf16* dyn, const float* d_carry_next, const ConvArgs& a,
                      cudaStream_t stream);

// 7. merge (near [+ far]) + sink + gate. o_* fp32 [T, H, 128], l_* fp32 [H, T] (-inf = no keys), sink bf16 [T, H],
//    gate bf16 [T, H*128]. result bf16 = bf16((o * sig(gate)) * sig(l - sink)) (the engine's order and sigmoid);
//    out_s fp32 [T, H, 128] = o * exp(l - lse_s), lse_s [H, T] = logaddexp(l, sink) (either may be null).
void attn_merge_fwd(bf16* result, float* out_s, float* lse_s, const float* o_n, const float* l_n, const float* o_f,
                    const float* l_f, const bf16* sink, const bf16* gate, int T, int H, cudaStream_t stream);
/// Per-part (standard chain rule) backward, g1's 3-op graph: d_o_* fp32 [T, H, 128], d_l_* [H, T], d_sink [T, H],
/// d_gate [T, H*128] (all written; far outputs ignored when o_f is null). d_result bf16 or fp32.
void attn_merge_bwd(float* d_o_n, float* d_l_n, float* d_o_f, float* d_l_f, float* d_sink, float* d_gate,
                    const void* d_result, bool d_result_fp32, const float* o_n, const float* l_n, const float* o_f,
                    const float* l_f, const bf16* sink, const bf16* gate, int T, int H, cudaStream_t stream);
/// Union-form backward (the composite `inkling_attention` op): dout bf16 [T, H, 128] = bf16(d_result * sig(gate)),
/// delta [H, T] = rowsum(dout * out_s), d_sink [T, H], d_gate [T, H*128].
void attn_merge_bwd_union(bf16* dout, float* delta, float* d_sink, float* d_gate, const void* d_result,
                          bool d_result_fp32, const float* out_s, const float* lse_s, const bf16* sink,
                          const bf16* gate, int T, int H, cudaStream_t stream);
/// Per-part attention backward operands: delta [H, T] = rowsum(d_o * o) - d_l; dout bf16 = bf16(d_o).
void attn_part_bwd_prep(bf16* dout, float* delta, const float* d_o, const float* o, const float* d_l, int T, int H,
                        cudaStream_t stream);

// 8. block AttnRes. Sources n < N: bf16 or fp32 [T, D]; the last may get an addend (fp32 sum). Queries RAW fp32
//    [Q, D] times query_scale (fp32, one rounding as vLLM's load). Output [Q, T, D]: bf16 with norm_w [Q, D], else fp32.
constexpr int kAttnResMaxSources = 8;
constexpr int kAttnResMaxQueries = 4;
struct AttnResArgs {
    const void* src[kAttnResMaxSources] = {};
    bool src_fp32[kAttnResMaxSources] = {};
    int N = 0;
    const void* addend = nullptr;  // added to source N-1
    bool addend_fp32 = false;
    const float* query = nullptr;  // [Q, D] raw
    float query_scale = 1.0f;
    const bf16* norm_w = nullptr;  // [Q, D] or null
    float eps = 1e-6f, norm_eps = 1e-6f;
    int Q = 1, T = 0, D = 0;
};
/// saved: probs [Q, N, T] (softmax), rstd [N, T], rstd_out [Q, T] (norm only).
void attn_res_fwd(void* out, float* probs, float* rstd, float* rstd_out, const AttnResArgs& a, cudaStream_t stream);
/// d_src[n] fp32 [T, D] ACCUMULATED (+=; null = not needed); d_addend fp32 accumulated (= d_src[N-1]'s contribution).
void attn_res_bwd(float* const* d_src, float* d_addend, const void* d_out, bool d_out_fp32, const float* probs,
                  const float* rstd, const float* rstd_out, const AttnResArgs& a, cudaStream_t stream);

// 9. loop blend, elementwise parts (the two GEMMs per stream run in the op wrapper with cuBLAS).
//    dest fp32 [S|1, T, D] (dest_stride_s = 0 when it broadcasts), msg fp32 [S, T, D].
void loop_blend_pair_fwd(bf16* pair, const float* dest, long dest_stride_s, const float* msg, int S, int T, int D,
                         cudaStream_t stream);
/// gf fp32 [S, T, 2D] (gain rows then forget rows); norm_w [S|1, D] (norm_stride_s = 0 when broadcast).
void loop_blend_out_fwd(bf16* out, float* rstd, const float* dest, long dest_stride_s, const float* msg,
                        const float* gf, const bf16* norm_w, long norm_stride_s, float eps, int S, int T, int D,
                        cudaStream_t stream);
/// From d_out: d_dest_s fp32 [S, T, D] (per stream, written), d_msg fp32 [S, T, D] (written), d_gf fp32 [S, T, 2D].
void loop_blend_out_bwd(float* d_dest_s, float* d_msg, float* d_gf, const void* d_out, bool d_out_fp32,
                        const float* dest, long dest_stride_s, const float* msg, const float* gf, const float* rstd,
                        const bf16* norm_w, long norm_stride_s, int S, int T, int D, cudaStream_t stream);
/// d_pair fp32 [S, T, 2D] -> adds the pair-norm backward into d_dest_s and d_msg.
void loop_blend_pair_bwd(float* d_dest_s, float* d_msg, const float* d_pair, const float* dest, long dest_stride_s,
                         const float* msg, int S, int T, int D, cudaStream_t stream);
/// d_dest[t, j] (+)= sum_s d_dest_s[s, t, j] in stream order (broadcast dest) or per stream (S_dest == S).
void loop_blend_reduce_dest(float* d_dest, const float* d_dest_s, bool broadcast, bool accumulate, int S, int T,
                            int D, cudaStream_t stream);

// 10. softcapped SwiGLU, routing weight before down, rows expert-sorted: assignment a = t*k + s has row scatter[a]
//     and gamma weights[a]. gate_up bf16 [R, 2I] interleaved (2j gate, 2j+1 up) -> y bf16 [R, I].
void softcap_swiglu_fwd(bf16* y, const bf16* gate_up, const float* weights, const int* scatter, int num_assign, int I,
                        float gate_cap, float up_cap, cudaStream_t stream);
/// d_gate_up bf16 [R, 2I] (written), d_weights fp32 [T*k] (written; null = not needed); dy bf16 or fp32.
void softcap_swiglu_bwd(bf16* d_gate_up, float* d_weights, const void* dy, bool dy_fp32, const bf16* gate_up,
                        const float* weights, const int* scatter, int num_assign, int I, float gate_cap, float up_cap,
                        cudaStream_t stream);

// 11. MoE combine with unit weights: out[t] = bf16(sum_s rows[scatter[t*k + s]]) (fp32, slot order).
void moe_combine_fwd(bf16* out, const bf16* rows, const int* scatter, int T, int top_k, int D, cudaStream_t stream);
/// d_rows[scatter[t*k + s]] = bf16(d_out[t]).
void moe_combine_bwd(bf16* d_rows, const void* d_out, bool d_out_fp32, const int* scatter, int T, int top_k, int D,
                     cudaStream_t stream);

// 12. glue of the attention op wrapper (attention_glue.cu).
/// y = e4m3(x) widened back to bf16 (unit scale, RNE, satfinite: the engine's FP8 KV store), n elements.
void round_e4m3(bf16* y, const bf16* x, long n, cudaStream_t stream);
/// cu[i] = i * step for i = 0 .. num_segs (uniform rows as segments).
void fill_uniform_cu(int* cu, int num_segs, int step, cudaStream_t stream);
/// Key offsets of the window-bounded key buffer [tail | chunk] whose row 0 is global row `base`, from the executor's
/// window-relative cu_k (global row = cu_k + win_start): cu_near[0] = max(cu_k[0] + win_start, base) - base,
/// cu_near[s] = cu_k[s] + win_start - base for s >= 1 (only the first segment can reach before the tail).
void near_cu_from_window(int* cu_near, const int* cu_k, int num_segs, int win_start, int base, cudaStream_t stream);

}  // namespace inkling
