// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer, lane g2: backward-only entry points of the engine-op IR (see visit_backward.h). Composes the
// kernels of inkling.h and the FA2-fork backward (flash_rel/); references: lanes/g2/ref/inkling_kernels_ref.py
// (merge_bwd, attn_part_bwd, qkv_norm_bwd, short_conv_bwd, dynamic_conv_bwd).

#include "kernels/inkling/visit_backward.h"

#include <stdexcept>
#include <string>

#include "kernels/inkling/inkling.h"
#include "kernels/inkling/inkling_common.cuh"

namespace inkling {
namespace {

constexpr int kHd = 128;

inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling visit bwd ") + what + ": " + cudaGetErrorString(e));
}

inline size_t align256(size_t b) {
    return (b + 255) / 256 * 256;
}

inline int grid_for(long n, int threads) {
    const long b = (n + threads - 1) / threads;
    return static_cast<int>(b < (1L << 30) ? b : (1L << 30));
}

__global__ void kv_plane_gather_kernel(bf16* dst, const std::uint8_t* plane, const int* block_table, long pos0, long rows,
                                       int Hkv) {
    const long n = rows * Hkv * kHd;
    for (long e = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; e < n; e += static_cast<long>(gridDim.x) * blockDim.x) {
        const long row = e / (static_cast<long>(Hkv) * kHd);
        const int h = static_cast<int>((e / kHd) % Hkv), d = static_cast<int>(e % kHd);
        const long pos = pos0 + row;
        const long page = block_table[pos >> 6];
        const long off = 128L * 64L * (h + static_cast<long>(Hkv) * page) + 128L * (pos & 63) + d;
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(static_cast<__nv_fp8_storage_t>(plane[off]), __NV_E4M3);
        dst[e] = __float2bfloat16_rn(__half2float(__half(hr)));  // exact: e4m3 values are bf16 values
    }
}

// rel[i, e] = bf16(sum_k r[i, k] proj[k, e]), k = 0 .. 15 in order (fmaf chain).
__global__ void rel_bias_fwd_kernel(bf16* rel, const bf16* r, const bf16* proj, long rowsH, int E) {
    const long n = rowsH * E;
    for (long o = blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x; o < n; o += static_cast<long>(gridDim.x) * blockDim.x) {
        const long i = o / E;
        const int e = static_cast<int>(o % E);
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < 16; ++k) acc = fmaf(__bfloat162float(r[i * 16 + k]), __bfloat162float(proj[static_cast<long>(k) * E + e]), acc);
        rel[o] = __float2bfloat16_rn(acc);
    }
}

// d_r[i, k] = sum_e d_rel[i, e] proj[k, e]: warp per row, lane-strided partial sums, fixed shuffle tree.
__global__ void rel_bias_bwd_kernel(float* d_r, const float* d_rel, const bf16* proj, long rowsH, int E) {
    const long i = (blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (i >= rowsH) return;
    float acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0.f;
    for (int e = lane; e < E; e += 32) {
        const float g = d_rel[i * E + e];
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = fmaf(g, __bfloat162float(proj[static_cast<long>(k) * E + e]), acc[k]);
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) {
        const float v = warp_sum(acc[k]);
        if (lane == 0) d_r[i * 16 + k] = v;
    }
}

template <typename T>
void copy_cols(T* dst, long dst_ld, const T* src, long src_ld, long cols, long rows, cudaStream_t s, const char* what) {
    if (rows == 0 || cols == 0) return;
    check(cudaMemcpy2DAsync(dst, sizeof(T) * dst_ld, src, sizeof(T) * src_ld, sizeof(T) * cols, rows,
                            cudaMemcpyDeviceToDevice, s),
          what);
}

}  // namespace

void kv_plane_gather_bf16(bf16* dst, const std::uint8_t* plane, const int* block_table, long pos0, long rows, int Hkv,
                          cudaStream_t stream) {
    const long n = rows * Hkv * kHd;
    if (n <= 0) return;
    kv_plane_gather_kernel<<<grid_for(n, 256), 256, 0, stream>>>(dst, plane, block_table, pos0, rows, Hkv);
    check(cudaGetLastError(), "kv_plane_gather");
}

void rel_bias_fwd(bf16* rel, const bf16* r, const bf16* rel_proj, long rowsH, int E, cudaStream_t stream) {
    const long n = rowsH * E;
    if (n <= 0) return;
    rel_bias_fwd_kernel<<<grid_for(n, 256), 256, 0, stream>>>(rel, r, rel_proj, rowsH, E);
    check(cudaGetLastError(), "rel_bias_fwd");
}

void rel_bias_bwd(float* d_r, const float* d_rel, const bf16* rel_proj, long rowsH, int E, cudaStream_t stream) {
    if (rowsH <= 0) return;
    rel_bias_bwd_kernel<<<grid_for(rowsH * 32, 256), 256, 0, stream>>>(d_r, d_rel, rel_proj, rowsH, E);
    check(cudaGetLastError(), "rel_bias_bwd");
}

// ---------------------------------------------------------------------------------------------------------------
// attention: union backward from the engine's (o, l)
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct VisitWs {
    bf16 *rel, *y, *gate, *sink, *dout;
    float *d_rel, *out_s, *lse_s, *delta;
    void* part;
};
VisitWs carve_visit(void* base, long T, int max_q, int segs, int H, int E) {
    char* p = static_cast<char*>(base);
    VisitWs w{};
    auto take = [&](size_t bytes) {
        char* r = p;
        p += align256(bytes);
        return static_cast<void*>(r);
    };
    const long TH = T * H, THD = T * H * kHd;
    w.rel = static_cast<bf16*>(take(sizeof(bf16) * TH * E));
    w.d_rel = static_cast<float*>(take(sizeof(float) * TH * E));
    w.out_s = static_cast<float*>(take(sizeof(float) * THD));
    w.lse_s = static_cast<float*>(take(sizeof(float) * TH));
    w.y = static_cast<bf16*>(take(sizeof(bf16) * THD));
    w.gate = static_cast<bf16*>(take(sizeof(bf16) * THD));
    w.sink = static_cast<bf16*>(take(sizeof(bf16) * TH));
    w.dout = static_cast<bf16*>(take(sizeof(bf16) * THD));
    w.delta = static_cast<float*>(take(sizeof(float) * TH));
    w.part = take(attention_bwd_workspace_bytes(static_cast<int>(T), max_q, segs, H));
    return w;
}
}  // namespace

size_t attention_visit_bwd_workspace_bytes(int total_q, int max_q, int num_segs, int Hq, int E) {
    const size_t TH = static_cast<size_t>(total_q) * Hq, THD = TH * kHd;
    return align256(sizeof(bf16) * TH * E) + align256(sizeof(float) * TH * E) + align256(sizeof(float) * THD) +
           align256(sizeof(float) * TH) + 3 * align256(sizeof(bf16) * THD) + align256(sizeof(bf16) * TH) +
           align256(sizeof(float) * TH) + align256(attention_bwd_workspace_bytes(total_q, max_q, num_segs, Hq));
}

void attention_visit_bwd(const AttnVisitBwdArgs& a, cudaStream_t s) {
    const AttnPartArgs& g = a.geo;
    const long T = g.total_q;
    const int H = g.Hq, E = g.rel_extent;
    if (T == 0) return;
    if (!a.r || !a.rel_proj || !a.o || !a.l || !a.gate || !a.sink || !a.d_result || !a.dq || !a.dk || !a.dv ||
        !a.d_r || !a.d_sink || !a.d_gate || !a.workspace)
        throw std::invalid_argument("inkling attention_visit_bwd: null argument");
    VisitWs w = carve_visit(a.workspace, T, g.max_q, g.num_segs, H, E);
    // gate / sink as dense rows (the engine leaves them as column views of the projection output)
    const bf16* gate = a.gate;
    const bf16* sink = a.sink;
    if (a.gate_ld && a.gate_ld != static_cast<long>(H) * kHd) {
        copy_cols(w.gate, static_cast<long>(H) * kHd, a.gate, a.gate_ld, static_cast<long>(H) * kHd, T, s, "gate copy");
        gate = w.gate;
    }
    if (a.sink_ld && a.sink_ld != H) {
        copy_cols(w.sink, static_cast<long>(H), a.sink, a.sink_ld, static_cast<long>(H), T, s, "sink copy");
        sink = w.sink;
    }
    rel_bias_fwd(w.rel, a.r, a.rel_proj, T * H, E, s);
    // lse_s = logaddexp(l, sink), out_s = o e^(l - lse_s) (the merge kernel with one part; its result is not used)
    attn_merge_fwd(w.y, w.out_s, w.lse_s, a.o, a.l, nullptr, nullptr, sink, gate, static_cast<int>(T), H, s);
    attn_merge_bwd_union(w.dout, w.delta, a.d_sink, a.d_gate, a.d_result, a.d_result_fp32, w.out_s, w.lse_s, sink, gate,
                         static_cast<int>(T), H, s);
    check(cudaMemsetAsync(a.dq, 0, sizeof(float) * T * H * kHd, s), "memset dq");
    AttnPartBwdArgs b;
    b.fwd = g;
    b.fwd.rel = w.rel;
    b.total_k = a.total_k;
    b.dout = w.dout;
    b.lse = w.lse_s;
    b.delta = w.delta;
    b.dq = a.dq;
    b.dk = a.dk;
    b.dv = a.dv;
    b.d_rel = w.d_rel;
    b.workspace = w.part;
    attention_near_bwd(b, s);
    if (a.is_global) {
        AttnPartBwdArgs f = b;
        f.d_rel = nullptr;
        attention_far_bwd(f, s);
    }
    rel_bias_bwd(a.d_r, w.d_rel, a.rel_proj, T * H, E, s);
}

// ---------------------------------------------------------------------------------------------------------------
// prologue: q / r scale + clip (+ tau), K / V convs + clip / RMS (e4m3 store straight-through)
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct ProWs {
    bf16 *q_raw, *r_raw, *kc, *vc;
    float *d_q_raw, *d_r_raw, *d_kc, *d_vc;
};
ProWs carve_pro(void* base, long T, int Hq, int Hkv, int dr) {
    char* p = static_cast<char*>(base);
    ProWs w{};
    auto take = [&](size_t bytes) {
        char* r = p;
        p += align256(bytes);
        return static_cast<void*>(r);
    };
    const long CQ = static_cast<long>(Hq) * kHd, CR = static_cast<long>(Hq) * dr, CK = static_cast<long>(Hkv) * kHd;
    w.q_raw = static_cast<bf16*>(take(sizeof(bf16) * T * CQ));
    w.r_raw = static_cast<bf16*>(take(sizeof(bf16) * T * CR));
    w.kc = static_cast<bf16*>(take(sizeof(bf16) * T * CK));
    w.vc = static_cast<bf16*>(take(sizeof(bf16) * T * CK));
    w.d_q_raw = static_cast<float*>(take(sizeof(float) * T * CQ));
    w.d_r_raw = static_cast<float*>(take(sizeof(float) * T * CR));
    w.d_kc = static_cast<float*>(take(sizeof(float) * T * CK));
    w.d_vc = static_cast<float*>(take(sizeof(float) * T * CK));
    return w;
}
}  // namespace

size_t attention_prologue_bwd_workspace_bytes(int T, int Hq, int Hkv, int d_rel) {
    const size_t CQ = static_cast<size_t>(Hq) * kHd, CR = static_cast<size_t>(Hq) * d_rel, CK = static_cast<size_t>(Hkv) * kHd;
    return align256(sizeof(bf16) * T * CQ) + align256(sizeof(bf16) * T * CR) + 2 * align256(sizeof(bf16) * T * CK) +
           align256(sizeof(float) * T * CQ) + align256(sizeof(float) * T * CR) + 2 * align256(sizeof(float) * T * CK);
}

void attention_prologue_bwd(const AttnPrologueBwdArgs& a, cudaStream_t s) {
    const long T = a.T;
    if (T == 0) return;
    if (!a.positions || !a.qside || !a.q_scale || !a.r_scale || !a.k_raw || !a.v_raw || !a.dq || !a.d_r || !a.dk ||
        !a.dv || !a.d_gate || !a.d_sink || !a.d_qside || !a.d_k_raw || !a.d_v_raw || !a.workspace)
        throw std::invalid_argument("inkling attention_prologue_bwd: null argument");
    if (a.is_global ? (!a.k_dyn || !a.v_dyn || !a.k_conv_w || !a.v_conv_w || !a.d_k_dyn || !a.d_v_dyn || !a.tau)
                    : (!a.k_conv_w || !a.v_conv_w))
        throw std::invalid_argument("inkling attention_prologue_bwd: conv weights / dynamic taps / tau");
    const int Hq = a.Hq, Hkv = a.Hkv, dr = a.d_rel;
    const long CQ = static_cast<long>(Hq) * kHd, CR = static_cast<long>(Hq) * dr, CK = static_cast<long>(Hkv) * kHd;
    const long qld = a.qside_ld ? a.qside_ld : 2 * CQ + CR + Hq;
    const long dld = a.d_qside_ld ? a.d_qside_ld : 2 * CQ + CR + Hq;
    ProWs w = carve_pro(a.workspace, T, Hq, Hkv, dr);

    // q / r: dx = clip-bwd(dy * tau) * scale, recomputed from the raw projection columns
    copy_cols(w.q_raw, CQ, a.qside, qld, CQ, T, s, "q_raw copy");
    copy_cols(w.r_raw, CR, a.qside + CQ, qld, CR, T, s, "r_raw copy");
    QkvNormArgs qa;
    qa.mode = NormMode::Clip;
    qa.scale = a.q_scale;
    qa.tau = a.is_global ? a.tau : nullptr;
    qa.threshold = a.q_clip;
    qa.eps = a.eps;
    qa.T = static_cast<int>(T);
    qa.H = Hq;
    qa.d = kHd;
    qkv_norm_bwd(w.d_q_raw, a.dq, true, w.q_raw, qa, s);
    QkvNormArgs ra = qa;
    ra.scale = a.r_scale;
    ra.threshold = a.r_clip;
    ra.d = dr;
    qkv_norm_bwd(w.d_r_raw, a.d_r, true, w.r_raw, ra, s);

    // K / V: recompute the conv outputs (bf16, as the engine's prologue rounds them), then norm-bwd, then conv-bwd
    ConvArgs ca;
    ca.positions = a.positions;
    ca.T = static_cast<int>(T);
    ca.C = static_cast<int>(CK);
    ca.head_dim = kHd;
    ca.scale = a.is_global ? 0.3333333333333333f : 0.4472135954999579f;
    const bf16* carries[2] = {a.k_carry, a.v_carry};
    const bf16* raws[2] = {a.k_raw, a.v_raw};
    const bf16* ws_[2] = {a.k_conv_w, a.v_conv_w};
    const bf16* dyns[2] = {a.k_dyn, a.v_dyn};
    bf16* conv_out[2] = {w.kc, w.vc};
    float* d_conv[2] = {w.d_kc, w.d_vc};
    const float* d_in[2] = {a.dk, a.dv};
    const float* d_next[2] = {a.d_k_carry_next, a.d_v_carry_next};
    float* d_raw[2] = {a.d_k_raw, a.d_v_raw};
    float* d_dyn[2] = {a.d_k_dyn, a.d_v_dyn};
    float* d_carry[2] = {a.d_k_carry, a.d_v_carry};
    for (int i = 0; i < 2; ++i) {
        ca.carry = carries[i];
        if (a.is_global)
            dynamic_conv_fwd(conv_out[i], raws[i], ws_[i], dyns[i], ca, s);
        else
            short_conv_fwd(conv_out[i], raws[i], ws_[i], nullptr, nullptr, ca, s);
        QkvNormArgs na;
        na.mode = i == 0 ? NormMode::Clip : NormMode::Rms;
        na.threshold = a.k_clip;
        na.eps = a.eps;
        na.T = static_cast<int>(T);
        na.H = Hkv;
        na.d = kHd;
        qkv_norm_bwd(d_conv[i], d_in[i], true, conv_out[i], na, s);  // the e4m3 store is straight-through
        if (a.is_global)
            dynamic_conv_bwd(d_raw[i], d_dyn[i], d_carry[i], d_conv[i], true, raws[i], ws_[i], dyns[i], d_next[i], ca, s);
        else
            short_conv_bwd(d_raw[i], d_carry[i], d_conv[i], true, ws_[i], d_next[i], ca, s);
    }

    // d_qside = [d_q_raw | d_r_raw | d_gate | d_sink]
    copy_cols(a.d_qside, dld, w.d_q_raw, CQ, CQ, T, s, "d_q copy");
    copy_cols(a.d_qside + CQ, dld, w.d_r_raw, CR, CR, T, s, "d_r copy");
    copy_cols(a.d_qside + CQ + CR, dld, a.d_gate, CQ, CQ, T, s, "d_gate copy");
    copy_cols(a.d_qside + 2 * CQ + CR, dld, a.d_sink, static_cast<long>(Hq), static_cast<long>(Hq), T, s, "d_sink copy");
}

}  // namespace inkling
