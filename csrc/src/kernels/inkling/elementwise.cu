// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels (lane g2): row / elementwise ops -- tau, q/r/k/v norms, short convs, attention merge,
// softcapped SwiGLU, MoE combine. Forward numerics: the engine's (inkling_common.cuh); references:
// lanes/g2/ref/inkling_kernels_ref.py.

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/inkling/inkling.h"
#include "kernels/inkling/inkling_common.cuh"

namespace inkling {
namespace {

constexpr int kThreads = 256;

inline int blocks_for(long n, int threads = kThreads) {
    return static_cast<int>((n + threads - 1) / threads);
}

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling ") + what + ": " + cudaGetErrorString(e));
}

template <bool kFp32>
__device__ __forceinline__ float ldv(const void* p, long i) {
    if constexpr (kFp32)
        return static_cast<const float*>(p)[i];
    else
        return __bfloat162float(static_cast<const bf16*>(p)[i]);
}

__device__ __forceinline__ float logaddexp(float a, float b) {
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    const float m = fmaxf(a, b);
    return m + log1pf(expf(-fabsf(a - b)));
}

// ---------------------------------------------------------------------------------------------------------------- 1
__global__ void log_tau_kernel(float* tau, const int* pos, int T, float inv_floor, float alpha) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    // `ops/attention.py` _log_scaling_tau_kernel (enable_fp_fusion=False): mul, max, log, mul, add.
    const float ratio = fmaxf(__fmul_rn(static_cast<float>(pos[t] + 1), inv_floor), 1.0f);
    tau[t] = __fadd_rn(1.0f, __fmul_rn(alpha, logf(ratio)));
}

// ---------------------------------------------------------------------------------------------------------------- 2
// One warp per (token, head) row.
__global__ void qkv_norm_fwd_kernel(bf16* y, const bf16* x, QkvNormArgs a) {
    const long row = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= static_cast<long>(a.T) * a.H) return;
    const int h = static_cast<int>(row % a.H);
    const long t = row / a.H;
    const bf16* xr = x + row * a.d;
    float ss = 0.f;
    for (int j = lane; j < a.d; j += 32) {
        float v = __bfloat162float(xr[j]);
        if (a.scale) v = __fmul_rn(v, __bfloat162float(a.scale[h * a.d + j]));
        ss = __fadd_rn(ss, __fmul_rn(v, v));
    }
    ss = warp_sum(ss);
    float f;
    if (a.mode == NormMode::Clip) {
        const float n = ss > 0.f ? __fmul_rn(ss, rsqrt_approx(ss)) : 0.f;
        f = fminf(__fdiv_rn(a.threshold, __fadd_rn(n, a.eps)), 1.0f);
    } else {
        f = rsqrt_approx(__fadd_rn(__fdiv_rn(ss, static_cast<float>(a.d)), a.eps));
    }
    const float tau = a.tau ? a.tau[t] : 1.0f;
    for (int j = lane; j < a.d; j += 32) {
        float v = __bfloat162float(xr[j]);
        if (a.scale) v = __fmul_rn(v, __bfloat162float(a.scale[h * a.d + j]));
        float o = __fmul_rn(v, f);
        if (a.tau) o = __fmul_rn(bf16_round(o), tau);
        if (a.kv_fp8) o = e4m3_round(o);
        y[row * a.d + j] = __float2bfloat16_rn(o);
    }
}

template <bool kDyFp32>
__global__ void qkv_norm_bwd_kernel(float* dx, const void* dy, const bf16* x, QkvNormArgs a) {
    const long row = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= static_cast<long>(a.T) * a.H) return;
    const int h = static_cast<int>(row % a.H);
    const long t = row / a.H;
    const float tau = a.tau ? a.tau[t] : 1.0f;
    float ss = 0.f, dot = 0.f;
    for (int j = lane; j < a.d; j += 32) {
        float v = __bfloat162float(x[row * a.d + j]);
        if (a.scale) v *= __bfloat162float(a.scale[h * a.d + j]);
        const float g = ldv<kDyFp32>(dy, row * a.d + j) * tau;
        ss += v * v;
        dot += v * g;
    }
    ss = warp_sum(ss);
    dot = warp_sum(dot);
    float c1, c2;  // dxs = c1 * g - c2 * xs
    if (a.mode == NormMode::Clip) {
        const float n = ss > 0.f ? ss * rsqrt_approx(ss) : 0.f;
        const float c = a.threshold / (n + a.eps);
        if (c < 1.0f) {
            c1 = c;
            c2 = c * dot / (n * (n + a.eps));
        } else {
            c1 = 1.0f;
            c2 = 0.0f;
        }
    } else {
        const float r = rsqrt_approx(ss / static_cast<float>(a.d) + a.eps);
        c1 = r;
        c2 = r * r * r / static_cast<float>(a.d) * dot;
    }
    for (int j = lane; j < a.d; j += 32) {
        const float s = a.scale ? __bfloat162float(a.scale[h * a.d + j]) : 1.0f;
        const float v = __bfloat162float(x[row * a.d + j]) * s;
        const float g = ldv<kDyFp32>(dy, row * a.d + j) * tau;
        dx[row * a.d + j] = (c1 * g - c2 * v) * s;
    }
}

// ---------------------------------------------------------------------------------------------------------- 3 / 4
// Input `lag` steps before token t of channel c: x, the carry (previous chunk), or 0 before the document start.
__device__ __forceinline__ float conv_tap(const bf16* x, const bf16* carry, const int* pos, int t, int c, int C,
                                          int lag) {
    if (pos[t] < lag) return 0.f;
    const int s = t - lag;
    if (s >= 0) return __bfloat162float(x[static_cast<long>(s) * C + c]);
    return carry ? __bfloat162float(carry[static_cast<long>(3 + s) * C + c]) : 0.f;
}

__global__ void short_conv_fwd_kernel(bf16* out, const bf16* x, const bf16* w, const bf16* addend,
                                      const bf16* residual, ConvArgs a) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long>(a.T) * a.C) return;
    const int t = static_cast<int>(i / a.C), c = static_cast<int>(i % a.C);
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k)  // oldest tap first, fma chain (`_tap_sum` static)
        acc = fmaf(conv_tap(x, a.carry, a.positions, t, c, a.C, 3 - k), __bfloat162float(w[c * 4 + k]), acc);
    const float conv = bf16_round(__fmul_rn(__fadd_rn(acc, __bfloat162float(x[i])), a.scale));
    float o = conv;
    if (addend) {  // `(res + addend) + y` in fp32 with y already rounded (`_sconv_kernel` HAS_ADDEND)
        float base = __bfloat162float(addend[i]);
        if (residual) base = __fadd_rn(__bfloat162float(residual[i]), base);
        o = __fadd_rn(base, conv);
    }
    out[i] = __float2bfloat16_rn(o);
}

template <bool kDyFp32>
__global__ void short_conv_bwd_kernel(float* dx, const void* dy, const bf16* w, const float* d_carry_next,
                                      ConvArgs a) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long>(a.T) * a.C) return;
    const int t = static_cast<int>(i / a.C), c = static_cast<int>(i % a.C);
    float acc = ldv<kDyFp32>(dy, i) * (__bfloat162float(w[c * 4 + 3]) + 1.0f);  // lag 0 tap + the conv's residual
#pragma unroll
    for (int lag = 1; lag < 4; ++lag) {
        const int u = t + lag;
        if (u < a.T && a.positions[u] >= lag)
            acc += __bfloat162float(w[c * 4 + 3 - lag]) * ldv<kDyFp32>(dy, static_cast<long>(u) * a.C + c);
    }
    acc *= a.scale;
    if (d_carry_next && t >= a.T - 3) acc += d_carry_next[static_cast<long>(t - (a.T - 3)) * a.C + c];
    dx[i] = acc;
}

// d_carry[r, c] (carry row r = input 3 - r steps before the chunk): tokens t <= r use it at lag t + 3 - r.
template <bool kDyFp32, bool kDynamic>
__global__ void conv_d_carry_kernel(float* d_carry, const void* dy, const bf16* w, const bf16* static_w,
                                    const bf16* dyn, ConvArgs a) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 3L * a.C) return;
    const int r = static_cast<int>(i / a.C), c = static_cast<int>(i % a.C);
    float acc = 0.f;
    for (int t = 0; t <= r && t < a.T; ++t) {
        const int lag = t + 3 - r;
        if (a.positions[t] < lag) continue;
        float tap;
        if constexpr (kDynamic) {
            const int heads = a.C / a.head_dim;
            tap = __bfloat162float(static_w[static_cast<long>(lag) * a.C + c]) +
                  __bfloat162float(dyn[static_cast<long>(t) * heads * 4 + (c / a.head_dim) * 4 + lag]);
        } else {
            tap = __bfloat162float(w[c * 4 + 3 - lag]);
        }
        acc += tap * ldv<kDyFp32>(dy, static_cast<long>(t) * a.C + c);
    }
    d_carry[i] = acc * a.scale;
}

__device__ __forceinline__ float dyn_tap(const bf16* static_w, const bf16* dyn, int t, int c, int C, int hd, int lag) {
    const int heads = C / hd;
    // fp32 add of the two bf16 values (`_tap_sum`: w[None, :] + dyn)
    return __fadd_rn(__bfloat162float(static_w[static_cast<long>(lag) * C + c]),
                     __bfloat162float(dyn[static_cast<long>(t) * heads * 4 + (c / hd) * 4 + lag]));
}

__global__ void dynamic_conv_fwd_kernel(bf16* out, const bf16* x, const bf16* static_w, const bf16* dyn,
                                        ConvArgs a) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long>(a.T) * a.C) return;
    const int t = static_cast<int>(i / a.C), c = static_cast<int>(i % a.C);
    const float xt = __bfloat162float(x[i]);
    float acc = __fmul_rn(xt, dyn_tap(static_w, dyn, t, c, a.C, a.head_dim, 0));  // lag 0: plain product
#pragma unroll
    for (int lag = 1; lag < 4; ++lag)
        acc = fmaf(conv_tap(x, a.carry, a.positions, t, c, a.C, lag), dyn_tap(static_w, dyn, t, c, a.C, a.head_dim, lag),
                   acc);
    out[i] = __float2bfloat16_rn(__fmul_rn(__fadd_rn(acc, xt), a.scale));
}

template <bool kDyFp32>
__global__ void dynamic_conv_dx_kernel(float* dx, const void* dy, const bf16* static_w, const bf16* dyn,
                                       const float* d_carry_next, ConvArgs a) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long>(a.T) * a.C) return;
    const int t = static_cast<int>(i / a.C), c = static_cast<int>(i % a.C);
    float acc = ldv<kDyFp32>(dy, i) * (dyn_tap(static_w, dyn, t, c, a.C, a.head_dim, 0) + 1.0f);
#pragma unroll
    for (int lag = 1; lag < 4; ++lag) {
        const int u = t + lag;
        if (u < a.T && a.positions[u] >= lag)
            acc += dyn_tap(static_w, dyn, u, c, a.C, a.head_dim, lag) * ldv<kDyFp32>(dy, static_cast<long>(u) * a.C + c);
    }
    acc *= a.scale;
    if (d_carry_next && t >= a.T - 3) acc += d_carry_next[static_cast<long>(t - (a.T - 3)) * a.C + c];
    dx[i] = acc;
}

// d_dyn[t, h, lag] = scale * sum_{c in head h} tap_input(t, c, lag) * dy[t, c]; one warp per (token, head).
template <bool kDyFp32>
__global__ void dynamic_conv_ddyn_kernel(float* d_dyn, const void* dy, const bf16* x, ConvArgs a) {
    const long wi = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int heads = a.C / a.head_dim;
    if (wi >= static_cast<long>(a.T) * heads) return;
    const int t = static_cast<int>(wi / heads), h = static_cast<int>(wi % heads);
    float s[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = lane; j < a.head_dim; j += 32) {
        const int c = h * a.head_dim + j;
        const float g = ldv<kDyFp32>(dy, static_cast<long>(t) * a.C + c);
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) s[lag] += conv_tap(x, a.carry, a.positions, t, c, a.C, lag) * g;
    }
#pragma unroll
    for (int lag = 0; lag < 4; ++lag) {
        const float v = warp_sum(s[lag]);
        if (lane == 0) d_dyn[wi * 4 + lag] = v * a.scale;
    }
}

// ---------------------------------------------------------------------------------------------------------------- 7
constexpr int kHd = 128;  // attention head dim (fixed geometry)

__global__ void attn_merge_fwd_kernel(bf16* result, float* out_s, float* lse_s, const float* o_n, const float* l_n,
                                      const float* o_f, const float* l_f, const bf16* sink, const bf16* gate, int T,
                                      int H) {
    const long wi = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wi >= static_cast<long>(T) * H) return;
    const int t = static_cast<int>(wi / H), h = static_cast<int>(wi % H);
    const float ln = l_n[static_cast<long>(h) * T + t];
    float l = ln, wn = 1.0f, wf = 0.0f;
    if (o_f) {
        const float lf = l_f[static_cast<long>(h) * T + t];
        l = logaddexp(ln, lf);
        wn = expf(ln - l);
        wf = lf == -INFINITY ? 0.0f : expf(lf - l);
    }
    const float sk = __bfloat162float(sink[wi]);
    const float a = sigmoid_gate_sink(l - sk);
    const float ls = logaddexp(l, sk);
    const float ws = expf(l - ls);
    for (int j = lane; j < kHd; j += 32) {
        const long e = wi * kHd + j;
        const float o = o_f ? o_n[e] * wn + o_f[e] * wf : o_n[e];
        // `_gate_sink_kernel`: (x * sigmoid(gate)) * sigmoid(lse - sink), one bf16 rounding
        result[e] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(o, sigmoid_gate_sink(__bfloat162float(gate[e]))), a));
        if (out_s) out_s[e] = o * ws;
    }
    if (lse_s && lane == 0) lse_s[static_cast<long>(h) * T + t] = ls;
}

template <bool kFp32>
__global__ void attn_merge_bwd_kernel(float* d_o_n, float* d_l_n, float* d_o_f, float* d_l_f, float* d_sink,
                                      float* d_gate, const void* d_res, const float* o_n, const float* l_n,
                                      const float* o_f, const float* l_f, const bf16* sink, const bf16* gate, int T,
                                      int H) {
    const long wi = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wi >= static_cast<long>(T) * H) return;
    const int t = static_cast<int>(wi / H), h = static_cast<int>(wi % H);
    const float ln = l_n[static_cast<long>(h) * T + t];
    float l = ln, wn = 1.0f, wf = 0.0f;
    if (o_f) {
        const float lf = l_f[static_cast<long>(h) * T + t];
        l = logaddexp(ln, lf);
        wn = expf(ln - l);
        wf = lf == -INFINITY ? 0.0f : expf(lf - l);
    }
    const float a = sigmoid_exact(l - __bfloat162float(sink[wi]));
    float da = 0.f, gn = 0.f, gf = 0.f;
    for (int j = lane; j < kHd; j += 32) {
        const long e = wi * kHd + j;
        const float dr = ldv<kFp32>(d_res, e);
        const float sg = sigmoid_exact(__bfloat162float(gate[e]));
        const float on = o_n[e], of = o_f ? o_f[e] : 0.f;
        const float o = on * wn + of * wf;
        const float d_o = dr * sg * a;
        da += dr * o * sg;
        gn += d_o * on;
        gf += d_o * of;
        d_gate[e] = dr * o * a * sg * (1.0f - sg);
        d_o_n[e] = d_o * wn;
        if (o_f) d_o_f[e] = d_o * wf;
    }
    da = warp_sum(da);
    gn = warp_sum(gn);
    gf = warp_sum(gf);
    if (lane == 0) {
        const float dxs = da * a * (1.0f - a);
        const float dl = dxs - gn * wn - gf * wf;
        d_sink[wi] = -dxs;
        d_l_n[static_cast<long>(h) * T + t] = wn * (gn + dl);
        if (o_f) d_l_f[static_cast<long>(h) * T + t] = wf * (gf + dl);
    }
}

template <bool kFp32>
__global__ void attn_merge_bwd_union_kernel(bf16* dout, float* delta, float* d_sink, float* d_gate, const void* d_res,
                                            const float* out_s, const float* lse_s, const bf16* sink,
                                            const bf16* gate, int T, int H) {
    const long wi = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wi >= static_cast<long>(T) * H) return;
    const int t = static_cast<int>(wi / H), h = static_cast<int>(wi % H);
    float dl = 0.f;
    for (int j = lane; j < kHd; j += 32) {
        const long e = wi * kHd + j;
        const float dr = ldv<kFp32>(d_res, e);
        const float sg = sigmoid_exact(__bfloat162float(gate[e]));
        const bf16 db = __float2bfloat16_rn(dr * sg);
        dout[e] = db;
        dl += __bfloat162float(db) * out_s[e];
        d_gate[e] = dr * out_s[e] * sg * (1.0f - sg);
    }
    dl = warp_sum(dl);
    if (lane == 0) {
        const long hi = static_cast<long>(h) * T + t;
        delta[hi] = dl;
        d_sink[wi] = -expf(__bfloat162float(sink[wi]) - lse_s[hi]) * dl;
    }
}

__global__ void attn_part_bwd_prep_kernel(bf16* dout, float* delta, const float* d_o, const float* o, const float* d_l,
                                          int T, int H) {
    const long wi = (static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (wi >= static_cast<long>(T) * H) return;
    const int t = static_cast<int>(wi / H), h = static_cast<int>(wi % H);
    float dl = 0.f;
    for (int j = lane; j < kHd; j += 32) {
        const long e = wi * kHd + j;
        const bf16 db = __float2bfloat16_rn(d_o[e]);
        dout[e] = db;
        dl += __bfloat162float(db) * o[e];
    }
    dl = warp_sum(dl);
    if (lane == 0) delta[static_cast<long>(h) * T + t] = dl - d_l[static_cast<long>(h) * T + t];
}

// --------------------------------------------------------------------------------------------------------------- 10
// One CTA per assignment a = t*k + s (row scatter[a], gamma weights[a]); interleaved (gate, up) pairs as 32-bit words.
__global__ void swiglu_fwd_kernel(bf16* y, const bf16* gate_up, const float* weights, const int* scatter, int I) {
    const int a = blockIdx.x;
    const long row = scatter[a];
    const float gamma = weights[a];
    const uint32_t* gu = reinterpret_cast<const uint32_t*>(gate_up) + row * I;
    for (int j = threadIdx.x; j < I; j += blockDim.x) {
        const uint32_t p = gu[j];
        const float g = __uint_as_float(p << 16), u = __uint_as_float(p & 0xFFFF0000u);
        const float sig = sigmoid_situ(g);
        const float v = __fmul_rn(__fmul_rn(__fmul_rn(softcap4(g), sig), softcap25(u)), gamma);
        y[row * I + j] = __float2bfloat16_rn(v);
    }
}

template <bool kFp32>
__global__ void swiglu_bwd_kernel(bf16* d_gate_up, float* d_weights, const void* dy, const bf16* gate_up,
                                  const float* weights, const int* scatter, int I) {
    __shared__ float scratch[32];
    const int a = blockIdx.x;
    const long row = scatter[a];
    const float gamma = weights[a];
    const uint32_t* gu = reinterpret_cast<const uint32_t*>(gate_up) + row * I;
    uint32_t* dgu = reinterpret_cast<uint32_t*>(d_gate_up) + row * I;
    float dgam[1] = {0.f};
    for (int j = threadIdx.x; j < I; j += blockDim.x) {
        const uint32_t p = gu[j];
        const float g = __uint_as_float(p << 16), u = __uint_as_float(p & 0xFFFF0000u);
        const float tg = tanhf(g * 0.25f), tu = tanhf(u / 25.0f);
        const float ga = 4.0f * tg, s = sigmoid_exact(g), b = 25.0f * tu;
        const float d = ldv<kFp32>(dy, row * I + j);
        const float dg = d * gamma * b * ((1.0f - tg * tg) * s + ga * s * (1.0f - s));
        const float du = d * gamma * ga * s * (1.0f - tu * tu);
        const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(dg));
        const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(du));
        dgu[j] = lo | (hi << 16);
        dgam[0] += d * ga * s * b;
    }
    if (d_weights) {
        block_sum<1>(dgam, 1, scratch);
        if (threadIdx.x == 0) d_weights[a] = dgam[0];
    }
}

// --------------------------------------------------------------------------------------------------------------- 11
__global__ void moe_combine_fwd_kernel(bf16* out, const bf16* rows, const int* scatter, int top_k, int D) {
    const int t = blockIdx.x;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < top_k; ++s)
            acc = __fadd_rn(acc, __bfloat162float(rows[static_cast<long>(scatter[t * top_k + s]) * D + c]));
        out[static_cast<long>(t) * D + c] = __float2bfloat16_rn(acc);
    }
}

template <bool kFp32>
__global__ void moe_combine_bwd_kernel(bf16* d_rows, const void* d_out, const int* scatter, int top_k, int D) {
    const int a = blockIdx.x;
    const long row = scatter[a];
    const long t = a / top_k;
    for (int c = threadIdx.x; c < D; c += blockDim.x)
        d_rows[row * D + c] = __float2bfloat16_rn(ldv<kFp32>(d_out, t * D + c));
}

}  // namespace

// ================================================================================================================
void log_tau(float* tau, const int* positions, int T, float n_floor, float alpha, cudaStream_t stream) {
    if (T <= 0) return;
    log_tau_kernel<<<blocks_for(T), kThreads, 0, stream>>>(tau, positions, T, 1.0f / n_floor, alpha);
    check_launch("log_tau");
}

void qkv_norm_fwd(bf16* y, const bf16* x, const QkvNormArgs& a, cudaStream_t stream) {
    const long rows = static_cast<long>(a.T) * a.H;
    if (rows == 0) return;
    if (a.tau && a.kv_fp8) throw std::invalid_argument("inkling qkv_norm: tau (q / r) and kv_fp8 (k / v) exclude");
    qkv_norm_fwd_kernel<<<blocks_for(rows * 32), kThreads, 0, stream>>>(y, x, a);
    check_launch("qkv_norm_fwd");
}

void qkv_norm_bwd(float* dx, const void* dy, bool dy_fp32, const bf16* x, const QkvNormArgs& a, cudaStream_t stream) {
    const long rows = static_cast<long>(a.T) * a.H;
    if (rows == 0) return;
    if (dy_fp32)
        qkv_norm_bwd_kernel<true><<<blocks_for(rows * 32), kThreads, 0, stream>>>(dx, dy, x, a);
    else
        qkv_norm_bwd_kernel<false><<<blocks_for(rows * 32), kThreads, 0, stream>>>(dx, dy, x, a);
    check_launch("qkv_norm_bwd");
}

void short_conv_fwd(bf16* out, const bf16* x, const bf16* w, const bf16* addend, const bf16* residual,
                    const ConvArgs& a, cudaStream_t stream) {
    const long n = static_cast<long>(a.T) * a.C;
    if (n == 0) return;
    if (residual && !addend) throw std::invalid_argument("inkling short_conv: residual needs an addend");
    short_conv_fwd_kernel<<<blocks_for(n), kThreads, 0, stream>>>(out, x, w, addend, residual, a);
    check_launch("short_conv_fwd");
}

void short_conv_bwd(float* dx, float* d_carry, const void* dy, bool dy_fp32, const bf16* w, const float* d_carry_next,
                    const ConvArgs& a, cudaStream_t stream) {
    const long n = static_cast<long>(a.T) * a.C;
    if (n == 0) return;
    if (a.T < 3 && (d_carry || d_carry_next)) throw std::invalid_argument("inkling short_conv: carries need T >= 3");
    if (dy_fp32)
        short_conv_bwd_kernel<true><<<blocks_for(n), kThreads, 0, stream>>>(dx, dy, w, d_carry_next, a);
    else
        short_conv_bwd_kernel<false><<<blocks_for(n), kThreads, 0, stream>>>(dx, dy, w, d_carry_next, a);
    if (d_carry) {
        if (dy_fp32)
            conv_d_carry_kernel<true, false><<<blocks_for(3L * a.C), kThreads, 0, stream>>>(d_carry, dy, w, nullptr,
                                                                                           nullptr, a);
        else
            conv_d_carry_kernel<false, false><<<blocks_for(3L * a.C), kThreads, 0, stream>>>(d_carry, dy, w, nullptr,
                                                                                            nullptr, a);
    }
    check_launch("short_conv_bwd");
}

void dynamic_conv_fwd(bf16* out, const bf16* x, const bf16* static_w, const bf16* dyn, const ConvArgs& a,
                      cudaStream_t stream) {
    const long n = static_cast<long>(a.T) * a.C;
    if (n == 0) return;
    if (a.C % a.head_dim) throw std::invalid_argument("inkling dynamic_conv: C % head_dim");
    dynamic_conv_fwd_kernel<<<blocks_for(n), kThreads, 0, stream>>>(out, x, static_w, dyn, a);
    check_launch("dynamic_conv_fwd");
}

void dynamic_conv_bwd(float* dx, float* d_dyn, float* d_carry, const void* dy, bool dy_fp32, const bf16* x,
                      const bf16* static_w, const bf16* dyn, const float* d_carry_next, const ConvArgs& a,
                      cudaStream_t stream) {
    const long n = static_cast<long>(a.T) * a.C;
    if (n == 0) return;
    if (a.T < 3 && (d_carry || d_carry_next)) throw std::invalid_argument("inkling dynamic_conv: carries need T >= 3");
    const long warps = static_cast<long>(a.T) * (a.C / a.head_dim);
    if (dy_fp32) {
        dynamic_conv_dx_kernel<true><<<blocks_for(n), kThreads, 0, stream>>>(dx, dy, static_w, dyn, d_carry_next, a);
        if (d_dyn) dynamic_conv_ddyn_kernel<true><<<blocks_for(warps * 32), kThreads, 0, stream>>>(d_dyn, dy, x, a);
        if (d_carry)
            conv_d_carry_kernel<true, true><<<blocks_for(3L * a.C), kThreads, 0, stream>>>(d_carry, dy, nullptr,
                                                                                          static_w, dyn, a);
    } else {
        dynamic_conv_dx_kernel<false><<<blocks_for(n), kThreads, 0, stream>>>(dx, dy, static_w, dyn, d_carry_next, a);
        if (d_dyn) dynamic_conv_ddyn_kernel<false><<<blocks_for(warps * 32), kThreads, 0, stream>>>(d_dyn, dy, x, a);
        if (d_carry)
            conv_d_carry_kernel<false, true><<<blocks_for(3L * a.C), kThreads, 0, stream>>>(d_carry, dy, nullptr,
                                                                                           static_w, dyn, a);
    }
    check_launch("dynamic_conv_bwd");
}

void attn_merge_fwd(bf16* result, float* out_s, float* lse_s, const float* o_n, const float* l_n, const float* o_f,
                    const float* l_f, const bf16* sink, const bf16* gate, int T, int H, cudaStream_t stream) {
    const long warps = static_cast<long>(T) * H;
    if (warps == 0) return;
    attn_merge_fwd_kernel<<<blocks_for(warps * 32), kThreads, 0, stream>>>(result, out_s, lse_s, o_n, l_n, o_f, l_f,
                                                                           sink, gate, T, H);
    check_launch("attn_merge_fwd");
}

void attn_merge_bwd(float* d_o_n, float* d_l_n, float* d_o_f, float* d_l_f, float* d_sink, float* d_gate,
                    const void* d_result, bool d_result_fp32, const float* o_n, const float* l_n, const float* o_f,
                    const float* l_f, const bf16* sink, const bf16* gate, int T, int H, cudaStream_t stream) {
    const long warps = static_cast<long>(T) * H;
    if (warps == 0) return;
    if (d_result_fp32)
        attn_merge_bwd_kernel<true><<<blocks_for(warps * 32), kThreads, 0, stream>>>(
            d_o_n, d_l_n, d_o_f, d_l_f, d_sink, d_gate, d_result, o_n, l_n, o_f, l_f, sink, gate, T, H);
    else
        attn_merge_bwd_kernel<false><<<blocks_for(warps * 32), kThreads, 0, stream>>>(
            d_o_n, d_l_n, d_o_f, d_l_f, d_sink, d_gate, d_result, o_n, l_n, o_f, l_f, sink, gate, T, H);
    check_launch("attn_merge_bwd");
}

void attn_merge_bwd_union(bf16* dout, float* delta, float* d_sink, float* d_gate, const void* d_result,
                          bool d_result_fp32, const float* out_s, const float* lse_s, const bf16* sink,
                          const bf16* gate, int T, int H, cudaStream_t stream) {
    const long warps = static_cast<long>(T) * H;
    if (warps == 0) return;
    if (d_result_fp32)
        attn_merge_bwd_union_kernel<true><<<blocks_for(warps * 32), kThreads, 0, stream>>>(
            dout, delta, d_sink, d_gate, d_result, out_s, lse_s, sink, gate, T, H);
    else
        attn_merge_bwd_union_kernel<false><<<blocks_for(warps * 32), kThreads, 0, stream>>>(
            dout, delta, d_sink, d_gate, d_result, out_s, lse_s, sink, gate, T, H);
    check_launch("attn_merge_bwd_union");
}

void attn_part_bwd_prep(bf16* dout, float* delta, const float* d_o, const float* o, const float* d_l, int T, int H,
                        cudaStream_t stream) {
    const long warps = static_cast<long>(T) * H;
    if (warps == 0) return;
    attn_part_bwd_prep_kernel<<<blocks_for(warps * 32), kThreads, 0, stream>>>(dout, delta, d_o, o, d_l, T, H);
    check_launch("attn_part_bwd_prep");
}

void softcap_swiglu_fwd(bf16* y, const bf16* gate_up, const float* weights, const int* scatter, int num_assign, int I,
                        float gate_cap, float up_cap, cudaStream_t stream) {
    if (num_assign == 0) return;
    if (gate_cap != 4.0f || up_cap != 25.0f)
        throw std::invalid_argument("inkling softcap_swiglu: the engine's instruction sequence is for caps 4 / 25");
    swiglu_fwd_kernel<<<num_assign, 128, 0, stream>>>(y, gate_up, weights, scatter, I);
    check_launch("softcap_swiglu_fwd");
}

void softcap_swiglu_bwd(bf16* d_gate_up, float* d_weights, const void* dy, bool dy_fp32, const bf16* gate_up,
                        const float* weights, const int* scatter, int num_assign, int I, float gate_cap, float up_cap,
                        cudaStream_t stream) {
    if (num_assign == 0) return;
    if (gate_cap != 4.0f || up_cap != 25.0f) throw std::invalid_argument("inkling softcap_swiglu: caps 4 / 25 only");
    if (dy_fp32)
        swiglu_bwd_kernel<true><<<num_assign, 128, 0, stream>>>(d_gate_up, d_weights, dy, gate_up, weights, scatter, I);
    else
        swiglu_bwd_kernel<false><<<num_assign, 128, 0, stream>>>(d_gate_up, d_weights, dy, gate_up, weights, scatter, I);
    check_launch("softcap_swiglu_bwd");
}

void moe_combine_fwd(bf16* out, const bf16* rows, const int* scatter, int T, int top_k, int D, cudaStream_t stream) {
    if (T == 0) return;
    moe_combine_fwd_kernel<<<T, 256, 0, stream>>>(out, rows, scatter, top_k, D);
    check_launch("moe_combine_fwd");
}

void moe_combine_bwd(bf16* d_rows, const void* d_out, bool d_out_fp32, const int* scatter, int T, int top_k, int D,
                     cudaStream_t stream) {
    if (T == 0) return;
    if (d_out_fp32)
        moe_combine_bwd_kernel<true><<<T * top_k, 256, 0, stream>>>(d_rows, d_out, scatter, top_k, D);
    else
        moe_combine_bwd_kernel<false><<<T * top_k, 256, 0, stream>>>(d_rows, d_out, scatter, top_k, D);
    check_launch("moe_combine_bwd");
}

}  // namespace inkling
