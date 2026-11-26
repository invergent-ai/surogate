// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inkling-2 trainer kernels: block AttnRes mix (fwd / bwd) and the loop controllers' elementwise parts.
// Forward fp32 order as `ops/attn_res.py` `_attn_res_kernel` and `ops/loop.py` (fma where Triton fuses, explicit
// rn ops where it compiles with enable_fp_fusion=False); reductions: fixed order (inkling_common.cuh block_sum).
// References: tests/inkling_ref/inkling_kernels_ref.py attn_res_*, loop_blend_*.

#include <stdexcept>
#include <string>

#include "kernels/inkling/inkling.h"
#include "kernels/inkling/inkling_common.cuh"

namespace inkling {
namespace {

constexpr int kBlock = 256;
constexpr int kN = kAttnResMaxSources;
constexpr int kQ = kAttnResMaxQueries;

inline void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("inkling ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float src_val(const AttnResArgs& a, int n, long i) {
    float v = a.src_fp32[n] ? static_cast<const float*>(a.src[n])[i] : __bfloat162float(static_cast<const bf16*>(a.src[n])[i]);
    if (n == a.N - 1 && a.addend)
        v = __fadd_rn(v, a.addend_fp32 ? static_cast<const float*>(a.addend)[i]
                                       : __bfloat162float(static_cast<const bf16*>(a.addend)[i]));
    return v;
}

// ------------------------------------------------------------------------------------------------- AttnRes forward
__global__ void __launch_bounds__(kBlock) attn_res_fwd_kernel(void* out, float* probs_out, float* rstd_out_src,
                                                              float* rstd_out, AttnResArgs a) {
    __shared__ float scratch[(kBlock / 32) * (kN + kQ * kN)];
    const long t = blockIdx.x;
    const int N = a.N, Q = a.Q, D = a.D;
    const long row = t * D;
    // pass 1: per-source sum of squares and query dots (fma accumulation as the reference)
    float v[kN + kQ * kN];
    for (int i = 0; i < N + Q * N; ++i) v[i] = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        for (int n = 0; n < N; ++n) {
            const float x = src_val(a, n, row + j);
            v[n] = fmaf(x, x, v[n]);
            for (int q = 0; q < Q; ++q)
                v[N + q * N + n] = fmaf(__fmul_rn(a.query[q * D + j], a.query_scale), x, v[N + q * N + n]);
        }
    }
    block_sum<kN + kQ * kN>(v, N + Q * N, scratch);
    float r[kN], pn[kQ][kN], inv_sum[kQ];
    for (int n = 0; n < N; ++n) r[n] = rsqrt_approx(fmaf(v[n], 1.0f / static_cast<float>(D), a.eps));
    for (int q = 0; q < Q; ++q) {
        float m = -INFINITY;
        for (int n = 0; n < N; ++n) {
            pn[q][n] = __fmul_rn(v[N + q * N + n], r[n]);
            m = fmaxf(m, pn[q][n]);
        }
        float s = 0.f;
        for (int n = 0; n < N; ++n) {
            pn[q][n] = expf(pn[q][n] - m);
            s += pn[q][n];
        }
        inv_sum[q] = __fdiv_rn(1.0f, s);
        for (int n = 0; n < N; ++n) {
            if (threadIdx.x == 0) probs_out[(static_cast<long>(q) * N + n) * gridDim.x + t] = pn[q][n] * inv_sum[q];
            pn[q][n] = __fmul_rn(pn[q][n], r[n]);  // normalize_values
        }
    }
    if (threadIdx.x == 0)
        for (int n = 0; n < N; ++n) rstd_out_src[static_cast<long>(n) * gridDim.x + t] = r[n];
    // pass 2 (norm only): RMS of each mix
    float ro[kQ];
    if (a.norm_w) {
        float ss[kQ];
        for (int q = 0; q < Q; ++q) ss[q] = 0.f;
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            for (int q = 0; q < Q; ++q) {
                float acc = 0.f;
                for (int n = 0; n < N; ++n) acc = fmaf(pn[q][n], src_val(a, n, row + j), acc);
                acc = __fmul_rn(acc, inv_sum[q]);
                ss[q] = __fadd_rn(ss[q], __fmul_rn(acc, acc));
            }
        }
        block_sum<kQ>(ss, Q, scratch);
        for (int q = 0; q < Q; ++q) {
            ro[q] = rsqrt_approx(__fadd_rn(__fdiv_rn(ss[q], static_cast<float>(D)), a.norm_eps));
            if (threadIdx.x == 0 && rstd_out) rstd_out[static_cast<long>(q) * gridDim.x + t] = ro[q];
        }
    }
    // pass 3: write
    const long TD = static_cast<long>(gridDim.x) * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        for (int q = 0; q < Q; ++q) {
            float acc = 0.f;
            for (int n = 0; n < N; ++n) acc = fmaf(pn[q][n], src_val(a, n, row + j), acc);
            acc = __fmul_rn(acc, inv_sum[q]);
            if (a.norm_w)
                static_cast<bf16*>(out)[q * TD + row + j] =
                    __float2bfloat16_rn(__fmul_rn(__fmul_rn(acc, ro[q]), __bfloat162float(a.norm_w[q * D + j])));
            else
                static_cast<float*>(out)[q * TD + row + j] = acc;
        }
    }
}

// ------------------------------------------------------------------------------------------------ AttnRes backward
struct DSrc {
    float* p[kN];
    float* addend;
};

template <bool kFp32>
__global__ void __launch_bounds__(kBlock) attn_res_bwd_kernel(DSrc d, const void* d_out, const float* probs,
                                                              const float* rstd, const float* rstd_out,
                                                              AttnResArgs a) {
    constexpr int kVals = kQ + 3 * kQ * kN;
    __shared__ float scratch[(kBlock / 32) * kVals];
    const long t = blockIdx.x, T = gridDim.x;
    const int N = a.N, Q = a.Q, D = a.D;
    const long row = t * D, TD = T * D;
    const bool norm = a.norm_w != nullptr;
    float p[kQ][kN], r[kN], ro[kQ];
    for (int n = 0; n < N; ++n) r[n] = rstd[n * T + t];
    for (int q = 0; q < Q; ++q) {
        ro[q] = norm ? rstd_out[q * T + t] : 1.f;
        for (int n = 0; n < N; ++n) p[q][n] = probs[(static_cast<long>(q) * N + n) * T + t];
    }
    auto mix_at = [&](int q, long j) {
        float acc = 0.f;
        for (int n = 0; n < N; ++n) acc += p[q][n] * r[n] * src_val(a, n, row + j);
        return acc;
    };
    auto g_at = [&](int q, long j) {
        float g = kFp32 ? static_cast<const float*>(d_out)[q * TD + row + j]
                        : __bfloat162float(static_cast<const bf16*>(d_out)[q * TD + row + j]);
        return norm ? g * __bfloat162float(a.norm_w[q * D + j]) : g;
    };
    // v = [dot_mg (Q) | a = q.x (Q*N) | G = g.x (Q*N) | M = mix.x (Q*N)]
    float v[kVals];
    const int nv = Q + 3 * Q * N;
    for (int i = 0; i < nv; ++i) v[i] = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float x[kN];
        for (int n = 0; n < N; ++n) x[n] = src_val(a, n, row + j);
        for (int q = 0; q < Q; ++q) {
            const float mix = mix_at(q, j), g = g_at(q, j), qv = a.query[q * D + j] * a.query_scale;
            v[q] += mix * g;
            for (int n = 0; n < N; ++n) {
                v[Q + q * N + n] += qv * x[n];
                v[Q + Q * N + q * N + n] += g * x[n];
                v[Q + 2 * Q * N + q * N + n] += mix * x[n];
            }
        }
    }
    block_sum<kVals>(v, nv, scratch);
    // per (q, n) scalars
    const float invD = 1.0f / static_cast<float>(D);
    float cmix[kQ], cA[kQ][kN], cB[kQ][kN], cC[kN];
    for (int n = 0; n < N; ++n) cC[n] = 0.f;
    for (int q = 0; q < Q; ++q) {
        cmix[q] = norm ? ro[q] * ro[q] * ro[q] * invD * v[q] : 0.f;  // dmix = ro g - cmix mix
        float u[kN], dp[kN], s = 0.f;
        for (int n = 0; n < N; ++n) {
            u[n] = norm ? ro[q] * v[Q + Q * N + q * N + n] - cmix[q] * v[Q + 2 * Q * N + q * N + n]
                        : v[Q + Q * N + q * N + n];
            dp[n] = r[n] * u[n];
            s += p[q][n] * dp[n];
        }
        for (int n = 0; n < N; ++n) {
            const float dlogit = p[q][n] * (dp[n] - s);
            cA[q][n] = p[q][n] * r[n];
            cB[q][n] = dlogit * r[n];
            cC[n] += p[q][n] * u[n] + dlogit * v[Q + q * N + n];  // dr_n
        }
    }
    for (int n = 0; n < N; ++n) cC[n] = -cC[n] * r[n] * r[n] * r[n] * invD;
    // pass 2: dx_n = sum_q (A dmix_q + B q_q) + C x_n, accumulated into the fp32 slots
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float dmix[kQ], qv[kQ];
        for (int q = 0; q < Q; ++q) {
            const float g = g_at(q, j);
            dmix[q] = norm ? ro[q] * g - cmix[q] * mix_at(q, j) : g;
            qv[q] = a.query[q * D + j] * a.query_scale;
        }
        for (int n = 0; n < N; ++n) {
            float dx = cC[n] * src_val(a, n, row + j);
            for (int q = 0; q < Q; ++q) dx += cA[q][n] * dmix[q] + cB[q][n] * qv[q];
            if (d.p[n]) d.p[n][row + j] += dx;
            if (n == N - 1 && d.addend) d.addend[row + j] += dx;
        }
    }
}

// --------------------------------------------------------------------------------------------------- loop blend
// One CTA per (stream s, token t). dest_stride_s = 0 broadcasts dest; norm_stride_s = 0 broadcasts the norm weight.
__global__ void __launch_bounds__(kBlock) blend_pair_fwd_kernel(bf16* pair, const float* dest, long dss,
                                                                const float* msg, int T, int D) {
    __shared__ float scratch[(kBlock / 32) * 2];
    const long s = blockIdx.y, t = blockIdx.x;
    const float* dr = dest + s * dss + t * D;
    const float* mr = msg + (s * T + t) * D;
    float v[2] = {0.f, 0.f};
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        v[0] += dr[j] * dr[j];
        v[1] += mr[j] * mr[j];
    }
    block_sum<2>(v, 2, scratch);
    const float rd = rsqrt_approx(__fdiv_rn(v[0], static_cast<float>(D)) + 1e-6f);
    const float rm = rsqrt_approx(__fdiv_rn(v[1], static_cast<float>(D)) + 1e-6f);
    bf16* out = pair + (s * T + t) * 2 * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        out[j] = __float2bfloat16_rn(dr[j] * rd);
        out[D + j] = __float2bfloat16_rn(mr[j] * rm);
    }
}

__device__ __forceinline__ float blend_x(float dest, float msg, float gf_gain, float gf_forget) {
    // `(1.0 - forget) * dest + gain * message`, unfused; accurate tanh (`tldevice.tanh`)
    const float gain = tanhf(gf_gain), forget = tanhf(gf_forget);
    return __fadd_rn(__fmul_rn(__fsub_rn(1.0f, forget), dest), __fmul_rn(gain, msg));
}

__global__ void __launch_bounds__(kBlock) blend_out_fwd_kernel(bf16* out, float* rstd, const float* dest, long dss,
                                                               const float* msg, const float* gf, const bf16* norm_w,
                                                               long nss, float eps, int T, int D) {
    __shared__ float scratch[kBlock / 32];
    const long s = blockIdx.y, t = blockIdx.x, st = s * T + t;
    const float* dr = dest + s * dss + t * D;
    const float* mr = msg + st * D;
    const float* g = gf + st * 2 * D;
    float v[1] = {0.f};
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float x = blend_x(dr[j], mr[j], g[j], g[D + j]);
        v[0] = __fadd_rn(v[0], __fmul_rn(x, x));
    }
    block_sum<1>(v, 1, scratch);
    const float rx = rsqrt_approx(__fadd_rn(__fdiv_rn(v[0], static_cast<float>(D)), eps));
    if (threadIdx.x == 0) rstd[st] = rx;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float x = blend_x(dr[j], mr[j], g[j], g[D + j]);
        out[st * D + j] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(x, rx), __bfloat162float(norm_w[s * nss + j])));
    }
}

template <bool kFp32>
__global__ void __launch_bounds__(kBlock) blend_out_bwd_kernel(float* d_dest_s, float* d_msg, float* d_gf,
                                                               const void* d_out, const float* dest, long dss,
                                                               const float* msg, const float* gf, const float* rstd,
                                                               const bf16* norm_w, long nss, int T, int D) {
    __shared__ float scratch[kBlock / 32];
    const long s = blockIdx.y, t = blockIdx.x, st = s * T + t;
    const float* dr = dest + s * dss + t * D;
    const float* mr = msg + st * D;
    const float* g = gf + st * 2 * D;
    const float rx = rstd[st];
    auto gw_at = [&](int j) {
        const float go = kFp32 ? static_cast<const float*>(d_out)[st * D + j]
                               : __bfloat162float(static_cast<const bf16*>(d_out)[st * D + j]);
        return go * __bfloat162float(norm_w[s * nss + j]);
    };
    float v[1] = {0.f};
    for (int j = threadIdx.x; j < D; j += blockDim.x) v[0] += blend_x(dr[j], mr[j], g[j], g[D + j]) * gw_at(j);
    block_sum<1>(v, 1, scratch);
    const float c = rx * rx * rx / static_cast<float>(D) * v[0];
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float gain = tanhf(g[j]), forget = tanhf(g[D + j]);
        const float x = blend_x(dr[j], mr[j], g[j], g[D + j]);
        const float dx = rx * gw_at(j) - c * x;
        d_dest_s[st * D + j] = (1.0f - forget) * dx;
        d_msg[st * D + j] = gain * dx;
        d_gf[st * 2 * D + j] = dx * mr[j] * (1.0f - gain * gain);
        d_gf[st * 2 * D + D + j] = -dx * dr[j] * (1.0f - forget * forget);
    }
}

__global__ void __launch_bounds__(kBlock) blend_pair_bwd_kernel(float* d_dest_s, float* d_msg, const float* d_pair,
                                                                const float* dest, long dss, const float* msg, int T,
                                                                int D) {
    __shared__ float scratch[(kBlock / 32) * 4];
    const long s = blockIdx.y, t = blockIdx.x, st = s * T + t;
    const float* dr = dest + s * dss + t * D;
    const float* mr = msg + st * D;
    const float* dp = d_pair + st * 2 * D;
    float v[4] = {0.f, 0.f, 0.f, 0.f};
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        v[0] += dr[j] * dr[j];
        v[1] += mr[j] * mr[j];
        v[2] += dr[j] * dp[j];
        v[3] += mr[j] * dp[D + j];
    }
    block_sum<4>(v, 4, scratch);
    const float rd = rsqrt_approx(__fdiv_rn(v[0], static_cast<float>(D)) + 1e-6f);
    const float rm = rsqrt_approx(__fdiv_rn(v[1], static_cast<float>(D)) + 1e-6f);
    const float cd = rd * rd * rd / static_cast<float>(D) * v[2];
    const float cm = rm * rm * rm / static_cast<float>(D) * v[3];
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        d_dest_s[st * D + j] += rd * dp[j] - cd * dr[j];
        d_msg[st * D + j] += rm * dp[D + j] - cm * mr[j];
    }
}

__global__ void blend_reduce_dest_kernel(float* d_dest, const float* d_dest_s, bool broadcast, bool accumulate, int S,
                                         long TD) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (broadcast) {
        if (i >= TD) return;
        float acc = 0.f;
        for (int s = 0; s < S; ++s) acc += d_dest_s[s * TD + i];  // stream order
        d_dest[i] = accumulate ? d_dest[i] + acc : acc;
    } else {
        if (i >= S * TD) return;
        d_dest[i] = accumulate ? d_dest[i] + d_dest_s[i] : d_dest_s[i];
    }
}

}  // namespace

// ================================================================================================================
void attn_res_fwd(void* out, float* probs, float* rstd, float* rstd_out, const AttnResArgs& a, cudaStream_t stream) {
    if (a.T == 0) return;
    if (a.N < 1 || a.N > kN || a.Q < 1 || a.Q > kQ) throw std::invalid_argument("inkling attn_res: N <= 8, Q <= 4");
    attn_res_fwd_kernel<<<a.T, kBlock, 0, stream>>>(out, probs, rstd, rstd_out, a);
    check_launch("attn_res_fwd");
}

void attn_res_bwd(float* const* d_src, float* d_addend, const void* d_out, bool d_out_fp32, const float* probs,
                  const float* rstd, const float* rstd_out, const AttnResArgs& a, cudaStream_t stream) {
    if (a.T == 0) return;
    if (a.N < 1 || a.N > kN || a.Q < 1 || a.Q > kQ) throw std::invalid_argument("inkling attn_res: N <= 8, Q <= 4");
    DSrc d{};
    for (int n = 0; n < a.N; ++n) d.p[n] = d_src ? d_src[n] : nullptr;
    d.addend = d_addend;
    if (d_out_fp32)
        attn_res_bwd_kernel<true><<<a.T, kBlock, 0, stream>>>(d, d_out, probs, rstd, rstd_out, a);
    else
        attn_res_bwd_kernel<false><<<a.T, kBlock, 0, stream>>>(d, d_out, probs, rstd, rstd_out, a);
    check_launch("attn_res_bwd");
}

void loop_blend_pair_fwd(bf16* pair, const float* dest, long dest_stride_s, const float* msg, int S, int T, int D,
                         cudaStream_t stream) {
    if (S * T == 0) return;
    blend_pair_fwd_kernel<<<dim3(T, S), kBlock, 0, stream>>>(pair, dest, dest_stride_s, msg, T, D);
    check_launch("loop_blend_pair_fwd");
}

void loop_blend_out_fwd(bf16* out, float* rstd, const float* dest, long dest_stride_s, const float* msg,
                        const float* gf, const bf16* norm_w, long norm_stride_s, float eps, int S, int T, int D,
                        cudaStream_t stream) {
    if (S * T == 0) return;
    blend_out_fwd_kernel<<<dim3(T, S), kBlock, 0, stream>>>(out, rstd, dest, dest_stride_s, msg, gf, norm_w,
                                                            norm_stride_s, eps, T, D);
    check_launch("loop_blend_out_fwd");
}

void loop_blend_out_bwd(float* d_dest_s, float* d_msg, float* d_gf, const void* d_out, bool d_out_fp32,
                        const float* dest, long dest_stride_s, const float* msg, const float* gf, const float* rstd,
                        const bf16* norm_w, long norm_stride_s, int S, int T, int D, cudaStream_t stream) {
    if (S * T == 0) return;
    if (d_out_fp32)
        blend_out_bwd_kernel<true><<<dim3(T, S), kBlock, 0, stream>>>(d_dest_s, d_msg, d_gf, d_out, dest, dest_stride_s,
                                                                      msg, gf, rstd, norm_w, norm_stride_s, T, D);
    else
        blend_out_bwd_kernel<false><<<dim3(T, S), kBlock, 0, stream>>>(d_dest_s, d_msg, d_gf, d_out, dest,
                                                                       dest_stride_s, msg, gf, rstd, norm_w,
                                                                       norm_stride_s, T, D);
    check_launch("loop_blend_out_bwd");
}

void loop_blend_pair_bwd(float* d_dest_s, float* d_msg, const float* d_pair, const float* dest, long dest_stride_s,
                         const float* msg, int S, int T, int D, cudaStream_t stream) {
    if (S * T == 0) return;
    blend_pair_bwd_kernel<<<dim3(T, S), kBlock, 0, stream>>>(d_dest_s, d_msg, d_pair, dest, dest_stride_s, msg, T, D);
    check_launch("loop_blend_pair_bwd");
}

void loop_blend_reduce_dest(float* d_dest, const float* d_dest_s, bool broadcast, bool accumulate, int S, int T,
                            int D, cudaStream_t stream) {
    const long TD = static_cast<long>(T) * D;
    const long n = broadcast ? TD : S * TD;
    if (n == 0) return;
    blend_reduce_dest_kernel<<<static_cast<int>((n + 255) / 256), 256, 0, stream>>>(d_dest, d_dest_s, broadcast,
                                                                                   accumulate, S, TD);
    check_launch("loop_blend_reduce_dest");
}

}  // namespace inkling
