// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
#include "kernels/glm5.h"

#include <cmath>
#include <stdexcept>
#include <cuda_bf16.h>
#include "utilities/utils.h"

namespace {
struct Ptr {
    void* data;
    bool bf16;
    __device__ float get(long i) const {
        if (!data) return 0.f;
        return bf16 ? __bfloat162float(static_cast<nv_bfloat16*>(data)[i]) : static_cast<float*>(data)[i];
    }
    __device__ void set(long i, float x) const {
        if (!data) return;
        if (bf16)
            static_cast<nv_bfloat16*>(data)[i] = __float2bfloat16(x);
        else
            static_cast<float*>(data)[i] = x;
    }
};
Ptr ptr(const Tensor& t) {
    if (t.Data && t.DType != ETensorDType::BF16 && t.DType != ETensorDType::FP32)
        throw std::runtime_error("GLM training kernels require BF16 or FP32 tensors");
    return {t.Data, t.DType == ETensorDType::BF16};
}
__device__ float round_to(float x, bool bf16) {
    return bf16 ? __bfloat162float(__float2bfloat16(x)) : x;
}
__device__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}
__device__ float reduce(float x, float* tmp) {
    int t = threadIdx.x;
    tmp[t] = x;
    __syncthreads();
    for (int s = blockDim.x / 2; s; s /= 2) {
        if (t < s) tmp[t] += tmp[t + s];
        __syncthreads();
    }
    return tmp[0];
}

// Thread zero executes the small Sinkhorn problem, in the reference's exact
// order: softmax + eps, column, then (row, column) * (iterations - 1).
__device__ void hc_gates(const float* logits,
                         Ptr base,
                         Ptr scale,
                         int H,
                         int iters,
                         float eps,
                         float* pre,
                         float* post,
                         float* comb,
                         float* history,
                         float* softmax) {
    for (int s = 0; s < H; ++s) {
        pre[s] = sigmoid(logits[s] * scale.get(0) + base.get(s)) + eps;
        post[s] = 2.f * sigmoid(logits[H + s] * scale.get(1) + base.get(H + s));
    }
    for (int r = 0; r < H; ++r) {
        float mx = -INFINITY, sum = 0.f;
        for (int c = 0; c < H; ++c)
            mx = fmaxf(mx, logits[2 * H + r * H + c] * scale.get(2) + base.get(2 * H + r * H + c));
        for (int c = 0; c < H; ++c) {
            float v = expf(logits[2 * H + r * H + c] * scale.get(2) + base.get(2 * H + r * H + c) - mx);
            comb[r * H + c] = v;
            sum += v;
        }
        for (int c = 0; c < H; ++c) {
            float v = comb[r * H + c] / sum;
            if (softmax) softmax[r * H + c] = v;
            comb[r * H + c] = v + eps;
        }
    }
    for (int step = 0; step < 2 * iters - 1; ++step) {
        if (history)
            for (int j = 0; j < H * H; ++j)
                history[step * H * H + j] = comb[j];
        for (int axis = 0; axis < H; ++axis) {
            float sum = eps;
            for (int j = 0; j < H; ++j)
                sum += comb[step % 2 ? axis * H + j : j * H + axis];
            for (int j = 0; j < H; ++j)
                comb[step % 2 ? axis * H + j : j * H + axis] /= sum;
        }
    }
}

__global__ void mhc_mix_fwd(Ptr x,
                            Ptr fn,
                            Ptr base,
                            Ptr scale,
                            Ptr y,
                            Ptr post_out,
                            Ptr comb_out,
                            int C,
                            int H,
                            int iters,
                            float eps,
                            float norm_eps) {
    __shared__ float scratch[256], logits[80], pre[8], post[8], comb[64], rstd;
    const int W = C * H, M = H * (H + 2), t = threadIdx.x;
    const long row = blockIdx.x;
    float ss = 0.f;
    for (int c = t; c < W; c += blockDim.x) {
        float v = x.get(row * W + c);
        ss += v * v;
    }
    ss = reduce(ss, scratch);
    if (t == 0) rstd = rsqrtf(ss / W + norm_eps);
    __syncthreads();
    for (int m = 0; m < M; ++m) {
        float dot = 0.f;
        for (int c = t; c < W; c += blockDim.x)
            dot += (x.get(row * W + c) * rstd) * fn.get(m * W + c);
        dot = reduce(dot, scratch);
        if (t == 0) logits[m] = dot;
        __syncthreads();
    }
    if (t == 0) hc_gates(logits, base, scale, H, iters, eps, pre, post, comb, nullptr, nullptr);
    __syncthreads();
    for (int c = t; c < C; c += blockDim.x) {
        float sum = 0.f;
        for (int s = 0; s < H; ++s)
            sum += pre[s] * x.get(row * W + s * C + c);
        y.set(row * C + c, sum);
    }
    if (t < H) post_out.set(row * H + t, post[t]);
    if (t < H * H) comb_out.set(row * H * H + t, comb[t]);
}

__global__ void mhc_mix_bwd(Ptr dy,
                            Ptr dpost,
                            Ptr dcomb,
                            Ptr x,
                            Ptr fn,
                            Ptr base,
                            Ptr scale,
                            float* dx,
                            float* dfn,
                            float* dbase,
                            float* dscale,
                            int C,
                            int H,
                            int iters,
                            float eps,
                            float norm_eps) {
    __shared__ float scratch[256], logits[80], dl[80], pre[8], post[8], comb[64], softmax[64];
    __shared__ float history[127 * 64], dc[64], dp[8], rstd;
    const int W = C * H, M = H * (H + 2), t = threadIdx.x;
    const long row = blockIdx.x;
    float ss = 0.f;
    for (int c = t; c < W; c += blockDim.x) {
        float v = x.get(row * W + c);
        ss += v * v;
    }
    ss = reduce(ss, scratch);
    if (t == 0) rstd = rsqrtf(ss / W + norm_eps);
    __syncthreads();
    for (int m = 0; m < M; ++m) {
        float dot = 0.f;
        for (int c = t; c < W; c += blockDim.x)
            dot += x.get(row * W + c) * rstd * fn.get(m * W + c);
        dot = reduce(dot, scratch);
        if (t == 0) logits[m] = dot;
        __syncthreads();
    }
    for (int s = 0; s < H; ++s) {
        float v = 0.f;
        for (int c = t; c < C; c += blockDim.x)
            v += dy.get(row * C + c) * x.get(row * W + s * C + c);
        v = reduce(v, scratch);
        if (t == 0) dp[s] = v;
        __syncthreads();
    }
    if (t == 0) {
        hc_gates(logits, base, scale, H, iters, eps, pre, post, comb, history, softmax);
        for (int j = 0; j < H * H; ++j)
            dc[j] = dcomb.get(row * H * H + j);
        for (int step = 2 * iters - 2; step >= 0; --step) {
            const float* old = history + step * H * H;
            for (int axis = 0; axis < H; ++axis) {
                float sum = eps, dot = 0.f;
                for (int j = 0; j < H; ++j) {
                    int i = step % 2 ? axis * H + j : j * H + axis;
                    sum += old[i];
                    dot += dc[i] * old[i];
                }
                for (int j = 0; j < H; ++j) {
                    int i = step % 2 ? axis * H + j : j * H + axis;
                    dc[i] = dc[i] / sum - dot / (sum * sum);
                }
            }
        }
        for (int r = 0; r < H; ++r) {
            float dot = 0.f;
            for (int c = 0; c < H; ++c)
                dot += dc[r * H + c] * softmax[r * H + c];
            for (int c = 0; c < H; ++c)
                dl[2 * H + r * H + c] = softmax[r * H + c] * (dc[r * H + c] - dot);
            float p = pre[r] - eps, q = post[r] * .5f;
            dl[r] = dp[r] * p * (1.f - p);
            dl[H + r] = dpost.get(row * H + r) * 2.f * q * (1.f - q);
        }
        float ds[3] = {};
        for (int m = 0; m < M; ++m) {
            int group = m < H ? 0 : m < 2 * H ? 1 : 2;
            if (dbase) atomicAdd(dbase + m, dl[m]);
            ds[group] += dl[m] * logits[m];
            dl[m] *= scale.get(group);
        }
        if (dscale)
            for (int g = 0; g < 3; ++g)
                atomicAdd(dscale + g, ds[g]);
    }
    __syncthreads();
    float dot = 0.f;
    for (int c = t; c < W; c += blockDim.x) {
        float gx = 0.f;
        for (int m = 0; m < M; ++m) {
            gx += dl[m] * fn.get(m * W + c);
            if (dfn) atomicAdd(dfn + m * W + c, dl[m] * x.get(row * W + c) * rstd);
        }
        if (dx) dx[row * W + c] = gx;
        dot += gx * x.get(row * W + c);
    }
    dot = reduce(dot, scratch);
    if (dx)
        for (int c = t; c < W; c += blockDim.x) {
            float v = rstd * (dx[row * W + c] - x.get(row * W + c) * rstd * rstd * dot / W);
            dx[row * W + c] = v + pre[c / C] * dy.get(row * C + c % C);
        }
}

__global__ void mhc_combine_fwd(Ptr x, Ptr y, Ptr post, Ptr comb, Ptr out, int C, int H) {
    long row = blockIdx.x;
    for (int c = threadIdx.x; c < C; c += blockDim.x)
        for (int s = 0; s < H; ++s) {
            float sum = 0.f;
            for (int j = 0; j < H; ++j)
                sum += round_to(comb.get(row * H * H + j * H + s), x.bf16) * x.get((row * H + j) * C + c);
            float placed = round_to(round_to(post.get(row * H + s), x.bf16) * y.get(row * C + c), x.bf16);
            out.set((row * H + s) * C + c, placed + round_to(sum, x.bf16));
        }
}

__global__ void
mhc_combine_bwd(Ptr dout, Ptr x, Ptr y, Ptr post, Ptr comb, float* dx, float* dy, float* dp, float* dc, int C, int H) {
    long row = blockIdx.x;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float gy = 0.f;
        for (int s = 0; s < H; ++s) {
            float d = dout.get((row * H + s) * C + c);
            gy += d * round_to(post.get(row * H + s), x.bf16);
            if (dp) atomicAdd(dp + row * H + s, d * y.get(row * C + c));
            if (dc)
                for (int j = 0; j < H; ++j)
                    atomicAdd(dc + row * H * H + j * H + s, d * x.get((row * H + j) * C + c));
        }
        if (dy) dy[row * C + c] = gy;
        if (dx)
            for (int j = 0; j < H; ++j) {
                float gx = 0.f;
                for (int s = 0; s < H; ++s)
                    gx += dout.get((row * H + s) * C + c) * round_to(comb.get(row * H * H + j * H + s), x.bf16);
                dx[(row * H + j) * C + c] = gx;
            }
    }
}

__global__ void decay_fwd(Ptr x, Ptr a, Ptr bias, Ptr out, long N, int H, int D, float bound) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out.set(i, bound * sigmoid(expf(a.get(i / D % H)) * (x.get(i) + bias.get(i % (H * D)))));
}
__global__ void
decay_bwd(Ptr dy, Ptr x, Ptr a, Ptr bias, float* dx, float* da, float* db, long N, int H, int D, float bound) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int h = i / D % H, c = i % (H * D);
    float rate = expf(a.get(h)), z = x.get(i) + bias.get(c), s = sigmoid(rate * z);
    float v = dy.get(i) * bound * s * (1.f - s) * rate;
    if (dx) dx[i] = v;
    if (da) atomicAdd(da + h, v * z);
    if (db) atomicAdd(db + c, v);
}

// The recurrent implementation is intentionally independent of FLA's chunk
// factorization. FP32 states and checkpoint/replay avoid products of inverse
// forget gates, which would be unstable for the model's -5 lower bound.
template <int K>
__device__ void kda_inputs(Ptr q, Ptr k, Ptr g, long offset, float* qs, float* ks, float* decay, float* qr, float* kr) {
    if (threadIdx.x == 0) {
        float qsum = 1e-6f, ksum = 1e-6f;
        for (int j = 0; j < K; ++j) {
            float a = q.get(offset + j), b = k.get(offset + j);
            qsum += a * a;
            ksum += b * b;
        }
        *qr = rsqrtf(qsum);
        *kr = rsqrtf(ksum);
        for (int j = 0; j < K; ++j) {
            qs[j] = q.get(offset + j) * (*qr);
            ks[j] = k.get(offset + j) * (*kr);
            decay[j] = expf(g.get(offset + j));
        }
    }
    __syncthreads();
}

template <int K>
__global__ void
kda_fwd(Ptr q, Ptr k, Ptr v, Ptr g, Ptr beta, Ptr out, float* checkpoints, const int* positions, int T, int H) {
    __shared__ float qs[K], ks[K], decay[K], qr, kr;
    const int h = blockIdx.x % H, b = blockIdx.x / H, c = threadIdx.x;
    const int NC = (T + GLM5_KDA_CHECKPOINT - 1) / GLM5_KDA_CHECKPOINT;
    float state[K] = {};
    for (int t = 0; t < T; ++t) {
        if (positions && positions[b * T + t] == 0)
            for (int j = 0; j < K; ++j)
                state[j] = 0.f;
        long offset = ((long(b) * T + t) * H + h) * K;
        if (checkpoints && t % GLM5_KDA_CHECKPOINT == 0)
            for (int j = 0; j < K; ++j)
                checkpoints[(((long(b) * NC + t / GLM5_KDA_CHECKPOINT) * H + h) * K + j) * K + c] = state[j];
        kda_inputs<K>(q, k, g, offset, qs, ks, decay, &qr, &kr);
        float memory = 0.f;
        for (int j = 0; j < K; ++j) {
            state[j] *= decay[j];
            memory += state[j] * ks[j];
        }
        float delta = beta.get(offset / K) * (v.get(offset + c) - memory), y = 0.f;
        for (int j = 0; j < K; ++j) {
            state[j] += ks[j] * delta;
            y += qs[j] * state[j];
        }
        out.set(offset + c, y * rsqrtf(float(K)));
        __syncthreads();
    }
}

template <int K>
__global__ void kda_bwd(Ptr dout,
                        Ptr q,
                        Ptr k,
                        Ptr v,
                        Ptr g,
                        Ptr beta,
                        float* dq,
                        float* dk,
                        float* dv,
                        float* dg,
                        float* db,
                        const float* checkpoints,
                        const int* positions,
                        int T,
                        int H) {
    __shared__ float qs[K], ks[K], decay[K], qr, kr, qgrad[K], kgrad[K], ggrad[K], bgrad;
    const int h = blockIdx.x % H, b = blockIdx.x / H, c = threadIdx.x;
    const int NC = (T + GLM5_KDA_CHECKPOINT - 1) / GLM5_KDA_CHECKPOINT;
    float adj[K] = {}, state[K];
    for (int t = T - 1; t >= 0; --t) {
        int start = t / GLM5_KDA_CHECKPOINT * GLM5_KDA_CHECKPOINT;
        for (int j = 0; j < K; ++j)
            state[j] = checkpoints[(((long(b) * NC + t / GLM5_KDA_CHECKPOINT) * H + h) * K + j) * K + c];
        for (int r = start; r < t; ++r) {
            if (positions && positions[b * T + r] == 0)
                for (int j = 0; j < K; ++j)
                    state[j] = 0.f;
            long off = ((long(b) * T + r) * H + h) * K;
            kda_inputs<K>(q, k, g, off, qs, ks, decay, &qr, &kr);
            float mem = 0.f;
            for (int j = 0; j < K; ++j) {
                state[j] *= decay[j];
                mem += state[j] * ks[j];
            }
            float delta = beta.get(off / K) * (v.get(off + c) - mem);
            for (int j = 0; j < K; ++j)
                state[j] += ks[j] * delta;
            __syncthreads();
        }
        long off = ((long(b) * T + t) * H + h) * K;
        const bool reset = positions && positions[b * T + t] == 0;
        if (reset)
            for (int j = 0; j < K; ++j)
                state[j] = 0.f;
        kda_inputs<K>(q, k, g, off, qs, ks, decay, &qr, &kr);
        qgrad[c] = kgrad[c] = ggrad[c] = 0.f;
        if (c == 0) bgrad = 0.f;
        __syncthreads();
        float mem = 0.f;
        for (int j = 0; j < K; ++j) {
            state[j] *= decay[j];
            mem += state[j] * ks[j];
        }
        float err = v.get(off + c) - mem, be = beta.get(off / K), delta = be * err;
        float dy = dout.get(off + c), du = 0.f, scale = rsqrtf(float(K));
        for (int j = 0; j < K; ++j) {
            adj[j] += qs[j] * scale * dy;
            du += ks[j] * adj[j];
        }
        if (dv) dv[off + c] = be * du;
        atomicAdd(&bgrad, du * err);
        for (int j = 0; j < K; ++j) {
            atomicAdd(qgrad + j, scale * dy * (state[j] + ks[j] * delta));
            atomicAdd(kgrad + j, adj[j] * delta - be * du * state[j]);
            float ds = adj[j] - ks[j] * be * du;
            atomicAdd(ggrad + j, ds * state[j]);
            adj[j] = reset ? 0.f : ds * decay[j];
        }
        __syncthreads();
        float qdot = 0.f, kdot = 0.f;
        for (int j = 0; j < K; ++j) {
            qdot += qgrad[j] * qs[j];
            kdot += kgrad[j] * ks[j];
        }
        if (dq) dq[off + c] = (qgrad[c] - qs[c] * qdot) * qr;
        if (dk) dk[off + c] = (kgrad[c] - ks[c] * kdot) * kr;
        if (dg) dg[off + c] = ggrad[c];
        if (db && c == 0) db[off / K] = bgrad;
        __syncthreads();
    }
}

// GLM stores its depthwise convolution in FP32 and applies SiLU before
// casting to activation precision. Position IDs reset history for packing.
__device__ float conv_value(Ptr x, Ptr w, long i, int t, int C, int K, int pos) {
    float z = 0.f;
    for (int j = 0; j < K && j <= t && j <= pos; ++j)
        z += x.get(i - long(j) * C) * w.get((i % C) * K + K - 1 - j);
    return z;
}
__global__ void conv_fwd(Ptr x, Ptr w, const int* pos, Ptr y, long n, int T, int C, int K) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float z = conv_value(x, w, i, (i / C) % T, C, K, pos[i / C]);
    y.set(i, z * sigmoid(z));
}
__global__ void conv_bwd(Ptr dy, Ptr x, Ptr w, const int* pos, float* dx, float* dw, long n, int T, int C, int K) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int t = (i / C) % T, c = i % C;
    float z = conv_value(x, w, i, t, C, K, pos[i / C]);
    float s = sigmoid(z), dz = dy.get(i) * s * (1.f + z * (1.f - s));
    for (int j = 0; j < K && j <= t && j <= pos[i / C]; ++j) {
        if (dx) atomicAdd(dx + i - long(j) * C, dz * w.get(c * K + K - 1 - j));
        if (dw) atomicAdd(dw + c * K + K - 1 - j, dz * x.get(i - long(j) * C));
    }
}

__global__ void clamp_fwd(Ptr x, Ptr y, long n, float lo, float hi, int width) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (width && i % width >= width / 2) lo = -INFINITY;
        y.set(i, fminf(fmaxf(x.get(i), lo), hi));
    }
}
__global__ void clamp_bwd(Ptr dy, Ptr x, float* dx, long n, float lo, float hi, int width) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && dx) {
        if (width && i % width >= width / 2) lo = -INFINITY;
        float v = x.get(i);
        dx[i] = (v >= lo && v <= hi) ? dy.get(i) : 0.f;
    }
}
__global__ void copy_gradient(Ptr src, Ptr dst, long n, bool add) {
    long i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst.set(i, src.get(i) + (add ? dst.get(i) : 0.f));
}

void validate_hc(const Glm5Options& o) {
    if (o.streams < 1 || o.streams > 8 || o.sinkhorn_iters < 1 || o.sinkhorn_iters > 64)
        throw std::runtime_error("mHC requires 1..8 streams and 1..64 Sinkhorn iterations");
}
}  // namespace

void glm5_forward(Glm5Kernel kind,
                  const std::vector<Tensor>& x,
                  std::vector<Tensor>& y,
                  const Glm5Options& o,
                  cudaStream_t stream) {
    auto p = [&](int i) {
        return ptr(x.at(i));
    };
    auto r = [&](int i) {
        return ptr(y.at(i));
    };
    long n = x[0].nelem();
    switch (kind) {
        case Glm5Kernel::MhcMix: {
            validate_hc(o);
            int C = x[0].Sizes[x[0].Rank - 1] / o.streams;
            mhc_mix_fwd<<<n / (C * o.streams), 256, 0, stream>>>(p(0),
                                                                 p(1),
                                                                 p(2),
                                                                 p(3),
                                                                 r(0),
                                                                 r(1),
                                                                 r(2),
                                                                 C,
                                                                 o.streams,
                                                                 o.sinkhorn_iters,
                                                                 o.hc_eps,
                                                                 o.norm_eps);
            break;
        }
        case Glm5Kernel::MhcCombine: {
            validate_hc(o);
            int C = x[0].Sizes[x[0].Rank - 1] / o.streams;
            mhc_combine_fwd<<<n / (C * o.streams), 256, 0, stream>>>(p(0), p(1), p(2), p(3), r(0), C, o.streams);
            break;
        }
        case Glm5Kernel::KdaDecay:
            decay_fwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                           p(1),
                                                           p(2),
                                                           r(0),
                                                           n,
                                                           x[0].Sizes[2],
                                                           x[0].Sizes[3],
                                                           o.lower_bound);
            break;
        case Glm5Kernel::KdaRule: {
            int T = x[0].Sizes[1], H = x[0].Sizes[2], D = x[0].Sizes[3], BH = x[0].Sizes[0] * H;
            const int* pos = x.size() == 6 ? x[5].get<int>() : nullptr;
#define KDA_FWD(DIM) \
    case DIM: kda_fwd<DIM><<<BH, DIM, 0, stream>>>(p(0), p(1), p(2), p(3), p(4), r(0), nullptr, pos, T, H); break
            switch (D) {
                KDA_FWD(8);
                KDA_FWD(16);
                KDA_FWD(32);
                KDA_FWD(64);
                KDA_FWD(128);
                default: throw std::runtime_error("KDA head dimension must be 8, 16, 32, 64 or 128");
            }
#undef KDA_FWD
            break;
        }
        case Glm5Kernel::Clamp:
            clamp_fwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                           r(0),
                                                           n,
                                                           o.clamp_min,
                                                           o.clamp_max,
                                                           o.fused_gate_up ? x[0].Sizes[x[0].Rank - 1] : 0);
            break;
        case Glm5Kernel::CausalConv1d:
            conv_fwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                          p(1),
                                                          x[2].get<int>(),
                                                          r(0),
                                                          n,
                                                          x[0].Sizes[1],
                                                          x[0].Sizes[2],
                                                          x[1].Sizes[2]);
            break;
    }
    CUDA_CHECK(cudaGetLastError());
}

void glm5_backward(Glm5Kernel kind,
                   const std::vector<Tensor>& x,
                   std::vector<Tensor>& y,
                   const Glm5Options& o,
                   Tensor checkpoints,
                   cudaStream_t stream) {
    auto p = [&](int i) {
        return ptr(x.at(i));
    };
    auto r = [&](int i) {
        return y.at(i).Data ? y.at(i).get<float>() : nullptr;
    };
    // Parameter/gate reductions use atomic adds; every output is a fresh temporary.
    for (auto& t : y)
        if (t.Data) CUDA_CHECK(cudaMemsetAsync(t.Data, 0, t.bytes(), stream));
    switch (kind) {
        case Glm5Kernel::MhcMix: {
            validate_hc(o);
            int C = x[3].Sizes[x[3].Rank - 1] / o.streams;
            mhc_mix_bwd<<<x[3].nelem() / (C * o.streams), 256, 0, stream>>>(p(0),
                                                                            p(1),
                                                                            p(2),
                                                                            p(3),
                                                                            p(4),
                                                                            p(5),
                                                                            p(6),
                                                                            r(0),
                                                                            r(1),
                                                                            r(2),
                                                                            r(3),
                                                                            C,
                                                                            o.streams,
                                                                            o.sinkhorn_iters,
                                                                            o.hc_eps,
                                                                            o.norm_eps);
            break;
        }
        case Glm5Kernel::MhcCombine: {
            int C = x[1].Sizes[x[1].Rank - 1] / o.streams;
            mhc_combine_bwd<<<x[1].nelem() / (C * o.streams), 256, 0, stream>>>(p(0),
                                                                                p(1),
                                                                                p(2),
                                                                                p(3),
                                                                                p(4),
                                                                                r(0),
                                                                                r(1),
                                                                                r(2),
                                                                                r(3),
                                                                                C,
                                                                                o.streams);
            break;
        }
        case Glm5Kernel::KdaDecay: {
            long n = x[1].nelem();
            decay_bwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                           p(1),
                                                           p(2),
                                                           p(3),
                                                           r(0),
                                                           r(1),
                                                           r(2),
                                                           n,
                                                           x[1].Sizes[2],
                                                           x[1].Sizes[3],
                                                           o.lower_bound);
            break;
        }
        case Glm5Kernel::KdaRule: {
            int T = x[1].Sizes[1], H = x[1].Sizes[2], D = x[1].Sizes[3], BH = x[1].Sizes[0] * H;
            const int* pos = x.size() == 7 ? x[6].get<int>() : nullptr;
#define KDA_BWD(DIM)                                                   \
    case DIM:                                                          \
        kda_fwd<DIM><<<BH, DIM, 0, stream>>>(p(1),                     \
                                             p(2),                     \
                                             p(3),                     \
                                             p(4),                     \
                                             p(5),                     \
                                             Ptr{nullptr, false},      \
                                             checkpoints.get<float>(), \
                                             pos,                      \
                                             T,                        \
                                             H);                       \
        kda_bwd<DIM><<<BH, DIM, 0, stream>>>(p(0),                     \
                                             p(1),                     \
                                             p(2),                     \
                                             p(3),                     \
                                             p(4),                     \
                                             p(5),                     \
                                             r(0),                     \
                                             r(1),                     \
                                             r(2),                     \
                                             r(3),                     \
                                             r(4),                     \
                                             checkpoints.get<float>(), \
                                             pos,                      \
                                             T,                        \
                                             H);                       \
        break
            switch (D) {
                KDA_BWD(8);
                KDA_BWD(16);
                KDA_BWD(32);
                KDA_BWD(64);
                KDA_BWD(128);
                default: throw std::runtime_error("unsupported KDA head dimension");
            }
#undef KDA_BWD
            break;
        }
        case Glm5Kernel::Clamp: {
            long n = x[1].nelem();
            clamp_bwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                           p(1),
                                                           r(0),
                                                           n,
                                                           o.clamp_min,
                                                           o.clamp_max,
                                                           o.fused_gate_up ? x[1].Sizes[x[1].Rank - 1] : 0);
            break;
        }
        case Glm5Kernel::CausalConv1d: {
            long n = x[1].nelem();
            conv_bwd<<<(n + 255) / 256, 256, 0, stream>>>(p(0),
                                                          p(1),
                                                          p(2),
                                                          x[3].get<int>(),
                                                          r(0),
                                                          r(1),
                                                          n,
                                                          x[1].Sizes[1],
                                                          x[1].Sizes[2],
                                                          x[2].Sizes[2]);
            break;
        }
    }
    CUDA_CHECK(cudaGetLastError());
}
void glm5_copy_gradient(const Tensor& src, Tensor& dst, bool accumulate, cudaStream_t stream) {
    copy_gradient<<<(src.nelem() + 255) / 256, 256, 0, stream>>>(ptr(src), ptr(dst), src.nelem(), accumulate);
    CUDA_CHECK(cudaGetLastError());
}
