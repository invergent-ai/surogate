// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "gated_delta_rule_v2.cuh"
#include "gdr_fwd_launchers.h"

namespace {

// ============================================================================
// Forward multi-kernel: Precompute (chunk-parallel)
// Grid: B*H*num_chunks. Computes A, M, u, w, S for each chunk.
// Saves results to forward workspace for state and output kernels.
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_fwd_precompute_wmma(
    float* __restrict__ fwd_workspace,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    int fwd_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int block_id = blockIdx.x;
    const int bh = block_id / num_chunks;
    const int chunk = block_id % num_chunks;
    const int b = bh / H;
    const int h = bh % H;
    const int Lp = kMaxC;

    const int cs = chunk * chunk_size;
    const int L = min(chunk_size, Tlen - cs);

    char* ws = reinterpret_cast<char*>(fwd_workspace) + (long)block_id * fwd_ws_stride;
    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);
    TQ* ws_u = reinterpret_cast<TQ*>(ws + fwl.u_off);
    TQ* ws_w = reinterpret_cast<TQ*>(ws + fwl.w_off);
    TQ* ws_k = reinterpret_cast<TQ*>(ws + fwl.k_off);
    TQ* ws_vnew_pre = reinterpret_cast<TQ*>(ws + fwl.vnew_pre_off);
    float* ws_gcum = reinterpret_cast<float*>(ws + fwl.gcum_off);

    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    buf1      = smem_k + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invk = smem_beta + Lp;
    float* smem_eg   = smem_invk + Lp;

    // Zero-fill
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        smem_k[idx] = from_float<TQ>(0.0f);
    }
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        buf2[idx] = from_float<TQ>(0.0f);
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        buf3[idx] = from_float<TQ>(0.0f);
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    if (tid < Lp) {
        smem_gcum[tid] = 0.0f;
        smem_beta[tid] = 0.0f;
        smem_eg[tid] = 0.0f;
    }
    __syncthreads();

    // Load k, v (vectorized for bf16/fp16 when possible).
    if constexpr (std::is_same_v<TQ, nv_bfloat16>) {
        if ((Kdim & 1) == 0) {
            const int k2 = Kdim >> 1;
            auto* smem_k2 = reinterpret_cast<__nv_bfloat162*>(smem_k);
            const auto* k2_ptr = reinterpret_cast<const __nv_bfloat162*>(k);
            for (int idx = tid; idx < L * k2; idx += nthr) {
                const int pos = idx / k2;
                const int kk2 = idx % k2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * k2 + kk2;
                smem_k2[pos * k2 + kk2] = k2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim, kk = idx % Kdim;
                const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                smem_k[pos * Kdim + kk] = k[gi];
            }
        }
        if ((Vdim & 1) == 0) {
            const int v2 = Vdim >> 1;
            auto* smem_v2 = reinterpret_cast<__nv_bfloat162*>(buf2);
            const auto* v2_ptr = reinterpret_cast<const __nv_bfloat162*>(v);
            for (int idx = tid; idx < L * v2; idx += nthr) {
                const int pos = idx / v2;
                const int vv2 = idx % v2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * v2 + vv2;
                smem_v2[pos * v2 + vv2] = v2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Vdim; idx += nthr) {
                const int pos = idx / Vdim, vv = idx % Vdim;
                buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
            }
        }
    } else if constexpr (std::is_same_v<TQ, half>) {
        if ((Kdim & 1) == 0) {
            const int k2 = Kdim >> 1;
            auto* smem_k2 = reinterpret_cast<__half2*>(smem_k);
            const auto* k2_ptr = reinterpret_cast<const __half2*>(k);
            for (int idx = tid; idx < L * k2; idx += nthr) {
                const int pos = idx / k2;
                const int kk2 = idx % k2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * k2 + kk2;
                smem_k2[pos * k2 + kk2] = k2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim, kk = idx % Kdim;
                const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
                smem_k[pos * Kdim + kk] = k[gi];
            }
        }
        if ((Vdim & 1) == 0) {
            const int v2 = Vdim >> 1;
            auto* smem_v2 = reinterpret_cast<__half2*>(buf2);
            const auto* v2_ptr = reinterpret_cast<const __half2*>(v);
            for (int idx = tid; idx < L * v2; idx += nthr) {
                const int pos = idx / v2;
                const int vv2 = idx % v2;
                const long gi2 = (((long)b * Tlen + cs + pos) * H + h) * v2 + vv2;
                smem_v2[pos * v2 + vv2] = v2_ptr[gi2];
            }
        } else {
            for (int idx = tid; idx < L * Vdim; idx += nthr) {
                const int pos = idx / Vdim, vv = idx % Vdim;
                buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
            }
        }
    } else {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < L; ++i) {
            const long gh = ((long)b * Tlen + cs + i) * H + h;
            acc += to_float(g[gh]);
            smem_gcum[i] = acc;
            smem_beta[i] = to_float(beta[gh]);
            smem_eg[i] = expf(acc);
        }
    }
    __syncthreads();

    // L2 norms + normalize (k only; q is normalized in output kernel).
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float kn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float kv2 = to_float(smem_k[pos * Kdim + kk]);
                kn2 += kv2 * kv2;
            }
            smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
        } else {
            smem_invk[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            smem_k[pos * Kdim + kk] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_invk[pos]));
        }
        __syncthreads();
    }

    // Save normalized k and gcum to workspace
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        ws_k[idx] = smem_k[idx];
    for (int idx = tid; idx < Lp; idx += nthr)
        ws_gcum[idx] = smem_gcum[idx];

    // k@k^T -> scratch1
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // Scale A
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i) {
            const float eg_j = smem_eg[j];
            const float exp_ij =
                (eg_j > 0.0f) ? (smem_eg[i] / eg_j) : expf(smem_gcum[i] - smem_gcum[j]);
            scratch1[idx] *= smem_beta[i] * exp_ij;
        } else {
            scratch1[idx] = 0.0f;
        }
    }
    __syncthreads();

    // M solve -> scratch2
    for (int idx = tid; idx < Lp * Lp; idx += nthr) scratch2[idx] = 0.0f;
    __syncthreads();
    for (int i = tid; i < Lp; i += nthr) scratch2[i * Lp + i] = 1.0f;
    __syncthreads();
    for (int row = 1; row < L; ++row) {
        for (int j = tid; j < row; j += nthr) {
            float s = 0.0f;
            for (int m = j; m < row; ++m)
                s += scratch1[row * Lp + m] * scratch2[m * Lp + j];
            scratch2[row * Lp + j] = -s;
        }
        __syncthreads();
    }

    // M->bf16 -> buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // beta*v -> buf2
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
    }
    __syncthreads();

    // M@bv -> scratch1 (u)
    wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
    __syncthreads();
    // Save u to workspace (bf16-truncated)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        ws_u[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    __syncthreads();

    // beta*k*exp(g) -> buf3
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * smem_eg[i] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // M@bkg -> scratch1 (w)
    wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();
    // Save w to workspace (bf16-truncated)
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        ws_w[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    // Keep vnew_pre pointer "used" for uniform typed layout.
    (void)ws_vnew_pre;
    __syncthreads();
}

} // namespace

// Launch wrapper
template<typename TQ, typename TG, typename TB>
void launch_gdr_fwd_precompute(
    float* fwd_workspace,
    const TQ* k, const TQ* v, const TG* g, const TB* beta,
    int fwd_ws_stride,
    int B, int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, bool use_qk_l2norm_in_kernel,
    int threads, std::size_t smem, cudaStream_t stream)
{
    CUDA_CHECK(cudaFuncSetAttribute(
        gdr_fwd_precompute_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem)));

    gdr_fwd_precompute_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, smem, stream>>>(
        fwd_workspace, k, v, g, beta,
        fwd_ws_stride, Tlen, H, Kdim, Vdim,
        num_chunks, chunk_size, use_qk_l2norm_in_kernel);
    CUDA_CHECK(cudaGetLastError());
}

// Explicit template instantiations (WMMA requires bf16 or half for TQ)
#define INSTANTIATE_PRECOMPUTE(TQ, TG, TB) \
    template void launch_gdr_fwd_precompute<TQ, TG, TB>( \
        float*, const TQ*, const TQ*, const TG*, const TB*, \
        int, int, int, int, int, int, int, int, bool, int, std::size_t, cudaStream_t);

#define INSTANTIATE_PRECOMPUTE_ALL_GB(TQ) \
    INSTANTIATE_PRECOMPUTE(TQ, float, float) \
    INSTANTIATE_PRECOMPUTE(TQ, float, nv_bfloat16) \
    INSTANTIATE_PRECOMPUTE(TQ, float, half) \
    INSTANTIATE_PRECOMPUTE(TQ, nv_bfloat16, float) \
    INSTANTIATE_PRECOMPUTE(TQ, nv_bfloat16, nv_bfloat16) \
    INSTANTIATE_PRECOMPUTE(TQ, nv_bfloat16, half) \
    INSTANTIATE_PRECOMPUTE(TQ, half, float) \
    INSTANTIATE_PRECOMPUTE(TQ, half, nv_bfloat16) \
    INSTANTIATE_PRECOMPUTE(TQ, half, half)

INSTANTIATE_PRECOMPUTE_ALL_GB(nv_bfloat16)
INSTANTIATE_PRECOMPUTE_ALL_GB(half)

#undef INSTANTIATE_PRECOMPUTE_ALL_GB
#undef INSTANTIATE_PRECOMPUTE
