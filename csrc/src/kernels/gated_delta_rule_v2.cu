// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Fully parallelized chunk-based gated delta rule kernels (v2).
// Same math and bf16 truncation points as the original, but all operations
// are distributed across threads for maximum GPU utilization.
//
// Key differences from v1:
//   - Forward: M matrix build parallelized across threads (was tid==0 only)
//   - Checkpoint: Fully parallelized (was tid==0 only)
//   - Backward: Fully parallelized (was 1-thread kernel)
//   - All intermediate data kept in shared memory where possible

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "kernels/kernels.h"
#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

#if defined(GDR_V2_COMPILE_FWD_ONLY) && defined(GDR_V2_COMPILE_BWD_ONLY)
#error "GDR_V2_COMPILE_FWD_ONLY and GDR_V2_COMPILE_BWD_ONLY are mutually exclusive."
#endif

namespace {

// ============================================================================
// Helpers (same as v1 to preserve exact numerics)
// ============================================================================

template<typename T>
__device__ __forceinline__ float to_float(T v) {
    return static_cast<float>(v);
}

template<>
__device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 v) {
    return __bfloat162float(v);
}

template<>
__device__ __forceinline__ float to_float<half>(half v) {
    return __half2float(v);
}

template<typename T>
__device__ __forceinline__ T from_float(float v);

template<>
__device__ __forceinline__ float from_float<float>(float v) {
    return v;
}

template<>
__device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template<>
__device__ __forceinline__ half from_float<half>(float v) {
    return __float2half(v);
}

template<typename T>
__device__ __forceinline__ float bf16_trunc(float x) {
    if constexpr (std::is_same_v<T, float>) {
        return x;
    } else {
        return to_float(from_float<T>(x));
    }
}

constexpr int kMaxC = 64;  // max chunk size

using namespace nvcuda;

// ============================================================================
// WMMA 16×16×16 matmul helpers (bf16/fp16 inputs, fp32 accumulator)
// 4 warps (128 threads), each warp handles ceil(M_tiles/4) row-blocks.
// M, N, K must be multiples of 16.
// ============================================================================

// C[M×N] = A[M×K] @ B[K×N], row-major @ row-major
template<typename TQ>
__device__ void wmma_nn(
    const TQ* A, int ldA, const TQ* B, int ldB,
    float* C, int ldC, int M, int N, int K) {
    const int wid = threadIdx.x / 32;
    const int nw = blockDim.x / 32;
    for (int m = wid; m < M/16; m += nw)
        for (int n = 0; n < N/16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < K/16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, TQ, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, TQ, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + m*16*ldA + kk*16, ldA);
                wmma::load_matrix_sync(bf, B + kk*16*ldB + n*16, ldB);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(C + m*16*ldC + n*16, acc, ldC, wmma::mem_row_major);
        }
}

// C[M×N] = A[M×K] @ B^T, where B is [N×K] row-major
template<typename TQ>
__device__ void wmma_nt(
    const TQ* A, int ldA, const TQ* B, int ldB,
    float* C, int ldC, int M, int N, int K) {
    const int wid = threadIdx.x / 32;
    const int nw = blockDim.x / 32;
    for (int m = wid; m < M/16; m += nw)
        for (int n = 0; n < N/16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < K/16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, TQ, wmma::row_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, TQ, wmma::col_major> bf;
                wmma::load_matrix_sync(af, A + m*16*ldA + kk*16, ldA);
                wmma::load_matrix_sync(bf, B + n*16*ldB + kk*16, ldB);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(C + m*16*ldC + n*16, acc, ldC, wmma::mem_row_major);
        }
}

// C[M×N] = A^T @ B, where A is [K×M] row-major (A^T is [M×K])
template<typename TQ>
__device__ void wmma_tn(
    const TQ* A, int ldA, const TQ* B, int ldB,
    float* C, int ldC, int M, int N, int K) {
    const int wid = threadIdx.x / 32;
    const int nw = blockDim.x / 32;
    for (int m = wid; m < M/16; m += nw)
        for (int n = 0; n < N/16; ++n) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < K/16; ++kk) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, TQ, wmma::col_major> af;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, TQ, wmma::row_major> bf;
                wmma::load_matrix_sync(af, A + kk*16*ldA + m*16, ldA);
                wmma::load_matrix_sync(bf, B + kk*16*ldB + n*16, ldB);
                wmma::mma_sync(acc, af, bf, acc);
            }
            wmma::store_matrix_sync(C + m*16*ldC + n*16, acc, ldC, wmma::mem_row_major);
        }
}

__device__ __forceinline__ float warp_reduce_sum_f32(float x) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffff, x, offset);
    }
    return x;
}

#if !defined(GDR_V2_COMPILE_BWD_ONLY)
// ============================================================================
// Forward multi-kernel workspace layout (per chunk)
// ============================================================================
struct FwdWorkspaceLayout {
    int u_off;         // [Lp×V] — M @ (beta*v) (bf16-truncated float)
    int w_off;         // [Lp×K] — M @ (beta*k*exp(g)) (bf16-truncated float)
    int S_off;         // [Lp×Lp] — causal masked gated q@k^T (bf16-truncated float)
    int k_off;         // [Lp×K] — normalized k (float, from bf16)
    int vnew_pre_off;  // [Lp×V] — u - w@h (filled by state kernel)
    int gcum_off;      // [Lp] — cumulative gate values
    int total;
};

__host__ __device__ FwdWorkspaceLayout make_fwd_ws(int Lp, int Kdim, int Vdim) {
    FwdWorkspaceLayout l;
    int off = 0;
    l.u_off = off; off += Lp * Vdim;
    l.w_off = off; off += Lp * Kdim;
    l.S_off = off; off += Lp * Lp;
    l.k_off = off; off += Lp * Kdim;
    l.vnew_pre_off = off; off += Lp * Vdim;
    l.gcum_off = off; off += Lp;
    l.total = off;
    return l;
}

// ============================================================================
// WMMA forward kernel — tensor core accelerated
// Requires K, V multiples of 16 and ≤ 64.
//
// Shared memory layout (K=V=64):
//   smem_h      [K×V] float       = 16 KB  (persistent state)
//   scratch1    [64×64] float     = 16 KB  (WMMA output 1)
//   scratch2    [64×64] float     = 16 KB  (WMMA output 2)
//   smem_k      [64×K] TQ         =  8 KB
//   smem_q      [64×K] TQ         =  8 KB
//   buf1        [64×64] TQ        =  8 KB  (M_bf16 → h_bf16 → S)
//   buf2        [64×V] TQ         =  8 KB  (v → bv → u → v_new_pre → v_scaled)
//   buf3        [64×K] TQ         =  8 KB  (bkg → w)
//   scalars     [4×64] float      =  1 KB
//   Total: ~89 KB
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_fwd_wmma(
    TQ* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    float* __restrict__ fwd_checkpoints,   // [B, H, num_chunks+1, K, V] or nullptr
    const TQ* __restrict__ q,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, float scale, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = (long)b * H + h;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;  // always pad to 64

    // Shared memory
    extern __shared__ char smem_raw[];
    float* smem_h    = (float*)smem_raw;
    float* scratch1  = smem_h + Kdim * Vdim;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;

    // Init state
    for (long idx = tid; idx < kv; idx += nthr)
        smem_h[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    __syncthreads();

    // Save checkpoint[0] = initial state
    if (fwd_checkpoints) {
        float* cp_base = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            cp_base[idx] = smem_h[idx];
    }
    __syncthreads();

    for (int cs = 0; cs < Tlen; cs += chunk_size) {
        const int L = min(chunk_size, Tlen - cs);

        // --- Zero-fill all buffers for WMMA padding ---
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            smem_k[idx] = from_float<TQ>(0.0f);
            smem_q[idx] = from_float<TQ>(0.0f);
        }
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        if (tid < Lp) { smem_gcum[tid] = 0.0f; smem_beta[tid] = 0.0f; }
        __syncthreads();

        // --- Load k, q, v from global ---
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k[gi];
            smem_q[pos * Kdim + kk] = q[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Vdim + vv;
            buf2[pos * Vdim + vv] = v[gi];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_gcum[i] = acc;
                smem_beta[i] = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // --- L2 norms + normalize ---
        for (int pos = tid; pos < L; pos += nthr) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv; kn2 += kv2 * kv2;
                }
                smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
            }
        }
        __syncthreads();
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim, kk = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_invk[pos]));
                smem_q[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_invq[pos]));
            }
            __syncthreads();
        }

        // --- Phase 2: A = beta * exp(g) * (k @ k^T), then solve M ---
        // kkt = k @ k^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // Scale A and zero upper triangle
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // Solve M in scratch2: M[i,i]=1, M[i,j] = -Σ_{m=j}^{i-1} A[i,m]*M[m,j]
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

        // Convert M → bf16 in buf1 with truncation
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // --- Phase 3: bv = beta*v, then u = M @ bv ---
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // --- Phase 4: bkg = beta*k*exp(g), then w = M @ bkg ---
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // --- Phase 5: h_bf16, then v_new_pre = u - w @ h_bf16 ---
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(smem_h[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        // wh = w[Lp×K] @ h_bf16[K×V] → scratch1[Lp×V]
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]));
        __syncthreads();

        // --- Phase 6: term1 = q @ h_bf16 → scratch1[Lp×V] ---
        // h_bf16 is still in buf1[K×V]
        wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // --- Phase 7: S = q @ k^T → scratch2, mask+gate → buf1 ---
        wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch2, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = (i < L && j <= i) ?
                bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch2[idx]) : 0.0f;
            buf1[idx] = from_float<TQ>(val);
        }
        __syncthreads();

        // --- Phase 8: term2 = S @ v_new_pre → scratch2[Lp×V] ---
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch2, Vdim, Lp, Vdim, Lp);
        __syncthreads();

        // --- Phase 9: Write output ---
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim, vv = idx % Vdim;
            const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + vv;
            out[oi] = from_float<TQ>(
                scale * (expf(smem_gcum[i]) * scratch1[i * Vdim + vv]
                        + scratch2[i * Vdim + vv]));
        }

        // --- Phase 10: State update ---
        // v_scaled[i,v] = bf16(v_new_pre[i,v] * exp(g_last - g_cum[i]))
        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ?
                bf16_trunc<TQ>(to_float(buf2[idx]) * expf(g_last - smem_gcum[i])) : 0.0f;
            buf2[idx] = from_float<TQ>(val);
        }
        __syncthreads();
        // K^T @ v_scaled → scratch1[K×V]
        wmma_tn<TQ>(smem_k, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        const float eg = expf(g_last);
        for (long idx = tid; idx < kv; idx += nthr)
            smem_h[idx] = eg * smem_h[idx] + scratch1[idx];
        __syncthreads();

        // Save checkpoint for this chunk boundary
        if (fwd_checkpoints) {
            const int chunk_idx = cs / chunk_size + 1;
            float* cp = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv + (long)chunk_idx * kv;
            for (long idx = tid; idx < kv; idx += nthr)
                cp[idx] = smem_h[idx];
            __syncthreads();
        }
    }

    // Write final state
    for (long idx = tid; idx < kv; idx += nthr) {
        final_state[bh * kv + idx] = smem_h[idx];
        state_scratch[bh * kv + idx] = smem_h[idx];
    }
}

// ============================================================================
// Forward multi-kernel: Precompute (chunk-parallel)
// Grid: B*H*num_chunks. Computes A, M, u, w, S for each chunk.
// Saves results to forward workspace for state and output kernels.
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_fwd_precompute_wmma(
    float* __restrict__ fwd_workspace,
    const TQ* __restrict__ q,
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
    const bool tail_chunk = (L < chunk_size);

    float* ws = fwd_workspace + (long)block_id * fwd_ws_stride;
    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);

    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Zero-fill
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        smem_k[idx] = from_float<TQ>(0.0f);
        smem_q[idx] = from_float<TQ>(0.0f);
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
    }
    __syncthreads();

    // Load k, q, v
    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
        smem_k[pos * Kdim + kk] = k[gi];
        smem_q[pos * Kdim + kk] = q[gi];
    }
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int pos = idx / Vdim, vv = idx % Vdim;
        buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < L; ++i) {
            const long gh = ((long)b * Tlen + cs + i) * H + h;
            acc += to_float(g[gh]);
            smem_gcum[i] = acc;
            smem_beta[i] = to_float(beta[gh]);
        }
    }
    __syncthreads();

    // L2 norms + normalize
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f, kn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float qv = to_float(smem_q[pos * Kdim + kk]);
                float kv2 = to_float(smem_k[pos * Kdim + kk]);
                qn2 += qv * qv; kn2 += kv2 * kv2;
            }
            smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
            smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
        } else {
            smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            smem_k[pos * Kdim + kk] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_invk[pos]));
            smem_q[pos * Kdim + kk] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // Save normalized k and gcum to workspace
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        ws[fwl.k_off + idx] = to_float(smem_k[idx]);
    for (int idx = tid; idx < Lp; idx += nthr)
        ws[fwl.gcum_off + idx] = smem_gcum[idx];

    // k@k^T → scratch1
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // Scale A
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i)
            scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
        else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();

    // M solve → scratch2
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

    // M→bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // beta*v → buf2
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
    }
    __syncthreads();

    // M@bv → scratch1 (u)
    wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
    __syncthreads();
    // Save u to workspace (bf16-truncated)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        ws[fwl.u_off + idx] = bf16_trunc<TQ>(scratch1[idx]);
    __syncthreads();

    // beta*k*exp(g) → buf3
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // M@bkg → scratch1 (w)
    wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();
    // Save w to workspace (bf16-truncated)
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        ws[fwl.w_off + idx] = bf16_trunc<TQ>(scratch1[idx]);
    __syncthreads();

    // S = q@k^T → scratch2
    wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch2, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // Mask+gate → save S to workspace (bf16-truncated)
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        float val = (i < L && j <= i) ?
            bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch2[idx]) : 0.0f;
        ws[fwl.S_off + idx] = val;
    }
    __syncthreads();
}

// ============================================================================
// Forward multi-kernel: State propagation (sequential)
// Grid: dim3(B, H). Propagates state h across chunks using precomputed u, w.
// Saves vnew_pre to workspace for output kernel, checkpoints for backward.
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_fwd_state_wmma(
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    float* __restrict__ fwd_checkpoints,
    float* __restrict__ fwd_workspace,
    const float* __restrict__ initial_state,
    int fwd_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int bh = b * H + h;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;

    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);

    extern __shared__ char smem_raw[];
    float* smem_h    = (float*)smem_raw;                            // [K×V]
    float* scratch1  = smem_h + Kdim * Vdim;                        // [Lp×V]
    TQ*    buf_w     = (TQ*)(scratch1 + Lp * Vdim);                 // [Lp×K]
    TQ*    buf_h     = buf_w + Lp * Kdim;                            // [K×V]
    TQ*    buf_k     = buf_h + Kdim * Vdim;                          // [Lp×K]
    TQ*    buf_vnp   = buf_k + Lp * Kdim;                            // [Lp×V]
    float* smem_gcum = (float*)(buf_vnp + Lp * Vdim);               // [Lp]

    // Init state
    for (long idx = tid; idx < kv; idx += nthr)
        smem_h[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    __syncthreads();

    // Save checkpoint[0]
    if (fwd_checkpoints) {
        float* cp_base = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            cp_base[idx] = smem_h[idx];
    }
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const int block_id = bh * num_chunks + chunk;
        float* ws = fwd_workspace + (long)block_id * fwd_ws_stride;

        // Load w, k, gcum from workspace
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            buf_w[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.w_off + idx]));
            buf_k[idx] = from_float<TQ>(bf16_trunc<TQ>(ws[fwl.k_off + idx]));
        }
        for (int idx = tid; idx < Lp; idx += nthr)
            smem_gcum[idx] = ws[fwl.gcum_off + idx];
        __syncthreads();

        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        const float eg = expf(g_last);

        // h→bf16
        for (long idx = tid; idx < kv; idx += nthr)
            buf_h[idx] = from_float<TQ>(bf16_trunc<TQ>(smem_h[idx]));
        __syncthreads();

        // w@h → scratch1[Lp×V]
        wmma_nn<TQ>(buf_w, Kdim, buf_h, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // vnew_pre = bf16_trunc(u - wh), compute v_scaled for state update
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float u_val = ws[fwl.u_off + idx];  // already bf16-truncated
            float vnew = bf16_trunc<TQ>(u_val - scratch1[idx]);
            ws[fwl.vnew_pre_off + idx] = vnew;  // save for output kernel
            float e_i = (i < L) ? expf(g_last - smem_gcum[i]) : 0.0f;
            buf_vnp[idx] = from_float<TQ>(bf16_trunc<TQ>(vnew * e_i));
        }
        __syncthreads();

        // delta = k^T @ v_scaled → scratch1[K×V]
        wmma_tn<TQ>(buf_k, Kdim, buf_vnp, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();

        // State update
        for (long idx = tid; idx < kv; idx += nthr)
            smem_h[idx] = eg * smem_h[idx] + scratch1[idx];
        __syncthreads();

        // Save checkpoint
        if (fwd_checkpoints) {
            const int chunk_idx = chunk + 1;
            float* cp = fwd_checkpoints + bh * (long)(num_chunks + 1) * kv + (long)chunk_idx * kv;
            for (long idx = tid; idx < kv; idx += nthr)
                cp[idx] = smem_h[idx];
            __syncthreads();
        }
    }

    // Write final state
    for (long idx = tid; idx < kv; idx += nthr) {
        final_state[bh * kv + idx] = smem_h[idx];
        state_scratch[bh * kv + idx] = smem_h[idx];
    }
}

// ============================================================================
// Forward multi-kernel: Output computation (chunk-parallel)
// Grid: B*H*num_chunks. Computes output using precomputed S, vnew_pre
// and state checkpoints.
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_fwd_output_wmma(
    TQ* __restrict__ out,
    const TQ* __restrict__ q,
    const float* __restrict__ fwd_checkpoints,
    const float* __restrict__ fwd_workspace,
    int fwd_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int block_id = blockIdx.x;
    const int bh = block_id / num_chunks;
    const int chunk = block_id % num_chunks;
    const int b = bh / H;
    const int h = bh % H;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;

    const int cs = chunk * chunk_size;
    const int L = min(chunk_size, Tlen - cs);
    const bool tail_chunk = (L < chunk_size);

    const float* ws = fwd_workspace + (long)block_id * fwd_ws_stride;
    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);

    // h_in from checkpoint (state entering this chunk)
    const float* h_in = fwd_checkpoints + (long)bh * (num_chunks + 1) * kv + (long)chunk * kv;

    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;                             // [Lp×V]
    float* scratch2  = scratch1 + Lp * Vdim;                         // [Lp×V]
    TQ*    smem_q    = (TQ*)(scratch2 + Lp * Vdim);                  // [Lp×K]
    TQ*    buf_h     = smem_q + Lp * Kdim;                            // [K×V]
    TQ*    buf_S     = buf_h + Kdim * Vdim;                           // [Lp×Lp]
    TQ*    buf_vnp   = buf_S + Lp * Lp;                               // [Lp×V]
    float* smem_gcum = (float*)(buf_vnp + Lp * Vdim);                // [Lp]
    float* smem_invq = smem_gcum + Lp;                                // [Lp]

    // Zero-fill q
    for (int idx = tid; idx < Lp * Kdim; idx += nthr)
        smem_q[idx] = from_float<TQ>(0.0f);
    if (tid < Lp) smem_gcum[tid] = 0.0f;
    __syncthreads();

    // Load q from global
    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        smem_q[pos * Kdim + kk] = q[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
    }
    // Load gcum from workspace
    for (int idx = tid; idx < Lp; idx += nthr)
        smem_gcum[idx] = ws[fwl.gcum_off + idx];
    __syncthreads();

    // L2 normalize q
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float qv = to_float(smem_q[pos * Kdim + kk]);
                qn2 += qv * qv;
            }
            smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
        } else {
            smem_invq[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // Load h_in → bf16
    for (long idx = tid; idx < kv; idx += nthr)
        buf_h[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
    __syncthreads();

    // term1 = q @ h_bf16 → scratch1[Lp×V]
    wmma_nn<TQ>(smem_q, Kdim, buf_h, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
    __syncthreads();

    // Load S and vnew_pre from workspace → bf16
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf_S[idx] = from_float<TQ>(ws[fwl.S_off + idx]);
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        buf_vnp[idx] = from_float<TQ>(ws[fwl.vnew_pre_off + idx]);
    __syncthreads();

    // term2 = S @ vnew_pre → scratch2[Lp×V]
    wmma_nn<TQ>(buf_S, Lp, buf_vnp, Vdim, scratch2, Vdim, Lp, Vdim, Lp);
    __syncthreads();

    // Write output: o = scale * (exp(gcum[i]) * term1 + term2)
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int i = idx / Vdim, vv = idx % Vdim;
        const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + vv;
        out[oi] = from_float<TQ>(
            scale * (expf(smem_gcum[i]) * scratch1[i * Vdim + vv]
                    + scratch2[i * Vdim + vv]));
    }
}

// ============================================================================
// Checkpoint kernel — WMMA tensor-core accelerated
// Computes state at each chunk boundary (no output computation).
// Shared memory: h[K×V]f + scratch1[C×C]f + scratch2[C×C]f
//   + k[C×K]TQ + buf1[C×C]TQ + buf2[C×V]TQ + buf3[C×K]TQ + scalars[3×C]f
// ============================================================================
#endif  // !GDR_V2_COMPILE_BWD_ONLY

#if !defined(GDR_V2_COMPILE_FWD_ONLY)
template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_checkpoint_wmma(
    float* __restrict__ checkpoints,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = (long)b * H + h;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;

    float* cp_base = checkpoints + bh * (long)(num_chunks + 1) * kv;

    extern __shared__ char smem_raw[];
    float* smem_h    = (float*)smem_raw;
    float* scratch1  = smem_h + Kdim * Vdim;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    buf1      = smem_k + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invk = smem_beta + Lp;

    // Checkpoint 0 = initial state
    for (long idx = tid; idx < kv; idx += nthr)
        smem_h[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    for (long idx = tid; idx < kv; idx += nthr)
        cp_base[idx] = smem_h[idx];
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);

        // Zero-fill buffers
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            smem_k[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        if (tid < Lp) { smem_gcum[tid] = 0.0f; smem_beta[tid] = 0.0f; }
        __syncthreads();

        // Load k, v
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            smem_k[pos * Kdim + kk] = k[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_gcum[i] = acc;
                smem_beta[i] = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norm + normalize k
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

        // A = beta * exp(g) * (k @ k^T)
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // Solve M
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

        // M → bf16 in buf1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // bv = beta*v, then u = M @ bv
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // bkg = beta*k*exp(g), then w = M @ bkg
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // h_bf16, then v_new_pre = u - w @ h_bf16
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(smem_h[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]));
        __syncthreads();

        // State update: v_scaled = bf16(v_new_pre * exp(g_last - g_cum[i]))
        const float g_last = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ?
                bf16_trunc<TQ>(to_float(buf2[idx]) * expf(g_last - smem_gcum[i])) : 0.0f;
            buf2[idx] = from_float<TQ>(val);
        }
        __syncthreads();
        // K^T @ v_scaled → scratch1[K×V]
        wmma_tn<TQ>(smem_k, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        const float eg = expf(g_last);
        for (long idx = tid; idx < kv; idx += nthr)
            smem_h[idx] = eg * smem_h[idx] + scratch1[idx];
        __syncthreads();

        // Write checkpoint
        float* cp_out = cp_base + (long)(chunk + 1) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            cp_out[idx] = smem_h[idx];
        __syncthreads();
    }
}

#endif  // !GDR_V2_COMPILE_FWD_ONLY

#if !defined(GDR_V2_COMPILE_BWD_ONLY)
// ============================================================================
// Forward kernel — fully parallelized (scalar fallback)
// ============================================================================
//
// Shared memory layout (for K=V=64, chunk_size=64):
//   smem_state  [K×V] float    = 16 KB
//   smem_M      [C×C] float    = 16 KB  (reused as S/h_bf16 in output phase)
//   smem_k      [C×K] TQ       =  8 KB  (normalized K)
//   smem_q      [C×K] TQ       =  8 KB  (normalized Q)
//   smem_v      [C×V] TQ       =  8 KB  (raw V, reused as S in output phase)
//   smem_u      [C×V] TQ       =  8 KB  (u then v_new_pre)
//   smem_w      [C×K] TQ       =  8 KB  (w, reused as h_bf16 in output phase)
//   scalars     [5×C] float    =  1.25KB (g_cum, beta, inv_q, inv_k, A_row_cache)
// Total: ~73 KB
//
template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_fwd_v2(
    TQ* __restrict__ out,
    float* __restrict__ final_state,
    float* __restrict__ state_scratch,
    const TQ* __restrict__ q,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const long bh = static_cast<long>(b) * H + h;
    const long kv = static_cast<long>(Kdim) * Vdim;

    // --- Shared memory pointers ---
    extern __shared__ char smem_raw[];
    float*  smem_state = reinterpret_cast<float*>(smem_raw);
    float*  smem_M     = smem_state + Kdim * Vdim;
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_M + kMaxC * kMaxC);
    TQ*     smem_q     = smem_k + kMaxC * Kdim;
    TQ*     smem_v     = smem_q + kMaxC * Kdim;
    TQ*     smem_u     = smem_v + kMaxC * Vdim;
    TQ*     smem_w     = smem_u + kMaxC * Vdim;
    float*  smem_g_cum = reinterpret_cast<float*>(smem_w + kMaxC * Kdim);
    float*  smem_beta  = smem_g_cum + kMaxC;
    float*  smem_inv_q = smem_beta  + kMaxC;
    float*  smem_inv_k = smem_inv_q + kMaxC;

    // Initialize state from initial_state or zeros
    for (long idx = tid; idx < kv; idx += nthreads) {
        smem_state[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int cs = 0; cs < Tlen; cs += chunk_size) {
        const int L = min(chunk_size, Tlen - cs);

        // ================================================================
        // Phase 1: Cooperative load of k, q, v + compute scalars
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k[gi];
            smem_q[pos * Kdim + kk] = q[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Vdim + vv;
            smem_v[pos * Vdim + vv] = v[gi];
        }
        // g_cum and beta — single thread (L ≤ 64, negligible)
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norms — one thread per position
        for (int pos = tid; pos < L; pos += nthreads) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv;
                    kn2 += kv2 * kv2;
                }
                smem_inv_q[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_inv_k[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_inv_q[pos] = 1.0f;
                smem_inv_k[pos] = 1.0f;
            }
        }
        __syncthreads();

        // Normalize k and q in-place (bf16 truncation)
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthreads) {
                const int pos = idx / Kdim;
                const int kk  = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_inv_k[pos]));
                smem_q[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_inv_q[pos]));
            }
            __syncthreads();
        }

        // ================================================================
        // Phase 2: Build M matrix (triangular solve)
        //   M[i,i] = 1,  M[i,j] = bf16(-Σ_{m=j}^{i-1} A[i,m]*M[m,j])
        //   where A[i,m] = beta[i] * dot(k_norm[i], k_norm[m]) * exp(g_cum[i] - g_cum[m])
        //
        //   Process row by row (sequential across rows).
        //   Within each row, parallelize across columns j.
        // ================================================================

        // Zero M
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();

        // Set diagonal
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            // Each thread computes M[row, j] for some j < row
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    // A[row, m]
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    float a_rm = smem_beta[row] * dot_k
                               * expf(smem_g_cum[row] - smem_g_cum[m]);
                    s += a_rm * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // ================================================================
        // Phase 3: u = M @ (beta * V)   [L×V]
        //   u[i,v] = bf16( Σ_m M[i,m] * bf16(V[m,v] * beta[m]) )
        // ================================================================
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float vb = bf16_trunc<TQ>(to_float(smem_v[m * Vdim + vv]) * smem_beta[m]);
                acc += smem_M[i * kMaxC + m] * vb;
            }
            smem_u[i * Vdim + vv] = from_float<TQ>(bf16_trunc<TQ>(acc));
        }
        __syncthreads();

        // ================================================================
        // Phase 4: w = M @ (beta * k_norm * exp(g_cum))   [L×K]
        //   w[i,k] = bf16( Σ_m M[i,m] * bf16(k_norm[m,k] * beta[m] * exp(g_cum[m])) )
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int i  = idx / Kdim;
            const int kk = idx % Kdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float kbg = bf16_trunc<TQ>(
                    to_float(smem_k[m * Kdim + kk]) * smem_beta[m] * expf(smem_g_cum[m]));
                acc += smem_M[i * kMaxC + m] * kbg;
            }
            smem_w[i * Kdim + kk] = from_float<TQ>(bf16_trunc<TQ>(acc));
        }
        __syncthreads();

        // ================================================================
        // Phase 5: v_new_pre = bf16(u - w @ bf16(state))   [L×V]
        //   Stored in smem_u (overwrites u).
        // ================================================================
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;
            float wh = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float h_val = bf16_trunc<TQ>(smem_state[kk * Vdim + vv]);
                wh += to_float(smem_w[i * Kdim + kk]) * h_val;
            }
            float u_val = to_float(smem_u[i * Vdim + vv]);
            smem_u[i * Vdim + vv] = from_float<TQ>(bf16_trunc<TQ>(u_val - wh));
        }
        __syncthreads();

        // ================================================================
        // Phase 6: Output computation
        //   out[i,v] = scale * (term1 + term2)
        //   term1 = exp(g_cum[i]) * Σ_k q_norm[i,k] * bf16(state[k,v])
        //   term2 = Σ_{j≤i} bf16(exp(g_cum[i]-g_cum[j]) * dot(q[i],k[j])) * v_new_pre[j,v]
        //
        //   We reuse smem_v and smem_w as scratch (no longer needed).
        //   smem_v → S[L×L] bf16 (attention scores, lower triangular)
        //   smem_w → h_bf16[K×V] bf16
        // ================================================================
        TQ* smem_S     = smem_v;   // [L×L] bf16 — fits in smem_v space (L*V == L*L for K=V)
        TQ* smem_h_bf16 = smem_w;  // [K×V] bf16

        // Prepare bf16-truncated state
        for (int idx = tid; idx < Kdim * Vdim; idx += nthreads) {
            smem_h_bf16[idx] = from_float<TQ>(smem_state[idx]);
        }
        __syncthreads();

        // Build S matrix (lower triangular attention scores)
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            if (j <= i) {
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_qk += to_float(smem_q[i * Kdim + kk])
                            * to_float(smem_k[j * Kdim + kk]);
                }
                smem_S[i * L + j] = from_float<TQ>(
                    expf(smem_g_cum[i] - smem_g_cum[j]) * dot_qk);
            } else {
                smem_S[i * L + j] = from_float<TQ>(0.0f);
            }
        }
        __syncthreads();

        // Compute output: term1 + term2
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;

            // term1 = exp(g_cum[i]) * Q_norm[i,:] @ h_bf16[:,v]
            float t1 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                t1 += to_float(smem_q[i * Kdim + kk])
                    * to_float(smem_h_bf16[kk * Vdim + vv]);
            }
            t1 *= expf(smem_g_cum[i]);

            // term2 = Σ_{j≤i} S[i,j] * v_new_pre[j,v]
            float t2 = 0.0f;
            for (int j = 0; j <= i; ++j) {
                t2 += to_float(smem_S[i * L + j])
                    * to_float(smem_u[j * Vdim + vv]);
            }

            const long oi = (((long)b * Tlen + cs + i) * H + h) * Vdim + vv;
            out[oi] = from_float<TQ>((t1 + t2) * scale);
        }
        __syncthreads();

        // ================================================================
        // Phase 7: State update (in-place)
        //   state[k,v] = exp(g_last) * state[k,v]
        //              + Σ_i k_norm[i,k] * bf16(v_new_pre[i,v] * exp(g_last - g_cum[i]))
        // ================================================================
        const float g_last = smem_g_cum[L - 1];
        const float eg_last = expf(g_last);

        for (int idx = tid; idx < Kdim * Vdim; idx += nthreads) {
            const int kk = idx / Vdim;
            const int vv = idx % Vdim;
            float acc = smem_state[kk * Vdim + vv] * eg_last;
            for (int i = 0; i < L; ++i) {
                float v_scaled = bf16_trunc<TQ>(
                    to_float(smem_u[i * Vdim + vv]) * expf(g_last - smem_g_cum[i]));
                acc += to_float(smem_k[i * Kdim + kk]) * v_scaled;
            }
            smem_state[kk * Vdim + vv] = acc;
        }
        __syncthreads();
    }

    // Write final state to global memory
    for (long idx = tid; idx < kv; idx += nthreads) {
        final_state[bh * kv + idx] = smem_state[idx];
    }
}

#endif  // !GDR_V2_COMPILE_BWD_ONLY

#if !defined(GDR_V2_COMPILE_FWD_ONLY)
// ============================================================================
// Checkpoint kernel — fully parallelized
// Recomputes state at each chunk boundary for the backward pass.
// Same math as forward state update (phases 2-4-5-7) without output.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_checkpoint_v2(
    float* __restrict__ checkpoints,
    const TQ* __restrict__ k,
    const TQ* __restrict__ v,
    const TG* __restrict__ g,
    const TB* __restrict__ beta,
    const float* __restrict__ initial_state,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    bool use_qk_l2norm_in_kernel) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;

    extern __shared__ char smem_raw[];
    float*  smem_M     = reinterpret_cast<float*>(smem_raw);
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_M + kMaxC * kMaxC);
    TQ*     smem_v_loc = smem_k + kMaxC * Kdim;
    TQ*     smem_u     = smem_v_loc + kMaxC * Vdim;
    TQ*     smem_w     = smem_u + kMaxC * Vdim;
    float*  smem_g_cum = reinterpret_cast<float*>(smem_w + kMaxC * Kdim);
    float*  smem_beta  = smem_g_cum + kMaxC;
    float*  smem_inv_k = smem_beta  + kMaxC;

    // Checkpoint 0 = initial state
    for (long idx = tid; idx < kv; idx += nthreads) {
        cp_base[idx] = initial_state ? initial_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        float* s_in  = cp_base + static_cast<long>(chunk) * kv;
        float* s_out = cp_base + static_cast<long>(chunk + 1) * kv;

        // Load k, v
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            smem_k[pos * Kdim + kk] = k[(((long)b * Tlen + cs + pos) * H + h) * Kdim + kk];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            smem_v_loc[pos * Vdim + vv] = v[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta[gh]);
            }
        }
        __syncthreads();

        // L2 norms + normalize k in-place
        for (int pos = tid; pos < L; pos += nthreads) {
            if (use_qk_l2norm_in_kernel) {
                float kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    kn2 += kv2 * kv2;
                }
                smem_inv_k[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_inv_k[pos] = 1.0f;
            }
        }
        __syncthreads();

        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthreads) {
                const int pos = idx / Kdim;
                const int kk  = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_inv_k[pos]));
            }
            __syncthreads();
        }

        // Build M matrix
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    float a_rm = smem_beta[row] * dot_k
                               * expf(smem_g_cum[row] - smem_g_cum[m]);
                    s += a_rm * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // Compute u, w, v_new_pre and update state -> s_out
        const float g_last = smem_g_cum[L - 1];
        const float eg_last = expf(g_last);

        // s_out = eg_last * s_in
        for (long idx = tid; idx < kv; idx += nthreads) {
            s_out[idx] = s_in[idx] * eg_last;
        }
        __syncthreads();

        // For each position i: compute v_new and accumulate into s_out
        for (int i = 0; i < L; ++i) {
            const float e_i = expf(g_last - smem_g_cum[i]);
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                // u_i_v
                float u_val = 0.0f;
                for (int m = 0; m <= i; ++m) {
                    float vb = bf16_trunc<TQ>(
                        to_float(smem_v_loc[m * Vdim + vv]) * smem_beta[m]);
                    u_val += smem_M[i * kMaxC + m] * vb;
                }
                u_val = bf16_trunc<TQ>(u_val);

                // w_i @ state -> wh_val
                float wh_val = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float w_ik = 0.0f;
                    for (int m = 0; m <= i; ++m) {
                        float kbg = bf16_trunc<TQ>(
                            to_float(smem_k[m * Kdim + kk]) * smem_beta[m]
                            * expf(smem_g_cum[m]));
                        w_ik += smem_M[i * kMaxC + m] * kbg;
                    }
                    w_ik = bf16_trunc<TQ>(w_ik);
                    float h_cs = bf16_trunc<TQ>(s_in[kk * Vdim + vv]);
                    wh_val += w_ik * h_cs;
                }

                float v_new_pre = bf16_trunc<TQ>(u_val - wh_val);
                float v_new = bf16_trunc<TQ>(v_new_pre * e_i);

                for (int kk = 0; kk < Kdim; ++kk) {
                    float ki = to_float(smem_k[i * Kdim + kk]);
                    atomicAdd(&s_out[kk * Vdim + vv], ki * v_new);
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Backward kernel — WMMA tensor-core accelerated
//
// Uses checkpoints (from checkpoint kernel) and a global workspace.
// WMMA accelerates: A=k@k^T, M@bkg, M@bv, W@h, and the key optimization:
//   Batched attention gradient via S=q@k^T, dS=d_out@vnew^T, WMMA matmuls
//   for DQ/DK/DU from attention gradient.
// Shared memory: scratch1[C×C]f + scratch2[C×C]f
//   + k[C×K]TQ + q[C×K]TQ + buf1[C×C]TQ + buf2[C×V]TQ + buf3[C×K]TQ
//   + scalars[4×C]f
// ============================================================================
template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_bwd_wmma(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    float* __restrict__ d_initial_state,
    const TQ* __restrict__ d_out,
    const float* __restrict__ d_final_state,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    float* __restrict__ workspace,
    int workspace_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int chunk_size, float scale, bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = (long)Kdim * Vdim;
    const long bh = (long)b * H + h;
    const float* cp_base = checkpoints + bh * (long)(num_chunks + 1) * kv;
    float* ds = d_initial_state + bh * kv;
    const int Lp = kMaxC;

    // Shared memory layout
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;       // [C×C] TQ
    TQ*    buf2      = buf1 + Lp * Lp;            // [C×V] TQ
    TQ*    buf3      = buf2 + Lp * Vdim;          // [C×K] TQ
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Global workspace per (b,h)
    float* ws = workspace + bh * workspace_stride;
    const int ws_W    = 0;
    const int ws_VNEW = ws_W    + chunk_size * Kdim;
    const int ws_DU   = ws_VNEW + chunk_size * Vdim;
    const int ws_DW   = ws_DU   + chunk_size * Vdim;
    const int ws_DQ   = ws_DW   + chunk_size * Kdim;
    const int ws_DK   = ws_DQ   + chunk_size * Kdim;
    const int ws_DG   = ws_DK   + chunk_size * Kdim;
    const int ws_DB   = ws_DG   + chunk_size;
    float* W    = ws + ws_W;
    float* VNEW = ws + ws_VNEW;
    float* DU   = ws + ws_DU;
    float* DW   = ws + ws_DW;
    float* DQ   = ws + ws_DQ;
    float* DK   = ws + ws_DK;
    float* DG   = ws + ws_DG;
    float* DB   = ws + ws_DB;

    // Seed ds
    for (long idx = tid; idx < kv; idx += nthr)
        ds[idx] = d_final_state ? d_final_state[bh * kv + idx] : 0.0f;
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const float* h_in = cp_base + (long)chunk * kv;
        float* dh_in = const_cast<float*>(cp_base + (long)(chunk + 1) * kv);

        // Zero-fill
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            smem_k[idx] = from_float<TQ>(0.0f);
            smem_q[idx] = from_float<TQ>(0.0f);
        }
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Kdim; idx += nthr)
            buf3[idx] = from_float<TQ>(0.0f);
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        if (tid < Lp) { smem_gcum[tid] = 0.0f; smem_beta[tid] = 0.0f; }
        __syncthreads();

        // Load k, q, v
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim, kk = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k_global[gi];
            smem_q[pos * Kdim + kk] = q_global[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[pos * Vdim + vv] = v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g_global[gh]);
                smem_gcum[i] = acc;
                smem_beta[i] = to_float(beta_global[gh]);
            }
        }
        __syncthreads();

        // L2 norms
        for (int pos = tid; pos < L; pos += nthr) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv; kn2 += kv2 * kv2;
                }
                smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
            }
        }
        __syncthreads();
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthr) {
                const int pos = idx / Kdim;
                smem_k[pos * Kdim + idx % Kdim] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[idx]) * smem_invk[pos]));
                smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
            }
            __syncthreads();
        }

        // ================================================================
        // Build M: A = beta*exp(g)*(k@k^T), solve M
        // ================================================================
        wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] *= smem_beta[i] * expf(smem_gcum[i] - smem_gcum[j]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // Solve M → scratch2
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

        // M → bf16 in buf1
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // ================================================================
        // Recompute W and VNEW using WMMA
        // ================================================================

        // bv = beta*v → buf2
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
        }
        __syncthreads();

        // u = M @ bv → scratch1[Lp×V]
        wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
        __syncthreads();

        // bkg = beta*k*exp(g) → buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // w = M @ bkg → scratch1[Lp×K]
        wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float wval = bf16_trunc<TQ>(scratch1[idx]);
            buf3[idx] = from_float<TQ>(wval);
            if (idx < L * Kdim) W[idx] = wval;
        }
        __syncthreads();

        // h_bf16 → buf1[K×V]
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();

        // wh = w @ h_bf16 → scratch1[Lp×V]
        wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // vnew_pre = u - wh, then vnew = vnew_pre * exp(g_last - g_cum[i])
        const float g_last_val = (L > 0) ? smem_gcum[L - 1] : 0.0f;
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float vnew_pre = bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]);
            float e_i = (i < L) ? expf(g_last_val - smem_gcum[i]) : 0.0f;
            float vnew = bf16_trunc<TQ>(vnew_pre * e_i);
            buf2[idx] = from_float<TQ>(vnew);
            if (idx < L * Vdim) VNEW[idx] = vnew;
        }
        __syncthreads();

        // ================================================================
        // dh_in = ds * eg_last; dg_last_extra
        // ================================================================
        const float eg_last = expf(g_last_val);

        if (tid == 0) scratch1[0] = 0.0f;
        __syncthreads();

        for (long idx = tid; idx < kv; idx += nthr) {
            dh_in[idx] = ds[idx] * eg_last;
            atomicAdd(&scratch1[0], ds[idx] * h_in[idx]);
        }
        __syncthreads();

        // Zero workspace arrays
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            DW[idx] = 0.0f; DQ[idx] = 0.0f; DK[idx] = 0.0f;
        }
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            DU[idx] = 0.0f;
        for (int idx = tid; idx < L; idx += nthr) {
            DG[idx] = 0.0f; DB[idx] = 0.0f;
        }
        if (tid == 0) DG[L - 1] += scratch1[0] * eg_last;
        __syncthreads();

        // ================================================================
        // Contribution from ds via WMMA:
        //   DU += k @ ds  (k[L×K] @ ds[K×V] → [L×V])
        //   DK += VNEW @ ds^T  (VNEW[L×V] @ ds^T[V×K] → [L×K])
        // ================================================================
        // Load ds[K×V] as bf16 → buf1[K×V] (fits in Lp×Lp since K=V=Lp)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(ds[idx]));
        __syncthreads();

        // DU_state = k @ ds_bf16 → scratch1[Lp×V]
        // Also save to ws area for reuse in v_new gating (d_vnew_state)
        float* ws_tmp = ws + ws_DB + chunk_size;
        float* ws_gs  = ws_tmp + Lp * Lp;
        wmma_nn<TQ>(smem_k, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            DU[idx] += scratch1[idx];
            ws_gs[idx] = scratch1[idx];  // save k@ds for v_new gating
        }
        __syncthreads();

        // DK_state = VNEW_bf16 @ ds_bf16^T → scratch1[Lp×K]
        // Load VNEW → buf2[Lp×V] as bf16
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? VNEW[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();

        // ================================================================
        // Output gradient — BATCHED with WMMA where possible
        // ================================================================

        // --- term1: q @ h contribution ---
        // DQ[i,k] += scale * exp(g_cum[i]) * Σ_v d_out[i,v] * h_bf16[k,v]
        // dh_in[k,v] += Σ_i scale * exp(g_cum[i]) * d_out[i,v] * q[i,k]
        // DG[i] += scale * exp(g_cum[i]) * Σ_v d_out[i,v] * Σ_k q[i,k] * h_bf16[k,v]

        // Load d_out into buf2 (overwriting vnew bf16 — we have VNEW in workspace)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr)
            buf2[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = d_out[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        __syncthreads();

        // h_bf16 is still in buf1[K×V] from earlier.
        // DQ_term1 = scale * diag(exp(g)) * d_out @ h_bf16^T → need d_out[L×V] @ h_bf16^T[V×K]
        // = d_out @ h_bf16^T = WMMA_NT(d_out[Lp×V], h_bf16[K×V]) → scratch1[Lp×K]
        // But h_bf16 is [K×V] not [Lp×V], so NT won't work directly.
        // Use WMMA_NN: d_out[Lp×V] @ h_bf16_T[V×K]. We need to transpose h.
        // Actually h_bf16 is Kdim×Vdim stored row-major. For A@B^T with B=[K×V]:
        //   A[Lp×V] @ B^T [V×K] where B is row-major [K×V]
        //   This is A @ B^T = wmma_nt(A, B) only if B is [K×V] row-major.
        //   Wait, wmma_nt loads B with col_major, so B^T is [V×K].
        //   If B is stored row-major [K×V], loading with col_major gives us V×K layout.
        //   So wmma_nt(d_out[Lp×Vdim], h_bf16[K×Vdim]) → [Lp×K] if K==Lp
        //   Actually the wmma_nt computes C[M×N] = A[M×K_inner] @ B[N×K_inner]^T
        //   So: M=Lp, N=Kdim, K_inner=Vdim
        //   C[Lp×Kdim] = d_out[Lp×Vdim] @ h_bf16[Kdim×Vdim]^T
        //   This works! But Kdim must be multiple of 16 (it is).
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();

        // Apply scale * exp(g) and accumulate to DQ
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            DQ[idx] += scale * expf(smem_gcum[i]) * scratch1[i * Kdim + idx % Kdim];
        }
        __syncthreads();

        // dh_in[k,v] += Σ_i scale*exp(g[i]) * q[i,k] * d_out[i,v]
        // = (scale*exp(g)*q)^T @ d_out = q_scaled^T[K×Lp] @ d_out[Lp×V] → [K×V]
        // Prepare q_scaled in buf3 (overwriting w — we have W in workspace)
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int i = idx / Kdim;
            float val = (i < L) ? to_float(smem_q[idx]) * scale * expf(smem_gcum[i]) : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // q_scaled^T @ d_out → scratch1[K×V]
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();

        // Accumulate to dh_in
        for (long idx = tid; idx < kv; idx += nthr)
            dh_in[idx] += scratch1[idx];
        __syncthreads();

        // DG[i] from term1: scale*exp(g[i]) * dot(d_out[i,:], q[i,:]@h[:,:])
        // We already computed d_out @ h^T → scratch1 was overwritten. Recompute per-row.
        // Actually let's compute it differently: DG[i] += scale*exp(g[i]) * q[i,:] @ h @ d_out[i,:]^T
        // = scale*exp(g[i]) * Σ_k q[i,k] * Σ_v h[k,v] * d_out[i,v]
        // We need q@h per row. Let's do q@h_bf16 → scratch1[Lp×V] using WMMA.
        // h_bf16 is still in buf1[K×V].
        wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // DG[i] += scale * exp(g[i]) * dot(scratch1[i,:], d_out[i,:])
        for (int i = tid; i < L; i += nthr) {
            float dot_val = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv)
                dot_val += scratch1[i * Vdim + vv] * to_float(buf2[i * Vdim + vv]);
            DG[i] += scale * expf(smem_gcum[i]) * dot_val;
        }
        __syncthreads();

        // --- term2: Batched intra-chunk attention gradient ---
        // S[i,j] = bf16_trunc(exp(g[i]-g[j]) * dot(q[i,:], k[j,:]))  for j<=i, else 0
        // Already have q, k in smem. Compute S = q @ k^T via WMMA.
        wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // Apply gating mask: S[i,j] = bf16_trunc(exp(g[i]-g[j]) * raw_S[i,j]) for j<=i
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] = bf16_trunc<TQ>(expf(smem_gcum[i] - smem_gcum[j]) * scratch1[idx]);
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // vnew_pre[j,v] = VNEW[j,v] / exp(g_last - g_cum[j])
        // Load vnew_pre into buf1 as bf16 (reusing buf1 — h_bf16 no longer needed)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        // Only need Lp×Vdim portion of buf1 for vnew_pre, but buf1 is Lp×Lp.
        // We'll store vnew_pre in the first Lp×Vdim of buf1 if Vdim<=Lp (true since Vdim<=64=Lp).
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int j = idx / Vdim;
            float e_j = (j < L) ? expf(g_last_val - smem_gcum[j]) : 1.0f;
            float vnew_pre = (j < L) ? VNEW[j * Vdim + idx % Vdim] / e_j : 0.0f;
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnew_pre));
        }
        // Zero rest of buf1 if Vdim < Lp
        for (int idx = Lp * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();

        // grad_S[i,j] = Σ_v scale * d_out[i,v] * vnew_pre[j,v]
        // = scale * d_out[Lp×V] @ vnew_pre[Lp×V]^T → [Lp×Lp]
        // Use WMMA_NT(scale*d_out, vnew_pre)
        // Scale d_out: buf2 currently has d_out. Multiply by scale.
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            float val = (i < L) ? to_float(buf2[idx]) * scale : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // Save S to workspace area 1 (ws_tmp, ws_gs already declared above)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_tmp[idx] = scratch1[idx];  // S saved
        __syncthreads();

        // Load vnew_pre into buf1[Lp×V] for grad_S computation
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int j = idx / Vdim;
            float e_j = (j < L) ? expf(g_last_val - smem_gcum[j]) : 1.0f;
            float vnp = (j < L) ? VNEW[j * Vdim + idx % Vdim] / e_j : 0.0f;
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnp));
        }
        __syncthreads();

        // grad_S_raw = scale_d_out @ vnew_pre^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // Save grad_S_raw to workspace area 2
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_gs[idx] = scratch1[idx];
        __syncthreads();

        // DG from S gradient without atomics:
        // DG[i] += sum_{j<=i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
        for (int i = tid; i < L; i += nthr) {
            float dg_pos = 0.0f;
            float dg_neg = 0.0f;
            for (int j = 0; j <= i; ++j) {
                dg_pos += ws_gs[i * Lp + j] * ws_tmp[i * Lp + j];
            }
            for (int r = i + 1; r < L; ++r) {
                dg_neg += ws_gs[r * Lp + i] * ws_tmp[r * Lp + i];
            }
            DG[i] += dg_pos - dg_neg;
        }
        __syncthreads();

        // DU[j,v] += Σ_i S[i,j]/e_j * scale*d_out[i,v]
        // S_scaled = S/e_j → buf1 as bf16
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i && j < L)
                val = ws_tmp[idx] / expf(g_last_val - smem_gcum[j]);
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DU_term2 = S_scaled^T @ scale_d_out → [Lp×V]
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            DU[idx] += scratch1[idx];
        __syncthreads();

        // grad_S_masked = grad_S_raw * exp(g[i]-g[j]) for j<=i → buf1 as bf16
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            float val = 0.0f;
            if (i < L && j <= i)
                val = ws_gs[idx] * expf(smem_gcum[i] - smem_gcum[j]);
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DQ_term2 = grad_S_masked @ k → [Lp×K]
        wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DQ[idx] += scratch1[idx];
        __syncthreads();

        // DK_term2 = grad_S_masked^T @ q → [Lp×K]
        wmma_tn<TQ>(buf1, Lp, smem_q, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DK[idx] += scratch1[idx];
        __syncthreads();

        // ================================================================
        // v_new gating gradients — batched with WMMA
        // ================================================================
        // Pass 1: compute d_pre[i,v] = DU[i,v] * e_i, DG contributions
        // Use precomputed k@ds from ws_gs for d_vnew_state
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim, vv = idx % Vdim;
            const float e_i = expf(g_last_val - smem_gcum[i]);
            const float v_new = VNEW[idx];
            const float pre = v_new / e_i;
            const float d_vnew = DU[idx];
            const float d_pre = d_vnew * e_i;
            const float d_vnew_state = ws_gs[idx]; // precomputed k@ds[i,v]
            const float d_e_state = d_vnew_state * pre;

            DU[idx] = d_pre;

            atomicAdd(&DG[L - 1],  d_e_state * e_i);
            atomicAdd(&DG[i],     -d_e_state * e_i);
        }
        __syncthreads();

        // Pass 2: DW = -d_pre @ h_bf16^T, dh_in += -W^T @ d_pre
        // Load d_pre (DU) → buf2 as bf16
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        // Load h_bf16 → buf1[K×V]
        for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
        for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        // Load W → buf3 as bf16
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float val = (idx < L * Kdim) ? W[idx] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // DW += -d_pre @ h_bf16^T = -(buf2[Lp×V] @ buf1[K×V]^T) → scratch1[Lp×K]
        wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
        __syncthreads();
        for (int idx = tid; idx < L * Kdim; idx += nthr)
            DW[idx] += -scratch1[idx];
        __syncthreads();

        // dh_in += -W^T @ d_pre = -(buf3[Lp×K]^T @ buf2[Lp×V]) → scratch1[K×V]
        wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
        __syncthreads();
        for (long idx = tid; idx < kv; idx += nthr)
            dh_in[idx] += -scratch1[idx];
        __syncthreads();

        // ================================================================
        // dM = DU @ (beta*V)^T + DW @ bkg^T via WMMA
        // ================================================================
        // Load raw v → buf2 (bf16, with beta scaling)
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            float val = 0.0f;
            if (pos < L) {
                float vraw = to_float(v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv]);
                val = bf16_trunc<TQ>(vraw * smem_beta[pos]);
            }
            buf2[idx] = from_float<TQ>(val);
        }
        // DU_bf16 → buf1 (note: buf1 is Lp×Lp, DU is L×V, Lp×V fits if V<=Lp)
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(0.0f);
        __syncthreads();
        for (int idx = tid; idx < L * Vdim; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(DU[idx]));
        __syncthreads();

        // dM_part1 = DU_bf16 @ (beta*V)^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf1, Vdim, buf2, Vdim, scratch1, Lp, Lp, Lp, Vdim);
        __syncthreads();

        // Prepare bkg → buf2[Lp×K] and DW_bf16 → buf1[Lp×K]
        // Reuse buf2 for bkg (Lp×K) — overwriting beta*V since we're done with it
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            float val = (pos < L) ? to_float(smem_k[idx]) * smem_beta[pos] * expf(smem_gcum[pos]) : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        // Need DW in bf16 for WMMA — put in buf3
        for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
            float val = (idx < L * Kdim) ? DW[idx] : 0.0f;
            buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // dM_part2 = DW_bf16 @ bkg^T → add to scratch1 via temp
        // But we only have one scratch float buffer free (scratch1 has dM_part1).
        // Compute part2 into ws_gs (workspace), then add.
        // Actually ws_gs has grad_S_raw — we're done with it. Reuse.
        // But ws_gs is not Lp-strided. Use it as L×L or Lp×Lp.
        // Let me use a different approach: compute into scratch1 directly by doing
        // the WMMA and adding results manually.
        // Save dM_part1 to ws_gs temporarily.
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            ws_gs[idx] = scratch1[idx];
        __syncthreads();

        // dM_part2 = DW_bf16 @ bkg^T → scratch1[Lp×Lp]
        wmma_nt<TQ>(buf3, Kdim, buf2, Kdim, scratch1, Lp, Lp, Lp, Kdim);
        __syncthreads();

        // dM = part1 + part2, masked to lower triangle
        for (int idx = tid; idx < Lp * Lp; idx += nthr) {
            const int i = idx / Lp, j = idx % Lp;
            if (i < L && j <= i)
                scratch1[idx] = ws_gs[idx] + scratch1[idx];
            else
                scratch1[idx] = 0.0f;
        }
        __syncthreads();

        // ================================================================
        // d_v, d_beta from M^T @ DU and M^T @ DW via WMMA
        // ================================================================
        // M_bf16 → buf1[Lp×Lp]
        for (int idx = tid; idx < Lp * Lp; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
        __syncthreads();

        // DU_bf16 → buf2[Lp×V]
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
        }
        __syncthreads();

        // MT_DU = M^T @ DU_bf16 → ws_gs[Lp×V] (float, via WMMA then copy)
        wmma_tn<TQ>(buf1, Lp, buf2, Vdim, ws_gs, Vdim, Lp, Vdim, Lp);
        __syncthreads();

        // d_v[j,v] = beta[j] * MT_DU[j,v]
        // DB[j] += Σ_v MT_DU[j,v] * v_raw[j,v]
        // Reload raw v into buf2 for DB computation
        for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
            const int pos = idx / Vdim, vv = idx % Vdim;
            buf2[idx] = (pos < L) ?
                v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv] :
                from_float<TQ>(0.0f);
        }
        __syncthreads();

        for (int j = tid; j < L; j += nthr) {
            const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;
            float db_acc = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv) {
                float mt_du = ws_gs[j * Vdim + vv];
                d_v[v_j_base + vv] = from_float<TQ>(mt_du * smem_beta[j]);
                db_acc += mt_du * to_float(buf2[j * Vdim + vv]);
            }
            DB[j] += db_acc;
        }
        __syncthreads();

        // DW_bf16 already in buf3. MT_DW = M^T @ DW_bf16 → ws_gs[Lp×K]
        wmma_tn<TQ>(buf1, Lp, buf3, Kdim, ws_gs, Kdim, Lp, Kdim, Lp);
        __syncthreads();

        // DK[j,k] += MT_DW[j,k] * beta[j] * exp(g[j])
        // DB[j] += Σ_k MT_DW[j,k] * exp(g[j]) * k[j,k]
        // DG[j] += Σ_k MT_DW[j,k] * beta[j] * exp(g[j]) * k[j,k]
        for (int j = tid; j < L; j += nthr) {
            const float egj = expf(smem_gcum[j]);
            float db_acc = 0.0f, dg_acc = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float mt_dw = ws_gs[j * Kdim + kk];
                float kj = to_float(smem_k[j * Kdim + kk]);
                DK[j * Kdim + kk] += mt_dw * smem_beta[j] * egj;
                db_acc += mt_dw * egj * kj;
                dg_acc += mt_dw * smem_beta[j] * egj * kj;
            }
            DB[j] += db_acc;
            DG[j] += dg_acc;
        }
        __syncthreads();

        // ================================================================
        // dA_grad = -M^T @ dM @ M^T
        // ================================================================
        // tmp1 = dM @ M^T
        for (int idx = tid; idx < L * L; idx += nthr) {
            const int i = idx / L, j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m)
                s += scratch1[i * Lp + m] * scratch2[j * Lp + m];
            ws_tmp[idx] = s;
        }
        __syncthreads();

        // dA_grad = -M^T @ tmp1
        for (int idx = tid; idx < L * L; idx += nthr) {
            const int i = idx / L, j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m)
                s += scratch2[m * Lp + i] * ws_tmp[m * L + j];
            scratch1[i * Lp + j] = -s;
        }
        __syncthreads();

        // dA_grad → DK, DB, DG
        for (int i = 0; i < L; ++i) {
            for (int j = tid; j < i; j += nthr) {
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk)
                    dot_k += to_float(smem_k[i * Kdim + kk])
                           * to_float(smem_k[j * Kdim + kk]);
                const float exp_ij = expf(smem_gcum[i] - smem_gcum[j]);
                const float a_grad = scratch1[i * Lp + j];
                const float val = dot_k * exp_ij;

                atomicAdd(&DB[i], a_grad * val);
                const float dval = a_grad * smem_beta[i];
                const float ddot = dval * exp_ij;
                const float dexp = dval * dot_k;

                atomicAdd(&DG[i],  dexp * exp_ij);
                atomicAdd(&DG[j], -dexp * exp_ij);

                for (int kk = 0; kk < Kdim; ++kk) {
                    atomicAdd(&DK[i * Kdim + kk], ddot * to_float(smem_k[j * Kdim + kk]));
                    atomicAdd(&DK[j * Kdim + kk], ddot * to_float(smem_k[i * Kdim + kk]));
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Write d_q, d_k, d_beta
        // ================================================================
        for (int i = tid; i < L; i += nthr) {
            const long qi_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
            if (use_qk_l2norm_in_kernel) {
                float dot_q = 0.0f, dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_q += DQ[i * Kdim + kk] * to_float(smem_q[i * Kdim + kk]);
                    dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[qi_base + kk] = from_float<TQ>(
                        (DQ[i * Kdim + kk] - to_float(smem_q[i * Kdim + kk]) * dot_q) * smem_invq[i]);
                    d_k[qi_base + kk] = from_float<TQ>(
                        (DK[i * Kdim + kk] - to_float(smem_k[i * Kdim + kk]) * dot_k) * smem_invk[i]);
                }
            } else {
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[qi_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                    d_k[qi_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
                }
            }
            d_beta[gh_idx] = from_float<TB>(DB[i]);
        }
        __syncthreads();

        // Reverse cumsum for d_g
        if (tid == 0) {
            float running = 0.0f;
            for (int i = L - 1; i >= 0; --i) {
                running += DG[i];
                d_g[((long)b * Tlen + cs + i) * H + h] = from_float<TG>(running);
            }
        }
        __syncthreads();

        // Propagate ds backward
        for (long idx = tid; idx < kv; idx += nthr)
            ds[idx] = dh_in[idx];
        __syncthreads();
    }
}

// ============================================================================
// Backward kernel — fully parallelized (scalar fallback)
//
// Uses checkpoints (from checkpoint kernel) and a global workspace.
// All heavy loops are distributed across threads.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_chunk_bwd_v2(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    float* __restrict__ d_initial_state,
    const TQ* __restrict__ d_out,
    const float* __restrict__ d_final_state,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    float* __restrict__ workspace,
    int workspace_stride,
    int Tlen,
    int H,
    int Kdim,
    int Vdim,
    int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel) {

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const long kv = static_cast<long>(Kdim) * Vdim;
    const long bh = static_cast<long>(b) * H + h;
    const float* cp_base = checkpoints + bh * static_cast<long>(num_chunks + 1) * kv;
    float* ds = d_initial_state + bh * kv;

    // --- Shared memory ---
    extern __shared__ char smem_raw[];
    float*  smem_M     = reinterpret_cast<float*>(smem_raw);                    // [C×C] float
    float*  smem_dM    = smem_M + kMaxC * kMaxC;                                // [C×C] float
    TQ*     smem_k     = reinterpret_cast<TQ*>(smem_dM + kMaxC * kMaxC);        // [C×K] TQ
    TQ*     smem_q     = smem_k + kMaxC * Kdim;                                 // [C×K] TQ
    TQ*     smem_v_loc = smem_q + kMaxC * Kdim;                                 // [C×V] TQ
    float*  smem_g_cum = reinterpret_cast<float*>(smem_v_loc + kMaxC * Vdim);   // [C] float
    float*  smem_beta  = smem_g_cum + kMaxC;                                     // [C] float
    float*  smem_inv_q = smem_beta  + kMaxC;                                     // [C] float
    float*  smem_inv_k = smem_inv_q + kMaxC;                                     // [C] float

    // --- Global workspace per (b,h) ---
    float* ws = workspace + bh * workspace_stride;
    const int ws_W    = 0;
    const int ws_VNEW = ws_W    + chunk_size * Kdim;
    const int ws_DU   = ws_VNEW + chunk_size * Vdim;
    const int ws_DW   = ws_DU   + chunk_size * Vdim;
    const int ws_DQ   = ws_DW   + chunk_size * Kdim;
    const int ws_DK   = ws_DQ   + chunk_size * Kdim;
    const int ws_DG   = ws_DK   + chunk_size * Kdim;
    const int ws_DB   = ws_DG   + chunk_size;
    float* W    = ws + ws_W;
    float* VNEW = ws + ws_VNEW;
    float* DU   = ws + ws_DU;
    float* DW   = ws + ws_DW;
    float* DQ   = ws + ws_DQ;
    float* DK   = ws + ws_DK;
    float* DG   = ws + ws_DG;
    float* DB   = ws + ws_DB;

    // Seed ds
    for (long idx = tid; idx < kv; idx += nthreads) {
        ds[idx] = d_final_state ? d_final_state[bh * kv + idx] : 0.0f;
    }
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        const int cs = chunk * chunk_size;
        const int L = min(chunk_size, Tlen - cs);
        const float* h_in = cp_base + static_cast<long>(chunk) * kv;
        // dh_in aliases the NEXT checkpoint slot (will be overwritten)
        float* dh_in = const_cast<float*>(cp_base + static_cast<long>(chunk + 1) * kv);

        // ================================================================
        // Load q, k, v for this chunk
        // ================================================================
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int pos = idx / Kdim;
            const int kk  = idx % Kdim;
            const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
            smem_k[pos * Kdim + kk] = k_global[gi];
            smem_q[pos * Kdim + kk] = q_global[gi];
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int pos = idx / Vdim;
            const int vv  = idx % Vdim;
            smem_v_loc[pos * Vdim + vv] = v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
        }
        __syncthreads();

        // Compute scalars and norms
        if (tid == 0) {
            float acc = 0.0f;
            for (int i = 0; i < L; ++i) {
                const long gh = ((long)b * Tlen + cs + i) * H + h;
                acc += to_float(g_global[gh]);
                smem_g_cum[i] = acc;
                smem_beta[i]  = to_float(beta_global[gh]);
            }
        }
        __syncthreads();

        for (int pos = tid; pos < L; pos += nthreads) {
            if (use_qk_l2norm_in_kernel) {
                float qn2 = 0.0f, kn2 = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    float qv = to_float(smem_q[pos * Kdim + kk]);
                    float kv2 = to_float(smem_k[pos * Kdim + kk]);
                    qn2 += qv * qv;
                    kn2 += kv2 * kv2;
                }
                smem_inv_q[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
                smem_inv_k[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
            } else {
                smem_inv_q[pos] = 1.0f;
                smem_inv_k[pos] = 1.0f;
            }
        }
        __syncthreads();

        // Normalize k, q in-place
        if (use_qk_l2norm_in_kernel) {
            for (int idx = tid; idx < L * Kdim; idx += nthreads) {
                const int pos = idx / Kdim;
                const int kk  = idx % Kdim;
                smem_k[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_k[pos * Kdim + kk]) * smem_inv_k[pos]));
                smem_q[pos * Kdim + kk] = from_float<TQ>(
                    bf16_trunc<TQ>(to_float(smem_q[pos * Kdim + kk]) * smem_inv_q[pos]));
            }
            __syncthreads();
        }

        // ================================================================
        // Build M matrix (same as forward)
        // ================================================================
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_M[idx] = 0.0f;
        }
        __syncthreads();
        for (int i = tid; i < L; i += nthreads) {
            smem_M[i * kMaxC + i] = 1.0f;
        }
        __syncthreads();

        for (int row = 1; row < L; ++row) {
            for (int j = tid; j < row; j += nthreads) {
                float s = 0.0f;
                for (int m = j; m < row; ++m) {
                    float dot_k = 0.0f;
                    for (int kk = 0; kk < Kdim; ++kk) {
                        dot_k += to_float(smem_k[row * Kdim + kk])
                               * to_float(smem_k[m   * Kdim + kk]);
                    }
                    s += smem_beta[row] * dot_k
                       * expf(smem_g_cum[row] - smem_g_cum[m])
                       * smem_M[m * kMaxC + j];
                }
                smem_M[row * kMaxC + j] = bf16_trunc<TQ>(-s);
            }
            __syncthreads();
        }

        // ================================================================
        // Recompute W [L×K] and VNEW [L×V] from M, k, v, beta, g_cum, h_in
        // ================================================================

        // Zero workspace arrays
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            W[idx] = 0.0f;
            DW[idx] = 0.0f;
            DQ[idx] = 0.0f;
            DK[idx] = 0.0f;
        }
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            VNEW[idx] = 0.0f;
            DU[idx] = 0.0f;
        }
        for (int idx = tid; idx < L; idx += nthreads) {
            DG[idx] = 0.0f;
            DB[idx] = 0.0f;
        }
        __syncthreads();

        // Compute W[i,k] = bf16(Σ_m M[i,m] * bf16(k_norm[m,k] * beta[m] * exp(g_cum[m])))
        for (int idx = tid; idx < L * Kdim; idx += nthreads) {
            const int i  = idx / Kdim;
            const int kk = idx % Kdim;
            float acc = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float kbg = bf16_trunc<TQ>(
                    to_float(smem_k[m * Kdim + kk]) * smem_beta[m] * expf(smem_g_cum[m]));
                acc += smem_M[i * kMaxC + m] * kbg;
            }
            W[i * Kdim + kk] = bf16_trunc<TQ>(acc);
        }
        __syncthreads();

        // Compute VNEW[i,v] = bf16((u_i - w_i @ bf16(h_in)) * exp(g_last - g_cum[i]))
        const float g_last_val = smem_g_cum[L - 1];
        for (int idx = tid; idx < L * Vdim; idx += nthreads) {
            const int i  = idx / Vdim;
            const int vv = idx % Vdim;

            // u_i_v
            float u_val = 0.0f;
            for (int m = 0; m <= i; ++m) {
                float vb = bf16_trunc<TQ>(
                    to_float(smem_v_loc[m * Vdim + vv]) * smem_beta[m]);
                u_val += smem_M[i * kMaxC + m] * vb;
            }
            u_val = bf16_trunc<TQ>(u_val);

            // wh
            float wh = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float h_cs = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                wh += W[i * Kdim + kk] * h_cs;
            }

            float e_i = expf(g_last_val - smem_g_cum[i]);
            VNEW[i * Vdim + vv] = bf16_trunc<TQ>((u_val - wh) * e_i);
        }
        __syncthreads();

        // ================================================================
        // dh_in = ds * eg_last; dg_last_extra
        // ================================================================
        const float eg_last = expf(g_last_val);

        // dh_in = ds * eg_last, and accumulate dg_last_extra using atomicAdd
        // We use smem_dM[0] as a temporary for dg_last_extra
        if (tid == 0) {
            smem_dM[0] = 0.0f;
        }
        __syncthreads();

        for (long idx = tid; idx < kv; idx += nthreads) {
            dh_in[idx] = ds[idx] * eg_last;
            float contrib = ds[idx] * h_in[idx];
            atomicAdd(&smem_dM[0], contrib);
        }
        __syncthreads();

        if (tid == 0) {
            DG[L - 1] += smem_dM[0] * eg_last;
        }
        __syncthreads();

        // Zero smem_dM properly now (was used as temp)
        for (int idx = tid; idx < kMaxC * kMaxC; idx += nthreads) {
            smem_dM[idx] = 0.0f;
        }
        __syncthreads();

        // ================================================================
        // Contribution from ds: DU += ds^T @ k_norm (per position)
        //                       DK += ds @ VNEW^T (per position)
        // ================================================================
        for (int i = 0; i < L; ++i) {
            // DU[i,v] += Σ_k ds[k,v] * k_norm[i,k]
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                float acc = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    acc += ds[kk * Vdim + vv] * to_float(smem_k[i * Kdim + kk]);
                }
                DU[i * Vdim + vv] += acc;
            }
            // DK[i,k] += Σ_v ds[k,v] * VNEW[i,v]
            for (int kk = tid; kk < Kdim; kk += nthreads) {
                float acc = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    acc += ds[kk * Vdim + vv] * VNEW[i * Vdim + vv];
                }
                DK[i * Kdim + kk] += acc;
            }
            __syncthreads();
        }

        // ================================================================
        // Output gradient contribution: d_out → DQ, DK, DG, DU, dh_in
        // ================================================================
        for (int i = 0; i < L; ++i) {
            const long q_i_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
            const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
            const float eg_i = expf(smem_g_cum[i]);

            // term1: d_out[i] @ h_in^T contribution to DQ, dh_in, DG
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                float qh = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = to_float(smem_q[i * Kdim + kk]);
                    const float hq = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    qh += qi * hq;
                    atomicAdd(&DQ[i * Kdim + kk], do_v * eg_i * hq);
                    atomicAdd(&dh_in[kk * Vdim + vv], do_v * eg_i * qi);
                }
                atomicAdd(&DG[i], do_v * eg_i * qh);
            }
            __syncthreads();

            // term2: intra-chunk attention gradient
            for (int j = 0; j <= i; ++j) {
                const float e_j = expf(g_last_val - smem_g_cum[j]);
                const float exp_ij = expf(smem_g_cum[i] - smem_g_cum[j]);

                // dot(q_norm[i], k_norm[j])
                float dot_qk = 0.0f;
                // Parallel reduction across threads
                for (int kk = tid; kk < Kdim; kk += nthreads) {
                    dot_qk += to_float(smem_q[i * Kdim + kk])
                            * to_float(smem_k[j * Kdim + kk]);
                }
                // Warp reduction
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot_qk += __shfl_down_sync(0xffffffff, dot_qk, offset);
                }
                // Cross-warp reduction via shared memory
                // Use smem_dM scratch space for reduction
                if (tid % 32 == 0) {
                    smem_dM[tid / 32] = dot_qk;
                }
                __syncthreads();
                if (tid == 0) {
                    float total = 0.0f;
                    for (int w = 0; w < (nthreads + 31) / 32; ++w) {
                        total += smem_dM[w];
                    }
                    smem_dM[kMaxC * kMaxC - 1] = total;  // store result
                }
                __syncthreads();
                dot_qk = smem_dM[kMaxC * kMaxC - 1];

                const float s_ij = bf16_trunc<TQ>(exp_ij * dot_qk);

                // grad_s = Σ_v d_out[i,v] * v_new_pre[j,v] / e_j
                float grad_s_local = 0.0f;
                for (int vv = tid; vv < Vdim; vv += nthreads) {
                    const float do_v = to_float(d_out[do_i_base + vv]) * scale;
                    const float v_new_pre = VNEW[j * Vdim + vv] / e_j;
                    grad_s_local += do_v * v_new_pre;
                    atomicAdd(&DU[j * Vdim + vv], do_v * (s_ij / e_j));
                }
                // Reduce grad_s
                for (int offset = 16; offset > 0; offset >>= 1) {
                    grad_s_local += __shfl_down_sync(0xffffffff, grad_s_local, offset);
                }
                if (tid % 32 == 0) {
                    smem_dM[tid / 32] = grad_s_local;
                }
                __syncthreads();
                float grad_s = 0.0f;
                if (tid == 0) {
                    for (int w = 0; w < (nthreads + 31) / 32; ++w) {
                        grad_s += smem_dM[w];
                    }
                    smem_dM[kMaxC * kMaxC - 2] = grad_s;
                }
                __syncthreads();
                grad_s = smem_dM[kMaxC * kMaxC - 2];

                const float coeff = grad_s * exp_ij;
                for (int kk = tid; kk < Kdim; kk += nthreads) {
                    const float qi = to_float(smem_q[i * Kdim + kk]);
                    const float kj = to_float(smem_k[j * Kdim + kk]);
                    atomicAdd(&DQ[i * Kdim + kk], coeff * kj);
                    atomicAdd(&DK[j * Kdim + kk], coeff * qi);
                }
                if (tid == 0) {
                    DG[i] += grad_s * s_ij;
                    DG[j] -= grad_s * s_ij;
                }
                __syncthreads();
            }
        }

        // ================================================================
        // v_new gating gradients: merge into DU, DW, DG, dh_in
        // ================================================================
        for (int i = 0; i < L; ++i) {
            const float e_i = expf(g_last_val - smem_g_cum[i]);

            for (int vv = tid; vv < Vdim; vv += nthreads) {
                const float v_new = VNEW[i * Vdim + vv];
                const float pre = v_new / e_i;
                const float d_vnew = DU[i * Vdim + vv];
                const float d_pre = d_vnew * e_i;

                // d_vnew from state contribution
                float d_vnew_state = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_vnew_state += ds[kk * Vdim + vv]
                                  * to_float(smem_k[i * Kdim + kk]);
                }
                const float d_e_state = d_vnew_state * pre;

                DU[i * Vdim + vv] = d_pre;  // overwrite with corrected value

                atomicAdd(&DG[L - 1],  d_e_state * e_i);
                atomicAdd(&DG[i],     -d_e_state * e_i);

                for (int kk = 0; kk < Kdim; ++kk) {
                    float h_cs = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    atomicAdd(&DW[i * Kdim + kk], -d_pre * h_cs);
                    atomicAdd(&dh_in[kk * Vdim + vv], -d_pre * W[i * Kdim + kk]);
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Compute dM[i,j] = DU[i,:] @ (beta*V)[j,:] + DW[i,:] @ (beta*K*exp(g))[j,:]
        // ================================================================
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            if (j <= i) {
                float s = 0.0f;
                for (int vv = 0; vv < Vdim; ++vv) {
                    float vb = bf16_trunc<TQ>(
                        to_float(smem_v_loc[j * Vdim + vv]) * smem_beta[j]);
                    s += DU[i * Vdim + vv] * vb;
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    float kbg = bf16_trunc<TQ>(
                        to_float(smem_k[j * Kdim + kk]) * smem_beta[j]
                        * expf(smem_g_cum[j]));
                    s += DW[i * Kdim + kk] * kbg;
                }
                smem_dM[i * kMaxC + j] = s;
            }
        }
        __syncthreads();

        // ================================================================
        // d_v, d_beta from M^T @ DU and M^T @ DW
        // ================================================================
        for (int j = 0; j < L; ++j) {
            const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;
            const long k_j_base = (((long)b * Tlen + cs + j) * H + h) * Kdim;

            // d_v[j,v] = beta[j] * Σ_i M[i,j] * DU[i,v]   (M^T column j)
            for (int vv = tid; vv < Vdim; vv += nthreads) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += smem_M[i * kMaxC + j] * DU[i * Vdim + vv];
                }
                d_v[v_j_base + vv] = from_float<TQ>(t * smem_beta[j]);
                atomicAdd(&DB[j], t * to_float(smem_v_loc[j * Vdim + vv]));
            }
            __syncthreads();

            // DK from M^T @ DW, and DB, DG contributions
            for (int kk = tid; kk < Kdim; kk += nthreads) {
                float t = 0.0f;
                for (int i = j; i < L; ++i) {
                    t += smem_M[i * kMaxC + j] * DW[i * Kdim + kk];
                }
                const float kj = to_float(smem_k[j * Kdim + kk]);
                const float egj = expf(smem_g_cum[j]);
                atomicAdd(&DK[j * Kdim + kk], t * smem_beta[j] * egj);
                atomicAdd(&DB[j], t * egj * kj);
                atomicAdd(&DG[j], t * smem_beta[j] * egj * kj);
            }
            __syncthreads();
        }

        // ================================================================
        // Back-propagate through A: dA_grad = -M^T @ dM @ M^T
        // tmp = dM @ M^T then result = -M^T @ tmp
        // Reuse smem_dM for tmp_mat after dM is consumed.
        // We need dM and M simultaneously, so use a 2-pass approach
        // writing tmp over dM.
        // ================================================================

        // Pass 1: tmp[i,j] = Σ_m dM[i,m] * M[j,m]  (= dM @ M^T)
        // Compute into smem_dM (overwriting dM)
        // But we need dM during this computation! Do it in-place with a
        // temporary in registers (process one row at a time).

        // Use workspace DB area as temporary (it's small and we're done with it
        // for the current row computations). Actually, let's use a different
        // approach: compute tmp into the workspace area after DB.
        // The workspace has extra space after DB.

        // Actually, we have space in the workspace after DB:
        // ws_after_DB = ws_DB + chunk_size (= DB end)
        float* ws_tmp = ws + ws_DB + chunk_size;
        // This overlaps nothing since workspace_stride >= ws_DB + chunk_size + 2*L*L
        // (caller must allocate enough). We'll check this in the launch function.

        // tmp1[i,j] = Σ_m dM[i,m] * M[j,m]
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m) {
                s += smem_dM[i * kMaxC + m] * smem_M[j * kMaxC + m];
            }
            ws_tmp[i * L + j] = s;
        }
        __syncthreads();

        // tmp2[i,j] = -Σ_m M[m,i] * tmp1[m,j]   (= -M^T @ tmp1)
        // Write result back to smem_dM (reuse as dA_grad)
        for (int idx = tid; idx < L * L; idx += nthreads) {
            const int i = idx / L;
            const int j = idx % L;
            float s = 0.0f;
            for (int m = 0; m < L; ++m) {
                s += smem_M[m * kMaxC + i] * ws_tmp[m * L + j];
            }
            smem_dM[i * kMaxC + j] = -s;
        }
        __syncthreads();

        // dA_grad is now in smem_dM. Use it to update DK, DB, DG.
        for (int i = 0; i < L; ++i) {
            for (int j = tid; j < i; j += nthreads) {
                // dot(k_norm[i], k_norm[j])
                float dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_k += to_float(smem_k[i * Kdim + kk])
                           * to_float(smem_k[j * Kdim + kk]);
                }
                const float exp_ij = expf(smem_g_cum[i] - smem_g_cum[j]);
                const float a_grad = smem_dM[i * kMaxC + j];
                const float val = dot_k * exp_ij;

                atomicAdd(&DB[i], a_grad * val);

                const float dval = a_grad * smem_beta[i];
                const float ddot = dval * exp_ij;
                const float dexp = dval * dot_k;

                atomicAdd(&DG[i],  dexp * exp_ij);
                atomicAdd(&DG[j], -dexp * exp_ij);

                for (int kk = 0; kk < Kdim; ++kk) {
                    const float ki = to_float(smem_k[i * Kdim + kk]);
                    const float kj = to_float(smem_k[j * Kdim + kk]);
                    atomicAdd(&DK[i * Kdim + kk], ddot * kj);
                    atomicAdd(&DK[j * Kdim + kk], ddot * ki);
                }
            }
            __syncthreads();
        }

        // ================================================================
        // Write d_q, d_k (with L2norm backward if needed), d_beta
        // ================================================================
        for (int i = tid; i < L; i += nthreads) {
            const long q_i_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
            const long k_i_base = q_i_base;
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;

            if (use_qk_l2norm_in_kernel) {
                float dot_q = 0.0f, dot_k = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_q += DQ[i * Kdim + kk] * to_float(smem_q[i * Kdim + kk]);
                    dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
                }
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qn = to_float(smem_q[i * Kdim + kk]);
                    const float kn = to_float(smem_k[i * Kdim + kk]);
                    d_q[q_i_base + kk] = from_float<TQ>(
                        (DQ[i * Kdim + kk] - qn * dot_q) * smem_inv_q[i]);
                    d_k[k_i_base + kk] = from_float<TQ>(
                        (DK[i * Kdim + kk] - kn * dot_k) * smem_inv_k[i]);
                }
            } else {
                for (int kk = 0; kk < Kdim; ++kk) {
                    d_q[q_i_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                    d_k[k_i_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
                }
            }
            d_beta[gh_idx] = from_float<TB>(DB[i]);
        }
        __syncthreads();

        // ================================================================
        // Reverse cumsum for d_g
        // ================================================================
        // DG[i] contains per-position gradients. d_g output needs reverse cumsum.
        if (tid == 0) {
            float running = 0.0f;
            for (int i = L - 1; i >= 0; --i) {
                running += DG[i];
                const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
                d_g[gh_idx] = from_float<TG>(running);
            }
        }
        __syncthreads();

        // ================================================================
        // Propagate ds backward: ds = dh_in
        // ================================================================
        for (long idx = tid; idx < kv; idx += nthreads) {
            ds[idx] = dh_in[idx];
        }
        __syncthreads();
    }
}

// ============================================================================
// Multi-kernel backward — Phase 1: chunk-parallel local gradients
//
// Grid: (B*H*num_chunks, 1, 1), each block processes one (b,h,chunk).
// Recomputes M, W, VNEW from checkpoints.
// Computes all ds-independent gradients: DQ, DK (term1+term2), DU (term2),
// DG (term1+term2), DB partial, DHT1 (dh contribution from term1).
// Stores results in per-chunk workspace for Phase 2 and 3.
// ============================================================================

// Per-chunk workspace offsets (all in floats, Lp-padded where needed)
struct ChunkWorkspaceLayout {
    int M_off;       // [Lp×Lp]
    int A_off;       // [Lp×Lp]
    int W_off;       // [Lp×K]
    int VNEW_off;    // [Lp×V]
    int DU_off;      // [Lp×V]
    int DW_off;      // [Lp×K]
    int DQ_off;      // [Lp×K]
    int DK_off;      // [Lp×K]
    int DG_off;      // [Lp]
    int DB_off;      // [Lp]
    int DHT1_off;    // [K×V]
    int C_off;       // [K×K] float — correction matrix for Phase 2 ds recurrence
    int EG_off;      // [1]   — exp(g_last) per chunk
    int total;       // total floats per chunk
};

__host__ __device__ __forceinline__ int align_up_int(int x, int align) {
    return ((x + align - 1) / align) * align;
}

template<typename TQ>
__host__ __device__ ChunkWorkspaceLayout make_chunk_ws(int Lp, int Kdim, int Vdim) {
    constexpr int kWsAlignFloats = 128 / sizeof(float);
    ChunkWorkspaceLayout l;
    int off = 0;
    const int c_storage_floats = Kdim * Kdim;
    l.M_off    = align_up_int(off, kWsAlignFloats); off = l.M_off + Lp * Lp;
    l.A_off    = align_up_int(off, kWsAlignFloats); off = l.A_off + Lp * Lp;
    l.W_off    = align_up_int(off, kWsAlignFloats); off = l.W_off + Lp * Kdim;
    l.VNEW_off = align_up_int(off, kWsAlignFloats); off = l.VNEW_off + Lp * Vdim;
    l.DU_off   = align_up_int(off, kWsAlignFloats); off = l.DU_off + Lp * Vdim;
    l.DW_off   = align_up_int(off, kWsAlignFloats); off = l.DW_off + Lp * Kdim;
    l.DQ_off   = align_up_int(off, kWsAlignFloats); off = l.DQ_off + Lp * Kdim;
    l.DK_off   = align_up_int(off, kWsAlignFloats); off = l.DK_off + Lp * Kdim;
    l.DG_off   = align_up_int(off, kWsAlignFloats); off = l.DG_off + Lp;
    l.DB_off   = align_up_int(off, kWsAlignFloats); off = l.DB_off + Lp;
    l.DHT1_off = align_up_int(off, kWsAlignFloats); off = l.DHT1_off + Kdim * Vdim;
    l.C_off    = align_up_int(off, kWsAlignFloats); off = l.C_off + c_storage_floats;
    l.EG_off   = align_up_int(off, kWsAlignFloats); off = l.EG_off + 1;
    l.total    = align_up_int(off, kWsAlignFloats);
    return l;
}

template<typename TQ, typename TG, typename TB>
__global__ void gdr_bwd_phase1_wmma(
    const TQ* __restrict__ d_out,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    const float* __restrict__ d_final_state,
    float* __restrict__ chunk_workspace,  // [B*H*num_chunks, chunk_ws_stride]
    float* __restrict__ ds_accum,         // [B, H, K*V] — each chunk atomicAdds dh_term1
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size, float scale,
    bool use_qk_l2norm_in_kernel)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int block_id = blockIdx.x;
    const int bh = block_id / num_chunks;
    const int chunk = block_id % num_chunks;
    const int b = bh / H;
    const int h = bh % H;
    const long kv = (long)Kdim * Vdim;
    const int Lp = kMaxC;

    const int cs = chunk * chunk_size;
    const int L = min(chunk_size, Tlen - cs);

    const float* cp_base = checkpoints + (long)bh * (num_chunks + 1) * kv;
    const float* h_in = cp_base + (long)chunk * kv;

    // Per-chunk workspace
    float* cws = chunk_workspace + (long)block_id * chunk_ws_stride;
    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    float* M    = cws + cwl.M_off;
    float* A    = cws + cwl.A_off;
    float* W    = cws + cwl.W_off;
    float* VNEW = cws + cwl.VNEW_off;
    float* DU   = cws + cwl.DU_off;
    float* DW   = cws + cwl.DW_off;
    float* DQ   = cws + cwl.DQ_off;
    float* DK   = cws + cwl.DK_off;
    float* DG   = cws + cwl.DG_off;
    float* DB   = cws + cwl.DB_off;
    float* DHT1 = cws + cwl.DHT1_off;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Zero-fill shared buffers
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        smem_k[idx] = from_float<TQ>(0.0f);
        smem_q[idx] = from_float<TQ>(0.0f);
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
        smem_e_last[tid] = 0.0f;
    }
    __syncthreads();

    // Load k, q, v
    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
        smem_k[pos * Kdim + kk] = k_global[gi];
        smem_q[pos * Kdim + kk] = q_global[gi];
    }
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int pos = idx / Vdim, vv = idx % Vdim;
        buf2[pos * Vdim + vv] = v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < L; ++i) {
            const long gh = ((long)b * Tlen + cs + i) * H + h;
            acc += to_float(g_global[gh]);
            smem_gcum[i] = acc;
            smem_beta[i] = to_float(beta_global[gh]);
        }
    }
    __syncthreads();

    for (int pos = tid; pos < L; pos += nthr) {
        smem_eg[pos] = expf(smem_gcum[pos]);
    }
    __syncthreads();
    const float g_last_val = (L > 0) ? smem_gcum[L - 1] : 0.0f;
    const float eg_last_val = (L > 0) ? smem_eg[L - 1] : 0.0f;
    const float inv_eg_last_val = (eg_last_val != 0.0f) ? (1.0f / eg_last_val) : 0.0f;
    for (int pos = tid; pos < L; pos += nthr) {
        smem_e_last[pos] = expf(g_last_val - smem_gcum[pos]);
    }
    __syncthreads();

    // L2 norms
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f, kn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float qv = to_float(smem_q[pos * Kdim + kk]);
                float kv2 = to_float(smem_k[pos * Kdim + kk]);
                qn2 += qv * qv; kn2 += kv2 * kv2;
            }
            smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
            smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
        } else {
            smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            smem_k[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_k[idx]) * smem_invk[pos]));
            smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // ================================================================
    // Build M: A = beta*exp(g)*(k@k^T), solve M
    // ================================================================
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i)
            scratch1[idx] *= smem_beta[i] * (smem_eg[i] * smem_e_last[j] * inv_eg_last_val);
        else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();
    // Save A for Phase 3
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        A[idx] = scratch1[idx];

    // Solve M → scratch2
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
    // Save M
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        M[idx] = scratch2[idx];
    __syncthreads();

    // M → bf16 in buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // ================================================================
    // Recompute W and VNEW
    // ================================================================
    // bv = beta*v → buf2
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(to_float(buf2[idx]) * smem_beta[i]));
    }
    __syncthreads();

    // u = M @ bv → scratch1[Lp×V]
    wmma_nn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    __syncthreads();

    // bkg = beta*k*exp(g) → buf3
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? to_float(smem_k[idx]) * smem_beta[i] * smem_eg[i] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // w = M @ bkg → scratch1[Lp×K]
    wmma_nn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        float wval = bf16_trunc<TQ>(scratch1[idx]);
        buf3[idx] = from_float<TQ>(wval);
        W[idx] = wval;
    }
    __syncthreads();

    // h_bf16 → buf1[K×V]
    for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
    for (int idx = Kdim * Vdim + tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    __syncthreads();

    // wh = w @ h_bf16 → scratch1[Lp×V]
    wmma_nn<TQ>(buf3, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
    __syncthreads();

    // vnew_pre = u - wh, then vnew = vnew_pre * exp(g_last - g_cum[i])
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        float vnew_pre = bf16_trunc<TQ>(to_float(buf2[idx]) - scratch1[idx]);
        float e_i = (i < L) ? smem_e_last[i] : 0.0f;
        float vnew = bf16_trunc<TQ>(vnew_pre * e_i);
        buf2[idx] = from_float<TQ>(vnew);
        VNEW[idx] = vnew;
    }
    __syncthreads();

    // ================================================================
    // Zero gradient accumulators
    // ================================================================
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        DQ[idx] = 0.0f; DK[idx] = 0.0f; DW[idx] = 0.0f;
    }
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        DU[idx] = 0.0f;
    for (int idx = tid; idx < Lp; idx += nthr) {
        DG[idx] = 0.0f; DB[idx] = 0.0f;
    }
    for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
        DHT1[idx] = 0.0f;
    __syncthreads();

    // ================================================================
    // term1: q @ h contribution
    // ================================================================
    // Load d_out → buf2 (overwriting vnew bf16, VNEW saved to workspace)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr)
        buf2[idx] = from_float<TQ>(0.0f);
    __syncthreads();
    for (int idx = tid; idx < L * Vdim; idx += nthr) {
        const int pos = idx / Vdim, vv = idx % Vdim;
        buf2[idx] = d_out[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv];
    }
    __syncthreads();

    // DQ_term1 = d_out @ h_bf16^T → scratch1[Lp×K]
    wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
    __syncthreads();
    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        DQ[idx] += scale * smem_eg[i] * scratch1[i * Kdim + idx % Kdim];
    }
    __syncthreads();

    // DHT1 = (scale*exp(g)*q)^T @ d_out → [K×V]
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? to_float(smem_q[idx]) * scale * smem_eg[i] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();
    wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
        DHT1[idx] = scratch1[idx];
    __syncthreads();

    // DG term1: q@h per row, dot with d_out
    wmma_nn<TQ>(smem_q, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
    __syncthreads();
    for (int i = tid; i < L; i += nthr) {
        float dot_val = 0.0f;
        for (int vv = 0; vv < Vdim; ++vv)
            dot_val += scratch1[i * Vdim + vv] * to_float(buf2[i * Vdim + vv]);
        DG[i] += scale * smem_eg[i] * dot_val;
    }
    __syncthreads();

    // ================================================================
    // term2: Batched intra-chunk attention gradient
    // ================================================================
    // S = q @ k^T
    wmma_nt<TQ>(smem_q, Kdim, smem_k, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i) {
            const float exp_ij = smem_eg[i] * smem_e_last[j] * inv_eg_last_val;
            scratch1[idx] = bf16_trunc<TQ>(exp_ij * scratch1[idx]);
        }
        else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();

    // vnew_pre[j,v] = VNEW[j,v] / exp(g_last - g_cum[j])
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    __syncthreads();
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int j = idx / Vdim;
        float e_j = (j < L) ? smem_e_last[j] : 1.0f;
        float vnp = (j < L) ? VNEW[j * Vdim + idx % Vdim] / e_j : 0.0f;
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(vnp));
    }
    __syncthreads();

    // Scale d_out
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        float val = (i < L) ? to_float(buf2[idx]) * scale : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // Save S to scratch2 for later
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch2[idx] = scratch1[idx];
    __syncthreads();

    // grad_S = scale_d_out @ vnew_pre^T → scratch1[Lp×Lp]
    wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Lp, Lp, Lp, Vdim);
    __syncthreads();

    // DG from S gradient without atomics:
    // DG[i] += sum_{j<i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
    for (int i = tid; i < L; i += nthr) {
        float dg_pos = 0.0f;
        float dg_neg = 0.0f;
        for (int j = 0; j < i; ++j) {
            dg_pos += scratch1[i * Lp + j] * scratch2[i * Lp + j];
        }
        for (int r = i + 1; r < L; ++r) {
            dg_neg += scratch1[r * Lp + i] * scratch2[r * Lp + i];
        }
        DG[i] += dg_pos - dg_neg;
    }
    __syncthreads();

    // grad_S_masked = grad_S * exp(g[i]-g[j]) → buf1
    // (scratch1 still has grad_S from the WMMA above)
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        float val = 0.0f;
        if (i < L && j <= i)
            val = scratch1[idx] * (smem_eg[i] * smem_e_last[j] * inv_eg_last_val);
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // DQ_term2 = grad_S_masked @ k
    wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < L * Kdim; idx += nthr)
        DQ[idx] += scratch1[idx];
    __syncthreads();

    // DK_term2 = grad_S_masked^T @ q
    wmma_tn<TQ>(buf1, Lp, smem_q, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < L * Kdim; idx += nthr)
        DK[idx] += scratch1[idx];
    __syncthreads();

    // DU_term2 = S_scaled^T @ scale_d_out
    // S_scaled = S / e_j → buf1 (scratch2 still has S)
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        float val = 0.0f;
        if (i < L && j <= i && j < L)
            val = scratch2[idx] / smem_e_last[j];
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();
    wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
    __syncthreads();
    for (int idx = tid; idx < L * Vdim; idx += nthr)
        DU[idx] += scratch1[idx];
    __syncthreads();

    // ================================================================
    // Precompute correction matrix C and correction_local for Phase 2
    // This allows Phase 2 to use a single matmul: ds = ds*eg + DHT1_corrected - C@ds
    // ================================================================
    float* C_mat = cws + cwl.C_off;
    const float g_last_val_p1 = g_last_val;

    // Store eg_last for Phase 2
    if (tid == 0) {
        float* eg_ptr = cws + cwl.EG_off;
        *eg_ptr = eg_last_val;
    }

    // W_gated → buf3[Lp×K]: W[i,k] * exp(g_last - g[i])
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int i = idx / Kdim;
        float val = (i < L) ? W[idx] * smem_e_last[i] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // C = W_gated^T @ k → scratch1[K×K]
    // buf3^T[K×Lp] @ smem_k[Lp×K] → scratch1[K×K]
    wmma_tn<TQ>(buf3, Kdim, smem_k, Kdim, scratch1, Kdim, Kdim, Kdim, Lp);
    __syncthreads();

    // Store C in workspace as fp32; Phase 2 converts to TQ for WMMA use.
    for (int idx = tid; idx < Kdim * Kdim; idx += nthr)
        C_mat[idx] = scratch1[idx];
    __syncthreads();

    // d_pre_local = DU * exp(g_last - g[i]) → buf2[Lp×V]
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int i = idx / Vdim;
        float val = (i < L) ? DU[idx] * smem_e_last[i] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // correction_local = W^T @ d_pre_local → scratch1[K×V]
    // W is in workspace (float). Load W → buf3 bf16 (we have it from W_gated, need ungated W)
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        float val = (idx < L * Kdim) ? W[idx] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // W^T @ d_pre_local → scratch1[K×V]
    wmma_tn<TQ>(buf3, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Lp);
    __syncthreads();

    // DHT1 -= correction_local
    for (int idx = tid; idx < Kdim * Vdim; idx += nthr)
        DHT1[idx] -= scratch1[idx];
    __syncthreads();
}

// ============================================================================
// Multi-kernel backward — Phase 2: sequential ds propagation
//
// Grid: (B*H, 1, 1), each block processes one (b,h) sequentially over chunks.
// Iterates chunks backward. For each chunk:
//   - Reads DU, W, VNEW, DHT1 from Phase 1 workspace
//   - Computes ds-dependent contributions: k@ds → DU_state, VNEW@ds^T → DK_state
//   - Computes d_pre = DU*e_i, DW = -d_pre@h^T, dh_in updates
//   - Propagates ds backward
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_bwd_phase2_wmma(
    const float* __restrict__ checkpoints,
    const float* __restrict__ d_final_state,
    float* __restrict__ d_initial_state,
    float* __restrict__ chunk_workspace,
    float* __restrict__ dh_storage,       // [B*H*num_chunks, K*V] per-chunk dh
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size)
{
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;
    const int bh = blockIdx.x;
    const long kv = (long)Kdim * Vdim;
    const long kk = (long)Kdim * Kdim;
    const int Lp = kMaxC;
    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);

    float* ds = d_initial_state + (long)bh * kv;

    // Shared memory: scratch1[K×V] + buf1[K×K bf16] + buf2[K×V bf16]
    extern __shared__ char smem_raw[];
    float* scratch1 = (float*)smem_raw;                          // [K×V]
    TQ*    buf1     = (TQ*)(scratch1 + Kdim * Vdim);             // [K×K] (C_bf16)
    TQ*    buf2     = buf1 + Kdim * Kdim;                        // [K×V] (ds_bf16)

    // Seed ds from d_final_state
    for (long idx = tid; idx < kv; idx += nthr)
        ds[idx] = d_final_state ? d_final_state[(long)bh * kv + idx] : 0.0f;
    __syncthreads();

    for (int chunk = num_chunks - 1; chunk >= 0; --chunk) {
        // Store entering ds for this chunk (Phase 3 needs it)
        float* dh_store = dh_storage + (long)(bh * num_chunks + chunk) * kv;
        for (long idx = tid; idx < kv; idx += nthr)
            dh_store[idx] = ds[idx];
        __syncthreads();

        // Per-chunk workspace
        const int chunk_block_id = bh * num_chunks + chunk;
        float* cws = chunk_workspace + (long)chunk_block_id * chunk_ws_stride;
        float* DHT1 = cws + cwl.DHT1_off;   // already corrected by Phase 1
        const float* C_mat = cws + cwl.C_off;
        float  eg_last = *(cws + cwl.EG_off);

        // Load C (fp32) and convert to TQ into buf1 [K×K]
        for (long idx = tid; idx < kk; idx += nthr)
            buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(C_mat[idx]));
        // Load ds → buf2 bf16 [K×V]
        for (long idx = tid; idx < kv; idx += nthr)
            buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(ds[idx]));
        __syncthreads();

        // C @ ds → scratch1[K×V]
        wmma_nn<TQ>(buf1, Kdim, buf2, Vdim, scratch1, Vdim, Kdim, Vdim, Kdim);
        __syncthreads();

        // ds = ds * eg_last + DHT1_corrected - C@ds
        for (long idx = tid; idx < kv; idx += nthr)
            ds[idx] = ds[idx] * eg_last + DHT1[idx] - scratch1[idx];
        __syncthreads();
    }
}

// ============================================================================
// Multi-kernel backward — Phase 3: chunk-parallel finalization
//
// Grid: (B*H*num_chunks, 1, 1)
// Computes dM, M^T@DU, M^T@DW, dA_grad, writes d_q/d_k/d_v/d_beta/d_g.
// ============================================================================

template<typename TQ, typename TG, typename TB>
__global__ void gdr_bwd_phase3_wmma(
    TQ* __restrict__ d_q,
    TQ* __restrict__ d_k,
    TQ* __restrict__ d_v,
    TG* __restrict__ d_g,
    TB* __restrict__ d_beta,
    const TQ* __restrict__ d_out_global,
    const TQ* __restrict__ q_global,
    const TQ* __restrict__ k_global,
    const TQ* __restrict__ v_global,
    const TG* __restrict__ g_global,
    const TB* __restrict__ beta_global,
    const float* __restrict__ checkpoints,
    const float* __restrict__ dh_storage,   // [B*H*num_chunks, K*V] per-chunk dh from Phase 2
    float* __restrict__ chunk_workspace,
    int chunk_ws_stride,
    int Tlen, int H, int Kdim, int Vdim,
    int num_chunks, int chunk_size,
    float scale,
    bool use_qk_l2norm_in_kernel)
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
    const bool tail_chunk = (L < chunk_size);

    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    float* cws = chunk_workspace + (long)block_id * chunk_ws_stride;
    float* M_ws  = cws + cwl.M_off;
    float* VNEW  = cws + cwl.VNEW_off;
    float* DU    = cws + cwl.DU_off;
    float* DW    = cws + cwl.DW_off;
    float* DQ    = cws + cwl.DQ_off;
    float* DK    = cws + cwl.DK_off;
    float* DG    = cws + cwl.DG_off;
    float* DB    = cws + cwl.DB_off;

    // Per-chunk dh from Phase 2
    const long kv = (long)Kdim * Vdim;
    const float* dh_chunk = dh_storage + (long)block_id * kv;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* scratch1  = (float*)smem_raw;
    float* scratch2  = scratch1 + Lp * Lp;
    TQ*    smem_k    = (TQ*)(scratch2 + Lp * Lp);
    TQ*    smem_q    = smem_k + Lp * Kdim;
    TQ*    buf1      = smem_q + Lp * Kdim;
    TQ*    buf2      = buf1 + Lp * Lp;
    TQ*    buf3      = buf2 + Lp * Vdim;
    float* smem_gcum = (float*)(buf3 + Lp * Kdim);
    float* smem_beta = smem_gcum + Lp;
    float* smem_invq = smem_beta + Lp;
    float* smem_invk = smem_invq + Lp;
    float* smem_eg   = smem_invk + Lp;
    float* smem_e_last = smem_eg + Lp;

    // Load k, q, scalars
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        smem_k[idx] = from_float<TQ>(0.0f);
        smem_q[idx] = from_float<TQ>(0.0f);
    }
    if (tid < Lp) {
        smem_gcum[tid] = 0.0f;
        smem_beta[tid] = 0.0f;
        smem_eg[tid] = 0.0f;
        smem_e_last[tid] = 0.0f;
    }
    __syncthreads();

    for (int idx = tid; idx < L * Kdim; idx += nthr) {
        const int pos = idx / Kdim, kk = idx % Kdim;
        const long gi = (((long)b * Tlen + cs + pos) * H + h) * Kdim + kk;
        smem_k[pos * Kdim + kk] = k_global[gi];
        smem_q[pos * Kdim + kk] = q_global[gi];
    }
    if (tid == 0) {
        float acc = 0.0f;
        for (int i = 0; i < L; ++i) {
            const long gh = ((long)b * Tlen + cs + i) * H + h;
            acc += to_float(g_global[gh]);
            smem_gcum[i] = acc;
            smem_beta[i] = to_float(beta_global[gh]);
        }
    }
    __syncthreads();

    for (int pos = tid; pos < L; pos += nthr) {
        smem_eg[pos] = expf(smem_gcum[pos]);
    }
    __syncthreads();
    const float g_last_p3 = (L > 0) ? smem_gcum[L - 1] : 0.0f;
    const float eg_last_p3 = (L > 0) ? smem_eg[L - 1] : 0.0f;
    const float inv_eg_last_p3 = (eg_last_p3 != 0.0f) ? (1.0f / eg_last_p3) : 0.0f;
    for (int pos = tid; pos < L; pos += nthr) {
        smem_e_last[pos] = expf(g_last_p3 - smem_gcum[pos]);
    }
    __syncthreads();

    // L2 norms
    for (int pos = tid; pos < L; pos += nthr) {
        if (use_qk_l2norm_in_kernel) {
            float qn2 = 0.0f, kn2 = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                float qv = to_float(smem_q[pos * Kdim + kk]);
                float kv2 = to_float(smem_k[pos * Kdim + kk]);
                qn2 += qv * qv; kn2 += kv2 * kv2;
            }
            smem_invq[pos] = 1.0f / sqrtf(qn2 + 1e-6f);
            smem_invk[pos] = 1.0f / sqrtf(kn2 + 1e-6f);
        } else {
            smem_invq[pos] = 1.0f; smem_invk[pos] = 1.0f;
        }
    }
    __syncthreads();
    if (use_qk_l2norm_in_kernel) {
        for (int idx = tid; idx < L * Kdim; idx += nthr) {
            const int pos = idx / Kdim;
            smem_k[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_k[idx]) * smem_invk[pos]));
            smem_q[pos * Kdim + idx % Kdim] = from_float<TQ>(
                bf16_trunc<TQ>(to_float(smem_q[idx]) * smem_invq[pos]));
        }
        __syncthreads();
    }

    // Checkpoint access for h_in
    const float* cp_base = checkpoints + (long)bh * (num_chunks + 1) * kv;
    const float* h_in = cp_base + (long)chunk * kv;
    // Reuse per-chunk DHT1 storage as temporary global staging for phase3.
    float* tmp_area = cws + cwl.DHT1_off;

    // Tail chunks are sensitive to padded WMMA numerics; recompute DG base terms
    // (term1 + intra-chunk term2) in scalar math to match reference behavior.
    if (tail_chunk) {
        for (int i = tid; i < L; i += nthr) {
            DG[i] = 0.0f;
        }
        __syncthreads();

        // term1: scale * exp(g[i]) * dot(d_out[i], q[i] @ bf16(h_in))
        for (int i = tid; i < L; i += nthr) {
            float dg_i = 0.0f;
            const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
            for (int vv = 0; vv < Vdim; ++vv) {
                const float do_v = to_float(d_out_global[do_i_base + vv]) * scale;
                float qh = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    const float qi = to_float(smem_q[i * Kdim + kk]);
                    const float hq = bf16_trunc<TQ>(h_in[kk * Vdim + vv]);
                    qh += qi * hq;
                }
                dg_i += do_v * smem_eg[i] * qh;
            }
            DG[i] = dg_i;
        }
        __syncthreads();

        // term2: DG[i] += sum_{j<i}(grad_S[i,j] * S[i,j]) - sum_{r>i}(grad_S[r,i] * S[r,i])
        for (int i = tid; i < L; i += nthr) {
            float dg_pos = 0.0f;
            float dg_neg = 0.0f;

            for (int j = 0; j < i; ++j) {
                const float exp_ij = smem_eg[i] * smem_e_last[j] * inv_eg_last_p3;
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_qk += to_float(smem_q[i * Kdim + kk]) * to_float(smem_k[j * Kdim + kk]);
                }
                const float s_ij = bf16_trunc<TQ>(exp_ij * dot_qk);

                const float e_j = smem_e_last[j];
                float grad_s = 0.0f;
                const long do_i_base = (((long)b * Tlen + cs + i) * H + h) * Vdim;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float do_v = to_float(d_out_global[do_i_base + vv]) * scale;
                    const float v_new_pre = VNEW[j * Vdim + vv] / e_j;
                    grad_s += do_v * v_new_pre;
                }
                dg_pos += grad_s * s_ij;
            }

            for (int r = i + 1; r < L; ++r) {
                const float exp_ri = smem_eg[r] * smem_e_last[i] * inv_eg_last_p3;
                float dot_qk = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    dot_qk += to_float(smem_q[r * Kdim + kk]) * to_float(smem_k[i * Kdim + kk]);
                }
                const float s_ri = bf16_trunc<TQ>(exp_ri * dot_qk);

                const float e_i = smem_e_last[i];
                float grad_s = 0.0f;
                const long do_r_base = (((long)b * Tlen + cs + r) * H + h) * Vdim;
                for (int vv = 0; vv < Vdim; ++vv) {
                    const float do_v = to_float(d_out_global[do_r_base + vv]) * scale;
                    const float v_new_pre = VNEW[i * Vdim + vv] / e_i;
                    grad_s += do_v * v_new_pre;
                }
                dg_neg += grad_s * s_ri;
            }

            DG[i] += dg_pos - dg_neg;
        }
        __syncthreads();
    }

    // ================================================================
    // Step 0-pre: k@ds → add to DU, apply gating, DG contributions
    // (moved from Phase 2 to enable single-matmul Phase 2)
    // ================================================================
    // Load dh_chunk (entering ds) → buf1[K×V] (zero-pad to Lp×Lp)
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    __syncthreads();
    for (long idx = tid; idx < kv; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(dh_chunk[idx]));
    __syncthreads();

    float dg_last_local = 0.0f;
    if (!tail_chunk) {
        // k @ ds → scratch1[Lp×V]
        wmma_nn<TQ>(smem_k, Kdim, buf1, Vdim, scratch1, Vdim, Lp, Vdim, Kdim);
        __syncthreads();

        // DU += k@ds, then apply gating: DU → d_pre
        for (int idx = tid; idx < L * Vdim; idx += nthr) {
            const int i = idx / Vdim;
            const float e_i = smem_e_last[i];
            const float kds_val = scratch1[idx];
            const float d_vnew = DU[idx] + kds_val;
            const float d_pre = d_vnew * e_i;

            DU[idx] = d_pre;  // overwrite DU with d_pre
        }
        __syncthreads();

        // DG gating contribution:
        //   DG[i]     -= Σ_v (k@ds)[i,v] * VNEW[i,v]
        //   DG[L - 1] += Σ_i,v (k@ds)[i,v] * VNEW[i,v]
        for (int i = tid; i < L; i += nthr) {
            float row_sum = 0.0f;
            for (int vv = 0; vv < Vdim; ++vv) {
                row_sum += scratch1[i * Vdim + vv] * VNEW[i * Vdim + vv];
            }
            DG[i] -= row_sum;
            dg_last_local += row_sum;
        }
    } else {
        // Tail chunks: compute k@ds in fp32 scalar math to match reference numerics.
        for (int i = tid; i < L; i += nthr) {
            float row_sum = 0.0f;
            const float e_i = smem_e_last[i];
            for (int vv = 0; vv < Vdim; ++vv) {
                float kds_val = 0.0f;
                for (int kk = 0; kk < Kdim; ++kk) {
                    kds_val += dh_chunk[(long)kk * Vdim + vv] * to_float(smem_k[i * Kdim + kk]);
                }
                const int idx = i * Vdim + vv;
                const float d_vnew = DU[idx] + kds_val;
                const float d_pre = d_vnew * e_i;
                DU[idx] = d_pre;
                row_sum += kds_val * VNEW[idx];
            }
            DG[i] -= row_sum;
            dg_last_local += row_sum;
        }
    }

    const float dg_last_warp = warp_reduce_sum_f32(dg_last_local);
    if ((tid & 31) == 0) {
        scratch1[tid / 32] = dg_last_warp;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int w = 0; w < nthr / 32; ++w) {
            block_sum += scratch1[w];
        }
        DG[L - 1] += block_sum;
    }
    __syncthreads();

    // DG[L-1] += eg_last * Σ(ds * h_in)
    float ds_h_local = 0.0f;
    for (long idx = tid; idx < kv; idx += nthr) {
        ds_h_local += dh_chunk[idx] * h_in[idx];
    }
    const float ds_h_warp = warp_reduce_sum_f32(ds_h_local);
    if ((tid & 31) == 0) {
        scratch1[tid / 32] = ds_h_warp;
    }
    __syncthreads();
    if (tid == 0) {
        float block_sum = 0.0f;
        for (int w = 0; w < nthr / 32; ++w) {
            block_sum += scratch1[w];
        }
        DG[L - 1] += block_sum * eg_last_p3;
    }
    __syncthreads();

    // ================================================================
    // Step 0a: VNEW @ dh^T → DK
    // ================================================================
    // Load VNEW → buf2[Lp×V]
    // buf1 still has dh_chunk from Step 0-pre (only used as read-only WMMA input since then)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        float val = (idx < L * Vdim) ? VNEW[idx] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // VNEW @ dh^T → scratch1[Lp×K]
    wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
    __syncthreads();
    for (int idx = tid; idx < L * Kdim; idx += nthr)
        DK[idx] += scratch1[idx];
    __syncthreads();

    // ================================================================
    // Step 0b: DW = -d_pre @ h_in^T
    // ================================================================
    // d_pre is now in DU[L×V] (computed in Step 0-pre)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    // Load h_in → buf1[K×V] (zero-pad to Lp×Lp)
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    __syncthreads();
    for (long idx = tid; idx < kv; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(h_in[idx]));
    __syncthreads();

    // d_pre @ h_in^T → scratch1[Lp×K]
    wmma_nt<TQ>(buf2, Vdim, buf1, Vdim, scratch1, Kdim, Lp, Kdim, Vdim);
    __syncthreads();
    for (int idx = tid; idx < L * Kdim; idx += nthr)
        DW[idx] = -scratch1[idx];
    __syncthreads();

    // Load active causal block of M and zero-pad the rest.
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp;
        const int j = idx % Lp;
        scratch2[idx] = (i < L && j <= i) ? M_ws[idx] : 0.0f;
    }
    __syncthreads();

    // ================================================================
    // Step 1: M^T @ DU → d_v, DB
    // ================================================================
    // M_bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    // DU_bf16 → buf2[Lp×V]
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        float val = (idx < L * Vdim) ? DU[idx] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // MT_DU → scratch1[Lp×V]
    wmma_tn<TQ>(buf1, Lp, buf2, Vdim, scratch1, Vdim, Lp, Vdim, Lp);
    __syncthreads();

    // Load raw v → buf2 for DB (and later reuse for beta*V in Step 3)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int pos = idx / Vdim, vv = idx % Vdim;
        buf2[idx] = (pos < L) ?
            v_global[(((long)b * Tlen + cs + pos) * H + h) * Vdim + vv] :
            from_float<TQ>(0.0f);
    }
    __syncthreads();

    for (int j = tid; j < L; j += nthr) {
        const long v_j_base = (((long)b * Tlen + cs + j) * H + h) * Vdim;
        float db_acc = 0.0f;
        for (int vv = 0; vv < Vdim; ++vv) {
            float mt_du = scratch1[j * Vdim + vv];
            d_v[v_j_base + vv] = from_float<TQ>(mt_du * smem_beta[j]);
            db_acc += mt_du * to_float(buf2[j * Vdim + vv]);
        }
        DB[j] += db_acc;
    }
    __syncthreads();

    // ================================================================
    // Step 2: M^T @ DW → DK, DB, DG
    // ================================================================
    // DW_bf16 → buf3
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        float val = (idx < L * Kdim) ? DW[idx] : 0.0f;
        buf3[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // MT_DW → scratch1[Lp×K]
    wmma_tn<TQ>(buf1, Lp, buf3, Kdim, scratch1, Kdim, Lp, Kdim, Lp);
    __syncthreads();

    for (int j = tid; j < L; j += nthr) {
        const float egj = smem_eg[j];
        float db_acc = 0.0f, dg_acc = 0.0f;
        for (int kk = 0; kk < Kdim; ++kk) {
            float mt_dw = scratch1[j * Kdim + kk];
            float kj = to_float(smem_k[j * Kdim + kk]);
            DK[j * Kdim + kk] += mt_dw * smem_beta[j] * egj;
            db_acc += mt_dw * egj * kj;
            dg_acc += mt_dw * smem_beta[j] * egj * kj;
        }
        DB[j] += db_acc;
        DG[j] += dg_acc;
    }
    __syncthreads();

    // ================================================================
    // Step 3: dM = DU @ (bv)^T + DW @ bkg^T
    // ================================================================
    // beta*V → buf2[Lp×V] (reuse raw-v values already in buf2)
    for (int idx = tid; idx < Lp * Vdim; idx += nthr) {
        const int pos = idx / Vdim;
        float val = 0.0f;
        if (pos < L) {
            float vraw = to_float(buf2[idx]);
            val = bf16_trunc<TQ>(vraw * smem_beta[pos]);
        }
        buf2[idx] = from_float<TQ>(val);
    }
    // DU_bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(0.0f);
    __syncthreads();
    for (int idx = tid; idx < L * Vdim; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(DU[idx]));
    __syncthreads();

    // dM_part1 = DU @ (bv)^T → scratch1
    wmma_nt<TQ>(buf1, Vdim, buf2, Vdim, scratch1, Lp, Lp, Lp, Vdim);
    __syncthreads();

    // Save dM_part1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        tmp_area[idx] = scratch1[idx];

    // bkg → buf2[Lp×K]
    for (int idx = tid; idx < Lp * Kdim; idx += nthr) {
        const int pos = idx / Kdim;
        float val = (pos < L) ? to_float(smem_k[idx]) * smem_beta[pos] * smem_eg[pos] : 0.0f;
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(val));
    }
    __syncthreads();

    // dM_part2 = DW @ bkg^T
    // buf3 still has DW_bf16 from step 2
    wmma_nt<TQ>(buf3, Kdim, buf2, Kdim, scratch1, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // dM = part1 + part2, masked
    for (int idx = tid; idx < Lp * Lp; idx += nthr) {
        const int i = idx / Lp, j = idx % Lp;
        if (i < L && j <= i)
            scratch1[idx] = tmp_area[idx] + scratch1[idx];
        else
            scratch1[idx] = 0.0f;
    }
    __syncthreads();

    // ================================================================
    // Step 4: dA_grad = -M^T @ dM @ M^T  (WMMA)
    // ================================================================
    // dM is in scratch1[Lp×Lp], M is in scratch2[Lp×Lp]
    // Convert both to bf16
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf2[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    // tmp1 = dM @ M^T → scratch1 (reuse since dM is now in buf1)
    wmma_nt<TQ>(buf1, Lp, buf2, Lp, scratch1, Lp, Lp, Lp, Lp);
    __syncthreads();

    // Convert tmp1 to bf16 → buf1
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch1[idx]));
    __syncthreads();

    // dA_grad = -M^T @ tmp1 → scratch1
    wmma_tn<TQ>(buf2, Lp, buf1, Lp, scratch1, Lp, Lp, Lp, Lp);
    __syncthreads();

    // Negate
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch1[idx] = -scratch1[idx];
    __syncthreads();

    // dA_grad → DK, DB, DG
    // Precompute kkt = k @ k^T via WMMA → scratch2[Lp×Lp]
    // dA_grad is in scratch1[Lp×Lp]
    wmma_nt<TQ>(smem_k, Kdim, smem_k, Kdim, scratch2, Lp, Lp, Lp, Kdim);
    __syncthreads();

    // Save kkt to global tmp_area, then reuse scratch2 for ddot_sym
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        tmp_area[idx] = scratch2[idx];
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        scratch2[idx] = 0.0f;
    __syncthreads();

    // Process lower triangle:
    //   - form ddot_sym for DK matmul in scratch2
    //   - stash DB pair contributions in scratch1 lower triangle
    //   - stash DG pair contributions in tmp_area lower triangle
    for (int idx = tid; idx < L * L; idx += nthr) {
        const int i = idx / L, j = idx % L;
        if (j >= i) continue;

        const float exp_ij = smem_eg[i] * smem_e_last[j] * inv_eg_last_p3;
        const float a_grad = scratch1[i * Lp + j];
        const float dot_k = tmp_area[i * Lp + j];
        const float val = dot_k * exp_ij;

        const float db_pair = a_grad * val;
        const float dval = a_grad * smem_beta[i];
        const float dg_pair = dval * dot_k * exp_ij;
        const float ddot = dval * exp_ij;

        scratch1[i * Lp + j] = db_pair;
        tmp_area[i * Lp + j] = dg_pair;

        // Store symmetric ddot for DK matmul
        scratch2[i * Lp + j] = ddot;
        scratch2[j * Lp + i] = ddot;
    }
    __syncthreads();

    // Reduce pairwise contributions into DB and DG without atomics.
    for (int i = tid; i < L; i += nthr) {
        float db_acc = 0.0f;
        float dg_acc = 0.0f;
        for (int j = 0; j < i; ++j) {
            db_acc += scratch1[i * Lp + j];
            dg_acc += tmp_area[i * Lp + j];
        }
        for (int j = i + 1; j < L; ++j) {
            dg_acc -= tmp_area[j * Lp + i];
        }
        DB[i] += db_acc;
        DG[i] += dg_acc;
    }
    __syncthreads();

    // DK += ddot_sym @ k  (WMMA matmul)
    for (int idx = tid; idx < Lp * Lp; idx += nthr)
        buf1[idx] = from_float<TQ>(bf16_trunc<TQ>(scratch2[idx]));
    __syncthreads();

    wmma_nn<TQ>(buf1, Lp, smem_k, Kdim, scratch2, Kdim, Lp, Kdim, Lp);
    __syncthreads();

    for (int idx = tid; idx < L * Kdim; idx += nthr)
        DK[idx] += scratch2[idx];
    __syncthreads();

    // ================================================================
    // Step 5: Write d_q, d_k (with L2norm backward), d_beta, d_g
    // ================================================================
    for (int i = tid; i < L; i += nthr) {
        const long q_i_base = (((long)b * Tlen + cs + i) * H + h) * Kdim;
        const long gh_idx = ((long)b * Tlen + cs + i) * H + h;

        if (use_qk_l2norm_in_kernel) {
            float dot_q = 0.0f, dot_k = 0.0f;
            for (int kk = 0; kk < Kdim; ++kk) {
                dot_q += DQ[i * Kdim + kk] * to_float(smem_q[i * Kdim + kk]);
                dot_k += DK[i * Kdim + kk] * to_float(smem_k[i * Kdim + kk]);
            }
            for (int kk = 0; kk < Kdim; ++kk) {
                const float qn = to_float(smem_q[i * Kdim + kk]);
                const float kn = to_float(smem_k[i * Kdim + kk]);
                d_q[q_i_base + kk] = from_float<TQ>(
                    (DQ[i * Kdim + kk] - qn * dot_q) * smem_invq[i]);
                d_k[q_i_base + kk] = from_float<TQ>(
                    (DK[i * Kdim + kk] - kn * dot_k) * smem_invk[i]);
            }
        } else {
            for (int kk = 0; kk < Kdim; ++kk) {
                d_q[q_i_base + kk] = from_float<TQ>(DQ[i * Kdim + kk]);
                d_k[q_i_base + kk] = from_float<TQ>(DK[i * Kdim + kk]);
            }
        }
        d_beta[gh_idx] = from_float<TB>(DB[i]);
    }
    __syncthreads();

    // Reverse cumsum for d_g
    if (tid == 0) {
        float running = 0.0f;
        for (int i = L - 1; i >= 0; --i) {
            running += DG[i];
            const long gh_idx = ((long)b * Tlen + cs + i) * H + h;
            d_g[gh_idx] = from_float<TG>(running);
        }
    }
}

// ============================================================================
// Multi-kernel backward launch
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_bwd_v2_multikernel(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {

    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const int Lp = kMaxC;
    const int threads = 128;

    ChunkWorkspaceLayout cwl = make_chunk_ws<TQ>(Lp, Kdim, Vdim);
    const int chunk_ws_stride = cwl.total;

    // Workspace layout: [chunk_workspace | dh_storage]
    // chunk_workspace: B*H*num_chunks * chunk_ws_stride floats
    // dh_storage: B*H*num_chunks * Kdim*Vdim floats (per-chunk dh for Phase 3)
    float* chunk_ws = workspace.get<float>();
    float* dh_storage = chunk_ws + static_cast<long>(B) * H * num_chunks * chunk_ws_stride;

    // Shared memory for all phases
    const std::size_t phase_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2    // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ) * 2   // k, q
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)         // buf1
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)       // buf2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)       // buf3
        + static_cast<std::size_t>(Lp) * 6 * sizeof(float);      // gcum, beta, invq, invk, eg, e_last

    // Phase 2: ultra-lightweight, just scratch1[K×V] + buf1[K×K bf16] + buf2[K×V bf16]
    const std::size_t phase2_smem =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)    // scratch1
        + static_cast<std::size_t>(Kdim) * Kdim * sizeof(TQ)     // buf1 (C_bf16)
        + static_cast<std::size_t>(Kdim) * Vdim * sizeof(TQ);    // buf2 (ds_bf16)

    // --- Checkpoint kernel (same as before) ---
    const std::size_t cp_smem =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)
        + static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)
        + static_cast<std::size_t>(Lp) * 3 * sizeof(float);

    // Optional per-phase timing via env var
    const bool profile = (std::getenv("SUROGATE_GDR_PROFILE") != nullptr);
    cudaEvent_t ev[5];
    if (profile) {
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }

    if (!skip_checkpoint) {
        cudaFuncSetAttribute(gdr_chunk_checkpoint_wmma<TQ, TG, TB>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(cp_smem));
        gdr_chunk_checkpoint_wmma<TQ, TG, TB><<<dim3(B, H), threads, cp_smem, stream>>>(
            checkpoints.get<float>(),
            k.get<TQ>(), v.get<TQ>(), g.get<TG>(), beta.get<TB>(),
            initial_state ? initial_state->get<float>() : nullptr,
            Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel);
    }

    if (profile) cudaEventRecord(ev[1], stream);

    // --- Phase 1: chunk-parallel ---
    cudaFuncSetAttribute(gdr_bwd_phase1_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(phase_smem));
    gdr_bwd_phase1_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, phase_smem, stream>>>(
        d_out.get<TQ>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        chunk_ws,
        d_initial_state.get<float>(),  // ds_accum (not used in Phase 1 directly)
        chunk_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel);

    if (profile) cudaEventRecord(ev[2], stream);

    // --- Phase 2: sequential (single matmul per chunk) ---
    cudaFuncSetAttribute(gdr_bwd_phase2_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(phase2_smem));
    gdr_bwd_phase2_wmma<TQ, TG, TB><<<B * H, threads, phase2_smem, stream>>>(
        checkpoints.get<float>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        d_initial_state.get<float>(),
        chunk_ws,
        dh_storage,
        chunk_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size);

    if (profile) cudaEventRecord(ev[3], stream);

    // --- Phase 3: chunk-parallel ---
    cudaFuncSetAttribute(gdr_bwd_phase3_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(phase_smem));
    gdr_bwd_phase3_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, phase_smem, stream>>>(
        d_q.get<TQ>(), d_k.get<TQ>(), d_v.get<TQ>(),
        d_g.get<TG>(), d_beta.get<TB>(),
        d_out.get<TQ>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        dh_storage,
        chunk_ws,
        chunk_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel);

    if (profile) {
        cudaEventRecord(ev[4], stream);
        cudaEventSynchronize(ev[4]);
        float ms[4];
        for (int i = 0; i < 4; ++i) cudaEventElapsedTime(&ms[i], ev[i], ev[i+1]);
        fprintf(stderr, "[GDR bwd] cp=%.3fms p1=%.3fms p2=%.3fms p3=%.3fms total=%.3fms\n",
                ms[0], ms[1], ms[2], ms[3], ms[0]+ms[1]+ms[2]+ms[3]);
        for (auto& e : ev) cudaEventDestroy(e);
    }
}

#endif  // !GDR_V2_COMPILE_FWD_ONLY

#if !defined(GDR_V2_COMPILE_BWD_ONLY)
// ============================================================================
// Multi-kernel forward launcher (3 kernels: precompute → state → output)
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_fwd_v2_multikernel(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, cudaStream_t stream) {

    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);
    const int num_chunks = (Tlen + chunk_size - 1) / chunk_size;
    const int Lp = kMaxC;
    const int threads = 128;

    FwdWorkspaceLayout fwl = make_fwd_ws(Lp, Kdim, Vdim);
    const int fwd_ws_stride = fwl.total;
    const long total_ws_floats = (long)B * H * num_chunks * fwd_ws_stride;

    // Allocate workspace (stream-ordered)
    float* fwd_workspace = nullptr;
    cudaMallocAsync(&fwd_workspace, total_ws_floats * sizeof(float), stream);

    static bool profile = (std::getenv("SUROGATE_GDR_PROFILE") != nullptr);
    std::vector<cudaEvent_t> ev;
    if (profile) {
        ev.resize(4);
        for (auto& e : ev) cudaEventCreate(&e);
        cudaEventRecord(ev[0], stream);
    }

    // --- Kernel 1: Precompute (chunk-parallel) ---
    // smem: scratch1[Lp×Lp]f + scratch2[Lp×Lp]f
    //     + k[Lp×K]TQ + q[Lp×K]TQ + buf1[Lp×Lp]TQ + buf2[Lp×V]TQ + buf3[Lp×K]TQ
    //     + scalars[4×Lp]f
    const std::size_t precompute_smem =
        static_cast<std::size_t>(Lp) * Lp * sizeof(float) * 2          // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ) * 2         // smem_k + smem_q
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)               // buf1
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)             // buf2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)             // buf3
        + static_cast<std::size_t>(Lp) * 4 * sizeof(float);            // gcum, beta, invk, invq

    cudaFuncSetAttribute(
        gdr_fwd_precompute_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(precompute_smem));

    gdr_fwd_precompute_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, precompute_smem, stream>>>(
        fwd_workspace,
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        fwd_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size,
        use_qk_l2norm_in_kernel);

    if (profile) cudaEventRecord(ev[1], stream);

    // --- Kernel 2: State propagation (sequential per B,H) ---
    // smem: smem_h[K×V]f + scratch1[Lp×V]f
    //     + buf_w[Lp×K]TQ + buf_h[K×V]TQ + buf_k[Lp×K]TQ + buf_vnp[Lp×V]TQ
    //     + gcum[Lp]f
    const std::size_t state_smem =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)         // smem_h
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(float)         // scratch1
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // buf_w
        + static_cast<std::size_t>(Kdim) * Vdim * sizeof(TQ)          // buf_h
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // buf_k
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)            // buf_vnp
        + static_cast<std::size_t>(Lp) * sizeof(float);               // gcum

    cudaFuncSetAttribute(
        gdr_fwd_state_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(state_smem));

    gdr_fwd_state_wmma<TQ, TG, TB><<<dim3(B, H), threads, state_smem, stream>>>(
        final_state.get<float>(), state_scratch.get<float>(),
        fwd_checkpoints_tensor->get<float>(),
        fwd_workspace,
        initial_state ? initial_state->get<float>() : nullptr,
        fwd_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size);

    if (profile) cudaEventRecord(ev[2], stream);

    // --- Kernel 3: Output (chunk-parallel) ---
    // smem: scratch1[Lp×V]f + scratch2[Lp×V]f
    //     + smem_q[Lp×K]TQ + buf_h[K×V]TQ + buf_S[Lp×Lp]TQ + buf_vnp[Lp×V]TQ
    //     + gcum[Lp]f + invq[Lp]f
    const std::size_t output_smem =
        static_cast<std::size_t>(Lp) * Vdim * sizeof(float) * 2       // scratch1 + scratch2
        + static_cast<std::size_t>(Lp) * Kdim * sizeof(TQ)            // smem_q
        + static_cast<std::size_t>(Kdim) * Vdim * sizeof(TQ)          // buf_h
        + static_cast<std::size_t>(Lp) * Lp * sizeof(TQ)              // buf_S
        + static_cast<std::size_t>(Lp) * Vdim * sizeof(TQ)            // buf_vnp
        + static_cast<std::size_t>(Lp) * 2 * sizeof(float);           // gcum + invq

    cudaFuncSetAttribute(
        gdr_fwd_output_wmma<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(output_smem));

    gdr_fwd_output_wmma<TQ, TG, TB><<<B * H * num_chunks, threads, output_smem, stream>>>(
        out.get<TQ>(),
        q.get<TQ>(),
        fwd_checkpoints_tensor->get<float>(),
        fwd_workspace,
        fwd_ws_stride,
        Tlen, H, Kdim, Vdim, num_chunks, chunk_size, scale,
        use_qk_l2norm_in_kernel);

    if (profile) {
        cudaEventRecord(ev[3], stream);
        cudaEventSynchronize(ev[3]);
        float ms[3];
        for (int i = 0; i < 3; ++i) cudaEventElapsedTime(&ms[i], ev[i], ev[i+1]);
        fprintf(stderr, "[GDR fwd multi] precompute=%.3fms state=%.3fms output=%.3fms total=%.3fms\n",
                ms[0], ms[1], ms[2], ms[0]+ms[1]+ms[2]);
        for (auto& e : ev) cudaEventDestroy(e);
    }

    // Free workspace (stream-ordered)
    cudaFreeAsync(fwd_workspace, stream);
}

// ============================================================================
// Launch helpers with dtype dispatch
// ============================================================================

template<typename TQ, typename TG, typename TB>
void launch_fwd_v2(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);

    const int threads = 128;
    dim3 grid(B, H);

    // Try multi-kernel WMMA path for bf16/fp16 with K,V multiples of 16 and <= 64
    // Requires checkpoints (output kernel reads h_in from them)
    constexpr bool can_wmma = std::is_same_v<TQ, nv_bfloat16> || std::is_same_v<TQ, half>;
    if constexpr (can_wmma) {
        if (Kdim <= 64 && Vdim <= 64 && Kdim % 16 == 0 && Vdim % 16 == 0) {
            if (fwd_checkpoints_tensor) {
                launch_fwd_v2_multikernel<TQ, TG, TB>(
                    out, final_state, state_scratch,
                    q, k, v, g, beta, initial_state,
                    scale, chunk_size, use_qk_l2norm_in_kernel,
                    fwd_checkpoints_tensor, stream);
                return;
            }

            // Single-kernel WMMA fallback (no checkpoints)
            const std::size_t wmma_smem =
                static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)           // h
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2   // scratch1 + scratch2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2       // k, q
                + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(TQ)          // buf1
                + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)           // buf2
                + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)           // buf3
                + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);          // scalars

            cudaFuncSetAttribute(
                gdr_chunk_fwd_wmma<TQ, TG, TB>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(wmma_smem));

            gdr_chunk_fwd_wmma<TQ, TG, TB><<<grid, threads, wmma_smem, stream>>>(
                out.get<TQ>(), final_state.get<float>(), state_scratch.get<float>(),
                nullptr,
                q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
                g.get<TG>(), beta.get<TB>(),
                initial_state ? initial_state->get<float>() : nullptr,
                Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
            return;
        }
    }

    // Fallback: scalar kernel
    const std::size_t smem_bytes =
        static_cast<std::size_t>(Kdim) * Vdim * sizeof(float)       // state
        + static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float)   // M
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2   // k, q
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ) * 2   // v, u
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)       // w
        + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);      // scalars

    cudaFuncSetAttribute(
        gdr_chunk_fwd_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem_bytes));

    gdr_chunk_fwd_v2<TQ, TG, TB><<<grid, threads, smem_bytes, stream>>>(
        out.get<TQ>(), final_state.get<float>(), state_scratch.get<float>(),
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
}

#endif  // !GDR_V2_COMPILE_BWD_ONLY

#if !defined(GDR_V2_COMPILE_FWD_ONLY)
template<typename TQ, typename TG, typename TB>
void launch_bwd_v2(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    const int B = static_cast<int>(q.Sizes[0]);
    const int Tlen = static_cast<int>(q.Sizes[1]);
    const int H = static_cast<int>(q.Sizes[2]);
    const int Kdim = static_cast<int>(q.Sizes[3]);
    const int Vdim = static_cast<int>(v.Sizes[3]);

    const int threads = 128;
    dim3 grid(B, H);
    const bool force_scalar_bwd = (std::getenv("SUROGATE_GDR_FORCE_SCALAR_BWD") != nullptr);

    // Try WMMA multi-kernel path for bf16/fp16 with K,V multiples of 16 and <= 64
    constexpr bool can_wmma = std::is_same_v<TQ, nv_bfloat16> || std::is_same_v<TQ, half>;
    if constexpr (can_wmma) {
        if (force_scalar_bwd) {
            // Debug switch to compare multikernel path against scalar fallback.
        } else
        if (Kdim <= 64 && Vdim <= 64 && Kdim % 16 == 0 && Vdim % 16 == 0) {
            launch_bwd_v2_multikernel<TQ, TG, TB>(
                d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state,
                scale, chunk_size, use_qk_l2norm_in_kernel,
                checkpoints, workspace, skip_checkpoint, stream);
            return;
        }
    }

    // Fallback: scalar kernels
    // Checkpoint kernel shared memory
    const std::size_t cp_smem =
        static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float)    // M
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)      // k
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)      // v
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)      // u
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ)      // w
        + static_cast<std::size_t>(kMaxC) * 3 * sizeof(float);     // g_cum, beta, inv_k

    cudaFuncSetAttribute(
        gdr_chunk_checkpoint_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(cp_smem));

    gdr_chunk_checkpoint_v2<TQ, TG, TB><<<grid, threads, cp_smem, stream>>>(
        checkpoints.get<float>(),
        k.get<TQ>(), v.get<TQ>(), g.get<TG>(), beta.get<TB>(),
        initial_state ? initial_state->get<float>() : nullptr,
        Tlen, H, Kdim, Vdim, chunk_size, use_qk_l2norm_in_kernel);

    // Backward kernel shared memory
    const std::size_t bwd_smem =
        static_cast<std::size_t>(kMaxC) * kMaxC * sizeof(float) * 2  // M + dM
        + static_cast<std::size_t>(kMaxC) * Kdim * sizeof(TQ) * 2    // k, q
        + static_cast<std::size_t>(kMaxC) * Vdim * sizeof(TQ)        // v
        + static_cast<std::size_t>(kMaxC) * 4 * sizeof(float);       // scalars

    cudaFuncSetAttribute(
        gdr_chunk_bwd_v2<TQ, TG, TB>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(bwd_smem));

    gdr_chunk_bwd_v2<TQ, TG, TB><<<grid, threads, bwd_smem, stream>>>(
        d_q.get<TQ>(), d_k.get<TQ>(), d_v.get<TQ>(),
        d_g.get<TG>(), d_beta.get<TB>(),
        d_initial_state.get<float>(),
        d_out.get<TQ>(),
        d_final_state ? d_final_state->get<float>() : nullptr,
        q.get<TQ>(), k.get<TQ>(), v.get<TQ>(),
        g.get<TG>(), beta.get<TB>(),
        checkpoints.get<float>(),
        workspace.get<float>(),
        static_cast<int>(workspace.Sizes[2]),
        Tlen, H, Kdim, Vdim, chunk_size, scale, use_qk_l2norm_in_kernel);
}

#endif  // !GDR_V2_COMPILE_FWD_ONLY

#if !defined(GDR_V2_COMPILE_BWD_ONLY)
// Dtype dispatch chain for forward
template<typename TQ, typename TG>
void dispatch_fwd_v2_beta(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_fwd_v2<TQ, TG, float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        case ETensorDType::BF16:
            launch_fwd_v2<TQ, TG, nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        case ETensorDType::FP16:
            launch_fwd_v2<TQ, TG, half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_fwd_v2_g(
    Tensor& out, Tensor& final_state, Tensor& state_scratch,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints_tensor, cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_fwd_v2_beta<TQ, float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        case ETensorDType::BF16:
            dispatch_fwd_v2_beta<TQ, nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        case ETensorDType::FP16:
            dispatch_fwd_v2_beta<TQ, half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints_tensor, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported g dtype");
    }
}

#endif  // !GDR_V2_COMPILE_BWD_ONLY

#if !defined(GDR_V2_COMPILE_FWD_ONLY)
// Dtype dispatch chain for backward
template<typename TQ, typename TG>
void dispatch_bwd_v2_beta(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    switch (beta.DType) {
        case ETensorDType::FP32:
            launch_bwd_v2<TQ, TG, float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::BF16:
            launch_bwd_v2<TQ, TG, nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::FP16:
            launch_bwd_v2<TQ, TG, half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported beta dtype");
    }
}

template<typename TQ>
void dispatch_bwd_v2_g(
    Tensor& d_q, Tensor& d_k, Tensor& d_v, Tensor& d_g, Tensor& d_beta,
    Tensor& d_initial_state, const Tensor& d_out, const Tensor* d_final_state,
    const Tensor& q, const Tensor& k, const Tensor& v,
    const Tensor& g, const Tensor& beta, const Tensor* initial_state,
    float scale, int chunk_size, bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints, Tensor& workspace, bool skip_checkpoint,
    cudaStream_t stream) {
    switch (g.DType) {
        case ETensorDType::FP32:
            dispatch_bwd_v2_beta<TQ, float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::BF16:
            dispatch_bwd_v2_beta<TQ, nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        case ETensorDType::FP16:
            dispatch_bwd_v2_beta<TQ, half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); return;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported g dtype");
    }
}

#endif  // !GDR_V2_COMPILE_FWD_ONLY

}  // namespace

// ============================================================================
// Public API
// ============================================================================

#if !defined(GDR_V2_COMPILE_BWD_ONLY)
void gated_delta_rule_chunk_forward_v2(
    Tensor& out,
    Tensor& final_state,
    Tensor& state_scratch,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor* fwd_checkpoints,
    cudaStream_t stream) {

    switch (q.DType) {
        case ETensorDType::FP32:
            dispatch_fwd_v2_g<float>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints, stream); break;
        case ETensorDType::BF16:
            dispatch_fwd_v2_g<nv_bfloat16>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints, stream); break;
        case ETensorDType::FP16:
            dispatch_fwd_v2_g<half>(out, final_state, state_scratch, q, k, v, g, beta,
                initial_state, scale, chunk_size, use_qk_l2norm_in_kernel, fwd_checkpoints, stream); break;
        default:
            throw std::logic_error("gated_delta_rule_chunk_forward_v2: unsupported q dtype");
    }
}
#endif  // !GDR_V2_COMPILE_BWD_ONLY

#if !defined(GDR_V2_COMPILE_FWD_ONLY)
void gated_delta_rule_chunk_backward_v2(
    Tensor& d_q,
    Tensor& d_k,
    Tensor& d_v,
    Tensor& d_g,
    Tensor& d_beta,
    Tensor& d_initial_state,
    const Tensor& d_out,
    const Tensor* d_final_state,
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor* initial_state,
    float scale,
    int chunk_size,
    bool use_qk_l2norm_in_kernel,
    Tensor& checkpoints,
    Tensor& workspace,
    bool skip_checkpoint,
    cudaStream_t stream) {

    switch (q.DType) {
        case ETensorDType::FP32:
            dispatch_bwd_v2_g<float>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        case ETensorDType::BF16:
            dispatch_bwd_v2_g<nv_bfloat16>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        case ETensorDType::FP16:
            dispatch_bwd_v2_g<half>(d_q, d_k, d_v, d_g, d_beta, d_initial_state,
                d_out, d_final_state, q, k, v, g, beta, initial_state, scale, chunk_size,
                use_qk_l2norm_in_kernel, checkpoints, workspace, skip_checkpoint, stream); break;
        default:
            throw std::logic_error("gated_delta_rule_chunk_backward_v2: unsupported q dtype");
    }
}
#endif  // !GDR_V2_COMPILE_FWD_ONLY
