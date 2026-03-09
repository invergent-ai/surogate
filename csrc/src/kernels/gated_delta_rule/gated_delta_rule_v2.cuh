// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_KERNELS_GATED_DELTA_RULE_V2_CUH
#define SUROGATE_SRC_KERNELS_GATED_DELTA_RULE_V2_CUH

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
    const int m_tiles = M / 16;
    const int n_tiles = N / 16;
    const int num_tiles = m_tiles * n_tiles;
    for (int tile = wid; tile < num_tiles; tile += nw) {
        const int m = tile / n_tiles;
        const int n = tile % n_tiles;
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
    const int m_tiles = M / 16;
    const int n_tiles = N / 16;
    const int num_tiles = m_tiles * n_tiles;
    for (int tile = wid; tile < num_tiles; tile += nw) {
        const int m = tile / n_tiles;
        const int n = tile % n_tiles;
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
    const int m_tiles = M / 16;
    const int n_tiles = N / 16;
    const int num_tiles = m_tiles * n_tiles;
    for (int tile = wid; tile < num_tiles; tile += nw) {
        const int m = tile / n_tiles;
        const int n = tile % n_tiles;
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

// Per-chunk workspace offsets (all in floats, Lp-padded where needed)
struct ChunkWorkspaceLayout {
    int M_off;       // [Lp×Lp]
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
    int KNORM_off;   // [Lp×K] float — normalized k cached from Phase 1
    int INVQ_off;    // [Lp] float — q inverse L2 norm per token
    int INVK_off;    // [Lp] float — k inverse L2 norm per token
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
    l.KNORM_off = align_up_int(off, kWsAlignFloats); off = l.KNORM_off + Lp * Kdim;
    l.INVQ_off  = align_up_int(off, kWsAlignFloats); off = l.INVQ_off + Lp;
    l.INVK_off  = align_up_int(off, kWsAlignFloats); off = l.INVK_off + Lp;
    l.total    = align_up_int(off, kWsAlignFloats);
    return l;
}

// ============================================================================
// Forward multi-kernel workspace layout (per chunk)
// ============================================================================
struct FwdWorkspaceLayout {
    // Byte offsets from chunk workspace base.
    int u_off;         // [Lp×V] packed as bf16/fp16 (2B element)
    int w_off;         // [Lp×K] packed as bf16/fp16
    int k_off;         // [Lp×K] packed as bf16/fp16
    int vnew_pre_off;  // [Lp×V] packed as bf16/fp16
    int gcum_off;      // [Lp] float32
    int total;         // total bytes per chunk
};

__host__ __device__ __forceinline__ int align_up_int_fwd(int x, int align) {
    return ((x + align - 1) / align) * align;
}

__host__ __device__ FwdWorkspaceLayout make_fwd_ws(int Lp, int Kdim, int Vdim) {
    constexpr int kWsAlignBytes = 128;
    constexpr int kPackedElemBytes = 2;  // bf16/fp16
    FwdWorkspaceLayout l;
    int off = 0;
    off = align_up_int_fwd(off, kWsAlignBytes); l.u_off = off; off += Lp * Vdim * kPackedElemBytes;
    off = align_up_int_fwd(off, kWsAlignBytes); l.w_off = off; off += Lp * Kdim * kPackedElemBytes;
    off = align_up_int_fwd(off, kWsAlignBytes); l.k_off = off; off += Lp * Kdim * kPackedElemBytes;
    off = align_up_int_fwd(off, kWsAlignBytes); l.vnew_pre_off = off; off += Lp * Vdim * kPackedElemBytes;
    off = align_up_int_fwd(off, kWsAlignBytes); l.gcum_off = off; off += Lp * static_cast<int>(sizeof(float));
    l.total = align_up_int_fwd(off, kWsAlignBytes);
    return l;
}


}  // namespace

#endif  // SUROGATE_SRC_KERNELS_GATED_DELTA_RULE_V2_CUH
