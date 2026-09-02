// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_mmvq.h"

#include "ops/common/warp.cuh"
#include "ops/linear/ggml/ggml_vecdot.cuh"

#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::ggml {
namespace {

using vec_dot_q_cuda_t = float (*)(const void* __restrict__, const block_q8_1* __restrict__,
                                   const int&, const int&);

template <GgmlType type> struct Traits;
template <> struct Traits<GgmlType::Q2_K> {
    static constexpr int qk = QK_K, qi = QI2_K, vdr = VDR_Q2_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q2_K_q8_1;
};
template <> struct Traits<GgmlType::Q3_K> {
    static constexpr int qk = QK_K, qi = QI3_K, vdr = VDR_Q3_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q3_K_q8_1;
};
template <> struct Traits<GgmlType::Q4_K> {
    static constexpr int qk = QK_K, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q4_K_q8_1;
};
template <> struct Traits<GgmlType::Q5_K> {
    static constexpr int qk = QK_K, qi = QI5_K, vdr = VDR_Q5_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q5_K_q8_1;
};
template <> struct Traits<GgmlType::Q6_K> {
    static constexpr int qk = QK_K, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q6_K_q8_1;
};

// llama.cpp's NVIDIA ("generic") schedule table: warps per CTA and rows per CTA by column count.
__host__ __device__ constexpr int calc_nwarps(int ncols_dst) { return ncols_dst <= 4 ? 4 : 2; }
__host__ __device__ constexpr int calc_rows_per_block(int ncols_dst) { return ncols_dst == 1 ? 1 : 2; }
constexpr int kWarpSize = 32;

template <typename DstT> __device__ __forceinline__ DstT to_dst(float v);
template <> __device__ __forceinline__ float to_dst<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 to_dst<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

template <GgmlType type, int ncols_dst, typename DstT>
__launch_bounds__(calc_nwarps(ncols_dst) * kWarpSize, 1)
__global__ void mul_mat_vec_q(const void* __restrict__ vx, const block_q8_1* __restrict__ vy,
                              DstT* __restrict__ dst, const int ncols_x, const int nrows_x,
                              const int stride_row_x, const int stride_col_y,
                              const int stride_col_dst) {
    constexpr int qk  = Traits<type>::qk;
    constexpr int qi  = Traits<type>::qi;
    constexpr int vdr = Traits<type>::vdr;
    constexpr int nwarps              = calc_nwarps(ncols_dst);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst);
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = Traits<type>::vec_dot;

    const int tid  = kWarpSize * threadIdx.y + threadIdx.x;
    const int row0 = rows_per_cuda_block * blockIdx.x;
    const int blocks_per_row_x         = ncols_x / qk;
    constexpr int blocks_per_iter      = vdr * nwarps * kWarpSize / qi;

    // partial sum for each thread
    float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

    const int kbx_offset = row0 * stride_row_x;
    for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x; kbx += blocks_per_iter) {
        const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx
        const int kqs = vdr * (tid % (qi / vdr)); // x block quant index when casting the quants to int
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                // The upstream kernel reads one row past the end for an odd row count and
                // relies on pool padding; the guard is uniform per CTA and costs nothing.
                if (rows_per_cuda_block == 1 || row0 + i < nrows_x) {
                    tmp[j][i] += vec_dot_q_cuda(vx, &vy[j * stride_col_y + kby],
                                                kbx_offset + i * stride_row_x + kbx, kqs);
                }
            }
        }
    }

    __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst][rows_per_cuda_block]
                               [kWarpSize];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
            for (int i = 0; i < rows_per_cuda_block; ++i) {
                tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
            }
        }
    }
    __syncthreads();
    if (threadIdx.y > 0) { return; }

    dst += row0;
    // sum up partial sums and write back result
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            tmp[j][i] = warp_sum(tmp[j][i]); // xor butterfly: every lane holds the total, lane i writes row i
            if (threadIdx.x == i && (rows_per_cuda_block == 1 || row0 + i < nrows_x)) {
                dst[j * stride_col_dst + i] = to_dst<DstT>(tmp[j][i]);
            }
        }
    }
}

template <GgmlType type, int ncols_dst, typename DstT>
void launch(const void* blocks, int n, int k, const block_q8_1* y, DstT* out, cudaStream_t stream) {
    constexpr int nwarps = calc_nwarps(ncols_dst);
    constexpr int rows   = calc_rows_per_block(ncols_dst);
    const dim3 block(kWarpSize, nwarps);
    const dim3 grid((n + rows - 1) / rows);
    mul_mat_vec_q<type, ncols_dst, DstT><<<grid, block, 0, stream>>>(
        blocks, y, out, k, n, k / Traits<type>::qk, k / QK8_1, n);
}

template <GgmlType type, typename DstT>
void launch_columns(const void* blocks, int n, int k, const block_q8_1* y, int tokens, DstT* out,
                    cudaStream_t stream) {
    switch (tokens) {
    case 1: launch<type, 1, DstT>(blocks, n, k, y, out, stream); return;
    case 2: launch<type, 2, DstT>(blocks, n, k, y, out, stream); return;
    case 3: launch<type, 3, DstT>(blocks, n, k, y, out, stream); return;
    case 4: launch<type, 4, DstT>(blocks, n, k, y, out, stream); return;
    case 5: launch<type, 5, DstT>(blocks, n, k, y, out, stream); return;
    case 6: launch<type, 6, DstT>(blocks, n, k, y, out, stream); return;
    case 7: launch<type, 7, DstT>(blocks, n, k, y, out, stream); return;
    case 8: launch<type, 8, DstT>(blocks, n, k, y, out, stream); return;
    default: break;
    }
    throw std::invalid_argument("mmvq: at most 8 columns per launch");
}

} // namespace

template <typename DstT>
void mmvq_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                 const block_q8_1* y, std::int32_t tokens, DstT* out, cudaStream_t stream) {
    if (blocks == nullptr || y == nullptr || out == nullptr || n <= 0 || k <= 0 ||
        (k % QK_K) != 0 || tokens <= 0 || tokens > kMmvqMaxColumns) {
        throw std::invalid_argument("mmvq: W[n, k] with k a multiple of 256, 1..8 columns");
    }
    switch (type) {
    case GgmlType::Q2_K: launch_columns<GgmlType::Q2_K>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q3_K: launch_columns<GgmlType::Q3_K>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q4_K: launch_columns<GgmlType::Q4_K>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q5_K: launch_columns<GgmlType::Q5_K>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q6_K: launch_columns<GgmlType::Q6_K>(blocks, n, k, y, tokens, out, stream); return;
    }
    throw std::invalid_argument("mmvq: unknown GGML type");
}

template void mmvq_launch<float>(GgmlType, const void*, std::int32_t, std::int32_t,
                                 const block_q8_1*, std::int32_t, float*, cudaStream_t);
template void mmvq_launch<__nv_bfloat16>(GgmlType, const void*, std::int32_t, std::int32_t,
                                         const block_q8_1*, std::int32_t, __nv_bfloat16*,
                                         cudaStream_t);

} // namespace sinfer::ops::detail::ggml
