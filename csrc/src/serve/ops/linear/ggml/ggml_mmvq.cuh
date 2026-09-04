// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

#include "ops/common/warp.cuh"
#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_mmvq.h"
#include "ops/linear/ggml/ggml_vecdot.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace sinfer::ops::detail::ggml {

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
template <> struct Traits<GgmlType::Q8_0> {
    static constexpr int qk = QK8_0, qi = QI8_0, vdr = VDR_Q8_0_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q8_0_q8_1;
};
template <> struct Traits<GgmlType::Q4_1> {
    static constexpr int qk = QK4_1, qi = QI4_1, vdr = VDR_Q4_1_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q4_1_q8_1;
};
template <> struct Traits<GgmlType::Q5_1> {
    static constexpr int qk = QK5_1, qi = QI5_1, vdr = VDR_Q5_1_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q5_1_q8_1;
};
template <> struct Traits<GgmlType::Q4_0> {
    static constexpr int qk = QK4_0, qi = QI4_0, vdr = VDR_Q4_0_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q4_0_q8_1;
};
template <> struct Traits<GgmlType::Q5_0> {
    static constexpr int qk = QK5_0, qi = QI5_0, vdr = VDR_Q5_0_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q5_0_q8_1;
};
template <> struct Traits<GgmlType::IQ4_NL> {
    static constexpr int qk = QK4_NL, qi = QI4_NL, vdr = VDR_IQ4_NL_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_iq4_nl_q8_1;
};
template <> struct Traits<GgmlType::Q6_K> {
    static constexpr int qk = QK_K, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ;
    static constexpr vec_dot_q_cuda_t vec_dot = vec_dot_q6_K_q8_1;
};

// llama.cpp's NVIDIA ("generic") schedule table: warps per CTA and rows per CTA by column count.
__host__ __device__ constexpr int calc_nwarps(int ncols_dst) { return ncols_dst <= 4 ? 4 : 2; }
__host__ __device__ // llama.cpp's small-K rule, and it matters here: "when K is small, increase rows_per_block to
// match nwarps so each warp has more work to do". At one column the default gives a CTA a
// single row, so for k = 1024 its 128 threads share four superblocks of work and most sit
// idle while the grid grows to one CTA per row. The rule triggers when a thread block covers
// every K block in one iteration.
__host__ __device__ constexpr int calc_rows_per_block(int ncols_dst, bool small_k = false,
                                                      int nwarps = 1) {
    return ncols_dst == 1 ? (small_k ? nwarps : 1) : 2;
}

/// Blocks one warp consumes per loop iteration, from the type's quant geometry.
template <GgmlType type> constexpr int blocks_per_iter_one_warp() {
    return Traits<type>::vdr * kWarpSize / Traits<type>::qi;
}

/// llama.cpp's own trigger for that rule -- and it is **measured worse on a 5090**, so this
/// returns false and the schedule stays one row per CTA. Enabling it (the commented condition)
/// cost 1.8 % of decode on Qwen3.5-0.8B-Q4_K_M: 862 -> 847 tok/s on the fused-parent artifact
/// and 835 -> 821 on the split one, medians of five over two rounds. Four rows per CTA quadruple
/// a CTA's register pressure and give it four separate weight streams, which on this card costs
/// more than the idle threads it recovers. Kept for the next card that disagrees.
template <GgmlType type> constexpr bool prefers_small_k(int ncols_dst, int k) {
    (void)ncols_dst;
    (void)k;
    // return calc_nwarps(ncols_dst) > 1 &&
    //        (k / Traits<type>::qk) < calc_nwarps(ncols_dst) * blocks_per_iter_one_warp<type>();
    return false;
}
constexpr int kWarpSize = 32;

template <typename DstT> __device__ __forceinline__ DstT to_dst(float v);
template <> __device__ __forceinline__ float to_dst<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 to_dst<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}
template <typename DstT> __device__ __forceinline__ float from_dst(DstT v);
template <> __device__ __forceinline__ float from_dst<float>(float v) { return v; }
template <> __device__ __forceinline__ float from_dst<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// What a finished row does with its accumulators. The default writes them to `dst`, which is
// every ordinary linear; a fused caller supplies its own and consumes the row in-kernel, which
// is how the GDN projection reaches its convolution without materialising a plane between them.
template <typename DstT, bool Accumulate, int Columns>
struct DirectStore {
    DstT* dst;
    int stride_col_dst;
    __device__ __forceinline__ void store(int row, const float (&value)[Columns]) const {
        DstT* slot = dst + row;
#pragma unroll
        for (int j = 0; j < Columns; ++j) {
            slot[j * stride_col_dst] =
                to_dst<DstT>(Accumulate ? from_dst<DstT>(slot[j * stride_col_dst]) + value[j]
                                        : value[j]);
        }
    }
};

template <GgmlType type, int ncols_dst, class Epilogue, bool SmallK = false>
__launch_bounds__(calc_nwarps(ncols_dst) * kWarpSize, 1)
__global__ void mul_mat_vec_q(const void* __restrict__ vx, const block_q8_1* __restrict__ vy,
                              const int ncols_x, const int nrows_x, const int stride_row_x,
                              const int stride_col_y, const Epilogue epilogue) {
    constexpr int qk  = Traits<type>::qk;
    constexpr int qi  = Traits<type>::qi;
    constexpr int vdr = Traits<type>::vdr;
    constexpr int nwarps              = calc_nwarps(ncols_dst);
    constexpr int rows_per_cuda_block = calc_rows_per_block(ncols_dst, SmallK, nwarps);
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

    // Reduce every column of a row before handing the row over: an epilogue that fuses a
    // downstream op (the GDN convolution) needs the row's whole token window at once.
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
        for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
            for (int l = 0; l < nwarps - 1; ++l) {
                tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
            }
            // xor butterfly: every lane ends up holding the total, and lane i owns row i
            tmp[j][i] = warp_sum(tmp[j][i]);
        }
    }
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
        if (threadIdx.x == i && (rows_per_cuda_block == 1 || row0 + i < nrows_x)) {
            float row_values[ncols_dst];
#pragma unroll
            for (int j = 0; j < ncols_dst; ++j) { row_values[j] = tmp[j][i]; }
            epilogue.store(row0 + i, row_values);
        }
    }
}

} // namespace sinfer::ops::detail::ggml
