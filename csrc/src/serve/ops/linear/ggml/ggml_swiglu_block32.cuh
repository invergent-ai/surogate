#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/ggml/ggml_dense_generic.cuh"

namespace sinfer::ops::detail::ggml {

// Gate and up occupy consecutive halves of the integer MMA tile and share its
// activation loads. Keep the projection's eight accumulator chains and BF16
// rounding boundaries before applying SwiGLU.
template <GgmlType Gate, GgmlType Up, int Rows, int Cols>
__global__ __launch_bounds__((Cols / 8) * 32) void swiglu_block32_prefill_kernel(
    const uint8_t* gate, const uint8_t* up, const int8_t* codes, const __half2* ds, int rows, int k,
    int tokens, __nv_bfloat16* out) {
    static_assert((Gate == GgmlType::Q8_0 || Gate == GgmlType::IQ4_NL) &&
                  (Up == GgmlType::Q8_0 || Up == GgmlType::IQ4_NL));
    static_assert(Rows == 16 || Rows == 32);
    constexpr int half_rows = Rows / 2, threads = (Cols / 8) * 32, stride = 272;
    __shared__ __align__(16) int8_t a[Rows * stride], b[Cols * stride];
    __shared__ float scales[8 * Rows], dx[Cols * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * half_rows, col0 = blockIdx.y * Cols;
    float acc[Rows / 16][8][4] = {};
    for (int first = 0; first < k; first += 256) {
        for (int item = tid; item < Cols * 16; item += threads) {
            const int col = item / 16, kk = item % 16 * 16;
            const bool valid = col0 + col < tokens && first + kk < k;
            cp_async_zfill<16, Cache::cg>(
                b + col * stride + kk, codes + (valid ? int64_t(col0 + col) * k + first + kk : 0),
                valid ? 16 : 0);
        }
        cp_commit();
        for (int item = tid; item < Cols * 8; item += threads) {
            const int col = item / 8, group = item % 8;
            const bool valid = col0 + col < tokens && first + 32 * group < k;
            dx[item] =
                valid ? __low2float(ds[int64_t(col0 + col) * (k / 32) + first / 32 + group]) : 0;
        }
        for (int item = tid; item < Rows * 32; item += threads) {
            const int row = item / 32, kk = item % 32 * 8;
            const int logical_row = row0 + row % half_rows;
            DenseAffineEight weight;
            if (first + kk < k && logical_row < rows) {
                if (row < half_rows) {
                    weight = dense_codes_eight<Gate>(
                        gate + int64_t(logical_row) * (k / 32) * block_bytes(Gate), first + kk);
                } else {
                    weight = dense_codes_eight<Up>(
                        up + int64_t(logical_row) * (k / 32) * block_bytes(Up), first + kk);
                }
            }
            *reinterpret_cast<uint2*>(a + row * stride + kk) = make_uint2(weight.lo, weight.hi);
            if (kk % 32 == 0) { scales[kk / 32 * Rows + row] = weight.scale; }
        }
        cp_wait<0>();
        __syncthreads();
#pragma unroll
        for (int group = 0; group < 8; ++group) {
            const int token = warp * 8 + (lane & 7);
            unsigned b0, b1;
            ldmatrix_x2(b0, b1,
                        smem_addr(b + token * stride + group * 32 + ((lane >> 3) & 1) * 16));
            const float x0 = dx[(warp * 8 + 2 * lid) * 8 + group];
            const float x1 = dx[(warp * 8 + 2 * lid + 1) * 8 + group];
#pragma unroll
            for (int mi = 0; mi < Rows / 16; ++mi) {
                const int ar = mi * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3,
                            smem_addr(a + ar * stride + group * 32 + (lane >> 4) * 16));
                int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
                const float s0    = scales[group * Rows + mi * 16 + gid];
                const float s1    = scales[group * Rows + mi * 16 + gid + 8];
                acc[mi][group][0] = __fmaf_rn(s0 * x0, float(d0), acc[mi][group][0]);
                acc[mi][group][1] = __fmaf_rn(s0 * x1, float(d1), acc[mi][group][1]);
                acc[mi][group][2] = __fmaf_rn(s1 * x0, float(d2), acc[mi][group][2]);
                acc[mi][group][3] = __fmaf_rn(s1 * x1, float(d3), acc[mi][group][3]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < (Rows == 16 ? 2 : 4); ++e) {
        const int row = row0 + gid + 8 * (e / 2), col = col0 + warp * 8 + 2 * lid + (e & 1);
        if (row < rows && col < tokens) {
            constexpr int up_tile = Rows == 16 ? 0 : 1;
            const int ue          = Rows == 16 ? e + 2 : e;
            const float g0        = (acc[0][0][e] + acc[0][1][e]) + (acc[0][2][e] + acc[0][3][e]);
            const float g1        = (acc[0][4][e] + acc[0][5][e]) + (acc[0][6][e] + acc[0][7][e]);
            const float u0        = (acc[up_tile][0][ue] + acc[up_tile][1][ue]) +
                             (acc[up_tile][2][ue] + acc[up_tile][3][ue]);
            const float u1 = (acc[up_tile][4][ue] + acc[up_tile][5][ue]) +
                             (acc[up_tile][6][ue] + acc[up_tile][7][ue]);
            const float g = __bfloat162float(__float2bfloat16_rn((g0 + g1) + 0.0f));
            const float u = __bfloat162float(__float2bfloat16_rn((u0 + u1) + 0.0f));
            out[int64_t(col) * rows + row] = __float2bfloat16_rn(swiglu_clamped(g, u, 0.0f));
        }
    }
}

} // namespace sinfer::ops::detail::ggml
