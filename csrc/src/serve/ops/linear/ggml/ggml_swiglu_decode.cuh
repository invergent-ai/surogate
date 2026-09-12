#pragma once

#include "ops/common/math.cuh"
#include "ops/linear/ggml/ggml_dense_block32.cuh"

namespace sinfer::ops::detail::ggml {

// Adjacent warps compute matching gate/up rows. Round both projections exactly
// as linear_launch does, then apply the same activation as the unfused path.
template <GgmlType Gate, GgmlType Up>
__global__ __launch_bounds__(128) void swiglu_decode_kernel(const uint8_t* gate, const uint8_t* up,
                                                            const int8_t* codes, const __half2* ds,
                                                            int rows, int k, int tokens,
                                                            __nv_bfloat16* out) {
    __shared__ __nv_bfloat16 halves[4];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int row = blockIdx.x * 2 + warp / 2, token = blockIdx.y;
    float value = 0;
    if (row < rows) {
        const auto* x      = codes + int64_t(token) * k;
        const auto* scales = ds + int64_t(token) * (k / 32);
        if ((warp & 1) == 0) {
            value = dense_block32_dot<Gate, 2, 1>(
                gate + int64_t(row) * (k / 32) * block_bytes(Gate), x, scales, k);
        } else {
            value = dense_block32_dot<Up, 2, 1>(up + int64_t(row) * (k / 32) * block_bytes(Up), x,
                                                scales, k);
        }
    }
    if (lane == 0) { halves[warp] = __float2bfloat16_rn(value + 0.0f); }
    __syncthreads();
    const int output_row = blockIdx.x * 2 + threadIdx.x;
    if (threadIdx.x < 2 && output_row < rows) {
        out[int64_t(token) * rows + output_row] = __float2bfloat16_rn(
            swiglu_clamped(__bfloat162float(halves[2 * threadIdx.x]),
                           __bfloat162float(halves[2 * threadIdx.x + 1]), 0.0f));
    }
}

template <GgmlType Type>
__device__ __forceinline__ int2 swiglu_block32_words(const uint8_t* block, int word) {
    const int lo = get_int_b2(block + 2, word);
    if constexpr (Type == GgmlType::IQ4_NL) {
        return table16_levels(lo, kIq4nlValues);
    } else {
        static_assert(Type == GgmlType::Q8_0);
        return make_int2(lo, get_int_b2(block + 2, word + 4));
    }
}

__device__ __forceinline__ float swiglu_sum_parts(const float* partial, int index) {
    constexpr int plane = 16 * 8;
    const float lo      = (partial[index] + partial[plane + index]) +
                     (partial[2 * plane + index] + partial[3 * plane + index]);
    const float hi = (partial[4 * plane + index] + partial[5 * plane + index]) +
                     (partial[6 * plane + index] + partial[7 * plane + index]);
    return lo + hi;
}

// Eight logical output rows occupy the two halves of one 16-row integer MMA
// tile. Each warp owns one K partition, just as in the projection-only kernel.
template <GgmlType Gate, GgmlType Up>
__global__ __launch_bounds__(256) void swiglu_decode_mma_kernel(const uint8_t* gate,
                                                                const uint8_t* up,
                                                                const int8_t* codes,
                                                                const __half2* ds, int rows, int k,
                                                                int tokens, __nv_bfloat16* out) {
    __shared__ float partial[8 * 16 * 8];
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * 8, row = min(row0 + gid, rows - 1);
    const auto* g = gate + int64_t(row) * (k / 32) * block_bytes(Gate);
    const auto* u = up + int64_t(row) * (k / 32) * block_bytes(Up);
    float acc[4]  = {};
#pragma unroll 2
    for (int group = part; group < k / 32; group += 8) {
        const auto* gb = g + group * block_bytes(Gate);
        const auto* ub = u + group * block_bytes(Up);
        const int2 a   = swiglu_block32_words<Gate>(gb, lid);
        const int2 b   = swiglu_block32_words<Up>(ub, lid);
        unsigned x0 = 0, x1 = 0;
        float dx = 0;
        if (gid < tokens) {
            const auto* input = reinterpret_cast<const int*>(codes + int64_t(gid) * k + group * 32);
            x0                = input[lid];
            x1                = input[lid + 4];
            dx                = __low2float(ds[int64_t(gid) * (k / 32) + group]);
        }
        const float dx0 = __shfl_sync(0xffffffffu, dx, lid * 8);
        const float dx1 = __shfl_sync(0xffffffffu, dx, lid * 8 + 4);
        const float gs  = __half2float(*reinterpret_cast<const __half*>(gb));
        const float us  = __half2float(*reinterpret_cast<const __half*>(ub));
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        mma_s8(d0, d1, d2, d3, a.x, b.x, a.y, b.y, x0, x1);
        acc[0] = __fmaf_rn(gs * dx0, float(d0), acc[0]);
        acc[1] = __fmaf_rn(gs * dx1, float(d1), acc[1]);
        acc[2] = __fmaf_rn(us * dx0, float(d2), acc[2]);
        acc[3] = __fmaf_rn(us * dx1, float(d3), acc[3]);
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        partial[part * 128 + (lid * 2 + (e & 1)) * 16 + gid + 8 * (e / 2)] = acc[e];
    }
    __syncthreads();
    if (threadIdx.x < 8 * tokens) {
        const int column = threadIdx.x / 8, output_row = row0 + threadIdx.x % 8;
        if (output_row < rows) {
            const int index = column * 16 + threadIdx.x % 8;
            const float gv =
                __bfloat162float(__float2bfloat16_rn(swiglu_sum_parts(partial, index) + 0.0f));
            const float uv =
                __bfloat162float(__float2bfloat16_rn(swiglu_sum_parts(partial, index + 8) + 0.0f));
            out[int64_t(column) * rows + output_row] =
                __float2bfloat16_rn(swiglu_clamped(gv, uv, 0.0f));
        }
    }
}

} // namespace sinfer::ops::detail::ggml
