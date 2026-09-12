#pragma once

#include "ops/linear/ggml/ggml_dense_generic.cuh"

namespace sinfer::ops::detail::ggml {

// Load several consecutive blocks together, then apply their contributions in
// the same eight-part order as the wide projection. Each warp owns one row.
template <GgmlType Type, int Lanes, int Warps, int Unroll>
__global__ __launch_bounds__(Warps * 32) void dense_block32_decode_kernel(
    const uint8_t* __restrict__ weights, const int8_t* __restrict__ codes,
    const __half2* __restrict__ ds, int rows, int k, int tokens, __nv_bfloat16* __restrict__ out,
    bool accumulate) {
    static_assert(Type == GgmlType::Q8_0 || Type == GgmlType::IQ4_NL);
    static_assert(Lanes == 1 || Lanes == 2 || Lanes == 4);
    const int lane = threadIdx.x & 31, row = blockIdx.x * Warps + (threadIdx.x >> 5),
              token = blockIdx.y;
    if (row >= rows) { return; }
    const int group = lane / Lanes, slice = lane % Lanes, part = group & 7;
    constexpr int groups = 32 / Lanes;
    const auto* w        = weights + int64_t(row) * (k / 32) * block_bytes(Type);
    const auto* x        = codes + int64_t(token) * k;
    const auto* scales   = ds + int64_t(token) * (k / 32);
    float acc            = 0;
#pragma unroll Unroll
    for (int chunk = 0; chunk < k; chunk += groups * 32) {
        const int first = chunk + group * 32;
        int dot         = 0;
        float scale     = 0;
        if (first < k) {
            const auto* b = w + (first / 32) * block_bytes(Type);
            scale =
                __half2float(*reinterpret_cast<const __half*>(b)) * __low2float(scales[first / 32]);
            if constexpr (Type == GgmlType::IQ4_NL) {
#pragma unroll
                for (int j = 0; j < 4 / Lanes; ++j) {
                    const int index = j * Lanes + slice;
                    const int q     = get_int_b2(reinterpret_cast<const int8_t*>(b + 2), index);
                    const int2 v    = table16_levels(q, kIq4nlValues);
                    dot = __dp4a(v.x, *reinterpret_cast<const int*>(x + first + index * 4), dot);
                    dot =
                        __dp4a(v.y, *reinterpret_cast<const int*>(x + first + 16 + index * 4), dot);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8 / Lanes; ++j) {
                    const int index = j * Lanes + slice;
                    const int q     = get_int_b2(reinterpret_cast<const int8_t*>(b + 2), index);
                    dot = __dp4a(q, *reinterpret_cast<const int*>(x + first + index * 4), dot);
                }
            }
        }
#pragma unroll
        for (int offset = 1; offset < Lanes; offset *= 2)
            dot += __shfl_xor_sync(0xffffffffu, dot, offset, Lanes);
#pragma unroll
        for (int phase = 0; phase < groups / 8; ++phase) {
            const int src = (part + phase * 8) * Lanes;
            acc           = __fmaf_rn(__shfl_sync(0xffffffffu, scale, src),
                                      float(__shfl_sync(0xffffffffu, dot, src)), acc);
        }
    }
    acc += __shfl_xor_sync(0xffffffffu, acc, Lanes, 8 * Lanes);
    acc += __shfl_xor_sync(0xffffffffu, acc, 2 * Lanes, 8 * Lanes);
    acc += __shfl_xor_sync(0xffffffffu, acc, 4 * Lanes, 8 * Lanes);
    if (lane == 0) {
        const auto i = int64_t(token) * rows + row;
        out[i]       = __float2bfloat16_rn(acc + (accumulate ? __bfloat162float(out[i]) : 0.0f));
    }
}

// Assign the eight K partitions to separate warps. Their integer MMA dots
// retain each block scale; the final sum and BF16 rounding match prefill.
template <GgmlType Type, int Rows>
__global__ __launch_bounds__(256) void dense_block32_decode_mma_kernel(
    const uint8_t* __restrict__ weights, const int8_t* __restrict__ codes,
    const __half2* __restrict__ ds, int rows, int k, int tokens, __nv_bfloat16* __restrict__ out,
    bool accumulate) {
    static_assert(Type == GgmlType::Q8_0 || Type == GgmlType::IQ4_NL);
    static_assert(Rows == 4 || Rows == 8 || Rows == 16);
    __shared__ float partial[8 * Rows * 8];
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5, gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * Rows, row = row0 + gid % Rows;
    const auto* w  = weights + int64_t(min(row, rows - 1)) * (k / 32) * block_bytes(Type);
    const auto* w1 = weights + int64_t(min(row + 8, rows - 1)) * (k / 32) * block_bytes(Type);
    float acc[4]   = {};
#pragma unroll 2
    for (int group = part; group < k / 32; group += 8) {
        unsigned a0, a1, a2, a3, b0 = 0, b1 = 0;
        const auto* block  = w + group * block_bytes(Type);
        const auto* block1 = w1 + group * block_bytes(Type);
        if constexpr (Type == GgmlType::IQ4_NL) {
            const int2 v = table16_levels(
                get_int_b2(reinterpret_cast<const int8_t*>(block + 2), lid), kIq4nlValues);
            a0 = v.x;
            a2 = v.y;
            if constexpr (Rows == 16) {
                const int2 v1 = table16_levels(
                    get_int_b2(reinterpret_cast<const int8_t*>(block1 + 2), lid), kIq4nlValues);
                a1 = v1.x;
                a3 = v1.y;
            } else {
                a1 = a0;
                a3 = a2;
            }
        } else {
            a0 = get_int_b2(reinterpret_cast<const int8_t*>(block + 2), lid);
            a2 = get_int_b2(reinterpret_cast<const int8_t*>(block + 2), lid + 4);
            if constexpr (Rows == 16) {
                a1 = get_int_b2(reinterpret_cast<const int8_t*>(block1 + 2), lid);
                a3 = get_int_b2(reinterpret_cast<const int8_t*>(block1 + 2), lid + 4);
            } else {
                a1 = a0;
                a3 = a2;
            }
        }
        float dx = 0;
        if (gid < tokens) {
            const auto* input = reinterpret_cast<const int*>(codes + int64_t(gid) * k + group * 32);
            b0                = input[lid];
            b1                = input[lid + 4];
            dx                = __low2float(ds[int64_t(gid) * (k / 32) + group]);
        }
        const float x0  = __shfl_sync(0xffffffffu, dx, lid * 8),
                    x1  = __shfl_sync(0xffffffffu, dx, lid * 8 + 4);
        const float ws  = __half2float(*reinterpret_cast<const __half*>(block));
        const float ws1 = Rows == 16 ? __half2float(*reinterpret_cast<const __half*>(block1)) : ws;
        int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        sinfer::ops::mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
        acc[0] = __fmaf_rn(ws * x0, float(d0), acc[0]);
        acc[1] = __fmaf_rn(ws * x1, float(d1), acc[1]);
        acc[2] = __fmaf_rn(ws1 * x0, float(d2), acc[2]);
        acc[3] = __fmaf_rn(ws1 * x1, float(d3), acc[3]);
    }
    if (gid < Rows) {
#pragma unroll
        for (int e = 0; e < (Rows == 16 ? 4 : 2); ++e)
            partial[part * Rows * 8 + (lid * 2 + (e & 1)) * Rows + gid + 8 * (e / 2)] = acc[e];
    }
    __syncthreads();
    dense_store_split<Rows, 8>(partial, row0, 0, rows, tokens, out, accumulate);
}

} // namespace sinfer::ops::detail::ggml
