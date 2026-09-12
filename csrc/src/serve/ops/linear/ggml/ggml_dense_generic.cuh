#pragma once

#include "ops/linear/ggml/ggml_dense_codec.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/memory.cuh"

namespace sinfer::ops::detail::ggml {

template <bool Bias>
__device__ __forceinline__ float dense_affine_fma(float acc, float scale, float bias,
                                                float dx, int dot, int sum) {
    if constexpr (Bias) {
        return __fmaf_rn(dx, __fmaf_rn(scale, float(dot), bias * float(sum)), acc);
    } else {
        return __fmaf_rn(scale * dx, float(dot), acc);
    }
}

// Eight interleaved scale-group accumulators match the wide tile exactly.
// Complete chunks use a uniform warp loop; short rows use independent groups
// whose masks exclude lanes that have already reached the end of their row.
template <GgmlType Type, int Columns, int CooperatingLanes, bool FullChunks>
__global__ __launch_bounds__(32) void dense_affine_decode_kernel(
    const std::uint8_t* weights, const std::int8_t* codes, const __half2* ds,
    int rows, int k, int tokens, __nv_bfloat16* out, bool accumulate) {
    static_assert(CooperatingLanes == 1 || CooperatingLanes == 2 || CooperatingLanes == 4);
    constexpr int width = DenseFormat<Type>::kWidth, lanes_per_row = 8 * CooperatingLanes;
    constexpr int RowsPerWarp = 32 / lanes_per_row, pieces = width / (8 * CooperatingLanes);
    const int lane = threadIdx.x, row = blockIdx.x * RowsPerWarp + lane / lanes_per_row;
    const int token0 = blockIdx.y * Columns, part = (lane / CooperatingLanes) & 7;
    const int slice = lane % CooperatingLanes;
    const unsigned group_mask = FullChunks ? 0xffffffffu
        : ((1u << CooperatingLanes) - 1u) << (lane - slice);
    const auto* w = weights + std::int64_t(min(row, rows - 1)) * (k / block_values(Type)) * block_bytes(Type);
    float acc[Columns] = {};
#pragma unroll 4
    for (int cursor = FullChunks ? 0 : part * width; cursor < k; cursor += 8 * width) {
        const int first = FullChunks ? cursor + part * width : cursor;
        unsigned lo[pieces], hi[pieces];
        float scale = 0, bias = 0;
#pragma unroll
        for (int j = 0; j < pieces; ++j) {
            const auto v = dense_codes_eight<Type>(w, first + (j * CooperatingLanes + slice) * 8);
            lo[j] = v.lo; hi[j] = v.hi;
            if (j == 0) { scale = v.scale; bias = v.bias; }
        }
#pragma unroll
        for (int c = 0; c < Columns; ++c) {
            const int token = token0 + c;
            if (token < tokens) {
                int dot = 0, sum = 0;
#pragma unroll
                for (int j = 0; j < pieces; ++j) {
                    const auto* x = codes + std::int64_t(token) * k + first;
                    const auto v = *reinterpret_cast<const int2*>(x + (j * CooperatingLanes + slice) * 8);
                    dot = __dp4a(int(hi[j]), v.y, __dp4a(int(lo[j]), v.x, dot));
                    if constexpr (DenseFormat<Type>::kBias) {
                        sum = __dp4a(0x01010101, v.y, __dp4a(0x01010101, v.x, sum));
                    }
                }
#pragma unroll
                for (int offset = 1; offset < CooperatingLanes; offset *= 2) {
                    dot += __shfl_xor_sync(group_mask, dot, offset, CooperatingLanes);
                    if constexpr (DenseFormat<Type>::kBias) {
                        sum += __shfl_xor_sync(group_mask, sum, offset, CooperatingLanes);
                    }
                }
                const float dx = __low2float(ds[std::int64_t(token) * (k / 32) + first / 32]);
                acc[c] = dense_affine_fma<DenseFormat<Type>::kBias>(acc[c], scale, bias, dx, dot, sum);
            }
        }
    }
#pragma unroll
    for (int c = 0; c < Columns; ++c) {
        acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], CooperatingLanes, lanes_per_row);
        acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 2 * CooperatingLanes, lanes_per_row);
        acc[c] += __shfl_xor_sync(0xffffffffu, acc[c], 4 * CooperatingLanes, lanes_per_row);
        if (lane % lanes_per_row == 0 && row < rows && token0 + c < tokens) {
            const auto i = std::int64_t(token0 + c) * rows + row;
            out[i] = __float2bfloat16_rn(acc[c] + (accumulate ? __bfloat162float(out[i]) : 0.0f));
        }
    }
}

template <int Rows, int Cols>
__device__ __forceinline__ void dense_store_split(const float* partial, int row0, int col0,
                                                  int rows, int tokens, __nv_bfloat16* out,
                                                  bool accumulate) {
    constexpr int plane = Rows * Cols;
    for (int i = threadIdx.x; i < plane; i += blockDim.x) {
        const int row = row0 + i % Rows, col = col0 + i / Rows;
        if (row < rows && col < tokens) {
            const float a = (partial[i] + partial[plane + i]) +
                            (partial[2 * plane + i] + partial[3 * plane + i]);
            const float b = (partial[4 * plane + i] + partial[5 * plane + i]) +
                            (partial[6 * plane + i] + partial[7 * plane + i]);
            const auto index = std::int64_t(col) * rows + row;
            out[index] = __float2bfloat16_rn((a + b) + (accumulate ? __bfloat162float(out[index]) : 0.0f));
        }
    }
}

__device__ __forceinline__ unsigned dense_ldmatrix_x1(unsigned address) {
    unsigned value;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];" : "=r"(value) : "r"(address));
    return value;
}

// A 256-value K tile makes each of the eight accumulator indices constant.
// Sixteen weight rows keep those accumulators in registers while wider token
// tiles amortize codebook decoding and metadata loads across the batch.
template <GgmlType Type, int Cols>
__global__ __launch_bounds__((Cols / 8) * 32) void dense_affine_mma_kernel(
    const std::uint8_t* weights, const std::int8_t* codes, const __half2* ds,
    int rows, int k, int tokens, __nv_bfloat16* out, bool accumulate) {
    constexpr int Rows = 16, threads = (Cols / 8) * 32, width = DenseFormat<Type>::kWidth;
    constexpr int stride = 272, groups = 256 / width;
    __shared__ __align__(16) std::int8_t a[Rows * stride], b[Cols * stride];
    __shared__ float2 scales[groups * Rows];
    __shared__ float dx[Cols * 8];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * Rows, col0 = blockIdx.y * Cols;
    float acc[8][4] = {};
    for (int first = 0; first < k; first += 256) {
        for (int item = tid; item < Cols * 16; item += threads) {
            const int col = item / 16, kk = item % 16 * 16;
            const bool valid = col0 + col < tokens && first + kk < k;
            cp_async_zfill<16, Cache::cg>(b + col * stride + kk,
                codes + (valid ? std::int64_t(col0 + col) * k + first + kk : 0), valid ? 16 : 0);
        }
        cp_commit();
        for (int item = tid; item < Cols * 8; item += threads) {
            const int col = item / 8, g = item % 8;
            const bool valid = col0 + col < tokens && first + 32 * g < k;
            dx[item] = valid ? __low2float(ds[std::int64_t(col0 + col) * (k / 32) + first / 32 + g]) : 0;
        }
        for (int item = tid; item < Rows * 32; item += threads) {
            const int row = item / 32, kk = item % 32 * 8;
            DenseAffineEight weight;
            if (first + kk < k && row0 + row < rows) {
                const auto* w = weights + std::int64_t(row0 + row) * (k / block_values(Type)) * block_bytes(Type);
                weight = dense_codes_eight<Type>(w, first + kk);
            }
            *reinterpret_cast<uint2*>(a + row * stride + kk) = make_uint2(weight.lo, weight.hi);
            if (kk % width == 0) { scales[kk / width * Rows + row] = make_float2(weight.scale, weight.bias); }
        }
        cp_wait<0>(); __syncthreads();
#pragma unroll
        for (int g = 0; g < groups; ++g) {
            const int token = warp * 8 + (lane & 7);
            const int ar = (lane & 7) + ((lane >> 3) & 1) * 8;
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            if constexpr (width == 32) {
                unsigned a0, a1, a2, a3, b0, b1;
                ldmatrix_x4(a0, a1, a2, a3, smem_addr(a + ar * stride + g * width + (lane >> 4) * 16));
                ldmatrix_x2(b0, b1, smem_addr(b + token * stride + g * width + ((lane >> 3) & 1) * 16));
                mma_s8(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1);
            } else {
                unsigned a0, a1;
                ldmatrix_x2(a0, a1, smem_addr(a + ar * stride + g * width));
                const unsigned b0 = dense_ldmatrix_x1(smem_addr(b + token * stride + g * width));
                mma_s8_k16(d0, d1, d2, d3, a0, a1, b0);
            }
            const float x0 = dx[(warp * 8 + 2 * lid) * 8 + g * width / 32];
            const float x1 = dx[(warp * 8 + 2 * lid + 1) * 8 + g * width / 32];
            int sum0 = 0, sum1 = 0;
            if constexpr (DenseFormat<Type>::kBias) {
                const int offset = (lane >> 3) % (width / 8) * 8;
                const auto* v = reinterpret_cast<const int*>(b + token * stride + g * width + offset);
                int sum = __dp4a(0x01010101, v[1], __dp4a(0x01010101, v[0], 0));
                sum += __shfl_xor_sync(0xffffffffu, sum, 8);
                if constexpr (width == 32) { sum += __shfl_xor_sync(0xffffffffu, sum, 16); }
                sum0 = __shfl_sync(0xffffffffu, sum, 2 * lid);
                sum1 = __shfl_sync(0xffffffffu, sum, 2 * lid + 1);
            }
            const float2 s0 = scales[g * Rows + gid], s1 = scales[g * Rows + gid + 8];
            constexpr bool bias = DenseFormat<Type>::kBias;
            acc[g % 8][0] = dense_affine_fma<bias>(acc[g % 8][0], s0.x, s0.y, x0, d0, sum0);
            acc[g % 8][1] = dense_affine_fma<bias>(acc[g % 8][1], s0.x, s0.y, x1, d1, sum1);
            acc[g % 8][2] = dense_affine_fma<bias>(acc[g % 8][2], s1.x, s1.y, x0, d2, sum0);
            acc[g % 8][3] = dense_affine_fma<bias>(acc[g % 8][3], s1.x, s1.y, x1, d3, sum1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int row = row0 + gid + 8 * (e / 2);
        const int col = col0 + warp * 8 + 2 * lid + (e & 1);
        if (row < rows && col < tokens) {
            const float p0 = (acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e]);
            const float p1 = (acc[4][e] + acc[5][e]) + (acc[6][e] + acc[7][e]);
            const auto i = std::int64_t(col) * rows + row;
            out[i] = __float2bfloat16_rn((p0 + p1) + (accumulate ? __bfloat162float(out[i]) : 0.0f));
        }
    }
}

// F16 weights and BF16 activations are represented exactly by TF32. Keep the
// same k8 MMA order at every width without adding activation quantization or
// rounding the stored weights to BF16.
template <int Rows>
__global__ __launch_bounds__(256) void dense_f16_decode_kernel(
    const __half* weights, const __nv_bfloat16* x, int rows, int k, int tokens,
    __nv_bfloat16* out, bool accumulate) {
    __shared__ float partial[8 * Rows * 8];
    const int lane = threadIdx.x & 31, part = threadIdx.x >> 5, gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * Rows, row = row0 + (gid % Rows);
    const auto* w = weights + std::int64_t(min(row, rows - 1)) * k;
    const auto* input = x + std::int64_t(gid < tokens ? gid : 0) * k;
    float acc[4] = {};
#pragma unroll 4
    for (int kk = part * 8; kk < k; kk += 64) {
        const float a0 = __half2float(w[kk + lid]);
        const float a2 = __half2float(w[kk + lid + 4]);
        const float b0 = gid < tokens ? __bfloat162float(input[kk + lid]) : 0;
        const float b1 = gid < tokens ? __bfloat162float(input[kk + lid + 4]) : 0;
        const auto* w1 = weights + std::int64_t(min(row + 8, rows - 1)) * k;
        const float a1 = Rows == 16 ? __half2float(w1[kk + lid]) : a0;
        const float a3 = Rows == 16 ? __half2float(w1[kk + lid + 4]) : a2;
        mma_tf32(acc[0], acc[1], acc[2], acc[3], a0, a1, a2, a3, b0, b1);
    }
    if (gid < Rows) {
#pragma unroll
        for (int e = 0; e < (Rows == 16 ? 4 : 2); ++e) {
            partial[part * Rows * 8 + (2 * lid + (e & 1)) * Rows + gid + 8 * (e / 2)] = acc[e];
        }
    }
    __syncthreads();
    dense_store_split<Rows, 8>(partial, row0, 0, rows, tokens, out, accumulate);
}

template <int Cols>
__global__ __launch_bounds__((Cols / 8) * 32) void dense_f16_mma_kernel(
    const __half* weights, const __nv_bfloat16* x, int rows, int k, int tokens,
    __nv_bfloat16* out, bool accumulate) {
    constexpr int Rows = 16, threads = (Cols / 8) * 32, stride = 136;
    __shared__ __align__(16) __half a[Rows * stride];
    __shared__ __align__(16) __nv_bfloat16 b[Cols * stride];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * Rows, col0 = blockIdx.y * Cols;
    float acc[8][4] = {};
    for (int first = 0; first < k; first += 128) {
        for (int item = tid; item < Rows * 16; item += threads) {
            const int row = item / 16, kk = item % 16 * 8;
            const bool valid = row0 + row < rows && first + kk < k;
            cp_async_zfill<16, Cache::cg>(a + row * stride + kk,
                weights + (valid ? std::int64_t(row0 + row) * k + first + kk : 0), valid ? 16 : 0);
        }
        for (int item = tid; item < Cols * 16; item += threads) {
            const int col = item / 16, kk = item % 16 * 8;
            const bool valid = col0 + col < tokens && first + kk < k;
            cp_async_zfill<16, Cache::cg>(b + col * stride + kk,
                x + (valid ? std::int64_t(col0 + col) * k + first + kk : 0), valid ? 16 : 0);
        }
        cp_commit(); cp_wait<0>(); __syncthreads();
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            const float b0 = __bfloat162float(b[(warp * 8 + gid) * stride + g * 8 + lid]);
            const float b1 = __bfloat162float(b[(warp * 8 + gid) * stride + g * 8 + lid + 4]);
            mma_tf32(acc[g % 8][0], acc[g % 8][1], acc[g % 8][2], acc[g % 8][3],
                __half2float(a[gid * stride + g * 8 + lid]), __half2float(a[(gid + 8) * stride + g * 8 + lid]),
                __half2float(a[gid * stride + g * 8 + lid + 4]), __half2float(a[(gid + 8) * stride + g * 8 + lid + 4]), b0, b1);
        }
        __syncthreads();
    }
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int row = row0 + gid + 8 * (e / 2);
        const int col = col0 + warp * 8 + 2 * lid + (e & 1);
        if (row < rows && col < tokens) {
            const float p0 = (acc[0][e] + acc[1][e]) + (acc[2][e] + acc[3][e]);
            const float p1 = (acc[4][e] + acc[5][e]) + (acc[6][e] + acc[7][e]);
            const auto i = std::int64_t(col) * rows + row;
            out[i] = __float2bfloat16_rn((p0 + p1) + (accumulate ? __bfloat162float(out[i]) : 0.0f));
        }
    }
}

} // namespace sinfer::ops::detail::ggml
