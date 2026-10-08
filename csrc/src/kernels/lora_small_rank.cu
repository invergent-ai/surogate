// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>

#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

template <int R>
__global__ void lora_accum_b_small_rank_bf16_kernel(nv_bfloat16* __restrict__ output,
                                                    const nv_bfloat16* __restrict__ B,
                                                    const nv_bfloat16* __restrict__ intermediate,
                                                    int BT,
                                                    int total_out_features,
                                                    int out_features,
                                                    int output_offset,
                                                    float scaling) {
    constexpr int kRows = 16;
    constexpr int kCols = 16;
    __shared__ nv_bfloat16 s_inter[kRows][R];
    __shared__ nv_bfloat16 s_b[kCols][R];

    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
    for (int r = threadIdx.x; r < R; r += kCols) {
        s_inter[threadIdx.y][r] = (row < BT) ? intermediate[row * R + r] : __float2bfloat16(0.0f);
    }
#pragma unroll
    for (int r = threadIdx.y; r < R; r += kRows) {
        s_b[threadIdx.x][r] = (col < out_features) ? B[col * R + r] : __float2bfloat16(0.0f);
    }

    __syncthreads();

    if (row >= BT || col >= out_features) return;

    float acc = 0.0f;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        acc += __bfloat162float(s_inter[threadIdx.y][r]) * __bfloat162float(s_b[threadIdx.x][r]);
    }

    const int out_idx = row * total_out_features + output_offset + col;
    const float value = __bfloat162float(output[out_idx]) + scaling * acc;
    output[out_idx] = __float2bfloat16(value);
}

// The same update, eight output columns a thread. A block owns 128 columns and keeps their B rows
// in shared memory as FP32, then walks a strip of rows: one 16-byte read and one write of the
// output per thread and row, and the intermediate row (R values, the same for the 16 threads of a
// row) from L1. That is all the memory traffic the update needs. A GEMM into a scratch buffer and
// an add afterwards move the output slice four times, and on a GPU short of bandwidth for its
// compute -- the DGX Spark's GB10 -- the LoRA projections are bound by exactly that traffic.
// The sum runs over r in order with fused multiply-adds and the output is rounded once.
constexpr int kVecCols = 8;
constexpr int kVecColThreads = 16;
constexpr int kVecRowThreads = 16;
constexpr int kVecBlockCols = kVecCols * kVecColThreads;
constexpr int kVecRowsPerBlock = 64;

template <int R>
__global__ void __launch_bounds__(kVecColThreads* kVecRowThreads)
    lora_accum_b_vec8_bf16_kernel(nv_bfloat16* __restrict__ output,
                                  const nv_bfloat16* __restrict__ B,
                                  const nv_bfloat16* __restrict__ intermediate,
                                  int BT,
                                  int total_out_features,
                                  int out_features,
                                  int output_offset,
                                  float scaling) {
    // Thread x's columns 8x..8x+3 sit at 4x and 8x+4..8x+7 at 64 + 4x, so the eight threads of a
    // quarter warp read distinct banks.
    __shared__ __align__(16) float s_b[R][kVecBlockCols];
    const int block_col = blockIdx.x * kVecBlockCols;
    const int thread = threadIdx.y * kVecColThreads + threadIdx.x;
    for (int i = thread; i < R * kVecBlockCols; i += kVecColThreads * kVecRowThreads) {
        const int c = i / R, r = i % R;
        const int slot = ((c & 7) >> 2) * (kVecBlockCols / 2) + (c >> 3) * 4 + (c & 3);
        s_b[r][slot] = (block_col + c < out_features) ? __bfloat162float(B[(long)(block_col + c) * R + r]) : 0.0f;
    }
    __syncthreads();

    const int col = block_col + threadIdx.x * kVecCols;
    if (col >= out_features) return;
    const int row_begin = static_cast<int>(blockIdx.y) * kVecRowsPerBlock;
    const int row_end = min(BT, row_begin + kVecRowsPerBlock);
    for (int row = row_begin + static_cast<int>(threadIdx.y); row < row_end; row += kVecRowThreads) {
        float inter[R];
        const uint4* inter_row = reinterpret_cast<const uint4*>(intermediate + (long)row * R);
#pragma unroll
        for (int v = 0; v < R / 8; ++v) {
            const uint4 packed = inter_row[v];
            const nv_bfloat16* values = reinterpret_cast<const nv_bfloat16*>(&packed);
#pragma unroll
            for (int j = 0; j < 8; ++j) inter[v * 8 + j] = __bfloat162float(values[j]);
        }
        float acc[kVecCols] = {};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float4 lo = *reinterpret_cast<const float4*>(&s_b[r][threadIdx.x * 4]);
            const float4 hi = *reinterpret_cast<const float4*>(&s_b[r][kVecBlockCols / 2 + threadIdx.x * 4]);
            acc[0] = __fmaf_rn(inter[r], lo.x, acc[0]);
            acc[1] = __fmaf_rn(inter[r], lo.y, acc[1]);
            acc[2] = __fmaf_rn(inter[r], lo.z, acc[2]);
            acc[3] = __fmaf_rn(inter[r], lo.w, acc[3]);
            acc[4] = __fmaf_rn(inter[r], hi.x, acc[4]);
            acc[5] = __fmaf_rn(inter[r], hi.y, acc[5]);
            acc[6] = __fmaf_rn(inter[r], hi.z, acc[6]);
            acc[7] = __fmaf_rn(inter[r], hi.w, acc[7]);
        }
        uint4* out = reinterpret_cast<uint4*>(output + (long)row * total_out_features + output_offset + col);
        uint4 packed = *out;
        nv_bfloat16* values = reinterpret_cast<nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < kVecCols; ++j)
            values[j] = __float2bfloat16_rn(__fmaf_rn(scaling, acc[j], __bfloat162float(values[j])));
        *out = packed;
    }
}

template <int R>
static void launch_lora_accum_b_small_rank_bf16(Tensor& output,
                                                const Tensor& B,
                                                const Tensor& intermediate,
                                                int BT,
                                                int total_out_features,
                                                int out_features,
                                                int output_offset,
                                                float scaling,
                                                cudaStream_t stream) {
    // Eight columns a thread wherever every row of the slice starts on a 16-byte boundary.
    const bool vectorized = out_features % kVecCols == 0 && output_offset % kVecCols == 0 &&
                            total_out_features % kVecCols == 0 &&
                            reinterpret_cast<std::uintptr_t>(output.Data) % 16 == 0 &&
                            reinterpret_cast<std::uintptr_t>(intermediate.Data) % 16 == 0;
    if (vectorized) {
        const dim3 block(kVecColThreads, kVecRowThreads);
        const dim3 grid((out_features + kVecBlockCols - 1) / kVecBlockCols,
                        (BT + kVecRowsPerBlock - 1) / kVecRowsPerBlock);
        lora_accum_b_vec8_bf16_kernel<R><<<grid, block, 0, stream>>>(output.get<nv_bfloat16>(),
                                                                     B.get<nv_bfloat16>(),
                                                                     intermediate.get<nv_bfloat16>(),
                                                                     BT,
                                                                     total_out_features,
                                                                     out_features,
                                                                     output_offset,
                                                                     scaling);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    dim3 block(16, 16);
    dim3 grid((out_features + block.x - 1) / block.x, (BT + block.y - 1) / block.y);
    lora_accum_b_small_rank_bf16_kernel<R><<<grid, block, 0, stream>>>(output.get<nv_bfloat16>(),
                                                                       B.get<nv_bfloat16>(),
                                                                       intermediate.get<nv_bfloat16>(),
                                                                       BT,
                                                                       total_out_features,
                                                                       out_features,
                                                                       output_offset,
                                                                       scaling);
    CUDA_CHECK(cudaGetLastError());
}

bool lora_accum_b_small_rank_bf16(Tensor& output,
                                  const Tensor& B,
                                  const Tensor& intermediate,
                                  int BT,
                                  int total_out_features,
                                  int out_features,
                                  int output_offset,
                                  int rank,
                                  float scaling,
                                  cudaStream_t stream) {
    const bool debug = std::getenv("SUROGATE_DEBUG_LORA_GEMM") != nullptr;
    auto reject = [&](const char* reason) -> bool {
        if (debug) {
            std::fprintf(stderr,
                         "[LORA-B-SMALL] reject=%s rank=%d BT=%d total_out=%d out=%d off=%d scaling=%g "
                         "out_ptr=%p B_ptr=%p inter_ptr=%p shapes(o,b,i)=[%ld,%ld]/[%ld,%ld]/[%ld,%ld] "
                         "ranks(o,b,i)=(%d,%d,%d) dtypes(o,b,i)=(%d,%d,%d)\n",
                         reason,
                         rank,
                         BT,
                         total_out_features,
                         out_features,
                         output_offset,
                         scaling,
                         (void*)output.Data,
                         (void*)B.Data,
                         (void*)intermediate.Data,
                         output.Sizes[0],
                         output.Sizes[output.Rank - 1],
                         B.Sizes[0],
                         B.Sizes[1],
                         intermediate.Sizes[0],
                         intermediate.Sizes[1],
                         output.Rank,
                         B.Rank,
                         intermediate.Rank,
                         (int)output.DType,
                         (int)B.DType,
                         (int)intermediate.DType);
        }
        return false;
    };

    if (output.DType != ETensorDType::BF16 || B.DType != ETensorDType::BF16 ||
        intermediate.DType != ETensorDType::BF16) {
        return reject("dtype");
    }
    if (!output.Data || !B.Data || !intermediate.Data) {
        return reject("null_data");
    }
    if (BT <= 0 || total_out_features <= 0 || out_features <= 0 || rank <= 0) {
        return reject("invalid_dims");
    }
    if (output_offset < 0 || output_offset + out_features > total_out_features) {
        return reject("offset");
    }
    if (output.Rank < 1 || output.Sizes[output.Rank - 1] < total_out_features) {
        return reject("output_shape");
    }
    if (B.Rank < 2 || B.Sizes[0] < out_features || B.Sizes[1] < rank) {
        return reject("B_shape");
    }
    if (intermediate.Rank < 1 || intermediate.Sizes[intermediate.Rank - 1] != rank) {
        return reject("intermediate_shape");
    }

    const std::size_t out_needed = (static_cast<std::size_t>(BT - 1) * static_cast<std::size_t>(total_out_features)) +
                                   static_cast<std::size_t>(output_offset + out_features);
    const std::size_t b_needed = static_cast<std::size_t>(out_features) * static_cast<std::size_t>(rank);
    const std::size_t inter_needed = static_cast<std::size_t>(BT) * static_cast<std::size_t>(rank);
    if (output.nelem() < out_needed || B.nelem() < b_needed || intermediate.nelem() < inter_needed) {
        return reject("nelem");
    }

    if (debug) {
        std::fprintf(stderr,
                     "[LORA-B-SMALL] launch rank=%d BT=%d total_out=%d out=%d off=%d scaling=%g\n",
                     rank,
                     BT,
                     total_out_features,
                     out_features,
                     output_offset,
                     scaling);
    }

    switch (rank) {
        case 8:
            launch_lora_accum_b_small_rank_bf16<8>(output,
                                                   B,
                                                   intermediate,
                                                   BT,
                                                   total_out_features,
                                                   out_features,
                                                   output_offset,
                                                   scaling,
                                                   stream);
            return true;
        case 16:
            launch_lora_accum_b_small_rank_bf16<16>(output,
                                                    B,
                                                    intermediate,
                                                    BT,
                                                    total_out_features,
                                                    out_features,
                                                    output_offset,
                                                    scaling,
                                                    stream);
            return true;
        case 32:
            launch_lora_accum_b_small_rank_bf16<32>(output,
                                                    B,
                                                    intermediate,
                                                    BT,
                                                    total_out_features,
                                                    out_features,
                                                    output_offset,
                                                    scaling,
                                                    stream);
            return true;
        case 64:
            launch_lora_accum_b_small_rank_bf16<64>(output,
                                                    B,
                                                    intermediate,
                                                    BT,
                                                    total_out_features,
                                                    out_features,
                                                    output_offset,
                                                    scaling,
                                                    stream);
            return true;
        default: return reject("unsupported_rank");
    }
}
