// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#include "kernels.h"

#include <cstdlib>
#include <cstdio>
#include <stdexcept>

#include "utilities/dtype.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

template <int R, bool A_TRANSPOSED>
__global__ void lora_project_small_rank_bf16_kernel(
    nv_bfloat16* out,
    const nv_bfloat16* A,      // [R, C] row-major or [C, R] row-major when A_TRANSPOSED=true
    const nv_bfloat16* input,  // [BT, C] row-major
    int BT,
    int C) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= BT) return;

    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r)
        acc[r] = 0.0f;

    const int x_base = n * C;
    for (int c = 0; c < C; ++c) {
        const float x = __bfloat162float(input[x_base + c]);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const int a_idx = A_TRANSPOSED ? (c * R + r) : (r * C + c);
            acc[r] += x * __bfloat162float(A[a_idx]);
        }
    }

    const int out_base = n * R;
#pragma unroll
    for (int r = 0; r < R; ++r) {
        out[out_base + r] = __float2bfloat16(acc[r]);
    }
}

template <int R>
static void launch_lora_project_small_rank_bf16(Tensor& out,
                                                const Tensor& A,
                                                const Tensor& input,
                                                int BT,
                                                int C,
                                                bool a_transposed,
                                                cudaStream_t stream) {
    constexpr int threads = 128;
    const int blocks = (BT + threads - 1) / threads;
    if (a_transposed) {
        lora_project_small_rank_bf16_kernel<R, true><<<blocks, threads, 0, stream>>>(out.get<nv_bfloat16>(),
                                                                                     A.get<nv_bfloat16>(),
                                                                                     input.get<nv_bfloat16>(),
                                                                                     BT,
                                                                                     C);
    } else {
        lora_project_small_rank_bf16_kernel<R, false><<<blocks, threads, 0, stream>>>(out.get<nv_bfloat16>(),
                                                                                      A.get<nv_bfloat16>(),
                                                                                      input.get<nv_bfloat16>(),
                                                                                      BT,
                                                                                      C);
    }
    CUDA_CHECK(cudaGetLastError());
}

bool lora_project_small_rank_bf16(Tensor& out,
                                  const Tensor& A,
                                  const Tensor& input,
                                  int BT,
                                  int in_features,
                                  int rank,
                                  cudaStream_t stream) {
    const bool debug = std::getenv("SUROGATE_DEBUG_LORA_GEMM") != nullptr;
    auto reject = [&](const char* reason) -> bool {
        if (debug) {
            std::fprintf(stderr,
                         "[LORA-SMALL] reject=%s rank=%d BT=%d in=%d out_ptr=%p A_ptr=%p inp_ptr=%p "
                         "out_shape=[%ld,%ld] out_rank=%d A_shape=[%ld,%ld] A_rank=%d inp_shape=[%ld,%ld,%ld] "
                         "inp_rank=%d dtypes(o,a,i)=(%d,%d,%d)\n",
                         reason,
                         rank,
                         BT,
                         in_features,
                         (void*)out.Data,
                         (void*)A.Data,
                         (void*)input.Data,
                         out.Sizes[0],
                         out.Sizes[1],
                         out.Rank,
                         A.Sizes[0],
                         A.Sizes[1],
                         A.Rank,
                         input.Sizes[0],
                         input.Sizes[1],
                         input.Sizes[2],
                         input.Rank,
                         (int)out.DType,
                         (int)A.DType,
                         (int)input.DType);
        }
        return false;
    };

    if (out.DType != ETensorDType::BF16 || A.DType != ETensorDType::BF16 || input.DType != ETensorDType::BF16) {
        return reject("dtype");
    }
    if (!out.Data || !A.Data || !input.Data) {
        return reject("null_data");
    }
    if (BT <= 0 || in_features <= 0 || rank <= 0) {
        return reject("invalid_dims");
    }

    // Accept flattened tensors as long as the trailing dimension matches.
    // This is required for model activations with shape [B, T, C] where BT = B*T.
    auto has_trailing_dim_and_rows = [](const Tensor& t, int min_rows, int cols) -> bool {
        if (t.Rank < 1) return false;
        if (t.Sizes[t.Rank - 1] != cols) return false;
        std::size_t rows = 1;
        for (int i = 0; i < t.Rank - 1; ++i) {
            rows *= static_cast<std::size_t>(t.Sizes[i]);
        }
        return rows >= static_cast<std::size_t>(min_rows);
    };

    if (!has_trailing_dim_and_rows(out, BT, rank)) {
        return reject("out_shape");
    }
    if (!has_trailing_dim_and_rows(input, BT, in_features)) {
        return reject("input_shape");
    }
    const bool a_rank_major = has_trailing_dim_and_rows(A, rank, in_features);
    const bool a_transposed = has_trailing_dim_and_rows(A, in_features, rank);
    if (!a_rank_major && !a_transposed) {
        return reject("A_shape");
    }

    const std::size_t out_needed = static_cast<std::size_t>(BT) * static_cast<std::size_t>(rank);
    const std::size_t in_needed = static_cast<std::size_t>(BT) * static_cast<std::size_t>(in_features);
    const std::size_t a_needed = static_cast<std::size_t>(rank) * static_cast<std::size_t>(in_features);
    if (out.nelem() < out_needed || input.nelem() < in_needed || A.nelem() < a_needed) {
        return reject("nelem");
    }

    if (debug) {
        std::fprintf(stderr,
                     "[LORA-SMALL] launch rank=%d BT=%d in=%d a_transposed=%d\n",
                     rank,
                     BT,
                     in_features,
                     (int)(a_transposed && !a_rank_major));
    }

    switch (rank) {
        case 8:
            launch_lora_project_small_rank_bf16<8>(out,
                                                   A,
                                                   input,
                                                   BT,
                                                   in_features,
                                                   a_transposed && !a_rank_major,
                                                   stream);
            return true;
        case 16:
            launch_lora_project_small_rank_bf16<16>(out,
                                                    A,
                                                    input,
                                                    BT,
                                                    in_features,
                                                    a_transposed && !a_rank_major,
                                                    stream);
            return true;
        case 32:
            launch_lora_project_small_rank_bf16<32>(out,
                                                    A,
                                                    input,
                                                    BT,
                                                    in_features,
                                                    a_transposed && !a_rank_major,
                                                    stream);
            return true;
        case 64:
            launch_lora_project_small_rank_bf16<64>(out,
                                                    A,
                                                    input,
                                                    BT,
                                                    in_features,
                                                    a_transposed && !a_rank_major,
                                                    stream);
            return true;
        default: return reject("unsupported_rank");
    }
}

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
        default: return reject("unsupported_rank");
    }
}
