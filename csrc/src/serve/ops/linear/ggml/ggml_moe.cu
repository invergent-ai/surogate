// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_moe.h"

#include "ops/linear/ggml/ggml_mmvq.cuh"

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// One CTA per (row block, slot); one warp per token within it. Ported from llama.cpp's
// mul_mat_vec_q_id: the only difference from the plain GEMV is where the weight starts, which
// the id lookup decides per (token, slot) rather than the block index.
template <GgmlType type, int RowsPerBlock>
__global__ void moe_gemv_kernel(const void* __restrict__ vx, const block_q8_1* __restrict__ vy,
                                const std::int32_t* __restrict__ ids, float* __restrict__ dst,
                                const int k, const int rows, const int tokens, const int slots,
                                const int ids_stride) {
    constexpr int qk  = Traits<type>::qk;
    constexpr int qi  = Traits<type>::qi;
    constexpr int vdr = Traits<type>::vdr;
    constexpr vec_dot_q_cuda_t vec_dot_q_cuda = Traits<type>::vec_dot;

    const int token = static_cast<int>(threadIdx.y);
    if (token >= tokens) { return; }
    const int row0            = RowsPerBlock * static_cast<int>(blockIdx.x);
    const int slot            = static_cast<int>(blockIdx.y);
    const int blocks_per_row  = k / qk;
    constexpr int per_iter    = vdr * kWarpSize / qi;

    const int expert = ids[slot + token * ids_stride];
    if (expert < 0) { return; } // a slot a token did not use

    const std::int64_t expert_stride = static_cast<std::int64_t>(rows) * blocks_per_row;
    const std::int64_t kbx_offset    = expert * expert_stride + static_cast<std::int64_t>(row0) * blocks_per_row;
    const block_q8_1* y              = vy + static_cast<std::int64_t>(token) * (k / QK8_1);

    float tmp[RowsPerBlock] = {0.0f};
    for (int kbx = static_cast<int>(threadIdx.x) / (qi / vdr); kbx < blocks_per_row; kbx += per_iter) {
        const int kby = kbx * (qk / QK8_1);
        const int kqs = vdr * (static_cast<int>(threadIdx.x) % (qi / vdr));
#pragma unroll
        for (int i = 0; i < RowsPerBlock; ++i) {
            if (RowsPerBlock == 1 || row0 + i < rows) {
                tmp[i] += vec_dot_q_cuda(vx, &y[kby],
                                         static_cast<int>(kbx_offset + i * blocks_per_row + kbx), kqs);
            }
        }
    }
    float* out = dst + (static_cast<std::int64_t>(token) * slots + slot) * rows + row0;
#pragma unroll
    for (int i = 0; i < RowsPerBlock; ++i) {
        const float total = warp_sum(tmp[i]);
        if (static_cast<int>(threadIdx.x) == i && (RowsPerBlock == 1 || row0 + i < rows)) {
            out[i] = total;
        }
    }
}

template <GgmlType type>
void launch(const void* blocks, std::int32_t rows, std::int32_t k, const block_q8_1* y,
            const std::int32_t* ids, std::int32_t tokens, std::int32_t slots,
            std::int32_t ids_stride, float* out, cudaStream_t stream) {
    constexpr int kRows = 1;
    const dim3 block(kWarpSize, static_cast<unsigned>(tokens));
    const dim3 grid(static_cast<unsigned>((rows + kRows - 1) / kRows), static_cast<unsigned>(slots));
    moe_gemv_kernel<type, kRows><<<grid, block, 0, stream>>>(blocks, y, ids, out, k, rows, tokens,
                                                             slots, ids_stride);
}

} // namespace

void moe_gemv_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                     const block_q8_1* y, const std::int32_t* ids, std::int32_t tokens,
                     std::int32_t slots, std::int32_t ids_stride, float* out, cudaStream_t stream) {
    if (blocks == nullptr || y == nullptr || ids == nullptr || out == nullptr || rows <= 0 ||
        k <= 0 || (k % QK_K) != 0 || tokens <= 0 || slots <= 0 || tokens > 32) {
        throw std::invalid_argument(
            "ggml moe_gemv: [experts, rows, k] with k a multiple of 256 and at most 32 tokens");
    }
    switch (type) {
    case GgmlType::Q2_K: launch<GgmlType::Q2_K>(blocks, rows, k, y, ids, tokens, slots, ids_stride, out, stream); return;
    case GgmlType::Q3_K: launch<GgmlType::Q3_K>(blocks, rows, k, y, ids, tokens, slots, ids_stride, out, stream); return;
    case GgmlType::Q4_K: launch<GgmlType::Q4_K>(blocks, rows, k, y, ids, tokens, slots, ids_stride, out, stream); return;
    case GgmlType::Q5_K: launch<GgmlType::Q5_K>(blocks, rows, k, y, ids, tokens, slots, ids_stride, out, stream); return;
    case GgmlType::Q6_K: launch<GgmlType::Q6_K>(blocks, rows, k, y, ids, tokens, slots, ids_stride, out, stream); return;
    }
    throw std::invalid_argument("ggml moe_gemv: unknown GGML type");
}

} // namespace sinfer::ops::detail::ggml
