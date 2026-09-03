// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_mmvq.h"

#include "ops/linear/ggml/ggml_mmvq.cuh"

#include <stdexcept>
#include <string>

namespace sinfer::ops::detail::ggml {
namespace {

template <GgmlType type, int ncols_dst, typename DstT, bool Accumulate, bool SmallK>
void launch_scheduled(const void* blocks, int n, int k, const block_q8_1* y, DstT* out,
                      cudaStream_t stream) {
    constexpr int nwarps = calc_nwarps(ncols_dst);
    constexpr int rows   = calc_rows_per_block(ncols_dst, SmallK, nwarps);
    const dim3 block(kWarpSize, nwarps);
    const dim3 grid(static_cast<unsigned>((n + rows - 1) / rows));
    const DirectStore<DstT, Accumulate, ncols_dst> epilogue{out, n};
    mul_mat_vec_q<type, ncols_dst, DirectStore<DstT, Accumulate, ncols_dst>, SmallK>
        <<<grid, block, 0, stream>>>(blocks, y, k, n, k / Traits<type>::qk, k / QK8_1, epilogue);
}

template <GgmlType type, int ncols_dst, typename DstT, bool Accumulate>
void launch(const void* blocks, int n, int k, const block_q8_1* y, DstT* out, cudaStream_t stream) {
    if (prefers_small_k<type>(ncols_dst, k)) {
        launch_scheduled<type, ncols_dst, DstT, Accumulate, true>(blocks, n, k, y, out, stream);
    } else {
        launch_scheduled<type, ncols_dst, DstT, Accumulate, false>(blocks, n, k, y, out, stream);
    }
}


template <GgmlType type, typename DstT, bool Accumulate>
void launch_columns(const void* blocks, int n, int k, const block_q8_1* y, int tokens, DstT* out,
                    cudaStream_t stream) {
    switch (tokens) {
    case 1: launch<type, 1, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 2: launch<type, 2, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 3: launch<type, 3, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 4: launch<type, 4, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 5: launch<type, 5, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 6: launch<type, 6, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 7: launch<type, 7, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    case 8: launch<type, 8, DstT, Accumulate>(blocks, n, k, y, out, stream); return;
    default: break;
    }
    throw std::invalid_argument("mmvq: at most 8 columns per launch");
}

} // namespace

template <typename DstT, bool Accumulate>
void mmvq_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                 const block_q8_1* y, std::int32_t tokens, DstT* out, cudaStream_t stream) {
    if (blocks == nullptr || y == nullptr || out == nullptr || n <= 0 || k <= 0 ||
        (k % block_values(type)) != 0 || tokens <= 0 || tokens > kMmvqMaxColumns) {
        throw std::invalid_argument("mmvq: W[n, k] with k a whole number of blocks, 1..8 columns");
    }
    switch (type) {
    case GgmlType::Q2_K: launch_columns<GgmlType::Q2_K, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q3_K: launch_columns<GgmlType::Q3_K, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q4_K: launch_columns<GgmlType::Q4_K, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q5_K: launch_columns<GgmlType::Q5_K, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q6_K: launch_columns<GgmlType::Q6_K, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    case GgmlType::Q8_0: launch_columns<GgmlType::Q8_0, DstT, Accumulate>(blocks, n, k, y, tokens, out, stream); return;
    }
    throw std::invalid_argument("mmvq: unknown GGML type");
}

template void mmvq_launch<float, false>(GgmlType, const void*, std::int32_t, std::int32_t, const block_q8_1*, std::int32_t, float*, cudaStream_t);
template void mmvq_launch<float, true>(GgmlType, const void*, std::int32_t, std::int32_t, const block_q8_1*, std::int32_t, float*, cudaStream_t);
template void mmvq_launch<__nv_bfloat16, false>(GgmlType, const void*, std::int32_t, std::int32_t, const block_q8_1*, std::int32_t, __nv_bfloat16*, cudaStream_t);
template void mmvq_launch<__nv_bfloat16, true>(GgmlType, const void*, std::int32_t, std::int32_t, const block_q8_1*, std::int32_t, __nv_bfloat16*, cudaStream_t);

} // namespace sinfer::ops::detail::ggml
