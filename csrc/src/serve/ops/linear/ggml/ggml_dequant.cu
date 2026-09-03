// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the vec-dot arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#include "ops/linear/ggml/ggml_dequant.h"

#include "ops/linear/ggml/ggml_dequant.cuh"

#include <stdexcept>

namespace sinfer::ops::detail::ggml {
namespace {

// One CTA per (row, superblock): the same per-type thread layout the gather uses, writing
// straight into the row-major BF16 tile.
template <GgmlType type>
__global__ void dequantize_rows_kernel(const void* __restrict__ blocks,
                                       __nv_bfloat16* __restrict__ out, const int k) {
    const int row   = blockIdx.x;
    const int block = blockIdx.y;
    constexpr int values  = block_values(type);
    const std::int64_t ib = static_cast<std::int64_t>(row) * (k / values) + block;
    dequantize_superblock<type>(blocks, ib,
                                out + static_cast<std::size_t>(row) * k + block * values,
                                threadIdx.x);
}

template <GgmlType type>
void launch(const void* blocks, int rows, int k, __nv_bfloat16* out, cudaStream_t stream) {
    const dim3 grid(rows, k / block_values(type));
    dequantize_rows_kernel<type><<<grid, dequant_threads<type>(), 0, stream>>>(blocks, out, k);
}

} // namespace

void dequantize_rows_launch(GgmlType type, const void* blocks, std::int32_t rows, std::int32_t k,
                            __nv_bfloat16* out, cudaStream_t stream) {
    if (blocks == nullptr || out == nullptr || rows <= 0 || k <= 0 ||
        (k % block_values(type)) != 0) {
        throw std::invalid_argument("ggml dequantize_rows: k must be a whole number of blocks");
    }
    switch (type) {
    case GgmlType::Q2_K: launch<GgmlType::Q2_K>(blocks, rows, k, out, stream); return;
    case GgmlType::Q3_K: launch<GgmlType::Q3_K>(blocks, rows, k, out, stream); return;
    case GgmlType::Q4_K: launch<GgmlType::Q4_K>(blocks, rows, k, out, stream); return;
    case GgmlType::Q5_K: launch<GgmlType::Q5_K>(blocks, rows, k, out, stream); return;
    case GgmlType::Q6_K: launch<GgmlType::Q6_K>(blocks, rows, k, out, stream); return;
    case GgmlType::Q8_0: launch<GgmlType::Q8_0>(blocks, rows, k, out, stream); return;
    }
    throw std::invalid_argument("ggml dequantize_rows: unknown GGML type");
}

} // namespace sinfer::ops::detail::ggml
