#include "ops/linear/ggml/ggml_linear.h"

#include "ops/linear/ggml/ggml_q8_1.h"

#include <algorithm>
#include <stdexcept>

namespace sinfer::ops::detail::ggml {

std::size_t linear_workspace_bytes(std::int32_t k, std::int32_t tokens) noexcept {
    return q8_1_bytes(k, tokens);
}

namespace {

template <typename DstT>
void run(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
         const __nv_bfloat16* x, std::int32_t tokens, DstT* out, void* scratch,
         std::size_t scratch_bytes, cudaStream_t stream) {
    if (tokens <= 0 || k <= 0 || (k % QK_K) != 0 || n <= 0) {
        throw std::invalid_argument("ggml linear: W[n, k] with k a multiple of 256");
    }
    if (scratch == nullptr || scratch_bytes < linear_workspace_bytes(k, tokens) ||
        (reinterpret_cast<std::uintptr_t>(scratch) & 15u) != 0) {
        throw std::invalid_argument("ggml linear: scratch too small or misaligned");
    }
    auto* y = static_cast<block_q8_1*>(scratch);
    quantize_q8_1_launch(x, k, tokens, y, stream);
    const std::int32_t blocks_per_token = k / QK8_1;
    for (std::int32_t column = 0; column < tokens; column += kMmvqMaxColumns) {
        const std::int32_t width = std::min(kMmvqMaxColumns, tokens - column);
        mmvq_launch<DstT>(type, blocks, n, k, y + static_cast<std::size_t>(column) * blocks_per_token,
                          width, out + static_cast<std::size_t>(column) * n, stream);
    }
}

} // namespace

void linear_launch(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                   const __nv_bfloat16* x, std::int32_t tokens, __nv_bfloat16* out,
                   void* scratch, std::size_t scratch_bytes, cudaStream_t stream) {
    run<__nv_bfloat16>(type, blocks, n, k, x, tokens, out, scratch, scratch_bytes, stream);
}

void linear_launch_f32(GgmlType type, const void* blocks, std::int32_t n, std::int32_t k,
                       const __nv_bfloat16* x, std::int32_t tokens, float* out, void* scratch,
                       std::size_t scratch_bytes, cudaStream_t stream) {
    run<float>(type, blocks, n, k, x, tokens, out, scratch, scratch_bytes, stream);
}

} // namespace sinfer::ops::detail::ggml
