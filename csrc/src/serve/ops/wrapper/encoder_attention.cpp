// sinfer::ops — encoder_attention wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/encoder_attention.h"

#include "ops/launcher/encoder_attention.h" // detail::encoder_attention_launch

#include <cuda_bf16.h>

#include <stdexcept>

namespace sinfer::ops {

std::size_t encoder_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens) {
    const auto matrix = static_cast<std::size_t>(tokens) * static_cast<std::size_t>(tokens) *
                        static_cast<std::size_t>(q_heads);
    // FP32 scores, then the BF16 probabilities the second product reads.
    return matrix * sizeof(float) + matrix * sizeof(__nv_bfloat16);
}

void encoder_attention_prewarm() { detail::encoder_attention_prewarm(); }

void encoder_attention(const Tensor& qkv, std::int32_t q_heads, std::int32_t head_dim,
                       std::int32_t window, float scale, Tensor& out, void* workspace,
                       std::size_t workspace_bytes, cudaStream_t stream) {
    if (qkv.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("encoder_attention: qkv/out must be BF16");
    }
    if (!qkv.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("encoder_attention: qkv/out must be contiguous");
    }
    if (q_heads <= 0 || head_dim <= 0) {
        throw std::invalid_argument("encoder_attention: q_heads and head_dim must be positive");
    }
    // One key head and one value head follow the query heads in the fused projection.
    if (qkv.ne[0] != static_cast<std::int64_t>(q_heads + 2) * head_dim) {
        throw std::invalid_argument("encoder_attention: qkv rows must be (q_heads + 2) * head_dim");
    }
    if (out.ne[0] != static_cast<std::int64_t>(q_heads) * head_dim || out.ne[1] != qkv.ne[1]) {
        throw std::invalid_argument("encoder_attention: out must be [q_heads * head_dim, tokens]");
    }
    if (qkv.ne[2] != 1 || qkv.ne[3] != 1) {
        throw std::invalid_argument("encoder_attention: qkv must be one sequence");
    }
    if (window < 0) { throw std::invalid_argument("encoder_attention: window must not be negative"); }
    const auto tokens = static_cast<std::int32_t>(qkv.ne[1]);
    if (tokens == 0) { return; }
    if (workspace == nullptr ||
        workspace_bytes < encoder_attention_workspace_bytes(q_heads, tokens)) {
        throw std::invalid_argument("encoder_attention: workspace too small");
    }
    if (qkv.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("encoder_attention: qkv/out data must be non-null");
    }

    detail::encoder_attention_launch(qkv, q_heads, head_dim, window, scale, out, workspace,
                                     stream);
}

} // namespace sinfer::ops
