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

void encoder_attention(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t window,
                       float scale, Tensor& out, void* workspace, std::size_t workspace_bytes,
                       cudaStream_t stream) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("encoder_attention: q/k/v/out must be BF16");
    }
    if (!q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("encoder_attention: q/k/v/out must be contiguous");
    }
    const std::int64_t head_dim = k.ne[0];
    const std::int64_t tokens   = q.ne[1];
    if (head_dim <= 0 || q.ne[0] <= 0 || q.ne[0] % head_dim != 0) {
        throw std::invalid_argument("encoder_attention: q rows must be a multiple of head_dim");
    }
    if (v.ne[0] != head_dim || k.ne[1] != tokens || v.ne[1] != tokens) {
        throw std::invalid_argument("encoder_attention: k/v must be [head_dim, tokens]");
    }
    if (out.ne[0] != q.ne[0] || out.ne[1] != tokens) {
        throw std::invalid_argument("encoder_attention: out must match q");
    }
    for (const Tensor* t : {&q, &k, &v}) {
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::invalid_argument("encoder_attention: inputs must be one sequence");
        }
    }
    if (window < 0) {
        throw std::invalid_argument("encoder_attention: window must not be negative");
    }
    if (tokens == 0) { return; }
    const auto q_heads = static_cast<std::int32_t>(q.ne[0] / head_dim);
    if (workspace == nullptr ||
        workspace_bytes <
            encoder_attention_workspace_bytes(q_heads, static_cast<std::int32_t>(tokens))) {
        throw std::invalid_argument("encoder_attention: workspace too small");
    }
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("encoder_attention: q/k/v/out data must be non-null");
    }

    detail::encoder_attention_launch(q, k, v, window, scale, out, workspace, stream);
}

} // namespace sinfer::ops
