// sinfer::ops — encoder_attention wrapper: implements the public api, validates parameters,
// and dispatches to the launcher. Host-compiled; never includes the kernel header.
// See docs/op-development.md §2.
#include "api/ops/encoder_attention.h"

#include "ops/launcher/encoder_attention.h" // detail::encoder_attention_launch

#include <cuda_bf16.h>

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <string>

namespace sinfer::ops {
namespace {

bool flash_enabled() {
    static const bool on = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_ENCODER_FLASH_ATTENTION");
        return raw == nullptr || std::string(raw) != "0";
    }();
    return on;
}

} // namespace

std::size_t encoder_attention_workspace_bytes(std::int32_t q_heads, std::int32_t tokens) {
    if (q_heads <= 0 || tokens < 0) { throw std::invalid_argument("invalid encoder attention dimensions"); }
    const auto matrix = static_cast<std::size_t>(tokens) * std::min(tokens, kEncoderAttentionQueryTile) *
                        static_cast<std::size_t>(q_heads);
    // FP32 scores, then the BF16 probabilities the second product reads.
    return matrix * sizeof(float) + matrix * sizeof(__nv_bfloat16);
}

void encoder_attention_prewarm() { detail::encoder_attention_prewarm(); }

void encoder_attention(const Tensor& q, const Tensor& k, const Tensor& v, std::int32_t window,
                       float scale, Tensor& out, void* workspace, std::size_t workspace_bytes,
                       cudaStream_t stream, std::int32_t kv_heads, bool causal) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("encoder_attention: q/k/v/out must be BF16");
    }
    if (!q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("encoder_attention: q/k/v/out must be contiguous");
    }
    if (kv_heads <= 0 || k.ne[0] % kv_heads || !std::isfinite(scale) || scale <= 0) {
        throw std::invalid_argument("encoder_attention: invalid KV heads or scale");
    }
    const std::int64_t head_dim = k.ne[0] / kv_heads;
    const std::int64_t tokens   = q.ne[1];
    if (head_dim <= 0 || q.ne[0] <= 0 || q.ne[0] % head_dim != 0) {
        throw std::invalid_argument("encoder_attention: q rows must be a multiple of head_dim");
    }
    if (v.ne[0] != k.ne[0] || k.ne[1] != tokens || v.ne[1] != tokens || q.ne[0] % k.ne[0]) {
        throw std::invalid_argument("encoder_attention: invalid grouped-query K/V shape");
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

    detail::encoder_attention_launch(q, k, v, window, scale, out, workspace, stream, kv_heads, causal);
}

bool encoder_attention_batch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& segments, std::int32_t longest, std::int32_t window,
                             float scale, Tensor& out, cudaStream_t stream, std::int32_t kv_heads,
                             bool causal) {
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument("encoder_attention_batch: q/k/v/out must be BF16");
    }
    if (!q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("encoder_attention_batch: q/k/v/out must be contiguous");
    }
    if (kv_heads <= 0 || k.ne[0] % kv_heads || !std::isfinite(scale) || scale <= 0) {
        throw std::invalid_argument("encoder_attention_batch: invalid KV heads or scale");
    }
    const std::int64_t head_dim = k.ne[0] / kv_heads;
    const std::int64_t columns  = q.ne[1];
    if (head_dim <= 0 || q.ne[0] % k.ne[0] != 0 || v.ne[0] != k.ne[0] || k.ne[1] != columns ||
        v.ne[1] != columns || out.ne[0] != q.ne[0] || out.ne[1] != columns) {
        throw std::invalid_argument("encoder_attention_batch: invalid grouped-query shapes");
    }
    for (const Tensor* t : {&q, &k, &v, static_cast<const Tensor*>(&out)}) {
        if (t->ne[2] != 1 || t->ne[3] != 1) {
            throw std::invalid_argument("encoder_attention_batch: q/k/v/out must be matrices");
        }
    }
    if (segments.dtype != DType::I32 || !segments.is_contiguous() || segments.ne[0] <= 0 ||
        segments.ne[1] != 2 || segments.ne[2] != 1 || segments.ne[3] != 1 ||
        segments.data == nullptr) {
        throw std::invalid_argument("encoder_attention_batch: segments must be contiguous I32 [batch, 2]");
    }
    if (longest <= 0 || longest > columns || window < 0) {
        throw std::invalid_argument("encoder_attention_batch: invalid longest length or window");
    }
    if (q.data == nullptr || k.data == nullptr || v.data == nullptr || out.data == nullptr) {
        throw std::invalid_argument("encoder_attention_batch: q/k/v/out data must be non-null");
    }
    if (!flash_enabled() || segments.ne[0] > 65535) { return false; }
    return detail::encoder_attention_batch_launch(q, k, v, segments, longest, window, scale, out,
                                                  stream, kv_heads, causal);
}

} // namespace sinfer::ops
