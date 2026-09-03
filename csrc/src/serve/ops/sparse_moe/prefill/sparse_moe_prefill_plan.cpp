#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"

#include <algorithm>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

std::int32_t prefill_min_tokens(QType routed_gate_up, QType routed_down) noexcept {
    // NVFP4's routed experts have no kernel of this family - the vendored TRT-LLM runner computes
    // them - so the family serves that profile at every width, down to a single token.
    if (routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4) { return 1; }
    if (routed_gate_up == QType::Q4G64_F16S) {
        if (routed_down == QType::Q5G64_F16S) { return kSparseMoePrefillQ4Q5Min; }
        if (routed_down == QType::Q6G64_F16S) { return kSparseMoePrefillQ4Q6Min; }
    }
    if (routed_gate_up == QType::W8G32_F16S && routed_down == QType::W8G32_F16S) {
        return kSparseMoePrefillW8W8Min;
    }
    const auto is_ggml_k = [](QType qtype) {
        return qtype == QType::Q4_K || qtype == QType::Q5_K || qtype == QType::Q6_K;
    };
    if (is_ggml_k(routed_gate_up) && is_ggml_k(routed_down)) { return kSparseMoePrefillGgmlKMin; }
    return 0;
}

} // namespace

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

std::size_t sparse_moe_prefill_workspace_bytes(const SparseMoeGeometry& geometry,
                                               std::int32_t max_tokens, bool routed_trtllm) {
    if (max_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: max_tokens must be at least 1");
    }
    const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_prefill_workspace(layout, geometry, capacity_tokens, routed_trtllm);
    return layout.peak_bytes(1);
}

SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(const SparseMoeGeometry& geometry,
                                                     std::int32_t tokens, QType routed_gate_up,
                                                     QType routed_down) {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    if (minimum == 0) {
        throw std::invalid_argument("sparse_moe prefill: unsupported routed codec profile");
    }
    if (tokens < minimum) {
        throw std::invalid_argument("sparse_moe prefill: unsupported token count");
    }
    const std::int32_t slice_tokens = std::min(tokens, kSparseMoePrefillSliceMax);
    const bool routed_trtllm =
        routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4;
    return {tokens, slice_tokens,
            sparse_moe_prefill_workspace_bytes(geometry, tokens, routed_trtllm), routed_trtllm};
}

} // namespace sinfer::ops::detail
