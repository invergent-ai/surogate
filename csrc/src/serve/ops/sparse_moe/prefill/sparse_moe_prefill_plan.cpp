#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

/// Veto for the K-quant gate/up beside a Q8_0 down (see `prefill_min_tokens`). Read once.
bool ggml_q80_down_vetoed() noexcept {
    static const bool vetoed = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MOE_GGML_Q80");
        return env != nullptr && env[0] == '0';
    }();
    return vetoed;
}

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
    // The prefill codec's own subset, not the whole GGML family. Two shapes of routed pair
    // qualify, and both run on the int8 tensor-core route:
    //
    //  - both sides a K-quant with a hand codec, the common K_M mix; and
    //  - a K-quant gate/up beside a Q8_0 down, which is what `llama-quantize` writes when the
    //    expert width is not a multiple of 256 and it cannot K-quant `ffn_down_exps` at all.
    //    Gemma 4's experts are 704 wide, so its Q6_K artifact is exactly this pair. Q8_0 is
    //    int8 codes under one scale per 32 -- the same affine form the route's MMA wants --
    //    so the down side needs no superblock, only a whole 64-wide tile, which 704 is.
    //
    // Q4_1 and Q5_1 (the Q3_K_M mix's down tensor) are the same story with a min term and
    // packed nibbles, and have no codec on either route yet; they still fall to small-T.
    const auto is_ggml_k = [](QType qtype) {
        return qtype == QType::Q4_K || qtype == QType::Q5_K || qtype == QType::Q6_K;
    };
    if (is_ggml_k(routed_gate_up) && is_ggml_k(routed_down)) { return kSparseMoePrefillGgmlKMin; }
    // The K-quant/Q8_0 pair is newer than the all-K one and moves a model that previously had no
    // prefill at all onto int8 activations, so it carries its own veto: SUROGATE_SERVE_MOE_GGML_Q80=0
    // returns that pair to the small-T slices without disturbing any other profile. It is the
    // rollback lever and the way to measure this path against the old one on one binary.
    if (is_ggml_k(routed_gate_up) && routed_down == QType::Q8_0 && !ggml_q80_down_vetoed()) {
        return kSparseMoePrefillGgmlKMin;
    }
    return 0;
}

} // namespace

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

bool sparse_moe_routed_int8_profile(QType routed_gate_up, QType routed_down) noexcept {
    const auto is_k = [](QType type) {
        return type == QType::Q4_K || type == QType::Q5_K || type == QType::Q6_K;
    };
    static const bool vetoed = [] {
        const char* env = std::getenv("SUROGATE_SERVE_MOE_INT8");
        return env != nullptr && env[0] == '0';
    }();
    // The same pairs `prefill_min_tokens` admits, and the Q8_0 down side answers to the same
    // veto there, so one switch returns that pair to small-T instead of leaving it half-routed.
    // Both sides have an int8 codec, so the whole round runs on the tensor-core route rather
    // than one side of it.
    return !vetoed && is_k(routed_gate_up) &&
           (is_k(routed_down) || (routed_down == QType::Q8_0 && !ggml_q80_down_vetoed()));
}

std::size_t sparse_moe_prefill_workspace_bytes(const SparseMoeGeometry& geometry,
                                               std::int32_t max_tokens, bool routed_trtllm,
                                               bool routed_int8) {
    if (max_tokens < 1) {
        throw std::invalid_argument("sparse_moe prefill: max_tokens must be at least 1");
    }
    const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_prefill_workspace(layout, geometry, capacity_tokens, routed_trtllm,
                                                routed_int8);
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
    const bool routed_int8 = sparse_moe_routed_int8_profile(routed_gate_up, routed_down);
    return {tokens, slice_tokens,
            sparse_moe_prefill_workspace_bytes(geometry, tokens, routed_trtllm, routed_int8),
            routed_trtllm, routed_int8};
}

} // namespace sinfer::ops::detail
