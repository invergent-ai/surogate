#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace sinfer::ops::detail {
namespace {

/// One veto per admission, each read once. `SUROGATE_SERVE_MOE_GGML_<X>=0` returns the pairs
/// that admission covers to the route they took before it, and nothing else moves, so every
/// admission can be measured against its predecessor on one binary and switched off alone in
/// production.
bool env_vetoed(const char* name) noexcept {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] == '0';
}
/// The K-quant gate/up beside a Q8_0 down (int8 route, or small-T when vetoed).
bool ggml_q80_down_vetoed() noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_GGML_Q80");
    return vetoed;
}
/// Every other GGML block pair on the wide BF16 route (small-T when vetoed).
bool ggml_wide_vetoed() noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_GGML_WIDE");
    return vetoed;
}
/// Q8_0 as the gate/up side on the int8 route (wide BF16 route when vetoed).
bool ggml_q80_gate_vetoed() noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_GGML_Q80_GATE");
    return vetoed;
}
/// Q5_0 and Q5_1 down on the int8 route (wide BF16 route when vetoed).
bool ggml_q5x_down_vetoed() noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_GGML_Q5X");
    return vetoed;
}
/// Q3_K gate/up on the int8 route (wide BF16 route when vetoed).
bool ggml_q3k_gate_vetoed() noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_GGML_Q3K");
    return vetoed;
}

/// The three K-quants with a hand codec on both prefill routes: the pairs served first.
bool is_ggml_k_hand(QType qtype) noexcept {
    return qtype == QType::Q4_K || qtype == QType::Q5_K || qtype == QType::Q6_K;
}

/// What a row's K must be a whole number of for this format's prefill codec, or zero for a
/// format the prefill family does not admit. Mirrors `GgmlBlockPrefill<T>::kKMultiple` and the
/// hand codecs: a superblock format stages per 256 values and a plain 32-value block format
/// per 64-wide tile. The plan sees only the pair of types, not the geometry, and that is
/// enough: a K-quant tensor exists only where K is a multiple of 256 (ggml refuses to write
/// one otherwise), and every registered geometry's reduction axes are multiples of 64.
///
/// Only the seven formats the op test has fixtures for (`test_sparse_moe_ggml_prefill.cpp`) --
/// the ones the published Gemma 4 artifacts hold. Q2_K, Q4_0, Q4_1, IQ4_NL and F16 decode on
/// the same generic codec and would very likely be right, but no artifact here holds experts
/// in them and nothing measures them, so they keep the small-T slices; to admit one, add its
/// fixture and mixture to that test and its case here.
std::int32_t ggml_prefill_k_multiple(QType qtype) noexcept {
    switch (qtype) {
    case QType::Q3_K:
    case QType::Q4_K:
    case QType::Q5_K:
    case QType::Q6_K:
        return 256;
    case QType::Q8_0:
    case QType::Q5_0:
    case QType::Q5_1:
        return 64;
    default:
        return 0;
    }
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
    // The GGML block formats. The wide prefill kernel is instantiated for every stored format
    // (`PrefillCodecFor<T>` names a hand codec or the generic one), so the only question here
    // is whether the pair's reduction axes can be a whole number of each codec's staging unit
    // -- and the type alone answers it, see `ggml_prefill_k_multiple`. Which route the pair
    // then takes (int8 tensor core or BF16 activations) is `sparse_moe_routed_int8_profile`'s.
    //
    // `llama-quantize` writes these pairs for a mixture whose expert width is not a multiple
    // of 256: it cannot K-quant `ffn_down_exps` at all and leaves it in Q8_0, Q5_0 or Q5_1
    // (Gemma 4's experts are 704 wide, so every one of its artifacts is such a pair), and a
    // Q3_K_M mix puts Q3_K on the gate/up side. Before this every pair but the all-K one and
    // the K-quant/Q8_0 one returned 0 here, which is "no wide prefill at all": those models
    // ran every prompt on the decode kernels in 46-token slices.
    if (ggml_prefill_k_multiple(routed_gate_up) == 0 || ggml_prefill_k_multiple(routed_down) == 0) {
        return 0;
    }
    // The two admissions that predate the general one keep their own vetoes, so each still
    // returns exactly the pairs it covered to small-T and no others.
    if (is_ggml_k_hand(routed_gate_up) && is_ggml_k_hand(routed_down)) {
        return kSparseMoePrefillGgmlKMin;
    }
    if (is_ggml_k_hand(routed_gate_up) && routed_down == QType::Q8_0) {
        return ggml_q80_down_vetoed() ? 0 : kSparseMoePrefillGgmlKMin;
    }
    // Every other block pair. SUROGATE_SERVE_MOE_GGML_WIDE=0 returns all of them to small-T.
    return ggml_wide_vetoed() ? 0 : kSparseMoePrefillGgmlKMin;
}

/// Whether this format has an int8 tensor-core codec for the side named, and its veto is not
/// set. The hand K-quant codecs and the Q8_0 down predate the vetoes here and answer to the
/// pair-level ones in `prefill_min_tokens`; each later codec carries its own.
bool int8_codec_admitted(QType qtype, bool gate_up_side) noexcept {
    switch (qtype) {
    case QType::Q4_K:
    case QType::Q5_K:
    case QType::Q6_K:
        return true;
    case QType::Q8_0:
        // `GgmlQ80Prefill` was written for the down side; on the gate/up side it is the same
        // codec over a 2,816-wide row, admitted separately so it can be measured alone.
        return gate_up_side ? !ggml_q80_gate_vetoed() : true;
    case QType::Q5_0:
    case QType::Q5_1:
        return !gate_up_side && !ggml_q5x_down_vetoed();
    case QType::Q3_K:
        return gate_up_side && !ggml_q3k_gate_vetoed();
    default:
        return false;
    }
}

} // namespace

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

bool sparse_moe_routed_int8_profile(QType routed_gate_up, QType routed_down) noexcept {
    static const bool vetoed = env_vetoed("SUROGATE_SERVE_MOE_INT8");
    // A pair the prefill family serves whose two sides both have an int8 codec. The round runs
    // on one route or the other, never one side of each: the plan's `routed_int8` covers the
    // pair. A pair `prefill_min_tokens` refuses is not int8 either, so a pair-level veto
    // returns it to small-T rather than leaving it half-routed.
    return !vetoed && prefill_min_tokens(routed_gate_up, routed_down) != 0 &&
           int8_codec_admitted(routed_gate_up, /*gate_up_side=*/true) &&
           int8_codec_admitted(routed_down, /*gate_up_side=*/false);
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
