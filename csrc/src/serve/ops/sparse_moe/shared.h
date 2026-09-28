#pragma once

#include "api/ops/sparse_moe.h"

namespace sinfer::ops::detail {

inline bool native_shared(const SparseMoeWeights& weights) {
    return weights.shared_down.qdata != nullptr &&
           (weights.shared_gate_up.qtype != QType::W8G32_F16S ||
            weights.shared_down.qtype != QType::W8G32_F16S);
}

// Native shared experts use the general GGML projections. The routing kernels still
// own their sigmoid gate and combine, using this ungated BF16 result in place of
// their fused W8 projection. This is transient state, never an artifact weight.
struct PreparedSparseMoeWeights : SparseMoeWeights {
    Tensor shared_output{};

    PreparedSparseMoeWeights(const SparseMoeWeights& weights) : SparseMoeWeights(weights) {}

    PreparedSparseMoeWeights slice(std::int32_t offset, std::int32_t tokens) const {
        auto out = *this;
        if (shared_output.data) { out.shared_output = shared_output.slice(1, offset, tokens); }
        return out;
    }
};

} // namespace sinfer::ops::detail
