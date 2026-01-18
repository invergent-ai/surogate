// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Declarative block specification for composable transformer blocks.

#ifndef SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_SPEC_H
#define SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_SPEC_H

#include <array>

namespace modules {

enum class BlockVariant {
    Dense,
    Parallel,
    MoE,
};

enum class BlockOp {
    LN1,
    QKV,
    QKNorm,
    RoPE,
    Attention,
    AttnOut,
    ResidualAdd,
    LN2,
    ResidualLN2,
    MLPUp,
    SwiGLU,
    MLPDown,
    Router,
    Experts,
    Combine,
};

struct BlockSpec {
    static constexpr int kMaxForwardOps = 16;

    BlockVariant variant = BlockVariant::Dense;
    bool use_qk_norm = false;
    bool ln2_on_residual_att = true;  // Dense pre-norm uses residual+att; parallel uses residual

    std::array<BlockOp, kMaxForwardOps> forward_ops{};
    int forward_ops_count = 0;

    void push_op(BlockOp op) {
        forward_ops[forward_ops_count++] = op;
    }
};

inline bool is_dense_like(const BlockSpec& spec) {
    return spec.variant == BlockVariant::Dense || spec.variant == BlockVariant::Parallel;
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_COMPOSITE_BLOCK_SPEC_H
