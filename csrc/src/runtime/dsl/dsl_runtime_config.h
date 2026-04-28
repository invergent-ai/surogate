// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime configuration derived from the Python DSL.

#ifndef SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
#define SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H

#include <string>
#include <vector>

#include "config/rope_config.h"

namespace dsl {

/// Per-layer dimensions for hybrid models where block types have different
/// head sizes, QKV channels, or intermediate sizes (e.g., Gemma4).
struct BlockTypeDims {
    long head_size = 0;
    long qkv_channels = 0;  ///< D * (Hq + 2*Hkv) or D * (Hq + Hkv) for k_eq_v
    long attn_dim = 0;      ///< Hq * D
    long intermediate = 0;  ///< M (may be 2x for double-wide MLP)
    long mlp_up = 0;        ///< up_factor * M
};

struct LayerRoPEConfig {
    long head_size = 0;
    RoPEConfig rope;
};

struct DslRuntimeConfig {
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    bool use_qk_norm = false;
    bool norm_topk_prob = false;
    bool use_shared_expert = false;
    int shared_expert_intermediate = 0;
    int linear_conv_kernel_dim = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int d_per_layer_input = 0;
    int mamba_num_heads = 0;
    int mamba_head_dim = 0;
    int ssm_state_size = 0;
    int n_groups = 0;

    /// Per-layer dimensions. Empty for homogeneous models (use global config).
    std::vector<BlockTypeDims> per_layer_dims;

    /// Per-layer RoPE parameters for hybrid models where layer types use
    /// different head sizes or rope formulas (e.g. Gemma4 full vs sliding).
    std::vector<LayerRoPEConfig> per_layer_rope;

    [[nodiscard]] bool is_moe() const {
        return num_experts > 0;
    }
    [[nodiscard]] bool has_per_layer_dims() const {
        return !per_layer_dims.empty();
    }
    [[nodiscard]] bool has_per_layer_rope() const {
        return !per_layer_rope.empty();
    }
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
