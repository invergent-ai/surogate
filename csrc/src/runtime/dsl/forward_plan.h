// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Per-layer forward execution plan metadata (used to replay recompute paths).

#ifndef SUROGATE_SRC_DSL_FORWARD_PLAN_H
#define SUROGATE_SRC_DSL_FORWARD_PLAN_H

namespace dsl {

struct MatmulForwardPlan {
    bool valid = false;
    bool use_recipe = false;
    bool allow_fp8 = false;
    bool allow_fp4 = false;
    bool has_bias = false;
    bool use_fp8_cache = false;
    bool use_fp4_cache = false;
    int delayed_quantizer_idx = -1;
};

struct AttnForwardPlan {
    bool valid = false;
    bool use_qk_norm = false;
    bool rope_fused = false;
    bool use_cudnn = true;
    int rotary_dim = 0;
};

struct LayerForwardPlan {
    MatmulForwardPlan qkv{};
    MatmulForwardPlan out_proj{};
    MatmulForwardPlan mlp_up{};
    MatmulForwardPlan mlp_down{};
    AttnForwardPlan attn{};
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_FORWARD_PLAN_H
