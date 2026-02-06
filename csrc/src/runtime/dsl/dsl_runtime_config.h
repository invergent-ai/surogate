// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime configuration derived from the Python DSL.

#ifndef SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
#define SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H

namespace dsl {

struct DslRuntimeConfig {
    int num_experts = 0;
    int num_experts_per_tok = 0;
    int moe_intermediate_size = 0;
    bool use_qk_norm = false;
    bool norm_topk_prob = false;
    bool use_shared_expert = false;
    int shared_expert_intermediate = 0;

    [[nodiscard]] bool is_moe() const { return num_experts > 0; }
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_RUNTIME_CONFIG_H
