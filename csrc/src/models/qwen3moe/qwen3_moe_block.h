// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_BLOCK_H
#define SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_BLOCK_H

#include "qwen3_router.h"
#include "modules/moe/moe_block.h"

namespace modules {

/**
 * @brief Type alias for Qwen3 MoE block
 *
 * Uses Qwen3-specific features:
 * - QK normalization in attention (use_qk_norm=true)
 * - Post-selection weight normalization in router (norm_topk_prob=true)
 * - Optional shared expert
 *
 * Note: AttentionModule already supports use_qk_norm via config.
 * The Qwen3RouterModule enforces norm_topk_prob=true.
 */
using Qwen3MoEBlock = MoETransformerBlock<AttentionModule, Qwen3RouterModule, FusedResidualRMSNormModule>;

/**
 * @brief Qwen3 MoE block configuration builder
 *
 * Extends the standard builder with Qwen3-specific defaults:
 * - QK normalization enabled
 * - norm_topk_prob enabled (handled by Qwen3RouterModule)
 */
class Qwen3MoEBlockConfigBuilder {
public:
    Qwen3MoEBlockConfigBuilder() {
        // Qwen3 defaults
        mConfig.use_qk_norm = true;  // QK normalization
    }

    Qwen3MoEBlockConfigBuilder& hidden_size(int size) { mConfig.hidden_size = size; return *this; }
    Qwen3MoEBlockConfigBuilder& num_query_heads(int heads) { mConfig.num_query_heads = heads; return *this; }
    Qwen3MoEBlockConfigBuilder& num_kv_heads(int heads) { mConfig.num_kv_heads = heads; return *this; }
    Qwen3MoEBlockConfigBuilder& head_size(int size) { mConfig.head_size = size; return *this; }
    Qwen3MoEBlockConfigBuilder& rms_norm_eps(float eps) { mConfig.rms_norm_eps = eps; return *this; }
    Qwen3MoEBlockConfigBuilder& rope_config(const RoPEConfig& cfg) { mConfig.rope = cfg; return *this; }
    Qwen3MoEBlockConfigBuilder& rope_theta(float theta) { mConfig.rope.theta = theta; return *this; }
    Qwen3MoEBlockConfigBuilder& max_seq_len(int len) { mConfig.max_seq_len = len; return *this; }

    Qwen3MoEBlockConfigBuilder& num_experts(int n) { mConfig.num_experts = n; return *this; }
    Qwen3MoEBlockConfigBuilder& top_k(int k) { mConfig.top_k = k; return *this; }
    Qwen3MoEBlockConfigBuilder& intermediate_size(int size) { mConfig.intermediate_size = size; return *this; }
    Qwen3MoEBlockConfigBuilder& aux_loss_coef(float coef) { mConfig.aux_loss_coef = coef; return *this; }
    Qwen3MoEBlockConfigBuilder& capacity_factor(float factor) { mConfig.capacity_factor = factor; return *this; }

    /**
     * @brief Enable shared expert (Qwen3 MoE uses this)
     *
     * @param intermediate_size Shared expert intermediate size (0 = same as regular experts)
     */
    Qwen3MoEBlockConfigBuilder& with_shared_expert(int intermediate_size = 0) {
        mConfig.use_shared_expert = true;
        mConfig.shared_expert_intermediate = intermediate_size;
        return *this;
    }

    typename Qwen3MoEBlock::Config build() const { return mConfig; }

private:
    typename Qwen3MoEBlock::Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_MOE_BLOCK_H
