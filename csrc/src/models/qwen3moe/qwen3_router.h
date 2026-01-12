// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_ROUTER_H
#define SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_ROUTER_H

#include "modules/moe/base_router.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Qwen3 Router - top-k routing with post-selection normalization
 *
 * Qwen3 MoE uses a variant of top-k routing where the routing weights
 * are normalized to sum to 1 AFTER the top-k selection, rather than
 * using the raw softmax probabilities.
 *
 * This is controlled by the norm_topk_prob flag in the base config.
 */
class Qwen3RouterModule : public BaseRouterModule<Qwen3RouterModule> {
public:
    /**
     * @brief Configuration for Qwen3 router
     *
     * Extends BaseConfig with Qwen3-specific defaults.
     */
    struct Config : public BaseConfig {
        Config() {
            norm_topk_prob = true;  // Qwen3 always normalizes after top-k selection
        }
    };

    // Use base types
    using Weights = BaseWeights;
    using Activations = BaseActivations;
    using Gradients = BaseGradients;
    using RouterOutput = typename BaseRouterModule<Qwen3RouterModule>::RouterOutput;

    explicit Qwen3RouterModule(Config config) : mConfig(config) {
        // Enforce norm_topk_prob for Qwen3
        mConfig.norm_topk_prob = true;
    }

    /**
     * @brief Forward pass for Qwen3 routing
     *
     * Uses top-k selection with post-selection normalization:
     * 1. Compute logits and softmax
     * 2. Select top-k experts
     * 3. Renormalize selected weights to sum to 1 (Qwen3 specific)
     *
     * @return RouterOutput with normalized routing weights
     */
    RouterOutput forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Save input for backward
        acts.input = input;

        // 1) Compute routing logits
        compute_routing_logits(ctx, mConfig, w.gate, w.bias, input, acts.logits);

        // 2) Apply softmax
        apply_softmax(ctx, mConfig, acts.logits, acts.softmax_probs);

        // 3) Top-k selection with post-selection normalization (Qwen3 style)
        RouterOutput output;
        select_topk(ctx, mConfig, acts.softmax_probs,
                    output.routing_weights, output.expert_indices,
                    true);  // norm_topk_prob = true for Qwen3

        // 4) Compute token counts
        compute_token_counts(ctx, mConfig, output.expert_indices, output.token_counts);

        // 5) Compute auxiliary losses
        compute_load_balance_loss(ctx, mConfig, acts.softmax_probs,
                                  output.expert_indices, &output.aux_loss);
        compute_z_loss(ctx, mConfig, acts.logits, &output.z_loss);

        acts.output = output;
        return output;
    }

    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_routing_weights, Gradients& grads, bool accumulate = false) {
        // Backward is the same as standard router
        // Note: The normalization gradient is handled within the top-k backward kernel
        Tensor d_logits;
        d_logits.DType = acts.logits.DType;
        apply_softmax_backward(ctx, mConfig, grad_routing_weights, acts.softmax_probs, d_logits);
        add_z_loss_gradient(ctx, mConfig, acts.logits, d_logits);
        return backward_gate_proj(ctx, mConfig, w.gate, acts.input, d_logits, grads.d_gate, accumulate);
    }

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODELS_QWEN3MOE_QWEN3_ROUTER_H
