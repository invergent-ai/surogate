// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_SWITCH_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_SWITCH_ROUTER_H

#include "base_router.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Switch Router - simplified routing with top-1 selection
 *
 * A simplified router that routes each token to exactly one expert.
 * Used in Switch Transformer architecture.
 *
 * Inherits from BaseRouterModule and uses top_k=1.
 */
class SwitchRouterModule : public BaseRouterModule<SwitchRouterModule> {
public:
    /**
     * @brief Configuration for Switch router
     *
     * Extends BaseConfig with Switch-specific defaults.
     */
    struct Config : public BaseConfig {
        Config() {
            top_k = 1;  // Switch always uses top-1
            capacity_factor = 1.0f;  // Tighter capacity for top-1
        }
    };

    // Use base types
    using Weights = BaseWeights;
    using Activations = BaseActivations;
    using Gradients = BaseGradients;
    using RouterOutput = typename BaseRouterModule<SwitchRouterModule>::RouterOutput;

    explicit SwitchRouterModule(Config config) : mConfig(config) {
        // Enforce top_k=1 for Switch routing
        mConfig.top_k = 1;
    }

    /**
     * @brief Forward pass for Switch routing (top-1 selection)
     *
     * Uses the same pipeline as standard router but with top_k=1.
     *
     * @return RouterOutput (same as standard router for MoE block compatibility)
     */
    RouterOutput forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Save input for backward
        acts.input = input;

        // 1) Compute routing logits
        compute_routing_logits(ctx, mConfig, w.gate, w.bias, input, acts.logits);

        // 2) Apply softmax
        apply_softmax(ctx, mConfig, acts.logits, acts.softmax_probs);

        // 3) Top-1 selection (Switch style)
        RouterOutput output;
        select_topk(ctx, mConfig, acts.softmax_probs,
                    output.routing_weights, output.expert_indices,
                    false);  // No post-selection normalization for top-1

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
        // Same backward as standard router
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

#endif // SUROGATE_SRC_MODULES_MOE_SWITCH_ROUTER_H
