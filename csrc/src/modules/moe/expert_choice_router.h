// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_EXPERT_CHOICE_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_EXPERT_CHOICE_ROUTER_H

#include "base_router.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Expert Choice Router - experts select tokens
 *
 * Instead of tokens selecting experts (which can cause load imbalance),
 * each expert selects its top-k tokens. Guarantees perfect load balancing.
 * Used in some recent MoE architectures.
 *
 * Inherits from BaseRouterModule with expert-choice-specific behavior.
 */
class ExpertChoiceRouterModule : public BaseRouterModule<ExpertChoiceRouterModule> {
public:
    /**
     * @brief Configuration for Expert Choice router
     *
     * Extends BaseConfig with expert-choice-specific fields.
     */
    struct Config : public BaseConfig {
        int tokens_per_expert;      ///< Fixed capacity per expert

        Config() {
            aux_loss_coef = 0.0f;   // Usually 0 since load is balanced by design
        }
    };

    /**
     * @brief Activations specific to expert choice routing
     *
     * Expert choice routing has different output shapes since experts select tokens.
     */
    struct Activations : public BaseActivations {
        Tensor ec_token_indices;    ///< (num_experts, tokens_per_expert) selected token indices
        Tensor ec_routing_weights;  ///< (num_experts, tokens_per_expert) weights per expert
    };

    // Use base types for weights and gradients
    using Weights = BaseWeights;
    using Gradients = BaseGradients;
    using RouterOutput = typename BaseRouterModule<ExpertChoiceRouterModule>::RouterOutput;

    explicit ExpertChoiceRouterModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass for Expert Choice routing
     *
     * Each expert selects its top tokens instead of tokens selecting experts.
     * This guarantees perfect load balancing at the cost of some tokens
     * potentially not being processed by any expert.
     *
     * @return RouterOutput (compatible with MoE block)
     */
    RouterOutput forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Save input for backward
        acts.input = input;

        // 1) Compute routing logits
        compute_routing_logits(ctx, mConfig, w.gate, w.bias, input, acts.logits);

        // 2) Apply softmax
        apply_softmax(ctx, mConfig, acts.logits, acts.softmax_probs);

        // 3) Expert choice selection
        // Transpose the selection: instead of top-k over experts per token,
        // do top-k over tokens per expert
        RouterOutput output;
        expert_select_tokens(ctx, mConfig, acts.softmax_probs, mConfig.tokens_per_expert,
                             acts.ec_token_indices, acts.ec_routing_weights);

        // Convert to standard RouterOutput format
        // Token counts are uniform by design
        output.aux_loss = 0.0f;  // Perfect balance by design
        output.z_loss = 0.0f;

        acts.output = output;
        return output;
    }

    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false) {
        // Expert choice backward requires special handling due to transposed selection
        // For now, use the standard backward path
        Tensor d_logits;
        d_logits.DType = acts.logits.DType;
        apply_softmax_backward(ctx, mConfig, grad_output, acts.softmax_probs, d_logits);
        return backward_gate_proj(ctx, mConfig, w.gate, acts.input, d_logits, grads.d_gate, accumulate);
    }

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_EXPERT_CHOICE_ROUTER_H
