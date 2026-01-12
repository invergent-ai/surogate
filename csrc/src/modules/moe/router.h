// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_ROUTER_H

#include "base_router.h"
#include "modules/primitives/linear.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Standard top-k router module for Mixture of Experts
 *
 * Computes routing probabilities for each token to each expert using a linear
 * projection followed by softmax. Supports top-k routing where only the top-k
 * experts are activated per token.
 *
 * The router also computes auxiliary losses for load balancing:
 * - Load balancing loss: encourages uniform expert utilization
 * - Router z-loss: regularizes logits to prevent routing collapse
 *
 * Inherits from BaseRouterModule and composes the protected helper methods.
 *
 * Input: (B, T, hidden_size) token representations
 * Output: RouterOutput containing routing weights, indices, and auxiliary loss
 */
class RouterModule : public BaseRouterModule<RouterModule> {
public:
    // Use base types
    using Config = BaseConfig;
    using Weights = BaseWeights;
    using Activations = BaseActivations;
    using Gradients = BaseGradients;
    using RouterOutput = typename BaseRouterModule<RouterModule>::RouterOutput;

    explicit RouterModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass: compute routing decisions
     *
     * @param ctx Module context
     * @param w Router weights
     * @param input Hidden states (B*T, hidden_size)
     * @param acts Activation storage
     * @return RouterOutput with routing weights, indices, and auxiliary losses
     */
    RouterOutput forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass: compute router gradients
     *
     * @param ctx Module context
     * @param w Router weights
     * @param acts Saved activations
     * @param grad_routing_weights Gradient w.r.t. routing weights (B*T, top_k)
     * @param grads Gradient storage
     * @param accumulate If true, accumulate into existing gradients
     * @return Gradient w.r.t. input (B*T, hidden_size)
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_routing_weights, Gradients& grads, bool accumulate = false);

    // Accessors
    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int num_experts() const { return mConfig.num_experts; }
    [[nodiscard]] int top_k() const { return mConfig.top_k; }

private:
    Config mConfig;
};

// ============================================================================
// Implementation
// ============================================================================

inline RouterModule::RouterOutput RouterModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    // Save input for backward
    acts.input = input;

    // 1) Compute routing logits using base class helper
    compute_routing_logits(ctx, mConfig, w.gate, w.bias, input, acts.logits);

    // Add noise during training for exploration
    if (mConfig.use_noisy_routing && ctx.use_quantization) {  // use_quantization as training flag
        // Add Gaussian noise to logits
        // add_gaussian_noise(acts.logits, mConfig.noise_std, ctx.stream);
    }

    // 2) Apply softmax to get routing probabilities
    apply_softmax(ctx, mConfig, acts.logits, acts.softmax_probs);

    // 3) Top-k selection
    // norm_topk_prob (Qwen3 style): normalize selected weights to sum to 1 after top-k selection
    RouterOutput output;
    select_topk(ctx, mConfig, acts.softmax_probs,
                output.routing_weights, output.expert_indices,
                mConfig.norm_topk_prob);

    // 4) Compute token counts per expert
    compute_token_counts(ctx, mConfig, output.expert_indices, output.token_counts);

    // 5) Compute auxiliary losses
    compute_load_balance_loss(ctx, mConfig, acts.softmax_probs,
                              output.expert_indices, &output.aux_loss);
    compute_z_loss(ctx, mConfig, acts.logits, &output.z_loss);

    acts.output = output;
    return output;
}

inline Tensor RouterModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_routing_weights, Gradients& grads, bool accumulate) {

    // 1) Gradient through softmax
    // The grad_routing_weights is sparse (only top-k elements are non-zero)
    // We need to scatter it back to full (BT, E) shape first
    // TODO: Add proper scatter logic for sparse top-k gradient
    Tensor d_logits;
    d_logits.DType = acts.logits.DType;
    apply_softmax_backward(ctx, mConfig, grad_routing_weights, acts.softmax_probs, d_logits);

    // 2) Add z-loss gradient contribution to logits
    add_z_loss_gradient(ctx, mConfig, acts.logits, d_logits);

    // 3) Backward through gate projection
    return backward_gate_proj(ctx, mConfig, w.gate, acts.input, d_logits, grads.d_gate, accumulate);
}

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_ROUTER_H
