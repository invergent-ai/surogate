// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_ROUTER_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief Router module for Mixture of Experts
 *
 * Computes routing probabilities for each token to each expert using a linear
 * projection followed by softmax. Supports top-k routing where only the top-k
 * experts are activated per token.
 *
 * The router also computes auxiliary losses for load balancing:
 * - Load balancing loss: encourages uniform expert utilization
 * - Router z-loss: regularizes logits to prevent routing collapse
 *
 * Input: (B, T, hidden_size) token representations
 * Output: RouterOutput containing routing weights, indices, and auxiliary loss
 */
class RouterModule : public ModuleBase<RouterModule> {
public:
    /**
     * @brief Configuration for the router
     */
    struct Config {
        int hidden_size;            ///< Input hidden dimension
        int num_experts;            ///< Total number of experts
        int top_k = 2;              ///< Number of experts to route each token to
        float aux_loss_coef = 0.01f;  ///< Coefficient for load balancing auxiliary loss
        float z_loss_coef = 0.001f;   ///< Coefficient for router z-loss
        bool use_noisy_routing = false;  ///< Add noise during training for exploration
        float noise_std = 0.1f;     ///< Standard deviation of routing noise
        float capacity_factor = 1.25f;  ///< Expert capacity factor (tokens per expert)
        bool normalize_routing = true;  ///< Normalize routing weights to sum to 1
    };

    /**
     * @brief Weight tensors for router
     */
    struct Weights {
        Tensor gate;                ///< (hidden_size, num_experts) routing projection
    };

    /**
     * @brief Router output structure
     */
    struct RouterOutput {
        Tensor routing_weights;     ///< (B*T, top_k) normalized weights for selected experts
        Tensor expert_indices;      ///< (B*T, top_k) indices of selected experts (int32)
        Tensor expert_mask;         ///< (B*T, num_experts) binary mask of expert selection
        float aux_loss;             ///< Load balancing auxiliary loss
        float z_loss;               ///< Router z-loss for regularization

        // For expert dispatch
        Tensor token_indices;       ///< (num_experts, capacity) token indices per expert
        Tensor token_counts;        ///< (num_experts,) number of tokens per expert
        Tensor dispatch_mask;       ///< (B*T, num_experts) dispatch weights
        Tensor combine_weights;     ///< (B*T, num_experts) combine weights for output
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input;               ///< (B*T, hidden_size) input to router
        Tensor logits;              ///< (B*T, num_experts) raw routing logits
        Tensor softmax_probs;       ///< (B*T, num_experts) softmax probabilities
        RouterOutput output;        ///< Full routing output
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_gate;              ///< (hidden_size, num_experts)
    };

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

    // Internal helpers
    void compute_aux_loss(Activations& acts, int B, int T, cudaStream_t stream);
};

// ============================================================================
// Implementation
// ============================================================================

inline RouterModule::RouterOutput RouterModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Save input for backward
    acts.input = input;

    // Compute routing logits: logits = input @ gate
    // gate is (hidden_size, num_experts), input is (BT, hidden_size)
    // logits is (BT, num_experts)
    matmul(
        acts.logits, w.gate, input, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        E, BT, C, EMMTranspose::TN, false,
        ctx.stream
    );

    // Add noise during training for exploration
    if (mConfig.use_noisy_routing && ctx.use_quantization) {  // use_quantization as training flag
        // Add Gaussian noise to logits
        // add_gaussian_noise(acts.logits, mConfig.noise_std, ctx.stream);
    }

    // Softmax to get routing probabilities
    if (acts.logits.DType == ETensorDType::BF16) {
        moe_softmax_forward(
            acts.softmax_probs.get<nv_bfloat16>(),
            acts.logits.get<nv_bfloat16>(),
            BT, E, ctx.stream
        );
    } else {
        moe_softmax_forward(
            acts.softmax_probs.get<float>(),
            acts.logits.get<float>(),
            BT, E, ctx.stream
        );
    }

    // Top-k selection
    RouterOutput output;
    if (acts.softmax_probs.DType == ETensorDType::BF16) {
        moe_topk_forward(
            output.expert_indices.get<int>(),
            output.routing_weights.get<nv_bfloat16>(),
            acts.softmax_probs.get<nv_bfloat16>(),
            BT, E, K, mConfig.normalize_routing, ctx.stream
        );
    } else {
        moe_topk_forward(
            output.expert_indices.get<int>(),
            output.routing_weights.get<float>(),
            acts.softmax_probs.get<float>(),
            BT, E, K, mConfig.normalize_routing, ctx.stream
        );
    }

    // Compute token counts per expert (for load balancing loss and dispatch)
    moe_compute_expert_counts(
        output.token_counts.get<int>(),
        output.expert_indices.get<int>(),
        BT, K, E, ctx.stream
    );

    // Compute auxiliary losses
    compute_aux_loss(acts, ctx.B, ctx.T, ctx.stream);
    output.aux_loss = acts.output.aux_loss;
    output.z_loss = acts.output.z_loss;

    acts.output = output;
    return output;
}

inline Tensor RouterModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_routing_weights, Gradients& grads, bool accumulate) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;

    // Gradient through softmax
    // The grad_routing_weights is sparse (only top-k elements are non-zero)
    // We need to scatter it back to full (BT, E) shape first
    // Then compute softmax backward: d_logits = softmax_probs * (d_probs - sum(d_probs * softmax_probs))
    Tensor d_probs;
    d_probs.DType = acts.softmax_probs.DType;
    // For simplicity, assume grad_routing_weights has been scattered to full (BT, E) by caller
    // TODO: Add proper scatter logic for sparse top-k gradient

    Tensor d_logits;
    d_logits.DType = acts.logits.DType;
    if (acts.softmax_probs.DType == ETensorDType::BF16) {
        moe_softmax_backward(
            d_logits.get<nv_bfloat16>(),
            grad_routing_weights.get<nv_bfloat16>(),
            acts.softmax_probs.get<nv_bfloat16>(),
            BT, E, ctx.stream
        );
    } else {
        moe_softmax_backward(
            d_logits.get<float>(),
            grad_routing_weights.get<float>(),
            acts.softmax_probs.get<float>(),
            BT, E, ctx.stream
        );
    }

    // Add z-loss gradient contribution to logits
    // d_logits += z_loss_coef * d_z_loss / d_logits
    if (mConfig.z_loss_coef > 0.0f) {
        if (acts.logits.DType == ETensorDType::BF16) {
            moe_router_z_loss_backward(
                d_logits.get<nv_bfloat16>(),
                acts.logits.get<nv_bfloat16>(),
                BT, E, mConfig.z_loss_coef, ctx.stream
            );
        } else {
            moe_router_z_loss_backward(
                d_logits.get<float>(),
                acts.logits.get<float>(),
                BT, E, mConfig.z_loss_coef, ctx.stream
            );
        }
    }

    // Gradient w.r.t. gate weights: d_gate = input^T @ d_logits
    matmul(
        grads.d_gate, acts.input, d_logits, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, E, BT, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Gradient w.r.t. input: d_input = d_logits @ gate^T
    Tensor d_input;
    d_input.DType = acts.input.DType;
    matmul(
        d_input, w.gate, d_logits, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, BT, E, EMMTranspose::NN, false,
        ctx.stream
    );

    return d_input;
}

inline void RouterModule::compute_aux_loss(Activations& acts, int B, int T, cudaStream_t stream) {
    const int BT = B * T;
    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Allocate device memory for loss outputs
    float* d_aux_loss = nullptr;
    float* d_z_loss = nullptr;
    cudaMallocAsync(&d_aux_loss, sizeof(float), stream);
    cudaMallocAsync(&d_z_loss, sizeof(float), stream);

    // Initialize z_loss to 0 (it gets accumulated via atomicAdd)
    cudaMemsetAsync(d_z_loss, 0, sizeof(float), stream);

    // Compute load balancing loss using kernel
    if (acts.softmax_probs.DType == ETensorDType::BF16) {
        moe_compute_aux_loss(
            d_aux_loss,
            acts.softmax_probs.get<nv_bfloat16>(),
            acts.output.expert_indices.get<int>(),
            BT, E, K, mConfig.aux_loss_coef, stream
        );
    } else {
        moe_compute_aux_loss(
            d_aux_loss,
            acts.softmax_probs.get<float>(),
            acts.output.expert_indices.get<int>(),
            BT, E, K, mConfig.aux_loss_coef, stream
        );
    }

    // Compute router z-loss for logit regularization (uses pre-softmax logits)
    if (mConfig.z_loss_coef > 0.0f) {
        if (acts.logits.DType == ETensorDType::BF16) {
            moe_router_z_loss_forward(
                d_z_loss,
                acts.logits.get<nv_bfloat16>(),
                BT, E, mConfig.z_loss_coef, stream
            );
        } else {
            moe_router_z_loss_forward(
                d_z_loss,
                acts.logits.get<float>(),
                BT, E, mConfig.z_loss_coef, stream
            );
        }
    }

    // Copy results back to host
    cudaMemcpyAsync(&acts.output.aux_loss, d_aux_loss, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&acts.output.z_loss, d_z_loss, sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_aux_loss, stream);
    cudaFreeAsync(d_z_loss, stream);
}

/**
 * @brief Switch Router - simplified routing with top-1 selection
 *
 * A simplified router that routes each token to exactly one expert.
 * Used in Switch Transformer architecture.
 */
class SwitchRouterModule : public ModuleBase<SwitchRouterModule> {
public:
    struct Config {
        int hidden_size;
        int num_experts;
        int top_k = 1;              ///< Number of experts per token (typically 1 for Switch)
        float aux_loss_coef = 0.01f;  ///< Coefficient for load balancing auxiliary loss
        float capacity_factor = 1.0f;  ///< Expert capacity factor
        bool use_expert_choice = false;  ///< Expert chooses tokens instead of tokens choosing experts
    };

    // RouterOutput type for compatibility with MoETransformerBlock
    using RouterOutput = RouterModule::RouterOutput;

    struct Weights {
        Tensor gate;  ///< (hidden_size, num_experts)
    };

    struct Activations {
        Tensor input;
        Tensor logits;
        Tensor expert_index;     ///< (B*T,) single expert per token
        Tensor routing_weight;   ///< (B*T,) weight for selected expert
        RouterOutput output;     ///< Full routing output for compatibility
    };

    struct Gradients {
        Tensor d_gate;
    };

    explicit SwitchRouterModule(Config config) : mConfig(config) {}

    Activations forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
};

/**
 * @brief Expert Choice Router - experts select tokens
 *
 * Instead of tokens selecting experts (which can cause load imbalance),
 * each expert selects its top-k tokens. Guarantees perfect load balancing.
 * Used in some recent MoE architectures.
 */
class ExpertChoiceRouterModule : public ModuleBase<ExpertChoiceRouterModule> {
public:
    struct Config {
        int hidden_size;
        int num_experts;
        int tokens_per_expert;  ///< Fixed capacity per expert
        float aux_loss_coef = 0.0f;  ///< Usually 0 since load is balanced by design
    };

    struct Weights {
        Tensor gate;  ///< (hidden_size, num_experts)
    };

    struct Activations {
        Tensor input;
        Tensor logits;
        Tensor token_indices;    ///< (num_experts, tokens_per_expert) selected token indices
        Tensor routing_weights;  ///< (num_experts, tokens_per_expert) weights
    };

    struct Gradients {
        Tensor d_gate;
    };

    explicit ExpertChoiceRouterModule(Config config) : mConfig(config) {}

    Activations forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_ROUTER_H
