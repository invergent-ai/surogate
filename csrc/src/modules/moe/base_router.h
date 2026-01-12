// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BaseRouterModule - CRTP base class for router module variants
//
// Provides common router functionality with protected helper methods that
// derived classes can compose differently:
//   - TopKRouterModule: Standard top-k routing (Mixtral, Qwen3)
//   - SwitchRouterModule: Top-1 routing (Switch Transformer)
//   - ExpertChoiceRouterModule: Experts select tokens
//   - Qwen3RouterModule: Top-k with post-selection normalization

#ifndef SUROGATE_SRC_MODULES_MOE_BASE_ROUTER_H
#define SUROGATE_SRC_MODULES_MOE_BASE_ROUTER_H

#include "modules/module_base.h"
#include "kernels/kernels.h"

namespace modules {

// Forward declaration
template<typename Derived>
class BaseRouterModule;

/**
 * @brief Common router output structure used by all router variants
 *
 * This is defined at namespace level so ExpertGroupModule and MoETransformerBlock
 * can use it without depending on a specific router type.
 */
struct MoERouterOutput {
    Tensor routing_weights;         ///< (B*T, top_k) normalized weights for selected experts
    Tensor expert_indices;          ///< (B*T, top_k) indices of selected experts (int32)
    Tensor expert_mask;             ///< (B*T, num_experts) binary mask of expert selection
    float aux_loss;                 ///< Load balancing auxiliary loss
    float z_loss;                   ///< Router z-loss for regularization

    // For expert dispatch
    Tensor token_indices;           ///< (num_experts, capacity) token indices per expert
    Tensor token_counts;            ///< (num_experts,) number of tokens per expert
    Tensor dispatch_mask;           ///< (B*T, num_experts) dispatch weights
    Tensor combine_weights;         ///< (B*T, num_experts) combine weights for output
};

/**
 * @brief CRTP base class for router module variants
 *
 * This class provides the common implementation for all router variants
 * (TopKRouter, SwitchRouter, ExpertChoiceRouter) through protected helper
 * methods. Derived classes implement forward_impl/backward_impl by composing
 * these helpers.
 *
 * The CRTP pattern ensures zero-overhead dispatch while allowing code reuse.
 *
 * @tparam Derived The concrete router module type (CRTP)
 */
template<typename Derived>
class BaseRouterModule : public ModuleBase<Derived> {
public:
    /**
     * @brief Base configuration for routers
     *
     * Contains all fields used by any router variant. Derived modules
     * can ignore fields they don't use.
     */
    struct BaseConfig {
        int hidden_size;                ///< Input hidden dimension
        int num_experts;                ///< Total number of experts
        int top_k = 2;                  ///< Number of experts to route each token to
        float aux_loss_coef = 0.01f;    ///< Coefficient for load balancing auxiliary loss
        float z_loss_coef = 0.001f;     ///< Coefficient for router z-loss
        bool use_noisy_routing = false; ///< Add noise during training for exploration
        float noise_std = 0.1f;         ///< Standard deviation of routing noise
        float capacity_factor = 1.25f;  ///< Expert capacity factor
        bool normalize_routing = true;  ///< Normalize routing weights to sum to 1

        // Qwen3 MoE-style normalization
        bool norm_topk_prob = false;    ///< Normalize top-k weights AFTER selection (Qwen3 style)
    };

    /**
     * @brief Base weight tensors for router
     */
    struct BaseWeights {
        Tensor gate;                    ///< (hidden_size, num_experts) routing projection
        std::optional<Tensor> bias;     ///< (num_experts,) optional routing bias
    };

    /**
     * @brief Router output structure - alias to common MoERouterOutput
     */
    using RouterOutput = MoERouterOutput;

    /**
     * @brief Base saved state for backward pass
     */
    struct BaseActivations {
        Tensor input;                   ///< (B*T, hidden_size) input to router
        Tensor logits;                  ///< (B*T, num_experts) raw routing logits
        Tensor softmax_probs;           ///< (B*T, num_experts) softmax probabilities
        RouterOutput output;            ///< Full routing output
    };

    /**
     * @brief Base weight gradients
     */
    struct BaseGradients {
        Tensor d_gate;                  ///< (hidden_size, num_experts)
    };

protected:
    // ========================================================================
    // Protected helper methods for derived classes to compose
    // ========================================================================

    /**
     * @brief Compute routing logits: logits = input @ gate
     *
     * @param ctx Module context with CUDA resources
     * @param config Router configuration
     * @param gate Gate weight tensor (hidden_size, num_experts)
     * @param bias Optional bias tensor (num_experts,)
     * @param input Input tensor (B*T, hidden_size)
     * @param logits Output tensor (B*T, num_experts)
     */
    static void compute_routing_logits(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& gate,
        const std::optional<Tensor>& bias,
        const Tensor& input,
        Tensor& logits
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int E = config.num_experts;

        matmul(
            logits, gate, input, bias,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            E, BT, C, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    /**
     * @brief Apply softmax to routing logits
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param logits Input logits (B*T, num_experts)
     * @param probs Output probabilities (B*T, num_experts)
     */
    static void apply_softmax(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& logits,
        Tensor& probs
    ) {
        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;

        if (logits.DType == ETensorDType::BF16) {
            moe_softmax_forward(
                probs.get<nv_bfloat16>(),
                logits.get<nv_bfloat16>(),
                BT, E, ctx.stream
            );
        } else {
            moe_softmax_forward(
                probs.get<float>(),
                logits.get<float>(),
                BT, E, ctx.stream
            );
        }
    }

    /**
     * @brief Softmax backward
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param grad_probs Gradient w.r.t. softmax output (B*T, num_experts)
     * @param probs Softmax output from forward (B*T, num_experts)
     * @param grad_logits Output: gradient w.r.t. logits (B*T, num_experts)
     */
    static void apply_softmax_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& grad_probs,
        const Tensor& probs,
        Tensor& grad_logits
    ) {
        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;

        if (probs.DType == ETensorDType::BF16) {
            moe_softmax_backward(
                grad_logits.get<nv_bfloat16>(),
                grad_probs.get<nv_bfloat16>(),
                probs.get<nv_bfloat16>(),
                BT, E, ctx.stream
            );
        } else {
            moe_softmax_backward(
                grad_logits.get<float>(),
                grad_probs.get<float>(),
                probs.get<float>(),
                BT, E, ctx.stream
            );
        }
    }

    /**
     * @brief Select top-k experts per token
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param probs Softmax probabilities (B*T, num_experts)
     * @param routing_weights Output: selected weights (B*T, top_k)
     * @param expert_indices Output: selected expert indices (B*T, top_k)
     * @param normalize_after Whether to normalize weights after selection (Qwen3 style)
     */
    static void select_topk(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& probs,
        Tensor& routing_weights,
        Tensor& expert_indices,
        bool normalize_after = false
    ) {
        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;
        const int K = config.top_k;

        if (probs.DType == ETensorDType::BF16) {
            moe_topk_forward(
                expert_indices.get<int>(),
                routing_weights.get<nv_bfloat16>(),
                probs.get<nv_bfloat16>(),
                BT, E, K, normalize_after, ctx.stream
            );
        } else {
            moe_topk_forward(
                expert_indices.get<int>(),
                routing_weights.get<float>(),
                probs.get<float>(),
                BT, E, K, normalize_after, ctx.stream
            );
        }
    }

    /**
     * @brief Compute token counts per expert
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param expert_indices Expert indices (B*T, top_k)
     * @param token_counts Output: count per expert (num_experts,)
     */
    static void compute_token_counts(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& expert_indices,
        Tensor& token_counts
    ) {
        const int BT = ctx.B * ctx.T;
        const int K = config.top_k;
        const int E = config.num_experts;

        moe_compute_expert_counts(
            token_counts.get<int>(),
            expert_indices.get<int>(),
            BT, K, E, ctx.stream
        );
    }

    /**
     * @brief Compute load balancing auxiliary loss
     *
     * The load balancing loss encourages uniform expert utilization:
     * aux_loss = num_experts * sum_e(f_e * P_e)
     * where f_e = fraction of tokens routed to expert e
     *       P_e = mean routing probability to expert e
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param probs Softmax probabilities (B*T, num_experts)
     * @param expert_indices Selected expert indices (B*T, top_k)
     * @param aux_loss_ptr Output: computed auxiliary loss
     */
    static void compute_load_balance_loss(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& probs,
        const Tensor& expert_indices,
        float* aux_loss_ptr
    ) {
        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;
        const int K = config.top_k;

        // Allocate device memory for loss output
        float* d_aux_loss = nullptr;
        cudaMallocAsync(&d_aux_loss, sizeof(float), ctx.stream);

        if (probs.DType == ETensorDType::BF16) {
            moe_compute_aux_loss(
                d_aux_loss,
                probs.get<nv_bfloat16>(),
                expert_indices.get<int>(),
                BT, E, K, config.aux_loss_coef, ctx.stream
            );
        } else {
            moe_compute_aux_loss(
                d_aux_loss,
                probs.get<float>(),
                expert_indices.get<int>(),
                BT, E, K, config.aux_loss_coef, ctx.stream
            );
        }

        // Copy result back to host
        cudaMemcpyAsync(aux_loss_ptr, d_aux_loss, sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaFreeAsync(d_aux_loss, ctx.stream);
    }

    /**
     * @brief Compute router z-loss for logit regularization
     *
     * The z-loss penalizes large logits to prevent routing collapse:
     * z_loss = (1/BT) * sum(log(sum(exp(logits))))^2
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param logits Raw routing logits (B*T, num_experts)
     * @param z_loss_ptr Output: computed z-loss
     */
    static void compute_z_loss(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& logits,
        float* z_loss_ptr
    ) {
        if (config.z_loss_coef <= 0.0f) {
            *z_loss_ptr = 0.0f;
            return;
        }

        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;

        // Allocate device memory for loss output
        float* d_z_loss = nullptr;
        cudaMallocAsync(&d_z_loss, sizeof(float), ctx.stream);
        cudaMemsetAsync(d_z_loss, 0, sizeof(float), ctx.stream);

        if (logits.DType == ETensorDType::BF16) {
            moe_router_z_loss_forward(
                d_z_loss,
                logits.get<nv_bfloat16>(),
                BT, E, config.z_loss_coef, ctx.stream
            );
        } else {
            moe_router_z_loss_forward(
                d_z_loss,
                logits.get<float>(),
                BT, E, config.z_loss_coef, ctx.stream
            );
        }

        // Copy result back to host
        cudaMemcpyAsync(z_loss_ptr, d_z_loss, sizeof(float),
                        cudaMemcpyDeviceToHost, ctx.stream);
        cudaFreeAsync(d_z_loss, ctx.stream);
    }

    /**
     * @brief Add z-loss gradient contribution to logit gradients
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param logits Raw routing logits from forward
     * @param grad_logits Logit gradients to modify (accumulated)
     */
    static void add_z_loss_gradient(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& logits,
        Tensor& grad_logits
    ) {
        if (config.z_loss_coef <= 0.0f) return;

        const int BT = ctx.B * ctx.T;
        const int E = config.num_experts;

        if (logits.DType == ETensorDType::BF16) {
            moe_router_z_loss_backward(
                grad_logits.get<nv_bfloat16>(),
                logits.get<nv_bfloat16>(),
                BT, E, config.z_loss_coef, ctx.stream
            );
        } else {
            moe_router_z_loss_backward(
                grad_logits.get<float>(),
                logits.get<float>(),
                BT, E, config.z_loss_coef, ctx.stream
            );
        }
    }

    /**
     * @brief Backward through gate projection
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param gate Gate weight tensor
     * @param input Saved input from forward
     * @param grad_logits Gradient w.r.t. logits
     * @param d_gate Output: gradient w.r.t. gate weight
     * @param accumulate Whether to accumulate gradients
     * @return Gradient w.r.t. input
     */
    static Tensor backward_gate_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& gate,
        const Tensor& input,
        const Tensor& grad_logits,
        Tensor& d_gate,
        bool accumulate
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int E = config.num_experts;

        // d_gate = input^T @ grad_logits
        matmul(
            d_gate, input, grad_logits, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, E, BT, EMMTranspose::NT, accumulate,
            ctx.stream
        );

        // d_input = grad_logits @ gate^T
        Tensor d_input;
        d_input.DType = input.DType;
        matmul(
            d_input, gate, grad_logits, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, BT, E, EMMTranspose::NN, false,
            ctx.stream
        );

        return d_input;
    }

    // ========================================================================
    // Expert Choice routing helpers (experts select tokens)
    // ========================================================================

    /**
     * @brief Expert choice routing: each expert selects its top tokens
     *
     * Instead of tokens selecting experts (which can cause load imbalance),
     * each expert selects its top tokens. This guarantees perfect load balancing.
     *
     * @param ctx Module context
     * @param config Router configuration
     * @param logits Routing logits (B*T, num_experts) - will be transposed
     * @param tokens_per_expert Number of tokens each expert should select
     * @param token_indices Output: selected token indices per expert (num_experts, tokens_per_expert)
     * @param routing_weights Output: routing weights (num_experts, tokens_per_expert)
     */
    static void expert_select_tokens(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& logits,
        int tokens_per_expert,
        Tensor& token_indices,
        Tensor& routing_weights
    ) {
        // Expert choice routing transposes the selection:
        // Instead of top-k over experts per token, we do top-k over tokens per expert
        // This is implemented by transposing logits and running top-k along the token dimension
        //
        // Note: This is a placeholder - the actual kernel would need to handle this efficiently
        // For now, expert choice routing uses the SwitchRouterModule/ExpertChoiceRouterModule
        // implementations which have their own forward_impl
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_BASE_ROUTER_H
