// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// BaseExpertModule - CRTP base class for expert module variants
//
// Provides common expert MLP functionality with protected helper methods that
// derived classes can compose differently:
//   - SwiGLUExpertModule: Standard gated MLP with SwiGLU activation
//   - GeGLUExpertModule: Gated MLP with GeGLU activation
//   - SharedExpertModule: Expert that processes all tokens (not routed)

#ifndef SUROGATE_SRC_MODULES_MOE_BASE_EXPERT_H
#define SUROGATE_SRC_MODULES_MOE_BASE_EXPERT_H

#include "modules/module_base.h"
#include "modules/forward_hooks.h"
#include "kernels/kernels.h"

namespace modules {

/**
 * @brief CRTP base class for expert module variants
 *
 * This class provides the common implementation for all expert variants
 * through protected helper methods. Derived classes implement forward_impl/
 * backward_impl by composing these helpers.
 *
 * An expert is essentially an MLP (same structure as the dense FFN).
 * In MoE, multiple experts share the same structure but have independent weights.
 *
 * Structure: input -> [gate_proj, up_proj] -> activation(gate) * up -> down_proj -> output
 *
 * @tparam Derived The concrete expert module type (CRTP)
 */
template<typename Derived>
class BaseExpertModule : public ModuleBase<Derived> {
public:
    /**
     * @brief Base configuration for experts
     */
    struct BaseConfig {
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension
        bool use_gated = true;      ///< Use gated activation (SwiGLU)
        float dropout = 0.0f;       ///< Dropout probability
    };

    /**
     * @brief Base weight tensors for a single expert
     *
     * For gated activations, gate_proj contains fused [gate, up] projections.
     */
    struct BaseWeights {
        Tensor gate_proj;           ///< (hidden_size, 2 * intermediate_size) - fused gate+up for gated
                                    ///< Or (hidden_size, intermediate_size) - up only for non-gated
        Tensor down_proj;           ///< (intermediate_size, hidden_size)
    };

    /**
     * @brief Base saved state for backward pass
     */
    struct BaseActivations {
        Tensor input;               ///< Input to expert
        Tensor gate_up;             ///< Output of gate+up projection (before activation)
        Tensor activated;           ///< Output after activation
        Tensor output;              ///< Final output
    };

    /**
     * @brief Base weight gradients
     */
    struct BaseGradients {
        Tensor d_gate_proj;         ///< (hidden_size, 2 * intermediate_size) or (hidden_size, intermediate_size)
        Tensor d_down_proj;         ///< (intermediate_size, hidden_size)
    };

protected:
    // ========================================================================
    // Protected helper methods for derived classes to compose
    // ========================================================================

    /**
     * @brief Forward through fused gate+up projection
     *
     * For gated activation: gate_up = input @ gate_proj^T
     * Result has shape (N, 2*intermediate_size) with [gate, up] concatenated.
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param gate_proj Fused gate+up weight (hidden_size, 2 * intermediate_size)
     * @param input Input tensor (N, hidden_size)
     * @param gate_up Output tensor (N, 2 * intermediate_size)
     */
    static void forward_gate_up_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& gate_proj,
        const Tensor& input,
        Tensor& gate_up
    ) {
        const int N = input.Sizes[0];
        const int C = config.hidden_size;
        const int D = config.intermediate_size;
        const int out_dim = config.use_gated ? 2 * D : D;

        matmul(
            gate_up, gate_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            out_dim, N, C, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    /**
     * @brief Forward through non-gated up projection only
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param up_proj Up projection weight (hidden_size, intermediate_size)
     * @param input Input tensor (N, hidden_size)
     * @param up_out Output tensor (N, intermediate_size)
     */
    static void forward_up_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& up_proj,
        const Tensor& input,
        Tensor& up_out
    ) {
        const int N = input.Sizes[0];
        const int C = config.hidden_size;
        const int D = config.intermediate_size;

        matmul(
            up_out, up_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            D, N, C, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    /**
     * @brief Apply SwiGLU activation: activated = SiLU(gate) * up
     *
     * Input gate_up has shape (N, 2*D) with [gate, up] layout or [up, gate] layout.
     * Output activated has shape (N, D).
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param gate_up Input (N, 2 * intermediate_size)
     * @param activated Output (N, intermediate_size)
     */
    static void apply_swiglu(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& gate_up,
        Tensor& activated
    ) {
        const int N = gate_up.Sizes[0];
        const int D = config.intermediate_size;

        swiglu_forward(activated, gate_up, nullptr, 1, N, D, ctx.stream);
    }

    /**
     * @brief SwiGLU backward
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param grad_activated Gradient w.r.t. activated (N, D)
     * @param gate_up Saved gate_up from forward (N, 2*D)
     * @param grad_gate_up Output: gradient w.r.t. gate_up (N, 2*D)
     */
    static void apply_swiglu_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& grad_activated,
        const Tensor& gate_up,
        Tensor& grad_gate_up
    ) {
        const int N = grad_activated.Sizes[0];
        const int D = config.intermediate_size;

        swiglu_backward(grad_gate_up, grad_activated, gate_up, nullptr, 1, N, D, ctx.stream);
    }

    /**
     * @brief Forward through down projection
     *
     * output = activated @ down_proj^T
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param down_proj Down projection weight (intermediate_size, hidden_size)
     * @param activated Input (N, intermediate_size)
     * @param output Output (N, hidden_size)
     */
    static void forward_down_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& down_proj,
        const Tensor& activated,
        Tensor& output
    ) {
        const int N = activated.Sizes[0];
        const int C = config.hidden_size;
        const int D = config.intermediate_size;

        matmul(
            output, down_proj, activated, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, N, D, EMMTranspose::TN, false,
            ctx.stream
        );
    }

    /**
     * @brief Backward through down projection
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param down_proj Down projection weight
     * @param activated Saved activated from forward
     * @param grad_output Gradient w.r.t. output (N, hidden_size)
     * @param d_down_proj Output: gradient w.r.t. down_proj
     * @param accumulate Whether to accumulate gradient
     * @return Gradient w.r.t. activated (N, intermediate_size)
     */
    static Tensor backward_down_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& down_proj,
        const Tensor& activated,
        const Tensor& grad_output,
        Tensor& d_down_proj,
        bool accumulate
    ) {
        const int N = grad_output.Sizes[0];
        const int C = config.hidden_size;
        const int D = config.intermediate_size;

        Tensor d_activated;
        d_activated.DType = grad_output.DType;

        // d_activated = grad_output @ down_proj (NN transpose)
        matmul(
            d_activated, down_proj, grad_output, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            D, N, C, EMMTranspose::NN, false,
            ctx.stream
        );

        // d_down_proj = activated^T @ grad_output (NT transpose)
        matmul(
            d_down_proj, activated, grad_output, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            D, C, N, EMMTranspose::NT, accumulate,
            ctx.stream
        );

        return d_activated;
    }

    /**
     * @brief Backward through gate+up projection
     *
     * @param ctx Module context
     * @param config Expert configuration
     * @param gate_proj Gate+up projection weight
     * @param input Saved input from forward
     * @param grad_gate_up Gradient w.r.t. gate_up
     * @param d_gate_proj Output: gradient w.r.t. gate_proj
     * @param accumulate Whether to accumulate gradient
     * @return Gradient w.r.t. input
     */
    static Tensor backward_gate_up_proj(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& gate_proj,
        const Tensor& input,
        const Tensor& grad_gate_up,
        Tensor& d_gate_proj,
        bool accumulate
    ) {
        const int N = input.Sizes[0];
        const int C = config.hidden_size;
        const int D = config.intermediate_size;
        const int out_dim = config.use_gated ? 2 * D : D;

        Tensor d_input;
        d_input.DType = input.DType;

        // d_input = grad_gate_up @ gate_proj (NN transpose)
        matmul(
            d_input, gate_proj, grad_gate_up, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, N, out_dim, EMMTranspose::NN, false,
            ctx.stream
        );

        // d_gate_proj = input^T @ grad_gate_up (NT transpose)
        matmul(
            d_gate_proj, input, grad_gate_up, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            C, out_dim, N, EMMTranspose::NT, accumulate,
            ctx.stream
        );

        return d_input;
    }
};

/**
 * @brief CRTP base class for expert group modules
 *
 * Manages multiple experts and handles the scatter/gather operations
 * for routing tokens to experts and combining outputs.
 *
 * Uses a permute-based approach:
 * 1. Permute tokens to expert-grouped order (scatter)
 * 2. Run batched expert computation
 * 3. Unpermute and weight-combine outputs (gather)
 *
 * @tparam Derived The concrete expert group module type (CRTP)
 */
template<typename Derived>
class BaseExpertGroupModule : public ModuleBase<Derived> {
public:
    /**
     * @brief Base configuration for expert group
     */
    struct BaseConfig {
        int num_experts;            ///< Number of experts in the group
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension per expert
        int top_k = 2;              ///< Number of experts per token
        int capacity_factor = 1;    ///< Capacity multiplier
        bool use_gated = true;      ///< Use gated activation
    };

    /**
     * @brief Base weights for all experts (batched layout)
     */
    struct BaseWeights {
        Tensor gate_up_proj;        ///< (num_experts, hidden_size, 2 * intermediate_size) - fused
        Tensor down_proj;           ///< (num_experts, intermediate_size, hidden_size)
    };

    /**
     * @brief Base activations for expert group
     */
    struct BaseActivations {
        // Permutation state
        Tensor gather_indices;      ///< (total_tokens,) - maps permuted position to original token
        Tensor scatter_indices;     ///< (total_tokens,) - maps original token to permuted position
        Tensor expert_offsets;      ///< (num_experts + 1,) - cumsum of tokens per expert

        // Dispatch/combine state
        Tensor permuted_input;      ///< (total_tokens, hidden_size) - tokens in expert-grouped order
        Tensor expert_outputs;      ///< (total_tokens, hidden_size) - outputs in expert-grouped order
        Tensor expert_gate_up;      ///< (total_tokens, 2 * intermediate_size) - for fused/fast path (Saved for backward)
        Tensor combined_output;     ///< (B*T, hidden_size) - final output
    };

    /**
     * @brief Base gradients for expert group (batched layout)
     */
    struct BaseGradients {
        Tensor d_gate_up_proj;      ///< (num_experts, hidden_size, 2 * intermediate_size)
        Tensor d_down_proj;         ///< (num_experts, intermediate_size, hidden_size)

        // Intermediate tensors for backward pass
        Tensor d_expert_outputs;    ///< (total_tokens, hidden_size) gradient w.r.t. expert outputs
        Tensor d_routing_weights;   ///< (B*T, top_k) gradient w.r.t. routing weights
        Tensor d_permuted_input;    ///< (total_tokens, hidden_size) gradient w.r.t. permuted input
        Tensor d_input;             ///< (B*T, hidden_size) gradient w.r.t. original input
    };

protected:
    // ========================================================================
    // Protected helper methods for derived classes
    // ========================================================================

    /**
     * @brief Permute tokens to expert-grouped order
     *
     * Reorders tokens so that all tokens for expert 0 come first, then expert 1, etc.
     *
     * @param ctx Module context
     * @param config Expert group configuration
     * @param input Input tokens (B*T, hidden_size)
     * @param gather_indices Indices computed from routing (total_tokens,)
     * @param permuted_output Output: permuted tokens (total_tokens, hidden_size)
     * @param top_k Number of experts per token
     */
    static void permute_tokens(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& input,
        const Tensor& gather_indices,
        Tensor& permuted_output,
        int top_k
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int total_tokens = BT * top_k;

        if (input.DType == ETensorDType::BF16) {
            moe_permute_tokens(
                permuted_output.get<nv_bfloat16>(),
                input.get<nv_bfloat16>(),
                gather_indices.get<int>(),
                total_tokens, BT, C, top_k, ctx.stream
            );
        } else {
            moe_permute_tokens(
                permuted_output.get<float>(),
                input.get<float>(),
                gather_indices.get<int>(),
                total_tokens, BT, C, top_k, ctx.stream
            );
        }
    }

    /**
     * @brief Unpermute and combine expert outputs with routing weights
     *
     * @param ctx Module context
     * @param config Expert group configuration
     * @param expert_outputs Expert outputs in permuted order (total_tokens, hidden_size)
     * @param routing_weights Routing weights (B*T, top_k)
     * @param scatter_indices Scatter indices (total_tokens,)
     * @param combined_output Output: combined result (B*T, hidden_size)
     * @param top_k Number of experts per token
     */
    static void unpermute_and_combine(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& expert_outputs,
        const Tensor& routing_weights,
        const Tensor& scatter_indices,
        Tensor& combined_output,
        int top_k
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int total_tokens = BT * top_k;

        if (expert_outputs.DType == ETensorDType::BF16) {
            moe_unpermute_and_combine(
                combined_output.get<nv_bfloat16>(),
                expert_outputs.get<nv_bfloat16>(),
                routing_weights.get<nv_bfloat16>(),
                scatter_indices.get<int>(),
                BT, total_tokens, C, top_k, ctx.stream
            );
        } else {
            moe_unpermute_and_combine(
                combined_output.get<float>(),
                expert_outputs.get<float>(),
                routing_weights.get<float>(),
                scatter_indices.get<int>(),
                BT, total_tokens, C, top_k, ctx.stream
            );
        }
    }

    /**
     * @brief Backward through unpermute+combine
     *
     * Computes gradients w.r.t. expert outputs and routing weights.
     *
     * @param ctx Module context
     * @param config Expert group configuration
     * @param grad_output Gradient w.r.t. combined output (B*T, hidden_size)
     * @param expert_outputs Expert outputs from forward (total_tokens, hidden_size)
     * @param routing_weights Routing weights (B*T, top_k)
     * @param scatter_indices Scatter indices (total_tokens,)
     * @param d_expert_outputs Output: gradient w.r.t. expert outputs
     * @param d_routing_weights Output: gradient w.r.t. routing weights
     * @param top_k Number of experts per token
     */
    static void combine_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& grad_output,
        const Tensor& expert_outputs,
        const Tensor& routing_weights,
        const Tensor& scatter_indices,
        Tensor& d_expert_outputs,
        Tensor& d_routing_weights,
        int top_k
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int total_tokens = BT * top_k;

        if (grad_output.DType == ETensorDType::BF16) {
            moe_combine_backward(
                d_expert_outputs.get<nv_bfloat16>(),
                d_routing_weights.get<nv_bfloat16>(),
                grad_output.get<nv_bfloat16>(),
                expert_outputs.get<nv_bfloat16>(),
                routing_weights.get<nv_bfloat16>(),
                scatter_indices.get<int>(),
                BT, total_tokens, C, top_k, ctx.stream
            );
        } else {
            moe_combine_backward(
                d_expert_outputs.get<float>(),
                d_routing_weights.get<float>(),
                grad_output.get<float>(),
                expert_outputs.get<float>(),
                routing_weights.get<float>(),
                scatter_indices.get<int>(),
                BT, total_tokens, C, top_k, ctx.stream
            );
        }
    }

    /**
     * @brief Backward through token permutation
     *
     * Accumulates gradients from multiple expert paths back to original tokens.
     *
     * @param ctx Module context
     * @param config Expert group configuration
     * @param d_permuted_input Gradient w.r.t. permuted input (total_tokens, hidden_size)
     * @param gather_indices Gather indices (total_tokens,)
     * @param d_input Output: gradient w.r.t. original input (B*T, hidden_size)
     * @param top_k Number of experts per token
     */
    static void permute_backward(
        ModuleContext& ctx,
        const BaseConfig& config,
        const Tensor& d_permuted_input,
        const Tensor& gather_indices,
        Tensor& d_input,
        int top_k
    ) {
        const int BT = ctx.B * ctx.T;
        const int C = config.hidden_size;
        const int total_tokens = BT * top_k;

        if (d_permuted_input.DType == ETensorDType::BF16) {
            moe_permute_backward(
                d_input.get<nv_bfloat16>(),
                d_permuted_input.get<nv_bfloat16>(),
                gather_indices.get<int>(),
                total_tokens, BT, C, top_k, ctx.stream
            );
        } else {
            moe_permute_backward(
                d_input.get<float>(),
                d_permuted_input.get<float>(),
                gather_indices.get<int>(),
                total_tokens, BT, C, top_k, ctx.stream
            );
        }
    }
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_BASE_EXPERT_H
