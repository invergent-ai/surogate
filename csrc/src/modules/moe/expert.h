// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_EXPERT_H
#define SUROGATE_SRC_MODULES_MOE_EXPERT_H

#include "modules/module_base.h"
#include "modules/primitives/linear.h"
#include "modules/primitives/swiglu.h"
#include "kernels/kernels.h"
#include "router.h"  // For RouterModule::RouterOutput

namespace modules {

/**
 * @brief Single Expert MLP module
 *
 * An expert is essentially an MLP (same structure as the dense FFN).
 * In MoE, multiple experts share the same structure but have independent weights.
 *
 * Structure: input -> up_proj -> activation -> down_proj -> output
 * With gated variants: input -> [gate_proj, up_proj] -> activation(gate) * up -> down_proj
 */
class ExpertModule : public ModuleBase<ExpertModule> {
public:
    /**
     * @brief Configuration for a single expert
     */
    struct Config {
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension
        bool use_gated = true;      ///< Use gated activation (SwiGLU)
        float dropout = 0.0f;       ///< Dropout probability
    };

    /**
     * @brief Weight tensors for a single expert
     */
    struct Weights {
        Tensor gate_proj;           ///< (hidden_size, intermediate_size) - only if gated
        Tensor up_proj;             ///< (hidden_size, intermediate_size)
        Tensor down_proj;           ///< (intermediate_size, hidden_size)
    };

    /**
     * @brief Saved state for backward pass
     */
    struct Activations {
        Tensor input;               ///< Input to expert
        Tensor gate_up;             ///< Output of gate+up projection (before activation)
        Tensor activated;           ///< Output after activation
        Tensor output;              ///< Final output
    };

    /**
     * @brief Weight gradients
     */
    struct Gradients {
        Tensor d_gate_proj;
        Tensor d_up_proj;
        Tensor d_down_proj;
    };

    explicit ExpertModule(Config config) : mConfig(config) {}

    /**
     * @brief Forward pass through expert MLP
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations routed to this expert (N, hidden_size)
     * @param acts Activation storage
     * @return Output (N, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts);

    /**
     * @brief Backward pass through expert MLP
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

// ============================================================================
// Implementation
// ============================================================================

inline Tensor ExpertModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    const int N = input.Sizes[0];  // Number of tokens routed to this expert
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    acts.input = input;

    if (mConfig.use_gated) {
        // Gated activation (SwiGLU style)
        // gate_up has shape (N, 2*D) with [gate, up] concatenated

        // Combined projection: gate_up = input @ [gate_proj; up_proj]
        // For efficiency, gate and up projections are often fused
        matmul(
            acts.gate_up, w.gate_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            2 * D, N, C, EMMTranspose::TN, false,
            ctx.stream
        );

        // SwiGLU activation: activated = swiglu(gate_up)
        swiglu_forward(acts.activated, acts.gate_up, nullptr, 1, N, D, ctx.stream);

    } else {
        // Simple activation (ReLU/GeLU)
        matmul(
            acts.gate_up, w.up_proj, input, std::nullopt,
            nullptr, nullptr,
            ctx.cublas_handle, *ctx.workspace,
            D, N, C, EMMTranspose::TN, false,
            ctx.stream
        );

        // Apply activation (GeLU)
        // gelu_forward(acts.activated, acts.gate_up, N * D, ctx.stream);
        acts.activated = acts.gate_up;  // Placeholder
    }

    // Down projection: output = activated @ down_proj
    matmul(
        acts.output, w.down_proj, acts.activated, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, N, D, EMMTranspose::TN, false,
        ctx.stream
    );

    return acts.output;
}

inline Tensor ExpertModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int N = acts.input.Sizes[0];
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    // Backward through down projection
    // d_activated = grad_output @ down_proj^T
    Tensor d_activated;
    d_activated.DType = grad_output.DType;
    matmul(
        d_activated, w.down_proj, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, N, C, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_down_proj = activated^T @ grad_output
    matmul(
        grads.d_down_proj, acts.activated, grad_output, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        D, C, N, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    // Backward through activation
    Tensor d_gate_up;
    d_gate_up.DType = d_activated.DType;
    if (mConfig.use_gated) {
        // SwiGLU backward
        swiglu_backward(d_gate_up, d_activated, acts.gate_up, nullptr, 1, N, D, ctx.stream);
    } else {
        // GeLU backward
        // gelu_backward(d_gate_up, d_activated, acts.gate_up, N * D, ctx.stream);
        d_gate_up = d_activated;  // Placeholder
    }

    // Backward through gate/up projection
    // d_input = d_gate_up @ gate_proj^T (or up_proj^T if not gated)
    Tensor d_input;
    d_input.DType = acts.input.DType;
    matmul(
        d_input, w.gate_proj, d_gate_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, N, 2 * D, EMMTranspose::NN, false,
        ctx.stream
    );

    // d_gate_proj = input^T @ d_gate_up
    matmul(
        grads.d_gate_proj, acts.input, d_gate_up, std::nullopt,
        nullptr, nullptr,
        ctx.cublas_handle, *ctx.workspace,
        C, 2 * D, N, EMMTranspose::NT, accumulate,
        ctx.stream
    );

    return d_input;
}

/**
 * @brief Expert group - collection of experts with shared dispatch logic
 *
 * Manages multiple experts and handles the scatter/gather operations
 * for routing tokens to experts and combining outputs.
 *
 * Uses a permute-based approach inspired by Unsloth's MoE implementation:
 * 1. Permute tokens to expert-grouped order (scatter)
 * 2. Run batched expert computation
 * 3. Unpermute and weight-combine outputs (gather)
 *
 * This approach enables efficient grouped GEMM computation across all experts.
 */
class ExpertGroupModule : public ModuleBase<ExpertGroupModule> {
public:
    /**
     * @brief Configuration for expert group
     */
    struct Config {
        int num_experts;            ///< Number of experts in the group
        int hidden_size;            ///< Input/output dimension
        int intermediate_size;      ///< FFN intermediate dimension per expert
        int top_k = 2;              ///< Number of experts per token
        int capacity_factor = 1;    ///< Capacity multiplier (tokens per expert = capacity_factor * tokens / num_experts * top_k)
        bool use_gated = true;      ///< Use gated activation
    };

    /**
     * @brief Weights for all experts
     *
     * Supports two layouts:
     * - Separate: std::vector<ExpertModule::Weights> - simple but less efficient
     * - Batched: single tensors with expert dimension - enables grouped GEMM
     */
    struct Weights {
        // Option 1: Separate weights per expert
        std::vector<ExpertModule::Weights> experts;

        // Option 2: Batched weights for grouped GEMM (preferred for performance)
        Tensor gate_up_proj;  ///< (num_experts, hidden_size, 2 * intermediate_size) - fused gate+up
        Tensor down_proj;     ///< (num_experts, intermediate_size, hidden_size)

        bool use_batched = false;  ///< Which layout to use
    };

    /**
     * @brief Activations for all experts
     */
    struct Activations {
        std::vector<ExpertModule::Activations> expert_acts;

        // Permutation state
        Tensor gather_indices;      ///< (total_tokens,) - maps permuted position to original token
        Tensor scatter_indices;     ///< (total_tokens,) - maps original token to permuted position
        Tensor expert_offsets;      ///< (num_experts + 1,) - cumsum of tokens per expert

        // Dispatch/combine state
        Tensor permuted_input;      ///< (total_tokens, hidden_size) - tokens in expert-grouped order
        Tensor expert_outputs;      ///< (total_tokens, hidden_size) - outputs in expert-grouped order
        Tensor combined_output;     ///< (B*T, hidden_size) - final output
    };

    /**
     * @brief Gradients for all experts
     */
    struct Gradients {
        // Per-expert gradients (for sequential execution)
        std::vector<ExpertModule::Gradients> expert_grads;

        // Batched gradients (for grouped GEMM)
        Tensor d_gate_up_proj;  ///< (num_experts, hidden_size, 2 * intermediate_size)
        Tensor d_down_proj;     ///< (num_experts, intermediate_size, hidden_size)

        // Intermediate tensors for backward pass (must be pre-allocated)
        Tensor d_expert_outputs;    ///< (total_tokens, hidden_size) gradient w.r.t. expert outputs
        Tensor d_routing_weights;   ///< (B*T, top_k) gradient w.r.t. routing weights
        Tensor d_permuted_input;    ///< (total_tokens, hidden_size) gradient w.r.t. permuted input
        Tensor d_input;             ///< (B*T, hidden_size) gradient w.r.t. original input
    };

    explicit ExpertGroupModule(Config config);

    /**
     * @brief Forward pass: dispatch tokens to experts and combine outputs
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations (B*T, hidden_size)
     * @param routing Router output with dispatch information
     * @param acts Activation storage
     * @return Combined output (B*T, hidden_size)
     */
    Tensor forward_impl(ModuleContext& ctx, Weights& w, Tensor& input,
                        const RouterModule::RouterOutput& routing, Activations& acts);

    /**
     * @brief Backward pass through expert group
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         const RouterModule::RouterOutput& routing,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

private:
    Config mConfig;
    std::vector<ExpertModule> mExperts;
};

// ============================================================================
// ExpertGroupModule Implementation
// ============================================================================

inline ExpertGroupModule::ExpertGroupModule(Config config) : mConfig(config) {
    ExpertModule::Config expert_config;
    expert_config.hidden_size = config.hidden_size;
    expert_config.intermediate_size = config.intermediate_size;
    expert_config.use_gated = config.use_gated;

    mExperts.reserve(config.num_experts);
    for (int i = 0; i < config.num_experts; ++i) {
        mExperts.emplace_back(expert_config);
    }
}

inline Tensor ExpertGroupModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input,
    const RouterModule::RouterOutput& routing, Activations& acts) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;
    const int total_tokens = BT * K;  // Each token is sent to K experts

    // Step 1: Permute tokens to expert-grouped order
    // This reorders tokens so that all tokens for expert 0 come first, then expert 1, etc.
    // Uses gather_indices computed from routing.expert_indices
    if (input.DType == ETensorDType::BF16) {
        moe_permute_tokens(
            acts.permuted_input.get<nv_bfloat16>(),
            input.get<nv_bfloat16>(),
            acts.gather_indices.get<int>(),
            total_tokens, BT, C, K, ctx.stream
        );
    } else {
        moe_permute_tokens(
            acts.permuted_input.get<float>(),
            input.get<float>(),
            acts.gather_indices.get<int>(),
            total_tokens, BT, C, K, ctx.stream
        );
    }

    // Step 2: Run experts on permuted tokens
    // For now, use sequential per-expert execution
    // TODO: Implement grouped GEMM for batched execution
    acts.expert_acts.resize(E);
    int offset = 0;

    for (int e = 0; e < E; ++e) {
        // Get token count for this expert from routing.token_counts
        const int* h_count_ptr = routing.token_counts.get<int>();
        int expert_tokens = h_count_ptr[e];  // Note: need to async copy this

        if (expert_tokens == 0) continue;

        // Slice of permuted input for this expert
        Tensor expert_input = slice(acts.permuted_input, 0, offset, offset + expert_tokens);

        // Slice for expert output
        Tensor expert_output = slice(acts.expert_outputs, 0, offset, offset + expert_tokens);

        // Run expert forward
        mExperts[e].forward(ctx, w.experts[e], expert_input, acts.expert_acts[e]);

        offset += expert_tokens;
    }

    // Step 3: Unpermute and combine expert outputs weighted by routing weights
    // For each original token, gather outputs from its assigned experts and weight-combine
    if (acts.expert_outputs.DType == ETensorDType::BF16) {
        moe_unpermute_and_combine(
            acts.combined_output.get<nv_bfloat16>(),
            acts.expert_outputs.get<nv_bfloat16>(),
            routing.routing_weights.get<nv_bfloat16>(),
            acts.scatter_indices.get<int>(),
            BT, total_tokens, C, K, ctx.stream
        );
    } else {
        moe_unpermute_and_combine(
            acts.combined_output.get<float>(),
            acts.expert_outputs.get<float>(),
            routing.routing_weights.get<float>(),
            acts.scatter_indices.get<int>(),
            BT, total_tokens, C, K, ctx.stream
        );
    }

    return acts.combined_output;
}

inline Tensor ExpertGroupModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    const RouterModule::RouterOutput& routing,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int BT = ctx.B * ctx.T;
    const int C = mConfig.hidden_size;
    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;
    const int total_tokens = BT * K;

    // Step 1: Backward through unpermute+combine
    // Computes:
    //   d_expert_outputs[permuted_idx] = routing_weights[token, k] * grad_output[token]
    //   d_routing_weights[token, k] = dot(expert_outputs[permuted_idx], grad_output[token])
    //
    // Note: grads.d_expert_outputs and grads.d_routing_weights should be pre-allocated

    if (grad_output.DType == ETensorDType::BF16) {
        moe_combine_backward(
            grads.d_expert_outputs.get<nv_bfloat16>(),
            grads.d_routing_weights.get<nv_bfloat16>(),
            grad_output.get<nv_bfloat16>(),
            acts.expert_outputs.get<nv_bfloat16>(),
            routing.routing_weights.get<nv_bfloat16>(),
            acts.scatter_indices.get<int>(),
            BT, total_tokens, C, K, ctx.stream
        );
    } else {
        moe_combine_backward(
            grads.d_expert_outputs.get<float>(),
            grads.d_routing_weights.get<float>(),
            grad_output.get<float>(),
            acts.expert_outputs.get<float>(),
            routing.routing_weights.get<float>(),
            acts.scatter_indices.get<int>(),
            BT, total_tokens, C, K, ctx.stream
        );
    }

    // Step 2: Backward through each expert
    // Note: grads.d_permuted_input should be pre-allocated
    grads.expert_grads.resize(E);
    int offset = 0;

    for (int e = 0; e < E; ++e) {
        const int* h_count_ptr = routing.token_counts.get<int>();
        int expert_tokens = h_count_ptr[e];

        if (expert_tokens == 0) continue;

        Tensor d_expert_output = slice(grads.d_expert_outputs, 0, offset, offset + expert_tokens);
        Tensor d_expert_input = slice(grads.d_permuted_input, 0, offset, offset + expert_tokens);

        mExperts[e].backward(ctx, w.experts[e], acts.expert_acts[e],
                             d_expert_output, grads.expert_grads[e], accumulate);

        offset += expert_tokens;
    }

    // Step 3: Unpermute gradient back to token order
    // Each original token receives gradients from K expert paths
    // Note: grads.d_input should be pre-allocated

    if (grads.d_permuted_input.DType == ETensorDType::BF16) {
        moe_permute_backward(
            grads.d_input.get<nv_bfloat16>(),
            grads.d_permuted_input.get<nv_bfloat16>(),
            acts.gather_indices.get<int>(),
            total_tokens, BT, C, K, ctx.stream
        );
    } else {
        moe_permute_backward(
            grads.d_input.get<float>(),
            grads.d_permuted_input.get<float>(),
            acts.gather_indices.get<int>(),
            total_tokens, BT, C, K, ctx.stream
        );
    }

    return grads.d_input;
}

/**
 * @brief Shared Expert - expert that processes all tokens (Nemotron/DeepSeek style)
 *
 * In some MoE architectures, a "shared expert" processes all tokens
 * in addition to the routed experts. This helps with representation quality.
 */
class SharedExpertModule : public ExpertModule {
public:
    struct Config : public ExpertModule::Config {
        float shared_expert_scale = 1.0f;  ///< Scale factor for shared expert output
    };

    explicit SharedExpertModule(Config config)
        : ExpertModule(static_cast<ExpertModule::Config&>(config))
        , mSharedConfig(config) {}

    /**
     * @brief Forward: run on all tokens (no routing)
     */
    Tensor forward_all(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        Tensor output = forward_impl(ctx, w, input, acts);

        // Scale output if configured
        if (mSharedConfig.shared_expert_scale != 1.0f) {
            // scale_tensor(output, mSharedConfig.shared_expert_scale, ctx.stream);
        }

        return output;
    }

private:
    Config mSharedConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_EXPERT_H
