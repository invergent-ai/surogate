// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_MOE_EXPERT_H
#define SUROGATE_SRC_MODULES_MOE_EXPERT_H

#include "base_expert.h"
#include "moe_types.h"
#include "modules/primitives/linear.h"
#include "modules/primitives/swiglu.h"
#include "modules/forward_hooks.h"
#include "modules/lora/lora_types.h"
#include "modules/lora/fast_expert_lora.h"
#include "kernels/kernels.h"
#include "base_router.h"  // For MoERouterOutput

namespace modules {

/**
 * @brief Single Expert MLP module
 *
 * An expert is essentially an MLP (same structure as the dense FFN).
 * In MoE, multiple experts share the same structure but have independent weights.
 *
 * Inherits from BaseExpertModule and composes the protected helper methods.
 *
 * Structure: input -> [gate_proj, up_proj] -> activation(gate) * up -> down_proj -> output
 */
class ExpertModule : public BaseExpertModule<ExpertModule> {
public:
    // Use base types
    using Config = BaseConfig;
    using Activations = BaseActivations;
    using Gradients = BaseGradients;

    /**
     * @brief Weight tensors for a single expert
     *
     * Note: Uses fused gate_proj layout where gate and up are concatenated.
     * This differs from BaseWeights for backwards compatibility.
     */
    struct Weights {
        Tensor gate_proj;           ///< (hidden_size, 2 * intermediate_size) - fused gate+up
        Tensor up_proj;             ///< (hidden_size, intermediate_size) - only used if not gated
        Tensor down_proj;           ///< (intermediate_size, hidden_size)
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
     * @brief Forward pass with LoRA hook for applying per-expert LoRA
     *
     * The hook is called at two points:
     * - AfterExpertUpProjection: After gate_up matmul, before SwiGLU activation
     * - AfterExpertDownProjection: After down_proj matmul
     *
     * The hook can modify acts.gate_up and acts.output in place to apply LoRA.
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations routed to this expert (N, hidden_size)
     * @param acts Activation storage
     * @param expert_hook Hook function called at specific points
     * @param layer_idx Layer index (passed to hook)
     * @param expert_idx Expert index (passed to hook)
     * @return Output (N, hidden_size)
     */
    Tensor forward_with_hook(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts,
                             const MoEExpertHook& expert_hook, int layer_idx, int expert_idx);

    /**
     * @brief Backward pass through expert MLP
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    /**
     * @brief Fast fused forward pass with LoRA for MoE experts.
     *
     * This method provides an optimized forward pass that:
     * 1. Fuses base expert computation with LoRA application
     * 2. Stores only e (gate output) and g (up output) instead of gate_up + activated
     * 3. Enables in-place backward computation for reduced memory traffic
     *
     * Memory savings: ~12% reduction in activation memory per expert
     * Performance: ~20-30% faster backward pass due to reduced bandwidth
     *
     * @param ctx Module context with cuBLAS handle and stream
     * @param w Expert base weights
     * @param input Expert input (N, C)
     * @param expert_lora LoRA weights for this expert
     * @param fast_state State to save for fast backward (e, g tensors)
     * @param output Output tensor (N, C)
     * @param scaling LoRA scaling factor
     * @param rank LoRA rank
     * @param intermediate Scratch tensor (N, rank)
     * @param h_buffer Scratch tensor (N, D) for h computation
     * @param gate_up_buffer Scratch tensor (N, 2*D) for gate_up projection
     * @param handle cuBLAS handle
     * @param workspace cuBLAS workspace
     */
    void forward_fast_lora(ModuleContext& ctx, Weights& w, Tensor& input,
                           const LoRAExpertWeights<Tensor>& expert_lora,
                           detail::FastExpertLoRAState& fast_state,
                           Tensor& output,
                           float scaling, int rank,
                           Tensor& intermediate,
                           Tensor& h_buffer,
                           Tensor& gate_up_buffer,
                           cublasLtHandle_t handle,
                           Tensor& workspace);

    /**
     * @brief Fast fused backward pass with LoRA for MoE experts.
     *
     * Computes:
     * - LoRA weight gradients (dA, dB for gate, up, down)
     * - Input gradient dx
     *
     * Key optimization: In-place SiLU backward overwrites e->de, g->dg,
     * eliminating the need to store h during forward.
     *
     * @param ctx Module context
     * @param w Expert base weights
     * @param expert_lora LoRA weights for this expert
     * @param expert_lora_grads LoRA gradient outputs
     * @param fast_state Saved state from forward (modified in-place)
     * @param dy Upstream gradient (N, C)
     * @param dx Input gradient output (N, C)
     * @param scaling LoRA scaling factor
     * @param rank LoRA rank
     * @param accumulate Whether to accumulate into gradient tensors
     * @param intermediate1 Scratch tensor (N, rank)
     * @param intermediate2 Scratch tensor (N, D)
     * @param d_gate_up_buffer Scratch tensor (N, 2*D)
     * @param handle cuBLAS handle
     * @param workspace cuBLAS workspace
     */
    void backward_fast_lora(ModuleContext& ctx, Weights& w,
                            const LoRAExpertWeights<Tensor>& expert_lora,
                            LoRAExpertWeights<Tensor>& expert_lora_grads,
                            detail::FastExpertLoRAState& fast_state,
                            const Tensor& dy,
                            Tensor& dx,
                            float scaling, int rank,
                            bool accumulate,
                            Tensor& intermediate1,
                            Tensor& intermediate2,
                            Tensor& d_gate_up_buffer,
                            cublasLtHandle_t handle,
                            Tensor& workspace);

    [[nodiscard]] const Config& config() const { return mConfig; }

private:
    Config mConfig;
};

// ============================================================================
// Implementation
// ============================================================================

inline Tensor ExpertModule::forward_impl(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {

    // Save input for backward
    acts.input = input;

    if (mConfig.use_gated) {
        // 1) Gated activation path: gate_up projection
        forward_gate_up_proj(ctx, mConfig, w.gate_proj, input, acts.gate_up);

        // 2) SwiGLU activation
        apply_swiglu(ctx, mConfig, acts.gate_up, acts.activated);
    } else {
        // Non-gated path: up projection only
        forward_up_proj(ctx, mConfig, w.up_proj, input, acts.gate_up);

        // Apply activation (GeLU) - placeholder
        // TODO: Add GeLU helper to base class
        acts.activated = acts.gate_up;
    }

    // 3) Down projection
    forward_down_proj(ctx, mConfig, w.down_proj, acts.activated, acts.output);

    return acts.output;
}

inline Tensor ExpertModule::forward_with_hook(
    ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts,
    const MoEExpertHook& expert_hook, int layer_idx, int expert_idx) {

    // Save input for backward
    acts.input = input;

    if (mConfig.use_gated) {
        // 1) Gated activation path: gate_up projection
        forward_gate_up_proj(ctx, mConfig, w.gate_proj, input, acts.gate_up);

        // Hook point: After gate_up projection, before SwiGLU
        // LoRA can modify acts.gate_up in place here
        if (expert_hook) {
            expert_hook(layer_idx, expert_idx, MoEExpertHookPoint::AfterExpertUpProjection, ctx.stream, nullptr);
        }

        // 2) SwiGLU activation
        apply_swiglu(ctx, mConfig, acts.gate_up, acts.activated);
    } else {
        // Non-gated path: up projection only
        forward_up_proj(ctx, mConfig, w.up_proj, input, acts.gate_up);

        if (expert_hook) {
            expert_hook(layer_idx, expert_idx, MoEExpertHookPoint::AfterExpertUpProjection, ctx.stream, nullptr);
        }

        acts.activated = acts.gate_up;
    }

    // 3) Down projection
    forward_down_proj(ctx, mConfig, w.down_proj, acts.activated, acts.output);

    // Hook point: After down projection
    // LoRA can modify acts.output in place here
    if (expert_hook) {
        expert_hook(layer_idx, expert_idx, MoEExpertHookPoint::AfterExpertDownProjection, ctx.stream, nullptr);
    }

    return acts.output;
}

inline Tensor ExpertModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    // 1) Backward through down projection
    Tensor d_activated = backward_down_proj(ctx, mConfig, w.down_proj, acts.activated,
                                             grad_output, grads.d_down_proj, accumulate);

    // 2) Backward through activation
    Tensor d_gate_up;
    d_gate_up.DType = d_activated.DType;
    if (mConfig.use_gated) {
        apply_swiglu_backward(ctx, mConfig, d_activated, acts.gate_up, d_gate_up);
    } else {
        // GeLU backward - placeholder
        d_gate_up = d_activated;
    }

    // 3) Backward through gate/up projection
    return backward_gate_up_proj(ctx, mConfig, w.gate_proj, acts.input, d_gate_up,
                                  grads.d_gate_proj, accumulate);
}

inline void ExpertModule::forward_fast_lora(
    ModuleContext& ctx, Weights& w, Tensor& input,
    const LoRAExpertWeights<Tensor>& expert_lora,
    detail::FastExpertLoRAState& fast_state,
    Tensor& output,
    float scaling, int rank,
    Tensor& intermediate,
    Tensor& h_buffer,
    Tensor& gate_up_buffer,
    cublasLtHandle_t handle,
    Tensor& workspace) {

    const int N = input.Sizes[0];
    const int C = mConfig.hidden_size;
    const int D = mConfig.intermediate_size;

    detail::fast_expert_lora_forward(
        output, input,
        w.gate_proj, w.down_proj,
        expert_lora,
        fast_state,
        scaling,
        N, C, D, rank,
        intermediate,
        h_buffer,
        gate_up_buffer,
        handle,
        workspace,
        ctx.stream);
}

inline void ExpertModule::backward_fast_lora(
    ModuleContext& ctx, Weights& w,
    const LoRAExpertWeights<Tensor>& expert_lora,
    LoRAExpertWeights<Tensor>& expert_lora_grads,
    detail::FastExpertLoRAState& fast_state,
    const Tensor& dy,
    Tensor& dx,
    float scaling, int rank,
    bool accumulate,
    Tensor& intermediate1,
    Tensor& intermediate2,
    Tensor& d_gate_up_buffer,
    cublasLtHandle_t handle,
    Tensor& workspace) {

    detail::fast_expert_lora_backward(
        expert_lora_grads,
        dx,
        dy,
        w.gate_proj, w.down_proj,
        expert_lora,
        fast_state,
        scaling,
        rank,
        accumulate,
        intermediate1,
        intermediate2,
        d_gate_up_buffer,
        handle,
        workspace,
        ctx.stream);
}

/**
 * @brief Expert group - collection of experts with shared dispatch logic
 *
 * Manages multiple experts and handles the scatter/gather operations
 * for routing tokens to experts and combining outputs.
 *
 * Inherits from BaseExpertGroupModule and composes the protected helper methods.
 *
 * Uses a permute-based approach:
 * 1. Permute tokens to expert-grouped order (scatter)
 * 2. Run batched expert computation
 * 3. Unpermute and weight-combine outputs (gather)
 *
 * This approach enables efficient grouped GEMM computation across all experts.
 */
class ExpertGroupModule : public BaseExpertGroupModule<ExpertGroupModule> {
public:
    // Use base config
    using Config = BaseConfig;

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
     *
     * Extends BaseActivations with per-expert activation storage.
     */
    struct Activations : public BaseActivations {
        std::vector<ExpertModule::Activations> expert_acts;
    };

    /**
     * @brief Gradients for all experts
     *
     * Extends BaseGradients with per-expert gradient storage.
     */
    struct Gradients : public BaseGradients {
        // Per-expert gradients (for sequential execution)
        std::vector<ExpertModule::Gradients> expert_grads;
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
                        const MoERouterOutput& routing, Activations& acts);

    /**
     * @brief Forward pass with per-expert hook for LoRA application
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input Token representations (B*T, hidden_size)
     * @param routing Router output with dispatch information
     * @param acts Activation storage
     * @param expert_hook Called after each expert's forward pass (layer_idx, expert_idx, point, stream)
     * @param layer_idx The layer index (passed to hook)
     * @return Combined output (B*T, hidden_size)
     */
    Tensor forward_with_hook(ModuleContext& ctx, Weights& w, Tensor& input,
                             const MoERouterOutput& routing, Activations& acts,
                             const MoEExpertHook& expert_hook, int layer_idx);

    /**
     * @brief Backward pass through expert group
     */
    Tensor backward_impl(ModuleContext& ctx, Weights& w, Activations& acts,
                         const MoERouterOutput& routing,
                         Tensor& grad_output, Gradients& grads, bool accumulate = false);

    [[nodiscard]] const Config& config() const { return mConfig; }

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
    const MoERouterOutput& routing, Activations& acts) {

    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Step 1: Permute tokens to expert-grouped order using base class helper
    permute_tokens(ctx, mConfig, input, acts.gather_indices, acts.permuted_input, K);

    // Step 2: Run experts on permuted tokens
    // Note: Grouped GEMM is used in modular_model_forward.hpp for the main training path.
    // This fallback uses sequential per-expert execution for the ModuleContext-based API.
    acts.expert_acts.resize(E);
    int offset = 0;

    for (int e = 0; e < E; ++e) {
        const int* h_count_ptr = routing.token_counts.get<int>();
        int expert_tokens = h_count_ptr[e];

        if (expert_tokens == 0) continue;

        Tensor expert_input = slice(acts.permuted_input, 0, offset, offset + expert_tokens);
        Tensor expert_output = slice(acts.expert_outputs, 0, offset, offset + expert_tokens);

        mExperts[e].forward(ctx, w.experts[e], expert_input, acts.expert_acts[e]);

        offset += expert_tokens;
    }

    // Step 3: Unpermute and combine using base class helper
    unpermute_and_combine(ctx, mConfig, acts.expert_outputs, routing.routing_weights,
                          acts.scatter_indices, acts.combined_output, K);

    return acts.combined_output;
}

inline Tensor ExpertGroupModule::forward_with_hook(
    ModuleContext& ctx, Weights& w, Tensor& input,
    const MoERouterOutput& routing, Activations& acts,
    const MoEExpertHook& expert_hook, int layer_idx) {

    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Step 1: Permute tokens using base class helper
    permute_tokens(ctx, mConfig, input, acts.gather_indices, acts.permuted_input, K);

    // Step 2: Try grouped manual hook (Fast Path)
    // If a manual group hook is provided (e.g., from ModularLoRAModel),
    // it can handle all experts in one fused call.
    if (expert_hook) {
        // We need expert offsets for grouped execution context
        if (acts.expert_offsets.is_null()) {
             // Fallback: this should ideally be handled by caller or activations allocation
        } else {
            moe_compute_expert_offsets(
                acts.expert_offsets.get<int>(),
                routing.token_counts.get<int>(),
                E, ctx.stream
            );

            MoEGroupedContext moe_ctx;
            moe_ctx.expert_offsets = &acts.expert_offsets;
            moe_ctx.permuted_input = &acts.permuted_input;
            moe_ctx.expert_outputs = &acts.expert_outputs;
            moe_ctx.expert_gate_up = &acts.expert_gate_up;  // Might be null
            moe_ctx.num_experts = E;
            moe_ctx.top_k = K;
            moe_ctx.total_tokens = acts.permuted_input.Sizes[0];
            moe_ctx.handled = false;

            expert_hook(layer_idx, -1, MoEExpertHookPoint::ManualGroup, ctx.stream, &moe_ctx);

            if (moe_ctx.handled) {
                // Step 3: Unpermute and combine using base class helper
                unpermute_and_combine(ctx, mConfig, acts.expert_outputs, routing.routing_weights,
                                      acts.scatter_indices, acts.combined_output, K);
                return acts.combined_output;
            }
        }
    }

    // Step 3: Fallback - Run experts on permuted tokens with hooks
    acts.expert_acts.resize(E);
    int offset = 0;

    for (int e = 0; e < E; ++e) {
        const int* h_count_ptr = routing.token_counts.get<int>();
        int expert_tokens = h_count_ptr[e];

        if (expert_tokens == 0) continue;

        // Slice of permuted input for this expert
        Tensor expert_input = slice(acts.permuted_input, 0, offset, offset + expert_tokens);
        Tensor expert_output = slice(acts.expert_outputs, 0, offset, offset + expert_tokens);

        // Run expert forward with hooks
        mExperts[e].forward_with_hook(ctx, w.experts[e], expert_input, acts.expert_acts[e],
                                       expert_hook, layer_idx, e);

        offset += expert_tokens;
    }

    // Step 3: Unpermute and combine using base class helper
    unpermute_and_combine(ctx, mConfig, acts.expert_outputs, routing.routing_weights,
                          acts.scatter_indices, acts.combined_output, K);

    return acts.combined_output;
}

inline Tensor ExpertGroupModule::backward_impl(
    ModuleContext& ctx, Weights& w, Activations& acts,
    const MoERouterOutput& routing,
    Tensor& grad_output, Gradients& grads, bool accumulate) {

    const int E = mConfig.num_experts;
    const int K = mConfig.top_k;

    // Step 1: Backward through unpermute+combine using base class helper
    combine_backward(ctx, mConfig, grad_output, acts.expert_outputs, routing.routing_weights,
                     acts.scatter_indices, grads.d_expert_outputs, grads.d_routing_weights, K);

    // Step 2: Backward through each expert
    // Note: Grouped GEMM backward is used in modular_model_block_ops.hpp for the main training path.
    // This fallback uses sequential per-expert backward for the ModuleContext-based API.
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

    // Step 3: Backward through token permutation using base class helper
    permute_backward(ctx, mConfig, grads.d_permuted_input, acts.gather_indices, grads.d_input, K);

    return grads.d_input;
}

/**
 * @brief Shared Expert - expert that processes all tokens (Nemotron/DeepSeek/Qwen3 style)
 *
 * In some MoE architectures, a "shared expert" processes all tokens
 * in addition to the routed experts. This helps with representation quality.
 *
 * Unlike routed experts, the shared expert:
 * - Processes ALL tokens (not just those routed to it)
 * - Has its output added to the combined routed expert output
 * - May have a different intermediate size than routed experts
 *
 * Inherits from ExpertModule to reuse the MLP computation.
 */
class SharedExpertModule : public ExpertModule {
public:
    /**
     * @brief Configuration for shared expert
     *
     * Extends ExpertModule::Config with shared-expert-specific settings.
     */
    struct Config : public ExpertModule::Config {
        float shared_expert_scale = 1.0f;  ///< Scale factor for shared expert output
    };

    explicit SharedExpertModule(Config config)
        : ExpertModule(static_cast<ExpertModule::Config&>(config))
        , mSharedConfig(config) {}

    /**
     * @brief Forward: run on all tokens (no routing)
     *
     * Unlike routed experts which only process assigned tokens,
     * the shared expert processes ALL tokens unconditionally.
     *
     * @param ctx Module context
     * @param w Expert weights
     * @param input ALL token representations (B*T, hidden_size)
     * @param acts Activation storage
     * @return Output (B*T, hidden_size) to be added to routed expert output
     */
    Tensor forward_all(ModuleContext& ctx, Weights& w, Tensor& input, Activations& acts) {
        // Use base class forward_impl for the MLP computation
        Tensor output = forward_impl(ctx, w, input, acts);

        // Scale output if configured (e.g., for DeepSeek-style shared expert scaling)
        if (mSharedConfig.shared_expert_scale != 1.0f) {
            // TODO: Add scale_tensor kernel call
            // scale_tensor(output, mSharedConfig.shared_expert_scale, ctx.stream);
        }

        return output;
    }

    [[nodiscard]] float shared_expert_scale() const { return mSharedConfig.shared_expert_scale; }

private:
    Config mSharedConfig;
};

} // namespace modules

#endif // SUROGATE_SRC_MODULES_MOE_EXPERT_H
