// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SUROGATE_SRC_MODULES_RUN_STATE_H
#define SUROGATE_SRC_MODULES_RUN_STATE_H

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <fmt/format.h>

#include "fp8_scaling_config.h"
#include "fp8_scaling_state.h"
#include "module_concept.h"
#include "run_state_types.h"
#include "run_state_config.h"
#include "residual_manager.h"
#include "modules/matmul_context.h"
#include "training/model.h"  // For IRunState base class
#include "utilities/allocator.h"
#include "utilities/tensor.h"
#include "utilities/stack.h"
#include "utilities/utils.h"
#include "kernels/kernels.h"  // For precompute_freqs_cis

namespace modules {

/**
 * @brief Modular run state manager template
 *
 * Manages activation tensors for a model composed of modular blocks. Handles:
 * - Per-layer activation storage during forward pass
 * - Per-layer gradient buffers during backward pass
 * - Residual offloading for memory efficiency
 * - Double-buffered prefetching for offloaded residuals
 *
 * The run state provides storage that mirrors the block structure,
 * enabling layer-by-layer processing with recomputation support.
 *
 * @tparam Block The transformer block module type
 */
template<typename Block>
class ModularRunState : public IRunState {
public:
    using BlockActivations = typename Block::Activations;
    using BlockGradients = typename Block::Gradients;

    /**
     * @brief Configuration for run state manager
     */
    using Config = ModularRunStateConfig<typename Block::Config>;

    ModularRunState(const Config& config, DeviceMemoryStack& stack,
                    const std::shared_ptr<TensorAllocator>& allocator);
    ~ModularRunState();

    // Non-copyable
    ModularRunState(const ModularRunState&) = delete;
    ModularRunState& operator=(const ModularRunState&) = delete;

    // Movable
    ModularRunState(ModularRunState&& other) noexcept;
    ModularRunState& operator=(ModularRunState&& other) noexcept;

    // ========================================================================
    // Block activation access
    // ========================================================================

    /**
     * @brief Get activation storage for a block
     *
     * Returns mutable reference to activation storage for the given layer.
     * Storage persists across forward pass for use in backward.
     */
    BlockActivations& get_block_activations(int layer_idx) {
        return mBlockActivations[layer_idx];
    }

    /**
     * @brief Get gradient buffer for a block
     *
     * Returns mutable reference to gradient buffer for the given layer.
     * Used during backward pass computation.
     */
    BlockGradients& get_block_gradients(int layer_idx) {
        return mBlockGradients[layer_idx];
    }

    // ========================================================================
    // Residual management (with optional offloading)
    // ========================================================================

    /**
     * @brief Initiate prefetch of residual from CPU
     *
     * Only does work if residuals are offloaded to CPU.
     */
    void fetch_residual(int layer_idx, cudaStream_t fetch_stream);

    /**
     * @brief Store residual to CPU
     *
     * Only does work if residuals are offloaded to CPU.
     */
    void put_residual(int layer_idx, cudaStream_t put_stream);

    /**
     * @brief Get residual tensor for a layer
     *
     * Waits for any pending fetch to complete.
     */
    Tensor& get_residual(int layer_idx, cudaStream_t stream);

    /**
     * @brief Get the final residual buffer (for output of fused_residual_rmsnorm of last layer).
     *
     * This buffer must not alias any per-layer residual buffers, because those are
     * needed for backward (LN1 backward for layer l consumes res_ffn(l-1)).
     */
    Tensor& get_final_residual() {
        return mResidualManager->get_final_residual();
    }

    /**
     * @brief Mark residual as ready for offloading
     */
    void mark_residual_ready(int layer_idx, cudaStream_t stream);

    /**
     * @brief Release residual (allow buffer reuse)
     */
    void release_residual(int layer_idx, cudaStream_t stream);

    // ========================================================================
    // MLP up buffer management (memory optimization)
    // ========================================================================

    /**
     * @brief Acquire MLP up buffer for a layer
     *
     * For memory efficiency, MLP up tensors can be shared between layers
     * when using recomputation (only one needed at a time).
     */
    Tensor acquire_mlp_up(int layer_idx);

    /**
     * @brief Release MLP up buffer
     */
    void release_mlp_up(Tensor& mlp_up);

    // ========================================================================
    // Non-block state access
    // ========================================================================

    NonBlockActivations& non_block_activations() { return mNonBlockActivations; }
    NonBlockGradientBuffers& non_block_gradients() { return mNonBlockGradients; }
    ScratchBuffers& scratch() { return mScratch; }
    [[nodiscard]] bool ffn_temps_on_stack() const { return mConfig.recompute_block; }
    [[nodiscard]] bool large_bwd_temps_on_stack() const { return mConfig.recompute_block; }

    /**
     * @brief Get simplified layer activations for a block
     *
     * Used for the initial forward/backward implementation until
     * the full modular activation system is in place.
     */
    SimplifiedLayerActivations& simplified_acts(int layer_idx) {
        return mSimplifiedActivations[layer_idx];
    }

    SimplifiedLayerGradients& simplified_grads(int layer_idx) {
        return mSimplifiedGradients[layer_idx];
    }

    SimplifiedLayerQuantActivations& simplified_quant_acts(int layer_idx) {
        return mSimplifiedQuantActivations[layer_idx];
    }

    SimplifiedQuantGradients& simplified_quant_grads() {
        return mSimplifiedQuantGrads;
    }

    [[nodiscard]] bool is_lora_only_mode() const { return mConfig.lora_only_mode; }
    [[nodiscard]] bool is_train_router() const { return mConfig.train_router; }

    // ========================================================================
    // IRunState virtual method overrides (for recipe-driven dispatch)
    // ========================================================================

    [[nodiscard]] bool has_activation_quants() const override { return mConfig.matmul_dtype != mConfig.activation_dtype; }
    [[nodiscard]] bool has_grad_quants() const override { return mConfig.grad_quant_dtype != mConfig.grad_dtype; }

    // FP8 forward-only mode accessors
    [[nodiscard]] bool has_fp8_forward() const override { return mConfig.enable_fp8_forward; }
    FP8ForwardQuantActivations& fp8_forward_quants() { return mFP8ForwardQuants; }

    // FP8 HYBRID backward mode: uses E4M3 weights × E5M2 gradients for backward matmuls
    // Enabled when: enable_fp8_forward=true AND grad_quant_dtype=E5M2
    [[nodiscard]] bool has_fp8_hybrid_backward() const override {
        return mConfig.enable_fp8_forward && mConfig.grad_quant_dtype == ETensorDType::FP8_E5M2;
    }

    // FP8 delayed scaling state (TransformerEngine-style)
    // Enabled automatically when FP8 HYBRID mode is active
    [[nodiscard]] bool has_fp8_delayed_scaling() const override { return mFP8ScalingState != nullptr; }
    [[nodiscard]] FP8ScalingState* get_fp8_scaling_state() override { return mFP8ScalingState.get(); }
    FP8ScalingState& fp8_scaling_state() { return *mFP8ScalingState; }
    const FP8ScalingState& fp8_scaling_state() const { return *mFP8ScalingState; }

    // FP4 forward-only mode accessors (Blackwell SM100+)
    [[nodiscard]] bool has_fp4_forward() const override {
        return mConfig.enable_fp4_forward || mConfig.enable_fp4_backward;
    }
    [[nodiscard]] bool has_fp4_backward() const override { return mConfig.enable_fp4_backward; }
    FP4ForwardQuantActivations& fp4_forward_quants() { return mFP4ForwardQuants; }

    /// @brief Provide FP4 forward buffers to recipes (data, scales, global amax).
    ///
    /// Note: The returned (data, scales) buffers are allocated for the cuDNN FP4 layout.
    /// Recipes that only need the global amax (e.g., CUTLASS NVFP4) can still reuse the
    /// amax pointer to avoid a separate abs_max reduction.
    [[nodiscard]] std::tuple<Tensor*, Tensor*, float*> get_fp4_forward_buffers(int op) override {
        if (!has_fp4_forward()) return {nullptr, nullptr, nullptr};
        const auto matmul_op = static_cast<MatmulOp>(op);
        switch (matmul_op) {
            case MatmulOp::QKV:
                return {&mFP4ForwardQuants.ln1_data, &mFP4ForwardQuants.ln1_scales, mFP4ForwardQuants.ln1_global_amax};
            case MatmulOp::MLPUp:
                return {&mFP4ForwardQuants.ln2_data, &mFP4ForwardQuants.ln2_scales, mFP4ForwardQuants.ln2_global_amax};
            case MatmulOp::AttnOut:
                return {&mFP4ForwardQuants.att_data, &mFP4ForwardQuants.att_scales, mFP4ForwardQuants.att_global_amax};
            case MatmulOp::MLPDown:
                return {&mFP4ForwardQuants.swiglu_data, &mFP4ForwardQuants.swiglu_scales, mFP4ForwardQuants.swiglu_global_amax};
            default:
                return {nullptr, nullptr, nullptr};
        }
    }

    [[nodiscard]] Tensor* get_hadamard_workspace() override {
        if (!has_fp4_forward()) return nullptr;
        if (!mFP4ForwardQuants.hadamard_workspace.Data) return nullptr;
        return &mFP4ForwardQuants.hadamard_workspace;
    }

    // ========================================================================
    // CUDA synchronization resources
    // ========================================================================

    cudaStream_t side_stream() const { return mSideStream; }
    cudaEvent_t side_stream_event() const { return mSideStreamEvent; }
    cudaEvent_t opt_embeddings_done() const { return mOptEmbeddingsDone; }
    cudaEvent_t layer_update_done(int layer_idx) const { return mLayerUpdateDone[layer_idx]; }

    // cuBLAS handle for grouped GEMM operations (MoE experts)
    // Public member for direct access (matches pattern of IRunState::MainStream)
    cublasHandle_t CublasHandle = nullptr;

    // MoE expert offsets cached on host to avoid repeated D2H sync in grouped GEMM
    // Set once per forward pass after moe_compute_expert_offsets, reused for all GEMMs
    std::vector<int> MoeHostExpertOffsets;
    bool MoeHostOffsetsValid = false;  // Set to false at start of each forward pass

    // CUDA graphs (exec instances live in the run state so they persist across steps).
    cudaGraphExec_t& forward_block_graph(int layer_idx) { return mForwardBlockGraphs.at((std::size_t)layer_idx); }
    // Select backward graph based on accumulate flag: [0]=first micro-step, [1]=accumulating
    cudaGraphExec_t& backward_block_graph(int layer_idx, bool accumulate) {
        return mBackwardBlockGraphs.at((std::size_t)layer_idx)[accumulate ? 1 : 0];
    }
    cudaGraphExec_t& global_norm_graph() { return mGlobalNormGraph; }

    // Stack checkpoints for graphs (ensures temp_alloc returns consistent addresses on graph replay).
    DeviceMemoryStack::Checkpoint& forward_block_stack_checkpoint(int layer_idx) {
        return mForwardBlockStackCheckpoints.at((std::size_t)layer_idx);
    }
    DeviceMemoryStack::Checkpoint& backward_block_stack_checkpoint(int layer_idx, bool accumulate) {
        return mBackwardBlockStackCheckpoints.at((std::size_t)layer_idx)[accumulate ? 1 : 0];
    }

    void configure_forward_graphs(bool hooked) {
        if (mForwardGraphsHooked == hooked) return;
        reset_graphs(mForwardBlockGraphs);
        mForwardGraphsHooked = hooked;
    }

    void configure_backward_graphs(bool hooked) {
        if (mBackwardGraphsHooked == hooked) return;
        reset_graphs(mBackwardBlockGraphs);
        mBackwardGraphsHooked = hooked;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    [[nodiscard]] const Config& config() const { return mConfig; }
    [[nodiscard]] int batch_size() const { return mConfig.batch_size; }
    [[nodiscard]] int seq_length() const { return mConfig.seq_length; }

    // ========================================================================
    // MoE Metrics
    // ========================================================================

    /// @brief Accumulate MoE stats from a single layer
    void accumulate_moe_stats(float aux_loss, float z_loss, int num_experts, const int* token_counts, cudaStream_t stream) {
        mMoEStats.aux_loss += aux_loss;
        mMoEStats.z_loss += z_loss;
        mMoEStats.num_layers++;
        mMoEStats.valid = true;

        // Copy token counts to host and compute utilization/imbalance
        if (num_experts > 0 && token_counts != nullptr) {
            std::vector<int> h_counts(num_experts);
            CUDA_CHECK(cudaMemcpyAsync(h_counts.data(), token_counts, num_experts * sizeof(int),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            int max_count = 0, total_count = 0, used_experts = 0;
            for (int e = 0; e < num_experts; ++e) {
                if (h_counts[e] > 0) {
                    used_experts++;
                    max_count = std::max(max_count, h_counts[e]);
                }
                total_count += h_counts[e];
            }

            float utilization = static_cast<float>(used_experts) / static_cast<float>(num_experts);
            float mean_count = static_cast<float>(total_count) / static_cast<float>(num_experts);
            float imbalance = (mean_count > 0) ? static_cast<float>(max_count) / mean_count : 1.0f;

            // Running average across layers
            float n = static_cast<float>(mMoEStats.num_layers);
            mMoEStats.expert_utilization = ((n - 1) * mMoEStats.expert_utilization + utilization) / n;
            mMoEStats.load_imbalance = ((n - 1) * mMoEStats.load_imbalance + imbalance) / n;
        }
    }

    /// @brief Get accumulated MoE stats
    [[nodiscard]] MoEStats get_moe_stats() const override { return mMoEStats; }

    /// @brief Reset MoE stats for next step
    void reset_moe_stats() override { mMoEStats = MoEStats{}; }

    /// @brief Check if this is an MoE model
    [[nodiscard]] bool is_moe_model() const override { return mConfig.is_moe; }

private:
    Config mConfig;
    std::shared_ptr<TensorAllocator> mAllocator;

    // MoE stats (accumulated across layers during forward pass)
    MoEStats mMoEStats{};

    // Per-block storage
    std::vector<BlockActivations> mBlockActivations;
    std::vector<BlockGradients> mBlockGradients;

    // Simplified layer activations (for initial forward/backward implementation)
    std::vector<SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<SimplifiedLayerGradients> mSimplifiedGradients;
    std::vector<SimplifiedLayerQuantActivations> mSimplifiedQuantActivations;
    SimplifiedQuantGradients mSimplifiedQuantGrads{};
    Tensor mQuantStats{};

    // FP8 forward-only buffers (shared across layers, transient)
    FP8ForwardQuantActivations mFP8ForwardQuants{};
    Tensor mFP8ForwardStats{};  ///< Stats buffer for FP8 forward quants (4 pairs = 8 floats)

    // FP8 delayed scaling state (TransformerEngine-style)
    // Allocated when FP8 HYBRID mode is enabled
    std::unique_ptr<FP8ScalingState> mFP8ScalingState;

    // FP4 forward buffers (shared across layers, transient)
    // FP4 uses E2M1 format with two-level block scaling (requires Blackwell SM100+)
    FP4ForwardQuantActivations mFP4ForwardQuants{};

    // Shared activation buffers for recomputation modes (checkpointing).
    // These are used to avoid storing large per-layer intermediates when recomputation is enabled.
    Tensor mSharedLn1;
    Tensor mSharedLn2;
    Tensor mSharedQKV;
    Tensor mSharedAtt;
    Tensor mSharedAttOut;
    Tensor mSharedMlpUp;
    Tensor mSharedSwiGlu;
    Tensor mSharedResidualAtt;
    Tensor mSharedMlpDown;
    // Shared rstd buffers for LoRA-only mode (can reuse across layers since we don't need per-layer values)
    Tensor mSharedLn1Rstd;
    Tensor mSharedLn2Rstd;
    Tensor mSharedQRstd;
    Tensor mSharedKRstd;
    // Shared activation quant buffers (FP8/int8) for matmuls.
    Tensor mSharedLn1Quant;
    Tensor mSharedLn2Quant;
    Tensor mSharedAttQuant;
    Tensor mSharedSwiGluQuant;
    std::array<Tensor, 2> mSharedDResFFN{};
    Tensor mSharedDResAtt;
    Tensor mSharedDLn2;
    Tensor mSharedDMlpUp;
    Tensor mSharedDSwiGlu;
    Tensor mSharedDAtt;
    Tensor mSharedDQKV;
    Tensor mSharedDLn1;

    // Non-block storage
    NonBlockActivations mNonBlockActivations;
    NonBlockGradientBuffers mNonBlockGradients;
    ScratchBuffers mScratch;

    // Residual management
    std::unique_ptr<ResidualManager> mResidualManager;

    // MLP up buffer pool (for memory optimization with recomputation)
    std::vector<Tensor> mMlpUpPool;
    std::vector<bool> mMlpUpInUse;

    // CUDA synchronization
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
    cudaEvent_t mOptEmbeddingsDone = nullptr;
    std::vector<cudaEvent_t> mLayerUpdateDone;

    // CUDA graph executables (optional, enabled by ModelOptions::use_cuda_graphs).
    std::vector<cudaGraphExec_t> mForwardBlockGraphs;
    // Two backward graphs per layer: [0] for accumulate=false (first micro-step),
    // [1] for accumulate=true (subsequent micro-steps with gradient accumulation).
    std::vector<std::array<cudaGraphExec_t, 2>> mBackwardBlockGraphs;
    cudaGraphExec_t mGlobalNormGraph = nullptr;
    bool mForwardGraphsHooked = false;
    bool mBackwardGraphsHooked = false;

    // Stack checkpoints for CUDA graph compatibility.
    // When graphs use temp_alloc, we must restore the stack to the same state
    // before each graph replay so that allocations return consistent addresses.
    std::vector<DeviceMemoryStack::Checkpoint> mForwardBlockStackCheckpoints;
    // Two checkpoints per layer corresponding to the two backward graphs.
    std::vector<std::array<DeviceMemoryStack::Checkpoint, 2>> mBackwardBlockStackCheckpoints;

    static void reset_graphs(std::vector<cudaGraphExec_t>& graphs) noexcept {
        for (auto& g : graphs) {
            if (g) {
                (void)cudaGraphExecDestroy(g);
                g = nullptr;
            }
        }
    }

    static void reset_graphs(std::vector<std::array<cudaGraphExec_t, 2>>& graphs) noexcept {
        for (auto& arr : graphs) {
            for (auto& g : arr) {
                if (g) {
                    (void)cudaGraphExecDestroy(g);
                    g = nullptr;
                }
            }
        }
    }

    // Internal helpers
    void allocate_block_activations(BlockActivations& acts);
    void allocate_block_gradients(BlockGradients& grads);
    void allocate_simplified_activations();
    void allocate_simplified_gradients();
    void allocate_simplified_quant_buffers();
    void allocate_non_block_state();
    void allocate_scratch_buffers(DeviceMemoryStack& stack);
    void allocate_residual_buffers();
    void create_cuda_resources();
    void release_cuda_resources() noexcept;
};

// ============================================================================
// Implementation
// ============================================================================

} // namespace modules

#include "run_state_impl.tpp"

#endif // SUROGATE_SRC_MODULES_RUN_STATE_H
