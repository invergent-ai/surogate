// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL run state - activation buffers, scratch, and CUDA graph management.

#ifndef SUROGATE_SRC_DSL_DSL_RUN_STATE_H
#define SUROGATE_SRC_DSL_DSL_RUN_STATE_H

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "runtime/dsl/dsl_runtime_config.h"
#include "runtime/dsl/tensor_slot_registry.h"
#include "utilities/stack.h"
#include "runtime/core/run_state_types.h"
#include "runtime/core/residual_manager.h"
#include "runtime/training/model.h"
#include "runtime/training/runtime_options.h"
#include "utilities/allocator.h"
namespace modules { class FP8ScalingState; }

namespace dsl {

// DSL run state for graph execution (activation buffers, scratch, etc).
class DslRunState final : public IRunState {
public:
    static constexpr std::size_t kDefaultStackBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2GB

    DslRunState(const PretrainedConfig& config,
                const DslRuntimeConfig& runtime_config,
                const RuntimeOptions& options,
                int B, int T,
                const std::shared_ptr<TensorAllocator>& allocator,
                bool lora_only_mode = false,
                bool prequantized = false,
                std::size_t stack_bytes = kDefaultStackBytes,
                bool allocate_stack = true,
                const ActivationLayoutIR* activation_layout = nullptr);
    ~DslRunState();

    void set_stack_buffer(Tensor buffer, const DeviceMemoryStack::AllocationList& high_mark = {});

    modules::SimplifiedLayerActivations& simplified_acts(int layer_idx) { return mSimplifiedActivations[layer_idx]; }
    modules::SimplifiedLayerGradients& simplified_grads(int layer_idx) { return mSimplifiedGradients[layer_idx]; }
    modules::SimplifiedQuantGradients& simplified_quant_grads() { return mSimplifiedQuantGrads; }
    modules::FP8ForwardQuantActivations& fp8_forward_quants() { return mFP8ForwardQuants; }

    void reset_simplified_gradients();

    /// @brief Zero all activation gradient buffers (d_res_ffn, d_res_att) for all layers.
    /// Call this at the start of each backward pass to prevent stale gradients from accumulating.
    void zero_activation_gradients(cudaStream_t stream);

    /// @brief Get the number of layers for gradient iteration
    std::size_t num_gradient_layers() const { return mSimplifiedGradients.size(); }

    modules::NonBlockActivations& non_block_activations() { return mNonBlockActivations; }
    modules::NonBlockGradientBuffers& non_block_gradients() { return mNonBlockGradients; }
    modules::ScratchBuffers& scratch() { return mScratch; }

    Tensor& get_residual(int layer_idx, cudaStream_t stream);
    Tensor& get_final_residual();

    // Residual offloading support (for large model training)
    void fetch_residual(int layer_idx, cudaStream_t stream);
    void put_residual(int layer_idx, cudaStream_t stream);
    void mark_residual_ready(int layer_idx, cudaStream_t stream);
    void release_residual(int layer_idx, cudaStream_t stream);
    bool has_residual_offloading() const { return mOffloadResiduals; }

    bool ffn_temps_on_stack() const { return mFfnTempsOnStack; }
    bool large_bwd_temps_on_stack() const { return mRecomputeLevel >= RecomputeLevel::Enabled; }
    bool is_lora_only_mode() const { return mLoraOnlyMode; }
    bool is_prequantized() const { return mPrequantized; }

    /// @brief Get the recompute level
    RecomputeLevel recompute_level() const { return mRecomputeLevel; }

    /// @brief Check if any recomputation is enabled
    bool recompute_enabled() const { return mRecomputeLevel != RecomputeLevel::None; }

    /// @brief Get temporary rstd buffer for recomputation (avoids overwriting saved values)
    Tensor& recompute_rstd() { return mRecomputeRstd; }

    /// @brief Get temporary LSE buffer for recomputation (avoids overwriting saved values)
    Tensor& recompute_lse() { return mRecomputeLSE; }

    cudaStream_t side_stream() const { return mSideStream; }
    cudaEvent_t side_stream_event() const { return mSideStreamEvent; }
    cudaEvent_t all_reduce_done_event() const { return mAllReduceDone; }
    cublasHandle_t cublas_handle() const { return mCublasHandle; }

    // ========================================================================
    // Per-layer CUDA graphs (for efficient layer-by-layer execution)
    // ========================================================================

    /// @brief Get forward CUDA graph exec for a layer (nullptr if not captured)
    cudaGraphExec_t& forward_block_graph(int layer_idx) {
        return mForwardBlockGraphs.at(static_cast<std::size_t>(layer_idx));
    }

    /// @brief Get backward CUDA graph exec for a layer
    /// @param layer_idx Layer index
    /// @param accumulate If true, returns graph for gradient accumulation mode
    cudaGraphExec_t& backward_block_graph(int layer_idx, bool accumulate) {
        return mBackwardBlockGraphs.at(static_cast<std::size_t>(layer_idx))[accumulate ? 1 : 0];
    }

    /// @brief Get stack checkpoint for forward graph replay (ensures consistent temp addresses)
    DeviceMemoryStack::Checkpoint& forward_block_stack_checkpoint(int layer_idx) {
        return mForwardBlockStackCheckpoints.at(static_cast<std::size_t>(layer_idx));
    }

    /// @brief Get stack checkpoint for backward graph replay
    DeviceMemoryStack::Checkpoint& backward_block_stack_checkpoint(int layer_idx, bool accumulate) {
        return mBackwardBlockStackCheckpoints.at(static_cast<std::size_t>(layer_idx))[accumulate ? 1 : 0];
    }

    /// @brief Reconfigure forward graphs when hook presence changes
    /// Destroys existing graphs to force re-capture with new topology
    void configure_forward_graphs(bool hooked);

    /// @brief Reconfigure backward graphs when hook presence changes
    void configure_backward_graphs(bool hooked);

    /// @brief Reset all CUDA graphs (call when batch/sequence dimensions change)
    void reset_cuda_graphs();

    /// @brief Check if per-layer CUDA graphs are enabled
    bool per_layer_graphs_enabled() const { return mPerLayerGraphsEnabled; }

    /// @brief Enable/disable per-layer CUDA graphs
    void set_per_layer_graphs_enabled(bool enabled) { mPerLayerGraphsEnabled = enabled; }

    /// @brief Get number of layers (for graph array sizing)
    int num_layers() const { return mNumLayers; }

    // IRunState overrides (quantization unsupported in DSL runtime for now).
    [[nodiscard]] bool has_activation_quants() const override { return mMatmulDtype != mActivationDtype; }
    [[nodiscard]] bool has_grad_quants() const override { return mGradQuantDtype != mGradDtype; }
    [[nodiscard]] bool has_fp8_forward() const override { return mEnableFp8Forward; }
    [[nodiscard]] bool has_fp8_hybrid_backward() const override {
        return mEnableFp8Forward && mGradQuantDtype == ETensorDType::FP8_E5M2;
    }
    [[nodiscard]] bool has_fp8_delayed_scaling() const override { return mFP8ScalingState != nullptr; }
    [[nodiscard]] bool has_fp4_forward() const override { return false; }
    [[nodiscard]] bool has_fp4_backward() const override { return false; }
    [[nodiscard]] Tensor* get_fp8_forward_buffer(int op) override;
    [[nodiscard]] Tensor* get_gradient_quant_buffer(int op) override;
    [[nodiscard]] modules::FP8ScalingState* get_fp8_scaling_state() override { return mFP8ScalingState.get(); }

    // MoE stats overrides
    [[nodiscard]] MoEStats get_moe_stats() const override;
    void reset_moe_stats() override;
    [[nodiscard]] bool is_moe_model() const override { return mNumMoEExperts > 0; }

    /// @brief Get device pointer to MoE stats buffer for kernel accumulation.
    /// Layout: [aux_loss_sum, z_loss_sum, utilization_sum, load_imbalance_sum, layer_count]
    float* moe_stats_device() { return mMoEStatsDevice; }

    /// @brief Set MoE config and allocate stats buffers (call once after construction)
    void set_moe_config(int num_experts, float aux_loss_coef);
    [[nodiscard]] int moe_num_experts() const { return mNumMoEExperts; }
    [[nodiscard]] float moe_aux_loss_coef() const { return mMoEAuxLossCoef; }

private:
    void allocate_non_block_state(const PretrainedConfig& cfg);
    void allocate_simplified_activations(const PretrainedConfig& cfg);
    void allocate_simplified_gradients(const PretrainedConfig& cfg);
    void build_activation_grad_zero_segments();
    void allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options);
    void allocate_scratch_buffers(const PretrainedConfig& cfg);
    void allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals);
    void create_cuda_resources();
    void release_cuda_resources() noexcept;

    std::shared_ptr<TensorAllocator> mAllocator;
    DslRuntimeConfig mRuntimeConfig;
    Tensor mStackBuffer{};
    bool mStackSimulate = false;
    RecomputeLevel mRecomputeLevel = RecomputeLevel::Enabled;
    bool mLoraOnlyMode = false;
    bool mPrequantized = false;
    bool mFfnTempsOnStack = false;
    ETensorDType mActivationDtype = ETensorDType::BF16;
    ETensorDType mGradDtype = ETensorDType::BF16;
    ETensorDType mMatmulDtype = ETensorDType::BF16;
    ETensorDType mGradQuantDtype = ETensorDType::BF16;
    bool mEnableFp8Forward = false;
    bool mOffloadResiduals = false;
    int mLMHeadChunks = 1;
    int mAttnBwdChunks = 1;

    modules::NonBlockActivations mNonBlockActivations;
    modules::NonBlockGradientBuffers mNonBlockGradients;
    modules::ScratchBuffers mScratch;
    TensorSlotRegistry mSlotRegistry;

    std::vector<modules::SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradients;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradientsBase;

    // Precomputed list of activation-gradient buffers to zero at the start of backward.
    // Stored as device arrays of (ptr, bytes) to zero in a single kernel launch.
    Tensor mActGradZeroPtrs{};
    Tensor mActGradZeroSizes{};
    int mActGradZeroCount = 0;

    // Shared gradient buffers (when recompute_block=true)
    std::array<Tensor, 2> mSharedDResFFN{};  ///< Alternating buffers for d_res_ffn
    std::array<Tensor, 2> mSharedDMlpDown{}; ///< Alternating buffers for d_mlp_down
    Tensor mSharedDResAtt{};
    Tensor mSharedDAttOut{};
    Tensor mSharedDLn2{};
    Tensor mSharedDAtt{};
    Tensor mSharedDLn1{};
    modules::SimplifiedQuantGradients mSimplifiedQuantGrads;
    modules::FP8ForwardQuantActivations mFP8ForwardQuants;
    Tensor mFP8ForwardStats{};
    Tensor mGradQuantStats{};

    // Temporary buffers for recomputation (to avoid overwriting saved activations)
    Tensor mRecomputeRstd{};  ///< Temp buffer for rstd during recomputation [B, T]
    Tensor mRecomputeLSE{};   ///< Temp buffer for LSE during recomputation [B, Hq, T]
    std::unique_ptr<modules::FP8ScalingState> mFP8ScalingState;

    std::unique_ptr<modules::ResidualManager> mResidualManager;

    // CUDA resources
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
    cudaEvent_t mAllReduceDone = nullptr;  ///< Recorded after async all-reduce completes
    cublasHandle_t mCublasHandle = nullptr;  ///< cuBLAS handle for MoE grouped GEMM

    // Per-layer CUDA graph support
    int mNumLayers = 0;
    bool mPerLayerGraphsEnabled = false;
    bool mForwardGraphsHooked = false;
    bool mBackwardGraphsHooked = false;

    // Per-layer CUDA graph executables
    std::vector<cudaGraphExec_t> mForwardBlockGraphs;
    // Two backward graphs per layer: [0] for accumulate=false (first micro-step),
    // [1] for accumulate=true (subsequent micro-steps with gradient accumulation)
    std::vector<std::array<cudaGraphExec_t, 2>> mBackwardBlockGraphs;

    // Stack checkpoints for CUDA graph compatibility
    // When graphs use temp_alloc, we must restore the stack to the same state
    // before each graph replay so allocations return consistent addresses
    std::vector<DeviceMemoryStack::Checkpoint> mForwardBlockStackCheckpoints;
    std::vector<std::array<DeviceMemoryStack::Checkpoint, 2>> mBackwardBlockStackCheckpoints;

    // Helper to destroy graph arrays
    void destroy_cuda_graphs() noexcept;
    void allocate_graph_arrays(int num_layers);

    // MoE routing stats accumulation buffer (GPU)
    // Layout: [aux_loss_sum, z_loss_sum, utilization_sum, load_imbalance_sum, layer_count]
    static constexpr int kMoEStatsSize = 5;
    float* mMoEStatsDevice = nullptr;   ///< Device buffer for kernel accumulation
    float* mMoEStatsHost = nullptr;     ///< Pinned host buffer for readback
    int mNumMoEExperts = 0;             ///< 0 = not MoE
    float mMoEAuxLossCoef = 0.01f;     ///< Auxiliary loss coefficient
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_RUN_STATE_H
