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

#include "utilities/stack.h"
#include "modules/run_state_types.h"
#include "modules/residual_manager.h"
#include "training/model.h"
#include "utilities/allocator.h"

struct RuntimeOptions;
namespace modules { class FP8ScalingState; }

namespace dsl {

// DSL run state for graph execution (activation buffers, scratch, etc).
class DslRunState final : public IRunState {
public:
    static constexpr std::size_t kDefaultStackBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;  // 2GB

    DslRunState(const PretrainedConfig& config,
                const RuntimeOptions& options,
                int B, int T,
                const std::shared_ptr<TensorAllocator>& allocator,
                bool lora_only_mode = false,
                std::size_t stack_bytes = kDefaultStackBytes,
                bool allocate_stack = true);
    ~DslRunState();

    void set_stack_buffer(Tensor buffer, const DeviceMemoryStack::AllocationList& high_mark = {});

    modules::SimplifiedLayerActivations& simplified_acts(int layer_idx) { return mSimplifiedActivations[layer_idx]; }
    modules::SimplifiedLayerGradients& simplified_grads(int layer_idx) { return mSimplifiedGradients[layer_idx]; }
    modules::SimplifiedQuantGradients& simplified_quant_grads() { return mSimplifiedQuantGrads; }
    modules::FP8ForwardQuantActivations& fp8_forward_quants() { return mFP8ForwardQuants; }

    void reset_simplified_gradients();

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
    bool large_bwd_temps_on_stack() const { return mRecomputeBlock; }
    bool is_lora_only_mode() const { return mLoraOnlyMode; }

    cudaStream_t side_stream() const { return mSideStream; }
    cudaEvent_t side_stream_event() const { return mSideStreamEvent; }
    cudaEvent_t all_reduce_done_event() const { return mAllReduceDone; }

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

private:
    void allocate_non_block_state(const PretrainedConfig& cfg);
    void allocate_simplified_activations(const PretrainedConfig& cfg);
    void allocate_simplified_gradients(const PretrainedConfig& cfg);
    void allocate_simplified_quant_buffers(const PretrainedConfig& cfg, const RuntimeOptions& options);
    void allocate_scratch_buffers(const PretrainedConfig& cfg);
    void allocate_residual_buffers(const PretrainedConfig& cfg, bool offload_residuals);
    void create_cuda_resources();
    void release_cuda_resources() noexcept;

    std::shared_ptr<TensorAllocator> mAllocator;
    Tensor mStackBuffer{};
    bool mStackSimulate = false;
    bool mRecomputeBlock = false;
    bool mRecomputeLoRA = false;
    bool mLoraOnlyMode = false;
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

    std::vector<modules::SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradients;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradientsBase;

    // Shared gradient buffers (when recompute_block=true)
    std::array<Tensor, 2> mSharedDResFFN{};  ///< Alternating buffers for d_res_ffn
    std::array<Tensor, 2> mSharedDMlpDown{}; ///< Alternating buffers for d_mlp_down
    Tensor mSharedDResAtt{};
    Tensor mSharedDLn2{};
    Tensor mSharedDAtt{};
    Tensor mSharedDLn1{};
    modules::SimplifiedQuantGradients mSimplifiedQuantGrads;
    modules::FP8ForwardQuantActivations mFP8ForwardQuants;
    Tensor mFP8ForwardStats{};
    Tensor mGradQuantStats{};
    std::unique_ptr<modules::FP8ScalingState> mFP8ScalingState;

    std::unique_ptr<modules::ResidualManager> mResidualManager;

    // CUDA resources
    cudaStream_t mSideStream = nullptr;
    cudaEvent_t mSideStreamEvent = nullptr;
    cudaEvent_t mAllReduceDone = nullptr;  ///< Recorded after async all-reduce completes

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
};

} // namespace dsl

#endif // SUROGATE_SRC_DSL_DSL_RUN_STATE_H
