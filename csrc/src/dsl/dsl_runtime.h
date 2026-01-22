// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL runtime components (run state + weights + gradients).

#ifndef SUROGATE_SRC_DSL_DSL_RUNTIME_H
#define SUROGATE_SRC_DSL_DSL_RUNTIME_H

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "utilities/stack.h"
#include "modules/run_state_types.h"
#include "modules/residual_manager.h"
#include "training/model.h"
#include "utilities/allocator.h"
#include "utilities/tensor_container.h"

struct RuntimeOptions;
class NCCLCommunicator;
namespace modules { struct ModularLoRAConfig; class FP8ScalingState; }

namespace dsl {

// Stores model parameters defined by the DSL IR.
class DslParamStore final : public ITensorContainer {
public:
    struct Entry {
        Tensor tensor;
        bool trainable = true;
    };

    DslParamStore(const Module& module,
                  const Graph& graph,
                  const RuntimeOptions& options,
                  const PretrainedConfig& config,
                  const std::shared_ptr<TensorAllocator>& allocator,
                  const modules::ModularLoRAConfig* lora_config = nullptr);

    Tensor& get(const std::string& name);
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;
    bool is_trainable(const std::string& name) const;

    const std::vector<std::string>& param_names() const { return mParamOrder; }

    void iterate_tensors(const std::function<void(std::string, const TensorShard&)>& callback) override;

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Entry> mParams;
    std::vector<std::string> mParamOrder;
};

// Stores parameter gradients for DSL execution.
class DslGradStore {
public:
    DslGradStore(const DslParamStore& params,
                 const std::shared_ptr<TensorAllocator>& allocator);

    void start_micro_step(cudaStream_t stream, int micro_step, int total_steps);
    void end_micro_step(cudaStream_t stream, NCCLCommunicator& comm);

    Tensor* get_param_grad(const std::string& name, bool& accumulate);

    void zero_all(cudaStream_t stream);
    void reduce_all(NCCLCommunicator& comm, cudaStream_t stream);

    /// Start async all-reduce on all gradients (non-blocking).
    /// Call wait_for_reduce() or synchronize on AllReduceDone event before using gradients.
    void reduce_all_async(NCCLCommunicator& comm, cudaStream_t stream, cudaEvent_t done_event);

    /// Check if async reduce has been started (for avoiding redundant reduce in update)
    bool is_reduce_pending() const { return mReducePending; }
    void clear_reduce_pending() { mReducePending = false; }

    const std::vector<std::string>& param_names() const { return mParamOrder; }
    const std::unordered_map<std::string, Tensor>& grads() const { return mGrads; }

private:
    std::shared_ptr<TensorAllocator> mAllocator;
    std::unordered_map<std::string, Tensor> mGrads;
    std::vector<std::string> mParamOrder;
    bool mAccumulate = false;
    bool mReducePending = false;  ///< True if async reduce has been started
};

// DSL run state for graph execution (activation buffers, scratch, etc).
class DslRunState final : public IRunState {
public:
    DslRunState(const PretrainedConfig& config,
                const RuntimeOptions& options,
                int B, int T,
                const std::shared_ptr<TensorAllocator>& allocator,
                bool lora_only_mode = false);
    ~DslRunState();

    modules::SimplifiedLayerActivations& simplified_acts(int layer_idx) { return mSimplifiedActivations[layer_idx]; }
    modules::SimplifiedLayerGradients& simplified_grads(int layer_idx) { return mSimplifiedGradients[layer_idx]; }
    modules::SimplifiedQuantGradients& simplified_quant_grads() { return mSimplifiedQuantGrads; }
    modules::FP8ForwardQuantActivations& fp8_forward_quants() { return mFP8ForwardQuants; }

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

    modules::NonBlockActivations mNonBlockActivations;
    modules::NonBlockGradientBuffers mNonBlockGradients;
    modules::ScratchBuffers mScratch;

    std::vector<modules::SimplifiedLayerActivations> mSimplifiedActivations;
    std::vector<modules::SimplifiedLayerGradients> mSimplifiedGradients;
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

#endif // SUROGATE_SRC_DSL_DSL_RUNTIME_H
