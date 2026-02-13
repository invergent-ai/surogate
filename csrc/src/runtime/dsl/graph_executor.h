// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (DSL-driven).

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/dsl/graph_executor_internal.h"
#include "runtime/dsl/ir.h"
#include "runtime/dsl/forward_plan.h"
#include "runtime/core/forward_hooks.h"
#include "runtime/core/backward_hooks.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"

class NCCLCommunicator;
struct RuntimeOptions;

namespace modules {
struct ModelConfig;
struct ModularLoRAConfig;
class ModularLoRAWeightsManager;
class ModularLoRAGradsManager;
struct LoRARunState;
struct MatmulContext;
enum class MatmulOp;
}
namespace dsl {
class DslRunState;
class DslParamStore;
class DslGradStore;
class DslWeightManager;
class GraphCompiler;
class CompiledExecutor;
struct CompiledGraph;
struct FP8WeightCacheEntry;
struct FP4WeightCacheEntry;
class RecomputePlan;
}

namespace dsl {

// Options for GraphExecutor construction
struct GraphExecutorOptions {
    // If true and module has no backward graph, derive one automatically using autodiff
    bool auto_backward = false;

    // Name of the loss tensor for autodiff (used when deriving backward)
    std::string loss_name = "loss";

    // Whether to print derived backward graph for debugging
    bool debug_print_backward = false;
};

class IGraphExecutor {
public:
    virtual ~IGraphExecutor() = default;

    virtual void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) = 0;
    virtual float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) = 0;
    virtual void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) = 0;

    // Hook-enabled forward/backward methods (matching ModularModel interface)
    virtual void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                                   const modules::ForwardHook& hook) {
        // Default: forward without hooks
        forward(inputs, position_ids, comm, micro_step);
    }

    virtual float validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm,
                                     int micro_step, const modules::ForwardHook& hook) {
        // Default: validate without hooks
        return validate(inputs, position_ids, targets, comm, micro_step);
    }

    virtual void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                                    int grad_accum_steps, int micro_step, const modules::BackwardHook& hook) {
        // Default: backward without hooks
        backward(inputs, targets, comm, grad_accum_steps, micro_step);
    }

    // Check if backward graph was auto-derived
    virtual bool has_derived_backward() const = 0;

    // Get the backward graph (either from module or derived)
    virtual const Graph* backward_graph() const = 0;

    // RNG state helpers (for checkpointing/repro)
    virtual std::vector<std::byte> rng_state() const = 0;
    virtual void set_rng_state(const std::vector<std::byte>& state) = 0;

    // Control internal CUDA graph usage (forward/backward graphs inside DSL executor).
    virtual void set_internal_graphs_enabled(bool enabled) { (void)enabled; }
    virtual bool internal_graphs_enabled() const { return false; }

    // Optional LoRA state wiring (no-op for implementations that don't support it).
    virtual void set_lora_state(const modules::ModularLoRAConfig*,
                                modules::ModularLoRAWeightsManager*,
                                modules::ModularLoRAGradsManager*,
                                modules::LoRARunState*) {}

    // Set optional hook context (opaque pointer passed to hooks)
    virtual void set_hook_context(void* context) { (void)context; }

    /// Total bytes of untracked persistent saved buffers (raw cudaMalloc).
    virtual size_t saved_buffers_total_bytes() const { return 0; }
    /// Number of persistent saved buffers.
    virtual int saved_buffers_count() const { return 0; }
    /// Per-buffer sizes for diagnostics.
    virtual const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const {
        static const std::unordered_map<std::string, size_t> empty;
        return empty;
    }
};

class GraphExecutor final : public IGraphExecutor {
public:
    GraphExecutor(const Module& module,
                  DslRunState& run_state,
                  DslParamStore& weights,
                  DslGradStore& grads,
                  const modules::ModelConfig& config,
                  const RuntimeOptions& options,
                  const GraphExecutorOptions& exec_options = {});
    ~GraphExecutor() override;

    void set_lora_state(const modules::ModularLoRAConfig* config,
                        modules::ModularLoRAWeightsManager* weights,
                        modules::ModularLoRAGradsManager* grads,
                        modules::LoRARunState* run_state);

    // Set optional weight manager for streaming/sharding
    void set_weight_manager(DslWeightManager* weight_manager);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;

    // Hook-enabled methods (matching ModularModel interface for LoRA integration)
    void forward_with_hook(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step,
                           const modules::ForwardHook& hook) override;
    float validate_with_hook(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm,
                             int micro_step, const modules::ForwardHook& hook) override;
    void backward_with_hook(Tensor inputs, Tensor targets, NCCLCommunicator& comm,
                            int grad_accum_steps, int micro_step, const modules::BackwardHook& hook) override;

    void set_hook_context(void* context) override { mHookContext = context; }

    bool has_derived_backward() const override { return mDerivedBackward.has_value(); }
    const Graph* backward_graph() const override { return mBackward; }

    std::vector<std::byte> rng_state() const;
    void set_rng_state(const std::vector<std::byte>& state);
    void set_internal_graphs_enabled(bool enabled) override;
    bool internal_graphs_enabled() const override;

    size_t saved_buffers_total_bytes() const override;
    int saved_buffers_count() const override;
    const std::unordered_map<std::string, size_t>& saved_buffers_sizes() const override;

private:
    friend class RecomputePlan;
    void init(const GraphExecutorOptions& options);
    void reset_cuda_graphs();
    void reset_forward_plan();
    void record_matmul_plan(int layer_idx, modules::MatmulOp op, const MatmulForwardPlan& plan);
    void record_attn_plan(int layer_idx, const AttnForwardPlan& plan);
    const LayerForwardPlan* forward_plan(int layer_idx) const;

    // Internal hook invocation helpers
    void invoke_forward_hook(int layer_idx, modules::ForwardHookPoint point, cudaStream_t stream,
                             const modules::ForwardHook* hook) {
        if (hook && *hook) {
            (*hook)(layer_idx, stream, point, mHookContext);
        }
    }

    void invoke_backward_hook(int layer_idx, bool accumulate, modules::BackwardHookPoint point,
                              cudaStream_t stream, const modules::BackwardHook* hook) {
        if (hook && *hook) {
            (*hook)(layer_idx, accumulate, stream, point, mHookContext);
        }
    }

    const Module& mModule;
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    // Optional LoRA state (owned by DslModel)
    const modules::ModularLoRAConfig* mLoRAConfig = nullptr;
    modules::ModularLoRAWeightsManager* mLoRAWeights = nullptr;
    modules::ModularLoRAGradsManager* mLoRAGrads = nullptr;
    modules::LoRARunState* mLoRARunState = nullptr;

    // Hook context (opaque pointer passed to hook callbacks)
    void* mHookContext = nullptr;

    // Optional weight manager for streaming/sharding (owned by DslModel)
    DslWeightManager* mWeightManager = nullptr;

    const Graph* mForward;
    const Graph* mBackward;

    // Holds derived backward graph if autodiff was used
    std::optional<Graph> mDerivedBackward;
    // Holds a reordered backward graph when we need a mutable copy
    std::optional<Graph> mReorderedBackward;

    // Combined save list (forward.save + autodiff-computed saves)
    std::vector<std::string> mSaveList;
    std::vector<std::string> mEmbeddingOutputs;

    std::unordered_map<std::string, Tensor> mSaved;
    std::unordered_map<std::string, std::string> mViewSources;
    std::unordered_map<std::string, std::string> mViewSourcesReverse;
    Tensor mLastInputsCpu{};
    bool mFP8ScalingInitialized = false;
    std::vector<LayerForwardPlan> mForwardPlan;
    std::unique_ptr<RecomputePlan> mRecomputePlan;

    // FP8/FP4 weight caches (use namespace-level types for compatibility with CompiledExecutor)
    std::unordered_map<std::string, FP8WeightCacheEntry> mFP8WeightCache;
    std::unordered_map<std::string, FP8WeightCacheEntry> mFP8WeightCacheT;  ///< Backward dinp (transposed layout)
    std::unordered_map<std::string, FP4WeightCacheEntry> mFP4WeightCache;    ///< Forward pass (normal layout)
    std::unordered_map<std::string, FP4WeightCacheEntry> mFP4WeightCacheT;   ///< Backward dgrad (transposed layout)

    unsigned int next_rng_seed();

    void prime_fp8_weight_cache(const std::vector<char>& required);
    const Tensor* get_fp8_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream);
    void prime_fp8_weight_cache_transposed(const std::vector<char>& required);
    const Tensor* get_fp8_cached_weight_transposed(const std::string& name, Tensor& weight, cudaStream_t stream);

    // FP4 weight cache helpers (for NVFP4 recipe on Blackwell+)
    void prime_fp4_weight_cache(const std::vector<char>& required);
    const FP4WeightCacheEntry* get_fp4_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream);
    const FP4WeightCacheEntry* get_fp4_cached_weight_transposed(const std::string& name, Tensor& weight, cudaStream_t stream);

    // Weight prefetching for layer-by-layer execution
    void prefetch_layer_weights(int layer_idx, cudaStream_t stream);
    void wait_for_prefetch(int layer_idx, cudaStream_t stream);
    void build_layer_weight_map();
    void build_layer_boundaries();  // Pre-compute layer start/end operation indices

    std::minstd_rand mRng{42};

    // Layer-to-weight-names map for prefetching
    std::vector<std::vector<std::string>> mLayerWeightNames;
    int mPrefetchedLayer = -1;
    cudaEvent_t mPrefetchEvent = nullptr;
    bool mPrefetchEnabled = false;
    bool mHasLossOp = false;
    bool mWeightCachesPrimed = false;  // True after first eager FP8/FP4 cache priming

    // Pre-computed layer boundaries for predictable prefetch
    struct LayerBoundary {
        int layer_idx = -1;
        std::size_t start_op_idx = 0;  // First operation for this layer
        std::size_t end_op_idx = 0;    // One past last operation for this layer
    };
    std::vector<LayerBoundary> mLayerBoundaries;  // Sorted by start_op_idx

    // CUDA graph capture (optional)
    bool mGraphsEnabled = false; // Forward graphs
    bool mBackwardGraphsEnabled = false;
    bool mBackwardGraphCapturable = true;
    std::size_t mBackwardGraphCut = 0;
    long mGraphB = 0;
    long mGraphT = 0;
    cudaGraphExec_t mForwardGraph = nullptr;
    cudaGraphExec_t mBackwardGraph[2]{nullptr, nullptr}; // [0]=accumulate false, [1]=true
    DeviceMemoryStack::Checkpoint mForwardCheckpoint{};
    DeviceMemoryStack::Checkpoint mBackwardCheckpoint[2]{};

    // Per-layer CUDA graph execution (more granular than whole-graph capture)
    bool mPerLayerGraphsEnabled = false;

    // ========================================================================
    // DSL-driven recomputation (plan-based)
    // ========================================================================
    // Execute full block recomputation (both segments)
    void recompute_block(int layer_idx, long B, long T);

    // ========================================================================
    // Compiled execution (operations pre-compiled into direct function calls)
    // ========================================================================
    std::unique_ptr<GraphCompiler> mCompiler;
    std::unique_ptr<CompiledExecutor> mCompiledExecutor;
    std::unique_ptr<CompiledGraph> mCompiledForward;
    std::unique_ptr<CompiledGraph> mCompiledBackward;
    long mCompiledB = 0;
    long mCompiledT = 0;

    void init_compiled_execution();
    void compile_graphs(long B, long T);
    void execute_forward(long B, long T, NCCLCommunicator& comm, bool full,
                         const modules::ForwardHook* hook);
    void execute_backward(long B, long T, NCCLCommunicator& comm, int grad_accum_steps,
                          int micro_step, const modules::BackwardHook* hook,
                          bool skip_zeroing = false);
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
