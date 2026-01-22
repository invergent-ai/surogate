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

#include "dsl/ir.h"
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
}
namespace dsl { class DslRunState; class DslParamStore; class DslGradStore; }

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

    // Check if backward graph was auto-derived
    virtual bool has_derived_backward() const = 0;

    // Get the backward graph (either from module or derived)
    virtual const Graph* backward_graph() const = 0;

    // RNG state helpers (for checkpointing/repro)
    virtual std::vector<std::byte> rng_state() const = 0;
    virtual void set_rng_state(const std::vector<std::byte>& state) = 0;

    // Optional LoRA state wiring (no-op for implementations that don't support it).
    virtual void set_lora_state(const modules::ModularLoRAConfig*,
                                modules::ModularLoRAWeightsManager*,
                                modules::ModularLoRAGradsManager*,
                                modules::LoRARunState*) {}
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

    void set_lora_state(const modules::ModularLoRAConfig* config,
                        modules::ModularLoRAWeightsManager* weights,
                        modules::ModularLoRAGradsManager* grads,
                        modules::LoRARunState* run_state);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;

    bool has_derived_backward() const override { return mDerivedBackward.has_value(); }
    const Graph* backward_graph() const override { return mBackward; }

    std::vector<std::byte> rng_state() const;
    void set_rng_state(const std::vector<std::byte>& state);

private:
    void init(const GraphExecutorOptions& options);
    void reset_cuda_graphs();

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
    struct FP8WeightCacheEntry {
        Tensor weight;
        Tensor stats;
        bool initialized = false;
    };
    std::unordered_map<std::string, FP8WeightCacheEntry> mFP8WeightCache;

    void execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full);
    void execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step);

    void run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy);

    unsigned int next_rng_seed();

    void prime_fp8_weight_cache(const std::vector<char>& required);
    const Tensor* get_fp8_cached_weight(const std::string& name, Tensor& weight, cudaStream_t stream);

    std::minstd_rand mRng{42};

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
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
