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

#include "dsl/ir.h"
#include "utilities/tensor.h"

class NCCLCommunicator;
struct RuntimeOptions;

namespace modules { struct ModelConfig; }
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

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step) override;
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step) override;
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step) override;

    bool has_derived_backward() const override { return mDerivedBackward.has_value(); }
    const Graph* backward_graph() const override { return mBackward; }

    std::vector<std::byte> rng_state() const;
    void set_rng_state(const std::vector<std::byte>& state);

private:
    void init(const GraphExecutorOptions& options);

    const Module& mModule;
    DslRunState& mRunState;
    DslParamStore& mWeights;
    DslGradStore& mGrads;
    const modules::ModelConfig& mConfig;
    const RuntimeOptions& mOptions;

    const Graph* mForward;
    const Graph* mBackward;

    // Holds derived backward graph if autodiff was used
    std::optional<Graph> mDerivedBackward;

    // Combined save list (forward.save + autodiff-computed saves)
    std::vector<std::string> mSaveList;

    std::unordered_map<std::string, Tensor> mSaved;
    Tensor mLastInputsCpu{};
    bool mFP8ScalingInitialized = false;

    void execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full);
    void execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step);

    void run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy);

    unsigned int next_rng_seed();

    std::minstd_rand mRng{42};
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
