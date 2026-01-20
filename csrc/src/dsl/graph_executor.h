// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (Qwen3-first).

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H

#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "utilities/tensor.h"

class NCCLCommunicator;

namespace modules {
class Qwen3Model;
}  // namespace modules

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

class GraphExecutor {
public:
    // Constructor with explicit forward/backward graphs from module
    GraphExecutor(const Module& module, modules::Qwen3Model& backend);

    // Constructor with options (enables autodiff)
    GraphExecutor(const Module& module, modules::Qwen3Model& backend, const GraphExecutorOptions& options);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step);
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step);
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step);

    // Check if backward graph was auto-derived
    bool has_derived_backward() const { return mDerivedBackward.has_value(); }

    // Get the backward graph (either from module or derived)
    const Graph* backward_graph() const { return mBackward; }

private:
    void init(const GraphExecutorOptions& options);

    const Module& mModule;
    modules::Qwen3Model& mBackend;
    const Graph* mForward;
    const Graph* mBackward;

    // Holds derived backward graph if autodiff was used
    std::optional<Graph> mDerivedBackward;

    // Combined save list (forward.save + autodiff-computed saves)
    std::vector<std::string> mSaveList;

    std::unordered_map<std::string, Tensor> mSaved;
    Tensor mLastInputsCpu{};
    bool mFP8ScalingInitialized = false;
    bool mLmHeadCached = false;

    void execute_forward_graph(long B, long T, NCCLCommunicator& comm, bool full);
    void execute_backward_graph(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step);

    void run_classifier(long B, long T, NCCLCommunicator& comm, int grad_accum_steps, int micro_step, bool compute_accuracy);

    unsigned int next_rng_seed();
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
