// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Graph executor (Qwen3-first).

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_H

#include <unordered_map>
#include <vector>

#include "dsl/ir.h"
#include "utilities/tensor.h"

class NCCLCommunicator;

namespace modules {
class Qwen3Model;
}  // namespace modules

namespace dsl {

class GraphExecutor {
public:
    GraphExecutor(const Module& module, modules::Qwen3Model& backend);

    void forward(Tensor inputs, Tensor position_ids, NCCLCommunicator& comm, int micro_step);
    float validate(Tensor inputs, Tensor position_ids, Tensor targets, NCCLCommunicator& comm, int micro_step);
    void backward(Tensor inputs, Tensor targets, NCCLCommunicator& comm, int grad_accum_steps, int micro_step);

private:
    const Module& mModule;
    modules::Qwen3Model& mBackend;
    const Graph* mForward;
    const Graph* mBackward;

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
