// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// NorMuon hybrid optimizer strategy.

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_NORMUON_OPTIMIZER_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_NORMUON_OPTIMIZER_H

#include <memory>

#include <cuda_runtime.h>

#include "runtime/optimizers/optimizer.h"

namespace optimizers {

/// Hybrid NorMuon: AdamW-8bit for 1D params (embeddings, norms, lm_head,
/// biases, MoE routers) + NorMuon with Polar Express orthogonalization on
/// 2D weight matrices (attention, MLP, expert MLP).
///
/// Parameter classification happens at state-alloc time based on name and
/// rank (see the internal ``is_normuon_param`` helper).
class NorMuonOptimizer final : public Optimizer {
public:
    NorMuonOptimizer();
    ~NorMuonOptimizer() override;

    [[nodiscard]] const char* name() const override {
        return "normuon";
    }

    [[nodiscard]] OptimizerType type() const override {
        return OptimizerType::NORMUON;
    }

    void step(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config, int step_idx) override;

    void step_graph(dsl::DslModel& model,
                    NCCLCommunicator& comm,
                    const OptimizerConfig& config,
                    const float* opt_params,
                    const int* opt_step) override;

    void prepare_for_graph(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;

    void init_state(dsl::DslModel& model, cudaStream_t stream);
};

}  // namespace optimizers

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_NORMUON_OPTIMIZER_H
