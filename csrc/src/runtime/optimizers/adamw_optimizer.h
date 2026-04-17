// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// FP32 AdamW optimizer strategy.

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_OPTIMIZER_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_OPTIMIZER_H

#include <memory>

#include <cuda_runtime.h>

#include "runtime/optimizers/optimizer.h"

namespace optimizers {

/// Full-precision AdamW. Momentum and variance kept in FP32.
///
/// Supports eager mode, CUDA-graph capture, and full-fine-tune paths.
/// LoRA dispatch is delegated to DslModel's LoRA-specific update methods.
class AdamWOptimizer final : public Optimizer {
public:
    AdamWOptimizer();
    ~AdamWOptimizer() override;

    [[nodiscard]] const char* name() const override {
        return "adamw";
    }

    [[nodiscard]] OptimizerType type() const override {
        return OptimizerType::ADAMW;
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

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_OPTIMIZER_H
