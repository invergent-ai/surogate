// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Flash-quantized 8-bit AdamW optimizer strategy (the default).

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_8BIT_OPTIMIZER_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_8BIT_OPTIMIZER_H

#include <memory>

#include <cuda_runtime.h>

#include "runtime/optimizers/optimizer.h"

namespace optimizers {

/// 8-bit block-quantized AdamW (FlashOptim-style). Momentum uses softsign
/// quantization, variance uses sqrt quantization — per-group scales held in
/// FP16 for a ~4x reduction in optimizer-state memory vs. FP32 AdamW.
///
/// Also handles the CPU-offload streaming-grads path when the gradient
/// store is configured for CPU-centric training.
class AdamW8BitOptimizer final : public Optimizer {
public:
    AdamW8BitOptimizer();
    ~AdamW8BitOptimizer() override;

    [[nodiscard]] const char* name() const override {
        return "adamw_8bit";
    }

    [[nodiscard]] OptimizerType type() const override {
        return OptimizerType::ADAMW_8BIT;
    }

    void step(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config, int step_idx) override;

    void step_graph(dsl::DslModel& model,
                    NCCLCommunicator& comm,
                    const OptimizerConfig& config,
                    const float* opt_params,
                    const int* opt_step) override;

    void prepare_for_graph(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config) override;

    [[nodiscard]] ITensorContainer* momentum_container() override;
    [[nodiscard]] ITensorContainer* variance_container() override;

    void on_restore_checkpoint(dsl::DslModel& model) override;
    void prepare_for_checkpoint_load(dsl::DslModel& model) override;

private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;

    void init_state(dsl::DslModel& model, cudaStream_t stream);
    void step_cpu_streaming(dsl::DslModel& model,
                            NCCLCommunicator& comm,
                            float learning_rate,
                            float beta_1,
                            float beta_2,
                            int t,
                            float epsilon,
                            float weight_decay,
                            float grad_clip);
};

}  // namespace optimizers

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_ADAMW_8BIT_OPTIMIZER_H
