// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Optimizer abstract interface — pluggable optimizer dispatch for DslModel.
//
// Follows the IQuantizer pattern: a small virtual interface, a factory that
// picks the concrete implementation from OptimizerConfig, and per-optimizer
// TUs that each own their own kernel dispatch AND state management.
//
// Concrete implementations (one per class):
//   - AdamWOptimizer      : FP32 AdamW
//   - AdamW8BitOptimizer  : Flash-quantized 8-bit AdamW (the default)
//   - NorMuonOptimizer    : Hybrid AdamW-8bit (1D) + NorMuon (2D weights)
//
// Adding a new optimizer means dropping a new TU that inherits Optimizer,
// wiring it into the factory, and adding the enum tag. No changes to
// dsl_model_optimizer.cpp.

#ifndef SUROGATE_SRC_RUNTIME_OPTIMIZERS_OPTIMIZER_H
#define SUROGATE_SRC_RUNTIME_OPTIMIZERS_OPTIMIZER_H

#include <memory>

#include "runtime/optimizers/optimizer_base.h"
#include "runtime/optimizers/optimizer_config.h"

class NCCLCommunicator;
class ITensorContainer;

namespace dsl {
class DslModel;
}  // namespace dsl

namespace optimizers {

/// Abstract optimizer strategy used by DslModel::update_with_config().
///
/// Each concrete implementation owns its own state (allocated lazily on the
/// first step) and is responsible for all execution modes:
///   - ``step``              : eager mode (non-captured)
///   - ``step_graph``        : CUDA-graph captured step (optional)
///   - ``prepare_for_graph`` : state initialization that must happen before
///                             graph capture begins (optional)
///
/// The optimizer accesses the model's params / grads / run-state through
/// the friend relationship declared on DslModel.
class Optimizer {
public:
    virtual ~Optimizer() = default;

    /// Short human-readable identifier (used in logs / error messages).
    [[nodiscard]] virtual const char* name() const = 0;

    /// Enum tag — matches OptimizerConfig::type.
    [[nodiscard]] virtual OptimizerType type() const = 0;

    /// Run one eager optimizer step across all parameters.
    ///
    /// Preconditions: ``model.allocate_run_state()`` has been called and
    /// gradients are populated. Implementations handle their own gradient
    /// all-reduce, gradient norm computation, and state lazy-init.
    virtual void step(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config, int step_idx) = 0;

    /// Run a CUDA-graph-captured optimizer step.
    ///
    /// ``opt_params`` / ``opt_step`` are device pointers into buffers the
    /// graph owner updates between replays. Default: throws — override if
    /// the concrete optimizer supports capture.
    virtual void step_graph(dsl::DslModel& model,
                            NCCLCommunicator& comm,
                            const OptimizerConfig& config,
                            const float* opt_params,
                            const int* opt_step);

    /// Initialize optimizer state eagerly before CUDA-graph capture begins.
    ///
    /// Graph capture cannot allocate or memset, so any first-step state
    /// seeding must happen here. Default: no-op.
    virtual void prepare_for_graph(dsl::DslModel& model, NCCLCommunicator& comm, const OptimizerConfig& config);

    /// Checkpoint/introspection containers for momentum and variance state.
    /// Default: return nullptr (no state visible).
    /// Concrete optimizers that persist per-param state (AdamW / AdamW-8bit)
    /// override these to expose their buffers through the ``IModel`` API.
    [[nodiscard]] virtual ITensorContainer* momentum_container() {
        return nullptr;
    }
    [[nodiscard]] virtual ITensorContainer* variance_container() {
        return nullptr;
    }
    [[nodiscard]] virtual ITensorContainer* momentum_scales_container() {
        return nullptr;
    }
    [[nodiscard]] virtual ITensorContainer* variance_scales_container() {
        return nullptr;
    }

    /// Pre-checkpoint-load hook: allocate and zero-init state buffers so the
    /// checkpoint reader has something to deserialize into. Default: no-op
    /// (for optimizers that don't persist state across runs).
    virtual void prepare_for_checkpoint_load(dsl::DslModel& /*model*/) {
    }

    /// Post-checkpoint-restore hook: mark state initialized and rebind any
    /// checkpoint containers to the newly-populated tensors. Default: no-op.
    virtual void on_restore_checkpoint(dsl::DslModel& /*model*/) {
    }
};

/// Factory: pick the concrete Optimizer for the given config.
///
/// Throws ``std::runtime_error`` if ``config.type`` is unsupported.
std::unique_ptr<Optimizer> create_optimizer(const OptimizerConfig& config);

}  // namespace optimizers

#endif  // SUROGATE_SRC_RUNTIME_OPTIMIZERS_OPTIMIZER_H
