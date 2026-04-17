// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Optimizer base defaults + factory.

#include "runtime/optimizers/optimizer.h"

#include <stdexcept>
#include <string>

#include "runtime/optimizers/adamw_8bit_optimizer.h"
#include "runtime/optimizers/adamw_optimizer.h"
#include "runtime/optimizers/normuon_optimizer.h"

namespace optimizers {

void Optimizer::step_graph(dsl::DslModel& /*model*/,
                           NCCLCommunicator& /*comm*/,
                           const OptimizerConfig& /*config*/,
                           const float* /*opt_params*/,
                           const int* /*opt_step*/) {
    throw std::runtime_error(std::string(name()) + ": CUDA graph capture not supported");
}

void Optimizer::prepare_for_graph(dsl::DslModel& /*model*/,
                                  NCCLCommunicator& /*comm*/,
                                  const OptimizerConfig& /*config*/) {
    // Default: no pre-capture initialization.
}

std::unique_ptr<Optimizer> create_optimizer(const OptimizerConfig& config) {
    switch (config.type) {
        case OptimizerType::ADAMW: return std::make_unique<AdamWOptimizer>();
        case OptimizerType::ADAMW_8BIT: return std::make_unique<AdamW8BitOptimizer>();
        case OptimizerType::NORMUON: return std::make_unique<NorMuonOptimizer>();
    }
    throw std::runtime_error(std::string("create_optimizer: unknown type ") +
                             std::string(optimizer_type_to_str(config.type)));
}

}  // namespace optimizers
