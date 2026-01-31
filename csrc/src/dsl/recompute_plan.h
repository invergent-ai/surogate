// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL-driven recompute plan construction + execution.

#ifndef SUROGATE_SRC_DSL_RECOMPUTE_PLAN_H
#define SUROGATE_SRC_DSL_RECOMPUTE_PLAN_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "dsl/ir.h"

namespace modules {
struct ModelConfig;
}

namespace dsl {

class GraphExecutor;

enum class RecomputePolicy : std::uint8_t {
    Always,
    LoraOnly,
    FftOnly,
    Never,
};

struct RecomputeOp {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    AttrMap attrs;
    RecomputePolicy policy = RecomputePolicy::Always;
    std::vector<std::string> lora_targets;
};

struct LayerRecomputePlan {
    std::vector<RecomputeOp> topo_ops;
    std::unordered_map<std::string, std::size_t> producer_index;
};

class RecomputePlan {
public:
    void init_from_layout(const ActivationLayoutIR& layout,
                          const AttrMap& module_config,
                          const modules::ModelConfig& model_cfg);

    void execute_layer(GraphExecutor& executor,
                       int layer_idx,
                       long B,
                       long T,
                       bool lora_only_mode,
                       cudaStream_t stream);

    bool empty() const { return mPlan.topo_ops.empty(); }

    bool can_recompute(const std::string& name) const;
    const std::vector<std::string>& get_dependencies(const std::string& name) const;

private:
    LayerRecomputePlan mPlan{};
    std::unordered_map<std::string, std::vector<std::string>> mDependencies;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_RECOMPUTE_PLAN_H
