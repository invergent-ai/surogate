// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Internal types and helpers for DSL Graph executor.

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_INTERNAL_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_INTERNAL_H

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dsl/dsl_runtime.h"
#include "dsl/ir.h"
#include "modules/matmul_context.h"
#include "modules/model_config.h"
#include "utilities/tensor.h"

class NCCLCommunicator;

namespace dsl {

class DslParamStore;
class DslGradStore;

// Prefix for saved tensors in backward pass
constexpr std::string_view kSavedPrefix = "saved.";

// Execution state passed through graph execution
struct ExecState {
    DslRunState& rs;
    DslParamStore& weights;
    DslGradStore& grads;
    const ::modules::ModelConfig& config;
    long B = 0;
    long T = 0;
    ShapeEnv shape_env{};
    const std::unordered_map<std::string, std::string>* view_sources = nullptr;
    const std::unordered_map<std::string, std::string>* view_sources_rev = nullptr;
    const std::vector<char>* recomputed_layers = nullptr;

    std::unordered_map<std::string, Tensor> tensors;
    std::unordered_set<std::string> zero_tensors;
    std::vector<Tensor> temps;
};

// FP8 weight cache entry
struct FP8WeightCacheEntry {
    Tensor weight;
    Tensor stats;
    bool initialized = false;
};

// FP4 weight cache entry for NVFP4 recipe (Blackwell+)
struct FP4WeightCacheEntry {
    Tensor data;      ///< FP4 packed data (N, K/2) for forward; (K, N/2) for transposed
    Tensor scales;    ///< Block scales (FP8 E4M3, CUTLASS layout)
    Tensor amax;      ///< Global amax (FP32, single element)
    bool initialized = false;
};

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_INTERNAL_H
