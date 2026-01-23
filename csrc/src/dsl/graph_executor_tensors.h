// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Tensor resolution functions for DSL Graph executor.

#ifndef SUROGATE_SRC_DSL_GRAPH_EXECUTOR_TENSORS_H
#define SUROGATE_SRC_DSL_GRAPH_EXECUTOR_TENSORS_H

#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/graph_executor_internal.h"
#include "utilities/tensor.h"

namespace dsl {

class DslRunState;

// Block activation tensor resolution
Tensor* resolve_block_activation_tensor(ExecState& st, const std::string& name, ETensorDType dtype,
                                        const std::vector<long>& shape);

Tensor* block_activation_base_ptr(DslRunState& rs, int layer_idx, const std::string& field);

Tensor* resolve_recomputed_block_tensor(ExecState& st, const std::string& name);

Tensor* resolve_block_activation_base(ExecState& st, const std::string& name);

// Block gradient tensor resolution
Tensor* resolve_block_gradient_tensor(ExecState& st, const std::string& name, ETensorDType dtype,
                                      const std::vector<long>& shape);

Tensor* resolve_gradient_view_tensor(ExecState& st,
                                     const std::string& name,
                                     const std::unordered_map<std::string, Tensor>& saved);

// General tensor resolution
Tensor& ensure_tensor(ExecState& st, const std::string& name, ETensorDType dtype, const std::vector<long>& shape);

Tensor& resolve_param_tensor(ExecState& st, const std::string& name);

Tensor& get_tensor(ExecState& st, const std::string& name, const std::unordered_map<std::string, Tensor>& saved);

Tensor* try_get_tensor(ExecState& st, const std::string& name, std::unordered_map<std::string, Tensor>& saved);

// View shape resolution
std::vector<long> resolve_view_shape(
    const Operation& op,
    const ShapeEnv& env,
    ExecState& st,
    std::unordered_map<std::string, Tensor>& saved);

}  // namespace dsl

#endif  // SUROGATE_SRC_DSL_GRAPH_EXECUTOR_TENSORS_H
