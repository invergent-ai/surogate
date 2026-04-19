// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Utility functions for DSL Graph executor.

#ifndef SUROGATE_SRC_EXECUTOR_GRAPH_EXECUTOR_UTILS_H
#define SUROGATE_SRC_EXECUTOR_GRAPH_EXECUTOR_UTILS_H

#include <optional>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/executor/graph_executor_internal.h"
#include "runtime/dsl/ir.h"
#include "kernels/kernels.h"
#include "utilities/stack.h"
#include "utilities/tensor.h"
#include "utilities/utils.h"

namespace dsl {

// String utilities
bool starts_with(std::string_view value, std::string_view prefix);
bool ends_with(std::string_view value, std::string_view suffix);

// Environment variable check
bool env_enabled(const char* name);

// *** COMPILE-TIME HEURISTIC ONLY — DO NOT CALL FROM RUNTIME DISPATCHERS. ***
//
// Returns Some(base) for ANY name starting with `d_` (after stripping
// `_from_N` / `_accum_N`). Does NOT verify the base is a real parameter — that
// is the caller's job (e.g. `mWeights.has(*base)` after the call). This
// pattern (heuristic + validate) is acceptable at compile time where we have
// no compiled graph yet. Any runtime dispatcher that has access to the
// compiled graph MUST use `base_param_from_grad_kind()` instead; the
// classifier is the only source of truth for "is this a parameter gradient?".
//
// Named `_heuristic` so greps for "base_param_from_grad(" land on the
// authoritative classifier overload first and nobody accidentally re-
// introduces the misclassification bug class we retired.
std::optional<std::string> base_param_from_grad_heuristic(std::string_view name);

struct CompiledGraph;  // fwd

// Classifier-backed resolution. Returns the parameter name ONLY when the
// tensor has TensorKind::ParamGrad. For ActivationGrad / AccumTemp /
// Scratch / etc. returns nullopt — which is the correct shape for callers
// that want to route tensors through `mGrads.get_param_grad`.
//
// This is the blessed runtime API. Use it from every op dispatcher that
// needs to ask "is this output a real parameter gradient?".
std::optional<std::string> base_param_from_grad_kind(int tensor_id, const CompiledGraph& graph);
std::optional<std::string> base_param_from_grad_kind(std::string_view name, const CompiledGraph& graph);

// Block parameter parsing (e.g., "blocks[0].qkv_weight" -> layer_idx=0, param_name="qkv_weight")
bool parse_block_param(std::string_view name, int& layer_idx, std::string& param_name);

// Shape inference and utilities
bool infer_block_tensor_shape(const ExecState& st, std::string_view name, std::vector<long>& shape);
std::string tensor_shape_str(const Tensor& t);
bool tensor_shape_matches(const Tensor& t, const std::vector<long>& shape);
std::size_t shape_nelem(const std::vector<long>& shape);

// Attribute access helpers
const AttrValue* find_attr(const AttrMap& attrs, std::string_view key);
std::optional<std::string> attr_string(const AttrValue& value);
std::optional<long> attr_int(const AttrValue& value);
std::optional<double> attr_double(const AttrValue& value);
std::optional<bool> attr_bool(const AttrValue& value);
std::optional<std::vector<long>> attr_list_int(const AttrValue& value);

// Shape environment augmentation
void augment_shape_env(ShapeEnv& env, const AttrMap& config);
std::vector<long> resolve_attr_shape(const AttrValue& value, const ShapeEnv& env);

// Tensor view utilities
Tensor view_tensor(const Tensor& src, const std::vector<long>& shape);
Tensor view_for_shape(const Tensor& src, const std::vector<long>& shape, const std::string& name);

/// Collapse a rank>2 tensor whose leading dims are ``[B, T, ...]`` into a
/// 2-D ``[B*T, D]`` view. Tensors already rank-2 or with non-matching
/// leading dims pass through unchanged.
inline Tensor flatten_bt(const Tensor& t, long B, long T) {
    if (t.Rank > 2 && t.Sizes[0] == B && t.Sizes[1] == T) {
        return view_tensor(t, {B * T, t.Sizes[t.Rank - 1]});
    }
    return t;
}

// Matmul utilities
std::optional<::modules::MatmulOp> matmul_op_from_weight(std::string_view name, int& layer_idx);
EMMTranspose parse_transpose(const AttrMap& attrs);
EMMTranspose swap_transpose(EMMTranspose mode);
void matmul_dims(const Tensor& a, const Tensor& b, EMMTranspose mode, int& M, int& N, int& K);

// Graph utilities
bool is_required_op(const Operation& op, const std::unordered_set<std::string>& needed);
std::vector<char> compute_required_ops(const Graph& graph, const std::vector<std::string>& outputs);

// Temporary memory management
void free_temps(ExecState& st);

// ---------------------------------------------------------------------------
// CUDA graph capture/replay with stack checkpoint management.
//
// Ensures temp_alloc returns the same addresses across graph replays by
// saving/restoring the stack allocator state.
// ---------------------------------------------------------------------------
template <typename Function>
inline void trace_or_execute_cuda_graph_with_stack(Function&& function,
                                                   cudaStream_t stream,
                                                   cudaGraphExec_t& instance,
                                                   bool enabled,
                                                   DeviceMemoryStack& stack,
                                                   DeviceMemoryStack::Checkpoint& checkpoint) {
    if (!enabled) {
        function();
        return;
    }

    // Fast path: restore stack state and replay existing executable.
    if (instance != nullptr) {
        stack.restore(checkpoint);
        CUDA_CHECK(cudaGraphLaunch(instance, stream));
        return;
    }

    // Capture path: save checkpoint before capture so we know where to restore to.
    checkpoint = stack.checkpoint();

    cudaGraph_t graph = nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    function();
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    CUDA_CHECK(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    CUDA_CHECK(cudaGraphDestroy(graph));
    CUDA_CHECK(cudaGraphLaunch(instance, stream));
}

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_GRAPH_EXECUTOR_UTILS_H
