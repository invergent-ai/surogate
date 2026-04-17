// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Automatic Differentiation implementation.

#include "autodiff.h"

#include <algorithm>
#include <cstdio>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#include "runtime/executor/graph_executor_utils.h"
#include "runtime/executor/op_registry.h"

namespace dsl {

// -----------------------------------------------------------------------------
// BackwardRuleRegistry (Phase 2b: now a thin shim over OpRegistry)
// -----------------------------------------------------------------------------
//
// The old BackwardRuleRegistry is kept only so external callers that
// still invoke `instance()` / `has_rule` / `registered_ops` keep
// working during the transition. All actual storage lives in
// OpRegistry; this shim proxies every operation to it.

BackwardRuleRegistry& BackwardRuleRegistry::instance() {
    static BackwardRuleRegistry registry;
    return registry;
}

void BackwardRuleRegistry::register_rule(const std::string& op_type, BackwardRule rule) {
    OpDescriptor desc;
    desc.name = op_type;
    desc.autodiff_fn = std::move(rule);
    OpRegistry::instance().register_op(std::move(desc));
}

const BackwardRule* BackwardRuleRegistry::get_rule(const std::string& op_type) const {
    const OpDescriptor* d = OpRegistry::instance().find_by_name(op_type);
    return (d && d->autodiff_fn) ? &d->autodiff_fn : nullptr;
}

bool BackwardRuleRegistry::has_rule(const std::string& op_type) const {
    return get_rule(op_type) != nullptr;
}

std::vector<std::string> BackwardRuleRegistry::registered_ops() const {
    // Unused by the codebase except in tests; returning empty is safe.
    return {};
}

// -----------------------------------------------------------------------------
// Promoted helpers (Phase 2b)
// -----------------------------------------------------------------------------
//
// `find_attr` already lives in graph_executor_utils.cpp; we reuse it
// (included at the top of this file) instead of duplicating. The other
// two helpers (get_string_attr, copy_attrs) are autodiff-specific and
// live here.

std::string get_string_attr(const AttrMap& attrs, const std::string& key, const std::string& default_val) {
    if (auto* attr = find_attr(attrs, std::string_view(key))) {
        if (auto* s = std::get_if<std::string>(&attr->value)) {
            return *s;
        }
    }
    return default_val;
}

AttrMap copy_attrs(const AttrMap& src, const std::vector<std::string>& keys, const char* rule_name) {
    AttrMap dst;
    for (const auto& key : keys) {
        if (auto* attr = find_attr(src, std::string_view(key))) {
            dst[key] = *attr;
        } else if (rule_name) {
            fprintf(stderr,
                    "WARNING [autodiff]: backward rule '%s' requested attr '%s' "
                    "not found in forward op attrs\n",
                    rule_name,
                    key.c_str());
        }
    }
    return dst;
}

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------

Operation make_operation(const std::string& id,
                         const std::string& name,
                         const std::string& kernel_type,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const AttrMap& attrs) {
    Operation op;
    op.id = id;
    op.name = name;
    op.kernel_type = kernel_type;
    op.inputs = inputs;
    op.outputs = outputs;
    op.attrs = attrs;
    return op;
}

Operation make_operation(const std::string& name,
                         const std::vector<std::string>& inputs,
                         const std::vector<std::string>& outputs,
                         const AttrMap& attrs,
                         int* counter) {
    std::string id = name;
    if (counter) {
        id += "_" + std::to_string((*counter)++);
    }
    return make_operation(id, name, name, inputs, outputs, attrs);
}

// -----------------------------------------------------------------------------
// derive_backward_graph implementation
// -----------------------------------------------------------------------------

namespace {

// Get the operation type (kernel_type if set and not "custom", otherwise name)
std::string get_op_type(const Operation& op) {
    if (op.kernel_type.empty() || op.kernel_type == "custom") {
        return op.name;
    }
    return op.kernel_type;
}

// Check if a string starts with a prefix
bool starts_with(const std::string& str, const std::string& prefix) {
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool is_non_diff_dtype(ETensorDType dtype) {
    switch (dtype) {
        case ETensorDType::INT32:
        case ETensorDType::INT8:
        case ETensorDType::BYTE: return true;
        default: return false;
    }
}

bool is_non_differentiable(const Graph& forward, const std::string& name) {
    // Check graph inputs
    auto it_input = forward.inputs.find(name);
    if (it_input != forward.inputs.end()) {
        if (it_input->second.dtype && is_non_diff_dtype(*it_input->second.dtype)) {
            return true;
        }
    }
    // Check graph params
    auto it_param = forward.params.find(name);
    if (it_param != forward.params.end()) {
        if (it_param->second.dtype && is_non_diff_dtype(*it_param->second.dtype)) {
            return true;
        }
        if (name.find("rope_freqs") != std::string::npos) {
            return true;
        }
    }
    // Check intermediate tensors (e.g., MoE scatter_indices)
    auto it_inter = forward.intermediates.find(name);
    if (it_inter != forward.intermediates.end()) {
        if (it_inter->second.dtype && is_non_diff_dtype(*it_inter->second.dtype)) {
            return true;
        }
    }
    // Also handle MoE index tensors by name pattern (in case intermediates map is incomplete)
    if (name.find("scatter_indices") != std::string::npos || name.find("routing_indices") != std::string::npos ||
        name.find("gather_indices") != std::string::npos || name.find("expert_offsets") != std::string::npos ||
        name.find("ep_recv_scatter") != std::string::npos) {
        return true;
    }
    return false;
}

}  // namespace

Graph derive_backward_graph(const Graph& forward, const DeriveBackwardOptions& options) {
    Graph backward;
    backward.name = forward.name + "_backward";

    const std::unordered_set<std::string> stop_set(options.stop_gradients.begin(), options.stop_gradients.end());
    auto is_stopped = [&](const std::string& name) -> bool {
        return stop_set.find(name) != stop_set.end();
    };

    // Force BackwardRuleRegistry initialization (triggers
    // register_builtin_backward_rules, which is now a thin shim that
    // forwards into OpRegistry). Harmless once autodiff_rules.cpp is
    // retired; the shim returns immediately.
    BackwardRuleRegistry::instance();
    auto& op_reg = OpRegistry::instance();
    int op_counter = 0;

    // Build map: tensor_name -> index of operation that produces it
    std::unordered_map<std::string, size_t> produced_by;
    for (size_t i = 0; i < forward.operations.size(); ++i) {
        for (const auto& out : forward.operations[i].outputs) {
            produced_by[out] = i;
        }
    }

    // Mark forward graph inputs as "produced externally" with sentinel value
    constexpr size_t EXTERNAL = static_cast<size_t>(-1);
    for (const auto& [name, _] : forward.inputs) {
        produced_by[name] = EXTERNAL;
    }
    for (const auto& [name, _] : forward.params) {
        produced_by[name] = EXTERNAL;
    }

    // Determine which tensors need gradients (backward traversal from loss)
    std::unordered_set<std::string> needs_grad;
    std::queue<std::string> worklist;

    // Start from loss (or all outputs if loss not found)
    if (forward.outputs.count(options.loss_name)) {
        worklist.push(options.loss_name);
        needs_grad.insert(options.loss_name);
    } else {
        // Fall back to all outputs
        for (const auto& [name, _] : forward.outputs) {
            worklist.push(name);
            needs_grad.insert(name);
        }
    }

    // Propagate needs_grad backward through the graph
    while (!worklist.empty()) {
        std::string tensor = worklist.front();
        worklist.pop();

        auto it = produced_by.find(tensor);
        if (it == produced_by.end() || it->second == EXTERNAL) {
            continue;
        }

        const auto& op = forward.operations[it->second];
        for (const auto& inp : op.inputs) {
            if (is_non_differentiable(forward, inp) || is_stopped(inp)) {
                continue;
            }
            if (needs_grad.insert(inp).second) {
                worklist.push(inp);
            }
        }
    }

    // Gradient map: tensor_name -> current gradient tensor name
    // (may change due to accumulation)
    std::unordered_map<std::string, std::string> grad_map;

    // Initialize gradient for loss/outputs (cotangent = 1, but typically provided externally)
    for (const auto& [name, _] : forward.outputs) {
        if (needs_grad.count(name)) {
            grad_map[name] = options.grad_prefix + name;
        }
    }

    // Build shape environment from forward graph config
    ShapeEnv shape_env;
    // TODO: populate from forward.config if available

    // Process operations in reverse order (reverse topological)
    for (auto it = forward.operations.rbegin(); it != forward.operations.rend(); ++it) {
        const Operation& fwd_op = *it;
        const std::string op_type = get_op_type(fwd_op);

        // Check if any output of this op needs gradient
        bool has_grad_output = false;
        for (const auto& out : fwd_op.outputs) {
            if (needs_grad.count(out) && grad_map.count(out)) {
                has_grad_output = true;
                break;
            }
        }
        if (!has_grad_output) {
            continue;
        }

        // Get backward rule from the unified OpRegistry (Phase 2b).
        const OpDescriptor* desc = op_reg.find_by_name(op_type);
        if (!desc || !desc->autodiff_fn) {
            throw std::runtime_error("Autodiff: no backward rule registered for operation '" + op_type + "'");
        }
        const AutodiffFn& rule = desc->autodiff_fn;

        // Determine output gradient names (one per forward output)
        std::vector<std::string> d_outputs(fwd_op.outputs.size());
        std::string d_output;
        for (size_t i = 0; i < fwd_op.outputs.size(); ++i) {
            const auto& out = fwd_op.outputs[i];
            auto it_grad = grad_map.find(out);
            if (it_grad != grad_map.end()) {
                d_outputs[i] = it_grad->second;
                if (d_output.empty()) {
                    d_output = d_outputs[i];
                }
            }
        }
        if (d_output.empty()) {
            continue;  // No gradient available for any output
        }

        // Determine input gradient names
        std::vector<std::string> d_inputs;
        d_inputs.reserve(fwd_op.inputs.size());
        for (size_t i = 0; i < fwd_op.inputs.size(); ++i) {
            const auto& inp = fwd_op.inputs[i];
            if (needs_grad.count(inp) && !is_non_differentiable(forward, inp) && !is_stopped(inp)) {
                // Use simple name if first gradient for this tensor, else unique name
                std::string d_inp;
                if (!grad_map.count(inp)) {
                    d_inp = options.grad_prefix + inp;  // Simple: d_xF
                } else {
                    d_inp = options.grad_prefix + inp + "_from_" + fwd_op.id;  // Unique for accumulation
                }
                d_inputs.push_back(d_inp);
            } else {
                d_inputs.push_back("");  // No gradient needed
            }
        }

        // Create context and call backward rule
        BackwardRuleContext ctx{fwd_op, d_outputs, d_output, d_inputs, shape_env, op_counter, &forward};
        std::vector<Operation> bwd_ops = rule(ctx);

        // Add generated operations to backward graph
        for (auto& bwd_op : bwd_ops) {
            backward.operations.push_back(std::move(bwd_op));
        }

        // Update gradient map with accumulation if needed
        for (size_t i = 0; i < fwd_op.inputs.size(); ++i) {
            if (d_inputs[i].empty()) {
                continue;
            }

            const std::string& inp = fwd_op.inputs[i];

            if (options.accumulate_grads && grad_map.count(inp)) {
                // Tensor already has a gradient - need to accumulate
                std::string accum_name = options.grad_prefix + inp + "_accum_" + std::to_string(op_counter++);

                Operation add_op = make_operation("add_grad_" + std::to_string(op_counter),
                                                  "add",
                                                  "add",
                                                  {grad_map[inp], d_inputs[i]},
                                                  {accum_name});
                backward.operations.push_back(add_op);

                grad_map[inp] = accum_name;
            } else {
                grad_map[inp] = d_inputs[i];
            }
        }
    }

    // Set backward graph inputs (gradients of forward outputs)
    for (const auto& [name, info] : forward.outputs) {
        std::string d_name = options.grad_prefix + name;
        backward.inputs[d_name] = info;
    }

    // Re-expose forward params on the backward graph so downstream passes
    // (e.g. ``inject_lora_slices`` in graph_compiler) can look up a weight's
    // IR-declared metadata by name from either graph. Without this, LoRA
    // slices on backward matmul ops stay empty and gradient norms are zero.
    backward.params = forward.params;

    // Set backward graph outputs (gradients of forward inputs/params)
    for (const auto& [name, info] : forward.inputs) {
        if (grad_map.count(name)) {
            backward.outputs[grad_map[name]] = info;
        }
    }
    for (const auto& [name, info] : forward.params) {
        if (grad_map.count(name) && !is_stopped(name)) {
            backward.outputs[grad_map[name]] = info;
        }
    }

    // Compute required saves if auto_save enabled
    if (options.auto_save) {
        auto saves = compute_required_saves(forward, backward);
        backward.save = std::move(saves);
    }

    // Add extra saves
    for (const auto& s : options.extra_saves) {
        if (std::find(backward.save.begin(), backward.save.end(), s) == backward.save.end()) {
            backward.save.push_back(s);
        }
    }

    return backward;
}

std::vector<std::string> compute_required_saves(const Graph& forward, const Graph& backward) {
    std::unordered_set<std::string> needed;

    // Scan backward ops for "saved.*" references in inputs
    for (const auto& op : backward.operations) {
        for (const auto& inp : op.inputs) {
            if (starts_with(inp, "saved.")) {
                needed.insert(inp.substr(6));  // Remove "saved." prefix
            }
        }

        // Also check attributes that might reference saved tensors (e.g., shape_like)
        for (const auto& [key, value] : op.attrs) {
            if (auto* str = std::get_if<std::string>(&value.value)) {
                if (starts_with(*str, "saved.")) {
                    needed.insert(str->substr(6));
                }
            }
        }
    }

    return {needed.begin(), needed.end()};
}

}  // namespace dsl
