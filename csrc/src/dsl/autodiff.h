// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// DSL Automatic Differentiation - derives backward graphs from forward graphs.

#ifndef SUROGATE_SRC_DSL_AUTODIFF_H
#define SUROGATE_SRC_DSL_AUTODIFF_H

#include "ir.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace dsl {

// Context passed to backward rules containing information about the forward op
// and what gradients are needed.
struct BackwardRuleContext {
    const Operation& fwd_op;                    // The forward operation
    const std::string& d_output;                // Name of output gradient (cotangent)
    const std::vector<std::string>& d_inputs;   // Names for input gradients to produce (empty = not needed)
    const ShapeEnv& shape_env;                  // Shape environment for resolving dimensions
    int& op_counter;                            // Counter for generating unique op IDs
    const Graph* forward_graph;                 // The full forward graph (for checking params)

    // Helper to check if gradient is needed for input at index
    bool needs_grad(size_t input_idx) const {
        return input_idx < d_inputs.size() && !d_inputs[input_idx].empty();
    }

    // Helper to check if a tensor is a parameter (available at backward time without saving)
    bool is_param(const std::string& name) const {
        if (!forward_graph) return false;
        return forward_graph->params.count(name) > 0;
    }

    // Helper to check if a tensor is a graph input
    bool is_input(const std::string& name) const {
        if (!forward_graph) return false;
        return forward_graph->inputs.count(name) > 0;
    }
};

// BackwardRule: given forward op context, generate backward operations.
// Returns: list of backward operations to add to backward graph.
using BackwardRule = std::function<std::vector<Operation>(const BackwardRuleContext& ctx)>;

// Registry for backward rules.
class BackwardRuleRegistry {
public:
    static BackwardRuleRegistry& instance();

    // Register a backward rule for an operation type
    void register_rule(const std::string& op_type, BackwardRule rule);

    // Get backward rule for an operation type (returns nullptr if not found)
    const BackwardRule* get_rule(const std::string& op_type) const;

    // Check if a rule exists
    bool has_rule(const std::string& op_type) const;

    // List all registered operation types
    std::vector<std::string> registered_ops() const;

private:
    BackwardRuleRegistry() = default;
    std::unordered_map<std::string, BackwardRule> rules_;
};

// Options for backward graph derivation
struct DeriveBackwardOptions {
    // Name of the loss tensor to differentiate from
    std::string loss_name = "loss";

    // Whether to automatically determine which tensors to save for backward
    bool auto_save = true;

    // Additional tensors to always save (beyond auto-detected)
    std::vector<std::string> extra_saves;

    // Whether to generate gradient accumulation ops for multi-use tensors
    bool accumulate_grads = true;

    // Prefix for generated gradient tensor names
    std::string grad_prefix = "d_";
};

// Derive a backward graph from a forward graph using registered backward rules.
// Throws if any operation lacks a backward rule.
Graph derive_backward_graph(const Graph& forward, const DeriveBackwardOptions& options = {});

// Compute which tensors need to be saved from forward for backward pass.
// This analyzes the backward graph to find all "saved.*" references.
std::vector<std::string> compute_required_saves(const Graph& forward, const Graph& backward);

// Helper to create an operation
Operation make_operation(
    const std::string& id,
    const std::string& name,
    const std::string& kernel_type,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const AttrMap& attrs = {});

// Convenience overload with auto-generated ID
Operation make_operation(
    const std::string& name,
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& outputs,
    const AttrMap& attrs = {},
    int* counter = nullptr);

// Helper to create a "saved.X" reference name
inline std::string saved_ref(const std::string& name) {
    return "saved." + name;
}

// Helper to create a gradient name
inline std::string grad_name(const std::string& name, const std::string& prefix = "d_") {
    return prefix + name;
}

// Initialize all built-in backward rules.
// Called automatically on first use, but can be called explicitly.
void register_builtin_backward_rules();

} // namespace dsl

#endif // SUROGATE_SRC_DSL_AUTODIFF_H
