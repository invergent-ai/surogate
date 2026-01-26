// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Operation shape signatures for compile-time validation.
//
// This module defines the expected input/output shapes for each DSL operation
// and provides validation functions to catch shape mismatches at graph
// compilation time rather than at runtime.

#ifndef SUROGATE_SRC_DSL_OP_SHAPE_SIGNATURES_H
#define SUROGATE_SRC_DSL_OP_SHAPE_SIGNATURES_H

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "dsl/ir.h"

namespace dsl {
namespace shape_checker {

// Shape validation result
struct ShapeValidationError {
    std::string message;
    std::string hint;  // Optional hint for fixing the error
};

// Operation shape signature with custom validation
struct OpShapeSignature {
    std::string op_name;

    // Custom validation function
    // Returns nullopt if valid, or error message if invalid
    using ValidatorFn = std::function<std::optional<ShapeValidationError>(
        const std::vector<std::vector<long>>& input_shapes,
        const std::vector<std::vector<long>>& output_shapes,
        const AttrMap& attrs,
        const ShapeEnv& env
    )>;

    ValidatorFn validator;

    // Optional: expected input/output counts
    int min_inputs = -1;   // -1 = any
    int max_inputs = -1;
    int min_outputs = -1;
    int max_outputs = -1;
};

// Shape signature registry (singleton)
class OpShapeRegistry {
public:
    static OpShapeRegistry& instance();

    void register_signature(const OpShapeSignature& sig);
    const OpShapeSignature* get_signature(const std::string& op_name) const;

    // Check if operation has a registered signature
    bool has_signature(const std::string& op_name) const {
        return signatures_.find(op_name) != signatures_.end();
    }

    // Get all registered operation names
    std::vector<std::string> registered_ops() const;

private:
    OpShapeRegistry() = default;
    std::unordered_map<std::string, OpShapeSignature> signatures_;
};

// Register all built-in operation signatures
void register_builtin_shape_signatures();

// Helper functions for common shape checks
namespace validators {

// Check that all input shapes have the same rank
std::optional<ShapeValidationError> check_same_rank(
    const std::vector<std::vector<long>>& shapes,
    const std::string& op_name);

// Check that tensor has expected rank
std::optional<ShapeValidationError> check_rank(
    const std::vector<long>& shape,
    int expected_rank,
    const std::string& tensor_name,
    const std::string& op_name);

// Check that two shapes have the same number of elements (for reshape/view)
std::optional<ShapeValidationError> check_same_numel(
    const std::vector<long>& shape1,
    const std::vector<long>& shape2,
    const std::string& name1,
    const std::string& name2,
    const std::string& op_name);

// Check that dimensions are compatible for matmul
std::optional<ShapeValidationError> check_matmul_dims(
    const std::vector<long>& a_shape,
    const std::vector<long>& b_shape,
    const std::vector<long>& out_shape,
    const AttrMap& attrs);

// Check that batch dimensions are broadcastable
std::optional<ShapeValidationError> check_broadcastable(
    const std::vector<long>& shape1,
    const std::vector<long>& shape2,
    const std::string& op_name);

}  // namespace validators

}}  // namespace dsl::shape_checker

#endif  // SUROGATE_SRC_DSL_OP_SHAPE_SIGNATURES_H
