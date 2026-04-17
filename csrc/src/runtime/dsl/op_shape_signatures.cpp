// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/op_shape_signatures.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"

namespace dsl {
namespace shape_checker {

// ============================================================================
// Registry Implementation
// ============================================================================

OpShapeRegistry& OpShapeRegistry::instance() {
    static OpShapeRegistry registry;
    static bool initialized = false;
    if (!initialized) {
        initialized = true;
        register_builtin_shape_signatures();
    }
    return registry;
}

void OpShapeRegistry::register_signature(const OpShapeSignature& sig) {
    signatures_[sig.op_name] = sig;
}

const OpShapeSignature* OpShapeRegistry::get_signature(const std::string& op_name) const {
    auto it = signatures_.find(op_name);
    return it != signatures_.end() ? &it->second : nullptr;
}

std::vector<std::string> OpShapeRegistry::registered_ops() const {
    std::vector<std::string> ops;
    ops.reserve(signatures_.size());
    for (const auto& [name, _] : signatures_) {
        ops.push_back(name);
    }
    return ops;
}

// ============================================================================
// Helper Validators
// ============================================================================

namespace validators {

std::optional<ShapeValidationError> check_same_rank(const std::vector<std::vector<long>>& shapes,
                                                    const std::string& op_name) {
    if (shapes.empty()) return std::optional<ShapeValidationError>();

    int expected_rank = static_cast<int>(shapes[0].size());
    for (size_t i = 1; i < shapes.size(); ++i) {
        if (static_cast<int>(shapes[i].size()) != expected_rank) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "rank mismatch in " << op_name << ": input[0] has rank " << expected_rank << " but input[" << i
                << "] has rank " << shapes[i].size();
            err.message = oss.str();
            return err;
        }
    }
    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_rank(const std::vector<long>& shape,
                                               int expected_rank,
                                               const std::string& tensor_name,
                                               const std::string& op_name) {
    if (static_cast<int>(shape.size()) != expected_rank) {
        ShapeValidationError err;
        std::ostringstream oss;
        oss << op_name << ": " << tensor_name << " has rank " << shape.size() << " but expected " << expected_rank;
        err.message = oss.str();
        return err;
    }
    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_same_numel(const std::vector<long>& shape1,
                                                     const std::vector<long>& shape2,
                                                     const std::string& name1,
                                                     const std::string& name2,
                                                     const std::string& op_name) {
    // Empty shape = unknown/not inferred — skip validation
    if (shape1.empty() || shape2.empty()) {
        return std::optional<ShapeValidationError>();
    }

    auto numel = [](const std::vector<long>& s) {
        return std::accumulate(s.begin(), s.end(), 1L, std::multiplies<long>());
    };

    long n1 = numel(shape1);
    long n2 = numel(shape2);

    if (n1 != n2) {
        ShapeValidationError err;
        std::ostringstream oss;
        oss << op_name << ": element count mismatch between " << name1 << " (" << n1 << " elements) and " << name2
            << " (" << n2 << " elements)";
        err.message = oss.str();

        // Add shape details to hint
        std::ostringstream hint_oss;
        hint_oss << name1 << " shape: (";
        for (size_t i = 0; i < shape1.size(); ++i) {
            if (i > 0) hint_oss << ", ";
            hint_oss << shape1[i];
        }
        hint_oss << "), " << name2 << " shape: (";
        for (size_t i = 0; i < shape2.size(); ++i) {
            if (i > 0) hint_oss << ", ";
            hint_oss << shape2[i];
        }
        hint_oss << ")";
        err.hint = hint_oss.str();

        return err;
    }
    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_matmul_dims(const std::vector<long>& a_shape,
                                                      const std::vector<long>& b_shape,
                                                      const std::vector<long>& out_shape,
                                                      const AttrMap& attrs) {
    if (a_shape.size() < 2 || b_shape.size() < 2) {
        ShapeValidationError err;
        err.message = "matmul: inputs must have at least rank 2";
        return err;
    }

    // Parse transpose mode
    EMMTranspose mode = parse_transpose(attrs);

    // Extract M, K, N based on transpose mode
    long M, K_a, K_b, N;
    if (mode == EMMTranspose::NN) {
        M = a_shape[a_shape.size() - 2];
        K_a = a_shape[a_shape.size() - 1];
        K_b = b_shape[b_shape.size() - 2];
        N = b_shape[b_shape.size() - 1];
    } else if (mode == EMMTranspose::NT) {
        M = a_shape[a_shape.size() - 2];
        K_a = a_shape[a_shape.size() - 1];
        N = b_shape[b_shape.size() - 2];
        K_b = b_shape[b_shape.size() - 1];
    } else if (mode == EMMTranspose::TN) {
        M = a_shape[a_shape.size() - 1];
        K_a = a_shape[a_shape.size() - 2];
        K_b = b_shape[b_shape.size() - 2];
        N = b_shape[b_shape.size() - 1];
    } else {  // TT
        M = a_shape[a_shape.size() - 1];
        K_a = a_shape[a_shape.size() - 2];
        N = b_shape[b_shape.size() - 2];
        K_b = b_shape[b_shape.size() - 1];
    }

    // Check K dimensions match
    if (K_a != K_b) {
        ShapeValidationError err;
        std::ostringstream oss;
        oss << "matmul: contraction dimension mismatch: K_a=" << K_a << " != K_b=" << K_b;
        err.message = oss.str();

        std::ostringstream hint;
        hint << "Transpose mode: "
             << (mode == EMMTranspose::NN   ? "NN"
                 : mode == EMMTranspose::NT ? "NT"
                 : mode == EMMTranspose::TN ? "TN"
                                            : "TT")
             << ", A shape: (";
        for (size_t i = 0; i < a_shape.size(); ++i) {
            if (i > 0) hint << ", ";
            hint << a_shape[i];
        }
        hint << "), B shape: (";
        for (size_t i = 0; i < b_shape.size(); ++i) {
            if (i > 0) hint << ", ";
            hint << b_shape[i];
        }
        hint << ")";
        err.hint = hint.str();

        return err;
    }

    // Check output shape if provided
    if (!out_shape.empty()) {
        if (out_shape.size() < 2) {
            ShapeValidationError err;
            err.message = "matmul: output must have at least rank 2";
            return err;
        }

        if (out_shape[out_shape.size() - 2] != M) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "matmul: output dim[-2] mismatch: expected " << M << " but got " << out_shape[out_shape.size() - 2];
            err.message = oss.str();
            return err;
        }

        if (out_shape[out_shape.size() - 1] != N) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "matmul: output dim[-1] mismatch: expected " << N << " but got " << out_shape[out_shape.size() - 1];
            err.message = oss.str();
            return err;
        }

        // Check batch dimensions
        size_t min_rank = std::min({a_shape.size(), b_shape.size(), out_shape.size()});
        for (size_t i = 0; i < min_rank - 2; ++i) {
            if (a_shape[i] != b_shape[i] || a_shape[i] != out_shape[i]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "matmul: batch dimension [" << i << "] mismatch: " << "A[" << i << "]=" << a_shape[i] << ", "
                    << "B[" << i << "]=" << b_shape[i] << ", " << "out[" << i << "]=" << out_shape[i];
                err.message = oss.str();
                return err;
            }
        }
    }

    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError>
check_broadcastable(const std::vector<long>& shape1, const std::vector<long>& shape2, const std::string& op_name) {
    // Broadcast rules: dimensions must be equal or one of them must be 1
    size_t max_rank = std::max(shape1.size(), shape2.size());

    for (size_t i = 0; i < max_rank; ++i) {
        // Index from the right
        long d1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        long d2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (d1 != d2 && d1 != 1 && d2 != 1) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << op_name << ": shapes not broadcastable at dimension " << (max_rank - 1 - i) << ": " << d1 << " vs "
                << d2;
            err.message = oss.str();
            return err;
        }
    }

    return std::optional<ShapeValidationError>();
}

}  // namespace validators

// ============================================================================
// Built-in Operation Signatures — Phase 2c: now a no-op.
//
// Signatures used to be registered in one centralized function here;
// Phase 2c moved each signature to its matching per-op file under
// runtime/ops/, where it registers itself via a static initializer.
// This function is kept so OpShapeRegistry::instance() still has
// something to call on first access, but it does nothing — by the time
// it runs, the per-op static initializers have already populated the
// registry.
// ============================================================================

void register_builtin_shape_signatures() {
}

}  // namespace shape_checker
}  // namespace dsl
