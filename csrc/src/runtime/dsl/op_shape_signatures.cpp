// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/op_shape_signatures.h"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "runtime/dsl/graph_executor_utils.h"
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

std::optional<ShapeValidationError> check_same_rank(
    const std::vector<std::vector<long>>& shapes,
    const std::string& op_name) {
    if (shapes.empty()) return std::optional<ShapeValidationError>();

    int expected_rank = static_cast<int>(shapes[0].size());
    for (size_t i = 1; i < shapes.size(); ++i) {
        if (static_cast<int>(shapes[i].size()) != expected_rank) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "rank mismatch in " << op_name << ": input[0] has rank "
                << expected_rank << " but input[" << i << "] has rank " << shapes[i].size();
            err.message = oss.str();
            return err;
        }
    }
    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_rank(
    const std::vector<long>& shape,
    int expected_rank,
    const std::string& tensor_name,
    const std::string& op_name) {
    if (static_cast<int>(shape.size()) != expected_rank) {
        ShapeValidationError err;
        std::ostringstream oss;
        oss << op_name << ": " << tensor_name << " has rank " << shape.size()
            << " but expected " << expected_rank;
        err.message = oss.str();
        return err;
    }
    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_same_numel(
    const std::vector<long>& shape1,
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
        oss << op_name << ": element count mismatch between " << name1 << " (" << n1
            << " elements) and " << name2 << " (" << n2 << " elements)";
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

std::optional<ShapeValidationError> check_matmul_dims(
    const std::vector<long>& a_shape,
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
        hint << "Transpose mode: " << (mode == EMMTranspose::NN ? "NN" :
                                       mode == EMMTranspose::NT ? "NT" :
                                       mode == EMMTranspose::TN ? "TN" : "TT")
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
            oss << "matmul: output dim[-2] mismatch: expected " << M
                << " but got " << out_shape[out_shape.size() - 2];
            err.message = oss.str();
            return err;
        }

        if (out_shape[out_shape.size() - 1] != N) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "matmul: output dim[-1] mismatch: expected " << N
                << " but got " << out_shape[out_shape.size() - 1];
            err.message = oss.str();
            return err;
        }

        // Check batch dimensions
        size_t min_rank = std::min({a_shape.size(), b_shape.size(), out_shape.size()});
        for (size_t i = 0; i < min_rank - 2; ++i) {
            if (a_shape[i] != b_shape[i] || a_shape[i] != out_shape[i]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "matmul: batch dimension [" << i << "] mismatch: "
                    << "A[" << i << "]=" << a_shape[i] << ", "
                    << "B[" << i << "]=" << b_shape[i] << ", "
                    << "out[" << i << "]=" << out_shape[i];
                err.message = oss.str();
                return err;
            }
        }
    }

    return std::optional<ShapeValidationError>();
}

std::optional<ShapeValidationError> check_broadcastable(
    const std::vector<long>& shape1,
    const std::vector<long>& shape2,
    const std::string& op_name) {
    // Broadcast rules: dimensions must be equal or one of them must be 1
    size_t max_rank = std::max(shape1.size(), shape2.size());

    for (size_t i = 0; i < max_rank; ++i) {
        // Index from the right
        long d1 = (i < shape1.size()) ? shape1[shape1.size() - 1 - i] : 1;
        long d2 = (i < shape2.size()) ? shape2[shape2.size() - 1 - i] : 1;

        if (d1 != d2 && d1 != 1 && d2 != 1) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << op_name << ": shapes not broadcastable at dimension "
                << (max_rank - 1 - i) << ": " << d1 << " vs " << d2;
            err.message = oss.str();
            return err;
        }
    }

    return std::optional<ShapeValidationError>();
}

}  // namespace validators

// ============================================================================
// Built-in Operation Signatures
// ============================================================================

void register_builtin_shape_signatures() {
    auto& reg = OpShapeRegistry::instance();

    // ------------------------------------------------------------------------
    // Matmul
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "matmul";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap& attrs, const ShapeEnv&) {
            if (inputs.size() < 2 || outputs.empty()) {
                ShapeValidationError err;
                err.message = "matmul requires 2 inputs and 1 output";
                return std::make_optional(err);
            }
            return validators::check_matmul_dims(inputs[0], inputs[1],
                                                  outputs[0], attrs);
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // Matmul + Bias
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "matmul_bias";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap& attrs, const ShapeEnv&) {
            if (inputs.size() < 3 || outputs.empty()) {
                ShapeValidationError err;
                err.message = "matmul_bias requires 3 inputs and 1 output";
                return std::make_optional(err);
            }

            // Check matmul dims
            auto matmul_err = validators::check_matmul_dims(inputs[0], inputs[1],
                                                             outputs[0], attrs);
            if (matmul_err) return matmul_err;

            // Check bias shape (should be broadcastable with output)
            const auto& bias_shape = inputs[2];
            const auto& out_shape = outputs[0];
            if (bias_shape.size() > out_shape.size()) {
                ShapeValidationError err;
                err.message = "matmul_bias: bias rank exceeds output rank";
                return std::make_optional(err);
            }

            // Bias last dim should match output last dim
            if (!bias_shape.empty() && !out_shape.empty()) {
                if (bias_shape.back() != out_shape.back() && bias_shape.back() != 1) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "matmul_bias: bias last dim (" << bias_shape.back()
                        << ") doesn't match output last dim (" << out_shape.back() << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // View / Reshape
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "view";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.empty() || outputs.empty()) {
                ShapeValidationError err;
                err.message = "view requires 1 input and 1 output";
                return std::make_optional(err);
            }
            if (outputs[0].empty()) {
                return std::optional<ShapeValidationError>();
            }
            return validators::check_same_numel(inputs[0], outputs[0],
                                                 "input", "output", "view");
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // Add (elementwise)
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "add";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.size() < 2 || outputs.empty()) {
                ShapeValidationError err;
                err.message = "add requires 2 inputs and 1 output";
                return std::make_optional(err);
            }
            return validators::check_broadcastable(inputs[0], inputs[1], "add");
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // SwiGLU
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "swiglu";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.empty() || outputs.empty()) {
                ShapeValidationError err;
                err.message = "swiglu requires 1 input and 1 output";
                return std::make_optional(err);
            }

            const auto& in_shape = inputs[0];
            const auto& out_shape = outputs[0];

            // Input last dim should be 2x output last dim
            if (!in_shape.empty() && !out_shape.empty()) {
                if (in_shape.back() != 2 * out_shape.back()) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "swiglu: input last dim (" << in_shape.back()
                        << ") should be 2x output last dim (" << out_shape.back() << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            // All other dims should match
            if (in_shape.size() != out_shape.size()) {
                ShapeValidationError err;
                err.message = "swiglu: input and output rank must match";
                return std::make_optional(err);
            }

            for (size_t i = 0; i + 1 < in_shape.size(); ++i) {
                if (in_shape[i] != out_shape[i]) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "swiglu: dimension [" << i << "] mismatch: "
                        << in_shape[i] << " != " << out_shape[i];
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // Embedding
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "embedding";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.size() < 2 || outputs.empty()) {
                ShapeValidationError err;
                err.message = "embedding requires 2 inputs (indices, weight) and 1 output";
                return std::make_optional(err);
            }

            const auto& indices_shape = inputs[0];
            const auto& weight_shape = inputs[1];
            const auto& out_shape = outputs[0];

            // Weight should be 2D: [vocab_size, embedding_dim]
            if (weight_shape.size() != 2) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "embedding: weight must be 2D, got rank " << weight_shape.size();
                err.message = oss.str();
                return std::make_optional(err);
            }

            // Output should be indices_shape + [embedding_dim]
            if (out_shape.size() != indices_shape.size() + 1) {
                ShapeValidationError err;
                err.message = "embedding: output rank should be indices rank + 1";
                return std::make_optional(err);
            }

            for (size_t i = 0; i < indices_shape.size(); ++i) {
                if (out_shape[i] != indices_shape[i]) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "embedding: output dim[" << i << "] (" << out_shape[i]
                        << ") doesn't match indices dim[" << i << "] (" << indices_shape[i] << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            if (out_shape.back() != weight_shape[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "embedding: output last dim (" << out_shape.back()
                    << ") doesn't match weight embedding dim (" << weight_shape[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // Zeros
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "zeros";
        sig.min_inputs = 0;
        sig.max_inputs = 0;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            // No inputs, output shape determined by allocation
            if (outputs.empty() || outputs[0].empty()) {
                ShapeValidationError err;
                err.message = "zeros: output shape not specified or could not be resolved";

                std::ostringstream hint;
                hint << "The 'zeros' operation requires an explicit output shape. ";
                hint << "This shape should be defined in the IR tensor definition or operation attributes. ";
                hint << "Check that the output tensor is properly declared in the DSL graph with a concrete shape.";
                err.hint = hint.str();

                return std::make_optional(err);
            }
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FusedResidualRMSNorm
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "fused_residual_rmsnorm";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 3;
        sig.max_outputs = 3;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& residual_in = inputs[0];
            const auto& input = inputs[1];
            const auto& weight = inputs[2];
            const auto& residual_out = outputs[0];
            const auto& y = outputs[1];
            const auto& rstd = outputs[2];

            // Check residual_in == input shape
            if (auto err = validators::check_same_numel(residual_in, input, "residual_in", "input", "fused_residual_rmsnorm")) {
                return err;
            }

            // Check weight is 1D
            if (auto err = validators::check_rank(weight, 1, "weight", "fused_residual_rmsnorm")) {
                return err;
            }

            // Check outputs match inputs (allow unknown/unspecified output shapes)
            if (!residual_out.empty()) {
                if (auto err = validators::check_same_numel(residual_out, residual_in, "residual_out", "residual_in", "fused_residual_rmsnorm")) {
                    return err;
                }
            }
            if (!y.empty()) {
                if (auto err = validators::check_same_numel(y, input, "y", "input", "fused_residual_rmsnorm")) {
                    return err;
                }
            }

            // rstd can be flattened [B*T], [B, T], or [B, T, 1] depending on allocation path
            if (rstd.empty()) {
                ShapeValidationError err;
                err.message = "fused_residual_rmsnorm: rstd shape is empty";
                return std::make_optional(err);
            }
            const bool rstd_ok = (rstd.size() == 1) || (rstd.size() == 2) ||
                                 (rstd.size() == 3 && rstd.back() == 1);
            if (!rstd_ok) {
                ShapeValidationError err;
                err.message = "fused_residual_rmsnorm: rstd must be [B*T], [B,T], or [B,T,1]";
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // BiasAdd
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "bias_add";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& x = inputs[0];
            const auto& bias = inputs[1];
            const auto& out = outputs[0];

            // Check bias is 1D
            if (auto err = validators::check_rank(bias, 1, "bias", "bias_add")) {
                return err;
            }

            // Check bias dimension matches last dimension of x
            if (!x.empty() && bias[0] != x.back()) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "bias_add: bias dim (" << bias[0]
                    << ") doesn't match input last dim (" << x.back() << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            // Check output matches input
            if (auto err = validators::check_same_numel(out, x, "out", "x", "bias_add")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MatmulSwiGLU
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "matmul_swiglu";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& a = inputs[0];
            const auto& b = inputs[1];
            const auto& out = outputs[0];
            const auto& up_out = outputs[1];

            // Check matmul dims
            if (auto err = validators::check_matmul_dims(a, b, up_out, attrs)) {
                return err;
            }

            // up_out should have N=2*D where out has N=D
            if (!up_out.empty() && !out.empty()) {
                long up_N = up_out.back();
                long out_N = out.back();
                if (up_N != 2 * out_N) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "matmul_swiglu: up_out last dim (" << up_N
                        << ") must be 2x out last dim (" << out_N << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            // Check batch dimensions match
            if (a.size() > 2 && out.size() > 2) {
                for (size_t i = 0; i < a.size() - 2 && i < out.size() - 2; ++i) {
                    if (a[i] != out[i]) {
                        ShapeValidationError err;
                        std::ostringstream oss;
                        oss << "matmul_swiglu: batch dim mismatch at [" << i << "]: "
                            << a[i] << " != " << out[i];
                        err.message = oss.str();
                        return std::make_optional(err);
                    }
                }
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // QKVQKNormRoPE
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "qkv_qk_norm_rope";
        sig.min_inputs = 5;
        sig.max_inputs = 5;
        sig.min_outputs = 3;
        sig.max_outputs = 3;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& qkv = inputs[0];
            const auto& q_norm = inputs[1];
            const auto& k_norm = inputs[2];
            const auto& freqs = inputs[3];
            const auto& pos_ids = inputs[4];
            const auto& qkv_out = outputs[0];
            const auto& q_rstd = outputs[1];
            const auto& k_rstd = outputs[2];

            // Check qkv rank >= 2
            if (qkv.size() < 2) {
                ShapeValidationError err;
                err.message = "qkv_qk_norm_rope: qkv must have rank >= 2";
                return std::make_optional(err);
            }

            // Check q_norm and k_norm are 1D
            if (auto err = validators::check_rank(q_norm, 1, "q_norm", "qkv_qk_norm_rope")) {
                return err;
            }
            if (auto err = validators::check_rank(k_norm, 1, "k_norm", "qkv_qk_norm_rope")) {
                return err;
            }

            // Check freqs rank >= 2
            if (freqs.size() < 2) {
                ShapeValidationError err;
                err.message = "qkv_qk_norm_rope: freqs must have rank >= 2";
                return std::make_optional(err);
            }

            // Check output shape matches input
            if (auto err = validators::check_same_numel(qkv_out, qkv, "qkv_out", "qkv", "qkv_qk_norm_rope")) {
                return err;
            }

            // q_rstd/k_rstd can be flattened [B*T*H], [B*T, H], or [B, T, H]
            // Skip validation if shapes weren't inferred (empty means "unknown")
            const auto rstd_rank_ok = [](const std::vector<long>& s) {
                return s.empty() || s.size() == 1 || s.size() == 2 || s.size() == 3;
            };
            if (!rstd_rank_ok(q_rstd)) {
                ShapeValidationError err;
                err.message = "qkv_qk_norm_rope: q_rstd must be rank 1, 2, or 3";
                return std::make_optional(err);
            }
            if (!rstd_rank_ok(k_rstd)) {
                ShapeValidationError err;
                err.message = "qkv_qk_norm_rope: k_rstd must be rank 1, 2, or 3";
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // RoPE
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "rope";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& qkv = inputs[0];
            const auto& freqs = inputs[1];
            const auto& pos_ids = inputs[2];
            const auto& out = outputs[0];

            // Check qkv rank >= 2
            if (qkv.size() < 2) {
                ShapeValidationError err;
                err.message = "rope: qkv must have rank >= 2";
                return std::make_optional(err);
            }

            // Check freqs rank >= 2
            if (freqs.size() < 2) {
                ShapeValidationError err;
                err.message = "rope: freqs must have rank >= 2";
                return std::make_optional(err);
            }

            // Check output matches input
            if (auto err = validators::check_same_numel(out, qkv, "out", "qkv", "rope")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FlashAttention
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "flash_attention";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& qkv = inputs[0];
            const auto& out = outputs[0];
            const auto& lse = outputs[1];

            // Check qkv rank = 3 or 4 (DSL uses rank 4: [B, T, H, D])
            if (qkv.size() < 3 || qkv.size() > 4) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "flash_attention: qkv has rank " << qkv.size() << " but expected 3 or 4";
                err.message = oss.str();
                return std::make_optional(err);
            }

            // Skip output shape checks if output shapes are unknown (empty)
            // FlashAttention output shape cannot be easily inferred from input
            // since input is [B, T, Hq+2*Hkv, D] but output is [B, T, Hq, D]
            if (out.empty()) {
                return std::optional<ShapeValidationError>();  // Skip validation
            }

            // Check output rank matches qkv rank (when known)
            if (out.size() != qkv.size()) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "flash_attention: out has rank " << out.size()
                    << " but expected " << qkv.size() << " (same as qkv)";
                err.message = oss.str();
                return std::make_optional(err);
            }

            // Check first two dimensions match (B, T)
            if (qkv.size() >= 2 && out.size() >= 2) {
                if (qkv[0] != out[0] || qkv[1] != out[1]) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "flash_attention: out batch dims [" << out[0] << "," << out[1]
                        << "] don't match qkv [" << qkv[0] << "," << qkv[1] << "]";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ========================================================================
    // BACKWARD OPERATIONS
    // ========================================================================

    // ------------------------------------------------------------------------
    // ViewBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "view_backward";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& d_in = outputs[0];

            // Check element count preserved
            if (auto err = validators::check_same_numel(d_in, d_out, "d_in", "d_out", "view_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // AddBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "add_backward";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& d_in1 = outputs[0];
            const auto& d_in2 = outputs[1];

            // Both outputs should match input gradient shape
            if (auto err = validators::check_same_numel(d_in1, d_out, "d_in1", "d_out", "add_backward")) {
                return err;
            }
            if (auto err = validators::check_same_numel(d_in2, d_out, "d_in2", "d_out", "add_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MatmulBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "matmul_backward";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& a = inputs[1];
            const auto& b = inputs[2];
            const auto& d_a = outputs[0];
            const auto& d_b = outputs[1];

            // Check shapes match forward matmul
            if (auto err = validators::check_matmul_dims(a, b, d_out, attrs)) {
                return err;
            }

            // Check gradient shapes match input shapes
            if (auto err = validators::check_same_numel(d_a, a, "d_a", "a", "matmul_backward")) {
                return err;
            }
            if (auto err = validators::check_same_numel(d_b, b, "d_b", "b", "matmul_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // BiasAddBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "bias_add_backward";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& d_input = outputs[0];
            const auto& d_bias = outputs[1];

            // d_input matches d_out
            if (auto err = validators::check_same_numel(d_input, d_out, "d_input", "d_out", "bias_add_backward")) {
                return err;
            }

            // d_bias is 1D
            if (auto err = validators::check_rank(d_bias, 1, "d_bias", "bias_add_backward")) {
                return err;
            }

            // d_bias dimension matches last dimension of d_out
            if (!d_out.empty() && d_bias[0] != d_out.back()) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "bias_add_backward: d_bias dim (" << d_bias[0]
                    << ") doesn't match d_out last dim (" << d_out.back() << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // SwiGLUBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "swiglu_backward";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& mlp_up = inputs[1];
            const auto& d_inp = outputs[0];

            // Check mlp_up last dim is 2x d_out last dim
            if (!mlp_up.empty() && !d_out.empty()) {
                long mlp_up_dim = mlp_up.back();
                long d_out_dim = d_out.back();
                if (mlp_up_dim != 2 * d_out_dim) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "swiglu_backward: mlp_up last dim (" << mlp_up_dim
                        << ") must be 2x d_out last dim (" << d_out_dim << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            // d_inp matches mlp_up shape
            if (auto err = validators::check_same_numel(d_inp, mlp_up, "d_inp", "mlp_up", "swiglu_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MatmulSwiGLUBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "matmul_swiglu_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& ln2 = inputs[1];
            const auto& weight = inputs[2];
            const auto& mlp_up = inputs[3];
            const auto& d_inp = outputs[0];
            const auto& d_weight = outputs[1];

            // Check mlp_up last dim is 2x d_out last dim
            if (!mlp_up.empty() && !d_out.empty()) {
                long mlp_up_dim = mlp_up.back();
                long d_out_dim = d_out.back();
                if (mlp_up_dim != 2 * d_out_dim) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "matmul_swiglu_backward: mlp_up last dim (" << mlp_up_dim
                        << ") must be 2x d_out last dim (" << d_out_dim << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }

            // d_inp should match ln2 (activation input)
            if (auto err = validators::check_same_numel(d_inp, ln2, "d_inp", "ln2", "matmul_swiglu_backward")) {
                return err;
            }

            // d_weight should match weight shape
            if (auto err = validators::check_same_numel(d_weight, weight, "d_weight", "weight", "matmul_swiglu_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // RoPEBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "rope_backward";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& freqs = inputs[1];
            const auto& pos_ids = inputs[2];
            const auto& d_qkv = outputs[0];

            // d_qkv should match d_out
            if (auto err = validators::check_same_numel(d_qkv, d_out, "d_qkv", "d_out", "rope_backward")) {
                return err;
            }

            // Check freqs rank >= 2
            if (freqs.size() < 2) {
                ShapeValidationError err;
                err.message = "rope_backward: freqs must have rank >= 2";
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // QKVQKNormRoPEBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "qkv_qk_norm_rope_backward";
        sig.min_inputs = 8;
        sig.max_inputs = 8;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& qkv = inputs[1];
            // inputs[2-7] are norm weights, rstds, freqs, pos_ids
            const auto& d_qkv = outputs[0];

            // d_qkv should match d_out and qkv
            if (auto err = validators::check_same_numel(d_qkv, d_out, "d_qkv", "d_out", "qkv_qk_norm_rope_backward")) {
                return err;
            }
            if (auto err = validators::check_same_numel(d_qkv, qkv, "d_qkv", "qkv", "qkv_qk_norm_rope_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FlashAttentionBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "flash_attention_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& att_out = inputs[1];
            const auto& lse = inputs[2];
            const auto& qkv = inputs[3];
            const auto& d_qkv = outputs[0];

            // d_qkv should match qkv shape
            if (auto err = validators::check_same_numel(d_qkv, qkv, "d_qkv", "qkv", "flash_attention_backward")) {
                return err;
            }

            // qkv should be rank 3
            if (auto err = validators::check_rank(qkv, 3, "qkv", "flash_attention_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // ZerosBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "zeros_backward";
        sig.min_inputs = 0;
        sig.max_inputs = 0;
        sig.min_outputs = 0;
        sig.max_outputs = 0;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            // No-op, no validation needed
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FusedResidualRMSNormBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "fused_residual_rmsnorm_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 5;
        sig.min_outputs = 2;
        sig.max_outputs = 3;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_y = inputs[0];
            // inputs[1] is d_residual_next (optional), inputs[2] is residual_out, inputs[3] is weight, inputs[4] is rstd
            const auto& residual_out = inputs[2];
            const auto& d_residual = outputs[0];
            const auto& d_input = outputs[1];

            // Check d_residual and d_input match residual_out and d_y
            if (auto err = validators::check_same_numel(d_residual, residual_out, "d_residual", "residual_out", "fused_residual_rmsnorm_backward")) {
                return err;
            }
            if (auto err = validators::check_same_numel(d_input, d_y, "d_input", "d_y", "fused_residual_rmsnorm_backward")) {
                return err;
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // EmbeddingBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "embedding_backward";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap& attrs,
                          const ShapeEnv& env) -> std::optional<ShapeValidationError> {
            const auto& d_out = inputs[0];
            const auto& d_embedding = outputs[0];

            // Check d_embedding is rank 2
            if (auto err = validators::check_rank(d_embedding, 2, "d_embedding", "embedding_backward")) {
                return err;
            }

            // Check d_out last dim matches d_embedding embedding dim
            if (!d_out.empty() && d_out.back() != d_embedding[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "embedding_backward: d_out last dim (" << d_out.back()
                    << ") doesn't match d_embedding embedding dim (" << d_embedding[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // CrossEntropyLoss
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "cross_entropy_loss";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap&,
                          const ShapeEnv&) -> std::optional<ShapeValidationError> {
            const auto& logits = inputs[0];
            const auto& targets = inputs[1];
            const auto& loss = outputs[0];

            // logits should be rank 2: [BT, V]
            if (auto err = validators::check_rank(logits, 2, "logits", "cross_entropy_loss")) {
                return err;
            }
            // targets should be rank 1: [BT] or rank 2: [B, T]
            if (targets.size() != 1 && targets.size() != 2) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_loss: targets has rank " << targets.size()
                    << " but expected 1 or 2";
                err.message = oss.str();
                return std::make_optional(err);
            }
            // loss should be rank 1: [BT]
            if (auto err = validators::check_rank(loss, 1, "loss", "cross_entropy_loss")) {
                return err;
            }

            if (!logits.empty() && !targets.empty()) {
                const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
                if (logits[0] != target_bt) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "cross_entropy_loss: logits BT (" << logits[0]
                        << ") doesn't match targets BT (" << target_bt << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }
            if (!logits.empty() && !loss.empty() && logits[0] != loss[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_loss: logits BT (" << logits[0]
                    << ") doesn't match loss BT (" << loss[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // CrossEntropyLossBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "cross_entropy_backward";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap&,
                          const ShapeEnv&) -> std::optional<ShapeValidationError> {
            const auto& d_loss = inputs[0];
            const auto& logits = inputs[1];
            const auto& targets = inputs[2];
            const auto& d_logits = outputs[0];

            if (auto err = validators::check_rank(d_loss, 1, "d_loss", "cross_entropy_backward")) {
                return err;
            }
            if (auto err = validators::check_rank(logits, 2, "logits", "cross_entropy_backward")) {
                return err;
            }
            if (targets.size() != 1 && targets.size() != 2) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_backward: targets has rank " << targets.size()
                    << " but expected 1 or 2";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (auto err = validators::check_rank(d_logits, 2, "d_logits", "cross_entropy_backward")) {
                return err;
            }

            if (!logits.empty() && !d_logits.empty() && logits[0] != d_logits[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_backward: logits BT (" << logits[0]
                    << ") doesn't match d_logits BT (" << d_logits[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!logits.empty() && !d_logits.empty() && logits[1] != d_logits[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_backward: logits V (" << logits[1]
                    << ") doesn't match d_logits V (" << d_logits[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!logits.empty() && !targets.empty()) {
                const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
                if (logits[0] != target_bt) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "cross_entropy_backward: logits BT (" << logits[0]
                        << ") doesn't match targets BT (" << target_bt << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }
            if (!logits.empty() && !d_loss.empty() && logits[0] != d_loss[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "cross_entropy_backward: logits BT (" << logits[0]
                    << ") doesn't match d_loss BT (" << d_loss[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FusedLMHeadLoss
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "fused_lm_head_loss";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap&,
                          const ShapeEnv&) -> std::optional<ShapeValidationError> {
            const auto& xF_flat = inputs[0];
            const auto& weight = inputs[1];
            const auto& targets = inputs[2];
            const auto& loss = outputs[0];

            if (auto err = validators::check_rank(xF_flat, 2, "xF_flat", "fused_lm_head_loss")) {
                return err;
            }
            if (auto err = validators::check_rank(weight, 2, "weight", "fused_lm_head_loss")) {
                return err;
            }
            if (targets.size() != 1 && targets.size() != 2) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss: targets has rank " << targets.size()
                    << " but expected 1 or 2";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (auto err = validators::check_rank(loss, 1, "loss", "fused_lm_head_loss")) {
                return err;
            }

            if (!xF_flat.empty() && !weight.empty() && xF_flat[1] != weight[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss: xF_flat C (" << xF_flat[1]
                    << ") doesn't match weight C (" << weight[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!xF_flat.empty() && !targets.empty()) {
                const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
                if (xF_flat[0] != target_bt) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "fused_lm_head_loss: xF_flat BT (" << xF_flat[0]
                        << ") doesn't match targets BT (" << target_bt << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }
            if (!xF_flat.empty() && !loss.empty() && xF_flat[0] != loss[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss: xF_flat BT (" << xF_flat[0]
                    << ") doesn't match loss BT (" << loss[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // FusedLMHeadLossBackward
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "fused_lm_head_loss_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const std::vector<std::vector<long>>& inputs,
                          const std::vector<std::vector<long>>& outputs,
                          const AttrMap&,
                          const ShapeEnv&) -> std::optional<ShapeValidationError> {
            const auto& d_loss = inputs[0];
            const auto& xF_flat = inputs[1];
            const auto& weight = inputs[2];
            const auto& targets = inputs[3];
            const auto& d_xF_flat = outputs[0];
            const auto& d_weight = outputs[1];

            if (auto err = validators::check_rank(d_loss, 1, "d_loss", "fused_lm_head_loss_backward")) {
                return err;
            }
            if (auto err = validators::check_rank(xF_flat, 2, "xF_flat", "fused_lm_head_loss_backward")) {
                return err;
            }
            if (auto err = validators::check_rank(weight, 2, "weight", "fused_lm_head_loss_backward")) {
                return err;
            }
            if (targets.size() != 1 && targets.size() != 2) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: targets has rank " << targets.size()
                    << " but expected 1 or 2";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (auto err = validators::check_rank(d_xF_flat, 2, "d_xF_flat", "fused_lm_head_loss_backward")) {
                return err;
            }
            if (auto err = validators::check_rank(d_weight, 2, "d_weight", "fused_lm_head_loss_backward")) {
                return err;
            }

            if (!xF_flat.empty() && !d_xF_flat.empty() && xF_flat[0] != d_xF_flat[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0]
                    << ") doesn't match d_xF_flat BT (" << d_xF_flat[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!xF_flat.empty() && !d_xF_flat.empty() && xF_flat[1] != d_xF_flat[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: xF_flat C (" << xF_flat[1]
                    << ") doesn't match d_xF_flat C (" << d_xF_flat[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!weight.empty() && !d_weight.empty() && weight[0] != d_weight[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: weight V (" << weight[0]
                    << ") doesn't match d_weight V (" << d_weight[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!weight.empty() && !d_weight.empty() && weight[1] != d_weight[1]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: weight C (" << weight[1]
                    << ") doesn't match d_weight C (" << d_weight[1] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
            if (!xF_flat.empty() && !targets.empty()) {
                const long target_bt = (targets.size() == 2) ? targets[0] * targets[1] : targets[0];
                if (xF_flat[0] != target_bt) {
                    ShapeValidationError err;
                    std::ostringstream oss;
                    oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0]
                        << ") doesn't match targets BT (" << target_bt << ")";
                    err.message = oss.str();
                    return std::make_optional(err);
                }
            }
            if (!xF_flat.empty() && !d_loss.empty() && xF_flat[0] != d_loss[0]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "fused_lm_head_loss_backward: xF_flat BT (" << xF_flat[0]
                    << ") doesn't match d_loss BT (" << d_loss[0] << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }

            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Sigmoid: probs = moe_sigmoid(logits)
    // Input: [num_tokens, num_experts], Output: [num_tokens, num_experts]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_sigmoid";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.empty() || outputs.empty()) {
                return std::make_optional(ShapeValidationError{"moe_sigmoid: missing inputs/outputs"});
            }
            // Output should have same shape as input
            if (inputs[0] != outputs[0]) {
                ShapeValidationError err;
                err.message = "moe_sigmoid: output shape must match input shape";
                return std::make_optional(err);
            }
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Softmax: probs = moe_softmax(logits)
    // Input: [num_tokens, num_experts], Output: [num_tokens, num_experts]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_softmax";
        sig.min_inputs = 1;
        sig.max_inputs = 1;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap&, const ShapeEnv&) {
            if (inputs.empty() || outputs.empty()) {
                return std::make_optional(ShapeValidationError{"moe_softmax: missing inputs/outputs"});
            }
            if (inputs[0] != outputs[0]) {
                ShapeValidationError err;
                err.message = "moe_softmax: output shape must match input shape";
                return std::make_optional(err);
            }
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE TopK: (weights, indices) = moe_topk(probs, top_k)
    // Input: [num_tokens, num_experts]
    // Output[0]: weights [num_tokens, top_k]
    // Output[1]: indices [num_tokens, top_k]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_topk";
        sig.min_inputs = 1;
        sig.max_inputs = 2;  // optional correction_bias as 2nd input
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap& attrs, const ShapeEnv&) {
            if (inputs.empty() || outputs.size() < 2) {
                return std::make_optional(ShapeValidationError{"moe_topk: requires 1 input, 2 outputs"});
            }
            const auto& probs = inputs[0];
            const auto& weights = outputs[0];
            const auto& indices = outputs[1];

            if (probs.size() != 2) {
                return std::make_optional(ShapeValidationError{"moe_topk: probs must be 2D [num_tokens, num_experts]"});
            }

            // Get top_k from attrs
            int top_k = 2;  // default
            auto it = attrs.find("top_k");
            if (it != attrs.end()) {
                if (auto* v = std::get_if<long>(&it->second.value)) {
                    top_k = static_cast<int>(*v);
                }
            }

            // Check output shapes
            if (!weights.empty()) {
                if (weights.size() != 2 || weights[0] != probs[0] || weights[1] != top_k) {
                    std::ostringstream oss;
                    oss << "moe_topk: weights shape mismatch, expected [" << probs[0] << ", " << top_k << "]";
                    return std::make_optional(ShapeValidationError{oss.str()});
                }
            }
            if (!indices.empty()) {
                if (indices.size() != 2 || indices[0] != probs[0] || indices[1] != top_k) {
                    std::ostringstream oss;
                    oss << "moe_topk: indices shape mismatch, expected [" << probs[0] << ", " << top_k << "]";
                    return std::make_optional(ShapeValidationError{oss.str()});
                }
            }
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Permute: (permuted, scatter_indices) = moe_permute(x, routing_indices)
    // Input[0]: x [num_tokens, hidden_size]
    // Input[1]: routing_indices [num_tokens, top_k]
    // Output[0]: permuted [num_tokens * top_k, hidden_size]
    // Output[1]: scatter_indices [num_tokens * top_k]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_permute";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const auto& inputs, const auto& outputs,
                          const AttrMap& attrs, const ShapeEnv&) {
            if (inputs.size() < 2 || outputs.size() < 2) {
                return std::make_optional(ShapeValidationError{"moe_permute: requires 2 inputs, 2 outputs"});
            }
            // Shape validation is complex due to dynamic routing; accept for now
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Grouped GEMM Gate+Up
    // Input[0]: x [total_tokens, hidden_size]
    // Input[1]: weights [num_experts, 2*intermediate, hidden_size]
    // Input[2]: scatter_indices
    // Output: [total_tokens, 2*intermediate]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_grouped_gemm_gate_up";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            // Complex grouped GEMM shapes; accept for now
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Grouped GEMM Down
    // Input[0]: x [total_tokens, intermediate]
    // Input[1]: weights [num_experts, hidden_size, intermediate]
    // Input[2]: scatter_indices
    // Output: [total_tokens, hidden_size]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_grouped_gemm_down";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            // Complex grouped GEMM shapes; accept for now
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE Unpermute: out = moe_unpermute(expert_out, routing_weights, scatter_indices)
    // Input[0]: expert_out [total_tokens, hidden_size]
    // Input[1]: routing_weights [num_tokens, top_k]
    // Input[2]: scatter_indices [total_tokens]
    // Output: [num_tokens, hidden_size]
    // ------------------------------------------------------------------------
    {
        OpShapeSignature sig;
        sig.op_name = "moe_unpermute";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            // Dynamic routing shapes; accept for now
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }

    // ------------------------------------------------------------------------
    // MoE backward operations (accept all - shapes match forward counterparts)
    // ------------------------------------------------------------------------
    {
        // moe_sigmoid_backward
        OpShapeSignature sig;
        sig.op_name = "moe_sigmoid_backward";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_softmax_backward";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_topk_backward";
        sig.min_inputs = 3;
        sig.max_inputs = 3;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_permute_backward";
        sig.min_inputs = 2;
        sig.max_inputs = 2;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_grouped_gemm_gate_up_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_grouped_gemm_down_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 1;
        sig.max_outputs = 1;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
    {
        OpShapeSignature sig;
        sig.op_name = "moe_unpermute_backward";
        sig.min_inputs = 4;
        sig.max_inputs = 4;
        sig.min_outputs = 2;
        sig.max_outputs = 2;
        sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
            return std::optional<ShapeValidationError>();
        };
        reg.register_signature(sig);
    }
}

}}  // namespace dsl::shape_checker
