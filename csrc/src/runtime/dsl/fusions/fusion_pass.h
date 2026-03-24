// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Inference fusion pass registry.
//
// Each fusion pass is a self-contained graph rewrite that matches by semantic
// roles (op attrs) rather than architecture identity.  Models declare which
// passes they want via _inference_opts_ in the Python DSL; the graph compiler
// runs them in declaration order, gated by kernel capabilities.

#ifndef SUROGATE_SRC_DSL_FUSIONS_FUSION_PASS_H
#define SUROGATE_SRC_DSL_FUSIONS_FUSION_PASS_H

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dsl {

struct Graph;
struct Operation;
struct Module;
enum class GraphCompileMode : std::uint8_t;

// ============================================================================
// Kernel capability identifiers
// ============================================================================
//
// String-typed for extensibility.  Conventions:
//   "compiled:matmul_swiglu"  — built-in CompiledOpType
//   "jit:gdn_fused_proj"     — JIT-compiled Triton/CuTe kernel
//   "arch:sm90+"             — GPU architecture gate
using KernelCapId = std::string;

// ============================================================================
// Fusion context (populated once per compile() call)
// ============================================================================

struct FusionContext {
    GraphCompileMode mode;
    bool is_forward_graph = false;
    int sm_version = 0;                                // e.g., 89, 100, 120
    std::unordered_set<KernelCapId> available_kernels;
    const Module* module = nullptr;

    bool has_kernel(const KernelCapId& id) const {
        return available_kernels.count(id) > 0;
    }
};

// ============================================================================
// Fusion pass definition
// ============================================================================

/// Signature for a fusion pass rewrite function.
/// Returns true if the graph was modified.
using FusionRunFn = bool(*)(Graph& graph, const FusionContext& ctx);

struct FusionPassInfo {
    std::string id;                              // e.g., "matmul_view_swiglu"
    std::string description;
    std::vector<KernelCapId> required_kernels;   // All must be present
    FusionRunFn run = nullptr;
};

// ============================================================================
// Fusion pass registry (singleton)
// ============================================================================

class FusionPassRegistry {
public:
    static FusionPassRegistry& instance();

    void register_pass(FusionPassInfo info);
    const FusionPassInfo* find(const std::string& id) const;

private:
    FusionPassRegistry() = default;
    std::vector<FusionPassInfo> passes_;
    std::unordered_map<std::string, std::size_t> id_to_index_;
};

/// Static registration helper — use at file scope in each pass .cpp file.
struct FusionPassRegistrar {
    FusionPassRegistrar(FusionPassInfo info) {
        FusionPassRegistry::instance().register_pass(std::move(info));
    }
};

// ============================================================================
// Shared utilities for fusion passes
// ============================================================================

/// Get the effective operation type for pattern matching.
/// Prefers kernel_type; falls back to name.
inline std::string op_type_for_rewrite(const Operation& op);

/// Case-insensitive substring check.
inline bool contains_ci(std::string_view haystack, std::string_view needle);

} // namespace dsl

// ============================================================================
// Inline implementations
// ============================================================================

#include "runtime/dsl/ir.h"

namespace dsl {

inline std::string op_type_for_rewrite(const Operation& op) {
    if (op.kernel_type.empty() || op.kernel_type == "custom") {
        return op.name;
    }
    return op.kernel_type;
}

inline bool contains_ci(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;
    auto tolower = [](char c) -> char {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
    };
    for (std::size_t i = 0; i <= haystack.size() - needle.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            if (tolower(haystack[i + j]) != tolower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

} // namespace dsl

#endif // SUROGATE_SRC_DSL_FUSIONS_FUSION_PASS_H
