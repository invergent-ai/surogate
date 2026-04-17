// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Op dispatch registry. Replaces the enum+switch pattern used in the
// CompiledExecutor hot path: each op registers a forward and/or backward
// dispatch function pointer at static-init time, and the graph compiler
// bakes the correct pointer into each CompiledOp::fn. Execute then does a
// single indirect call per op — no switch, no registry lookup.
//
// Phase 2b also stores the autodiff rule here, so deriving a backward
// graph is a single name-based lookup against the same registry.

#ifndef SUROGATE_SRC_EXECUTOR_OP_REGISTRY_H
#define SUROGATE_SRC_EXECUTOR_OP_REGISTRY_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/dsl/ir.h"

namespace dsl {

class CompiledExecutor;
struct BackwardRuleContext;

// Uniform dispatch signature. The hook pointer is type-erased because
// forward and backward hooks are distinct types; each wrapper casts it
// back to the type it expects. Forward-graph ops receive a
// `const modules::ForwardHook*`; backward-graph ops receive a
// `const modules::BackwardHook*`. Ops that don't use hooks ignore it.
using OpExecFn = void (*)(CompiledExecutor&, const CompiledOp&, const void* hook);

// Autodiff rule: given a forward op and its context, returns the list
// of backward operations to emit. Phase 2b: replaces the separate
// BackwardRuleRegistry, unifying everything in OpRegistry.
using AutodiffFn = std::function<std::vector<Operation>(const BackwardRuleContext&)>;

struct OpDescriptor {
    std::string name;                               // e.g. "embedding", "softmax"
    CompiledOpType type = CompiledOpType::Unknown;  // may be Unknown for name-only rules
    OpExecFn forward_fn = nullptr;                  // forward-graph dispatch
    OpExecFn backward_fn = nullptr;                 // backward-graph dispatch
    AutodiffFn autodiff_fn;                         // backward-graph generator (autodiff rule)
};

class OpRegistry {
public:
    static OpRegistry& instance();

    // Register an op. If an entry already exists for the same name, the
    // new descriptor is merged with the existing one field-by-field
    // (non-null forward_fn / backward_fn / autodiff_fn overwrite,
    // nulls are ignored). This lets REGISTER_OP and REGISTER_AUTODIFF
    // coexist in different TUs for the same op type.
    int register_op(OpDescriptor desc);

    // Lookup by enum type. Returns nullptr if no descriptor exists for
    // that type (which is also the case for name-only rules).
    const OpDescriptor* find(CompiledOpType type) const;

    // Lookup by string name. Returns nullptr if unknown.
    const OpDescriptor* find_by_name(const std::string& name) const;

private:
    OpRegistry() = default;

    // Primary storage. References into this map stay valid because
    // registrations happen only at static init, and we look up a
    // reference after all inserts are complete — by construction there
    // is no concurrent modification during lookup.
    std::unordered_map<std::string, OpDescriptor> mByName;

    // Type → canonical name. A CompiledOpType may alias multiple names
    // (e.g. "matmul" and "matmul_bias" both map to Matmul); the first
    // registration wins as the canonical.
    std::unordered_map<CompiledOpType, std::string> mTypeToName;
};

}  // namespace dsl

// Concatenation helpers so we can mint a unique static-variable name per
// REGISTER_OP call in the same translation unit.
#define SUROGATE_OP_REG_CONCAT_(a, b) a##b
#define SUROGATE_OP_REG_CONCAT(a, b) SUROGATE_OP_REG_CONCAT_(a, b)

// Register an op with forward/backward dispatch. `name_str` is the
// kernel-type string used by the graph compiler (e.g. "matmul").
// `op_type_enum` is the CompiledOpType tag (e.g. Matmul). `fwd_fn` /
// `bwd_fn` are `OpExecFn` pointers (or `nullptr` when the op has no
// handler for that direction).
#define REGISTER_OP(name_str, op_type_enum, fwd_fn, bwd_fn)                   \
    static const int SUROGATE_OP_REG_CONCAT(_surogate_op_reg_, __COUNTER__) = \
        ::dsl::OpRegistry::instance().register_op(                            \
            ::dsl::OpDescriptor{name_str, ::dsl::CompiledOpType::op_type_enum, fwd_fn, bwd_fn, {}})

// Register an op with forward/backward dispatch AND an autodiff rule.
// Used by ops whose backward graph is derived by this rule during
// autodiff (which is most ops that appear in a trainable forward graph).
#define REGISTER_OP_FULL(name_str, op_type_enum, fwd_fn, bwd_fn, autodiff_fn_) \
    static const int SUROGATE_OP_REG_CONCAT(_surogate_op_reg_, __COUNTER__) =  \
        ::dsl::OpRegistry::instance().register_op(                             \
            ::dsl::OpDescriptor{name_str, ::dsl::CompiledOpType::op_type_enum, fwd_fn, bwd_fn, autodiff_fn_})

// Register an autodiff rule only — for names that don't have a
// CompiledOpType counterpart (e.g. "softmax", "attention", "identity"
// generics) or whose dispatch is handled elsewhere. Type defaults to
// Unknown so it's excluded from the type → descriptor index.
#define REGISTER_AUTODIFF(name_str, autodiff_fn_)                             \
    static const int SUROGATE_OP_REG_CONCAT(_surogate_op_reg_, __COUNTER__) = \
        ::dsl::OpRegistry::instance().register_op(                            \
            ::dsl::OpDescriptor{name_str, ::dsl::CompiledOpType::Unknown, nullptr, nullptr, autodiff_fn_})

#endif  // SUROGATE_SRC_EXECUTOR_OP_REGISTRY_H
