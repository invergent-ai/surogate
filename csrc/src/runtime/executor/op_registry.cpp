// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/executor/op_registry.h"

namespace dsl {

const char* op_semantic_kind_name(OpSemanticKind kind) {
    switch (kind) {
        case OpSemanticKind::Unknown: return "Unknown";
        case OpSemanticKind::Dense: return "Dense";
        case OpSemanticKind::MoE: return "MoE";
        case OpSemanticKind::Collective: return "Collective";
        case OpSemanticKind::Attention: return "Attention";
        case OpSemanticKind::Normalization: return "Normalization";
        case OpSemanticKind::Elementwise: return "Elementwise";
        case OpSemanticKind::View: return "View";
        case OpSemanticKind::Loss: return "Loss";
        case OpSemanticKind::Sequence: return "Sequence";
    }
    return "Unknown";
}

OpRegistry& OpRegistry::instance() {
    // Meyers singleton. Thread-safe init per C++11; we only write to it
    // during static initialization (REGISTER_OP) before any thread is
    // spawned, so no lock is needed.
    static OpRegistry registry;
    return registry;
}

int OpRegistry::register_op(OpDescriptor desc) {
    const std::string name = desc.name;

    // Merge with any existing descriptor for the same name. This lets a
    // REGISTER_OP in one TU and a REGISTER_AUTODIFF (or REGISTER_OP_FULL)
    // in another TU both contribute fields for the same op.
    auto [it, inserted] = mByName.try_emplace(name, std::move(desc));
    if (!inserted) {
        auto& existing = it->second;
        if (desc.type != CompiledOpType::Unknown) existing.type = desc.type;
        if (desc.forward_fn) existing.forward_fn = desc.forward_fn;
        if (desc.backward_fn) existing.backward_fn = desc.backward_fn;
        if (desc.autodiff_fn) existing.autodiff_fn = std::move(desc.autodiff_fn);
        if (desc.stack_bound_fn) existing.stack_bound_fn = desc.stack_bound_fn;
        if (desc.semantic_kind != OpSemanticKind::Unknown) existing.semantic_kind = desc.semantic_kind;
        if (desc.distribution_kind != DistributionKind::Replicated) {
            existing.distribution_kind = desc.distribution_kind;
        }
        existing.descriptor_flags |= desc.descriptor_flags;
    }

    // Mirror type → name. Only record the first type-bearing registration
    // as canonical so aliased names (e.g. "matmul" + "matmul_bias" both
    // on Matmul) leave "matmul" as the canonical.
    if (it->second.type != CompiledOpType::Unknown) {
        mTypeToName.try_emplace(it->second.type, name);
    }
    return 0;
}

const OpDescriptor* OpRegistry::find(CompiledOpType type) const {
    auto it = mTypeToName.find(type);
    if (it == mTypeToName.end()) return nullptr;
    auto n = mByName.find(it->second);
    return (n == mByName.end()) ? nullptr : &n->second;
}

const OpDescriptor* OpRegistry::find_by_name(const std::string& name) const {
    auto it = mByName.find(name);
    return (it == mByName.end()) ? nullptr : &it->second;
}

}  // namespace dsl
