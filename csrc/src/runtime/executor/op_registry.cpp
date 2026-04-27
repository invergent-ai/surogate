// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/executor/op_registry.h"

namespace dsl {
namespace {

void append_flag(std::string& out, bool enabled, const char* name) {
    if (!enabled) return;
    if (!out.empty()) out += "|";
    out += name;
}

}  // namespace

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

const char* communication_kind_name(CommunicationKind kind) {
    switch (kind) {
        case CommunicationKind::NoComm: return "NoComm";
        case CommunicationKind::AllReduceAfter: return "AllReduceAfter";
        case CommunicationKind::ReduceScatterAfter: return "ReduceScatterAfter";
        case CommunicationKind::AllToAllIn: return "AllToAllIn";
        case CommunicationKind::AllToAllOut: return "AllToAllOut";
        case CommunicationKind::ExpertParallelRouted: return "ExpertParallelRouted";
        case CommunicationKind::WeightStreamFromCpu: return "WeightStreamFromCpu";
        case CommunicationKind::WeightTransferP2P: return "WeightTransferP2P";
    }
    return "NoComm";
}

std::string op_capability_flags_string(OpCapabilities caps) {
    std::string out;
    append_flag(out, caps.has(OpCapabilityDenseMatmul), "DenseMatmul");
    append_flag(out, caps.has(OpCapabilityGroupedMatmul), "GroupedMatmul");
    append_flag(out, caps.has(OpCapabilityMoeRouted), "MoERouted");
    append_flag(out, caps.has(OpCapabilityFp8Eligible), "FP8Eligible");
    append_flag(out, caps.has(OpCapabilityFp4Eligible), "FP4Eligible");
    append_flag(out, caps.has(OpCapabilityLoRACompatible), "LoRACompatible");
    append_flag(out, caps.has(OpCapabilityWeightCacheEligible), "WeightCacheEligible");
    return out.empty() ? "None" : out;
}

std::string epilogue_support_flags_string(EpilogueSupport support) {
    std::string out;
    append_flag(out, support.has(EpilogueSupportBias), "Bias");
    append_flag(out, support.has(EpilogueSupportActivation), "Activation");
    append_flag(out, support.has(EpilogueSupportResidual), "Residual");
    append_flag(out, support.has(EpilogueSupportNormalization), "Normalization");
    return out.empty() ? "None" : out;
}

std::string storage_compatibility_flags_string(StorageCompatibility compat) {
    std::string out;
    append_flag(out, compat.supports(StorageTier::GpuResident), "GpuResident");
    append_flag(out, compat.supports(StorageTier::CpuPinnedStream), "CpuPinnedStream");
    append_flag(out, compat.supports(StorageTier::CpuPageable), "CpuPageable");
    append_flag(out, compat.supports(StorageTier::NvmeOffload), "NvmeOffload");
    return out.empty() ? "None" : out;
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
        existing.default_caps.flags |= desc.default_caps.flags;
        existing.epilogue_support.flags |= desc.epilogue_support.flags;
        existing.storage_compat.flags |= desc.storage_compat.flags;
        if (desc.comm_profile.kind != CommunicationKind::NoComm || desc.comm_profile.can_overlap_with_compute ||
            desc.comm_profile.reduction_priority != 0) {
            existing.comm_profile = desc.comm_profile;
        }
        if (desc.grouped_semantics.is_grouped || desc.grouped_semantics.routes_tokens ||
            desc.grouped_semantics.expert_dim >= 0 || desc.grouped_semantics.ep_aware) {
            existing.grouped_semantics = desc.grouped_semantics;
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
