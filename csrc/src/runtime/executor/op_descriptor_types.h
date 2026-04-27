// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H
#define SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H

#include <cstdint>
#include <string>

#include "runtime/dsl/tensor_role.h"

namespace dsl {

enum class OpSemanticKind : std::uint8_t {
    Unknown = 0,
    Dense,
    MoE,
    Collective,
    Attention,
    Normalization,
    Elementwise,
    View,
    Loss,
    Sequence,
};

enum class CommunicationKind : std::uint8_t {
    NoComm = 0,
    AllReduceAfter,
    ReduceScatterAfter,
    AllToAllIn,
    AllToAllOut,
    ExpertParallelRouted,
    WeightStreamFromCpu,
    WeightTransferP2P,
};

enum OpCapabilityBits : std::uint32_t {
    OpCapabilityNone = 0,
    OpCapabilityDenseMatmul = 1u << 0,
    OpCapabilityGroupedMatmul = 1u << 1,
    OpCapabilityMoeRouted = 1u << 2,
    OpCapabilityFp8Eligible = 1u << 3,
    OpCapabilityFp4Eligible = 1u << 4,
    OpCapabilityLoRACompatible = 1u << 5,
    OpCapabilityWeightCacheEligible = 1u << 6,
};

enum EpilogueSupportBits : std::uint32_t {
    EpilogueSupportNone = 0,
    EpilogueSupportBias = 1u << 0,
    EpilogueSupportActivation = 1u << 1,
    EpilogueSupportResidual = 1u << 2,
    EpilogueSupportNormalization = 1u << 3,
};

enum StorageCompatibilityBits : std::uint32_t {
    StorageCompatibilityGpuResident = 1u << 0,
    StorageCompatibilityCpuPinnedStream = 1u << 1,
    StorageCompatibilityCpuPageable = 1u << 2,
    StorageCompatibilityNvmeOffload = 1u << 3,
};

enum MoECapabilityBits : std::uint32_t {
    MoECapabilityNone = 0,
    MoECapabilityGroupedGemmEligible = 1u << 0,
    MoECapabilityFp8GroupedEligible = 1u << 1,
    MoECapabilityFp4GroupedEligible = 1u << 2,
    MoECapabilityCudnnMoeGraphEligible = 1u << 3,
    MoECapabilityPerExpertQuant = 1u << 4,
    MoECapabilityRoutingAwareFusion = 1u << 5,
    MoECapabilityFp8BackwardImplemented = 1u << 6,
    MoECapabilityNvfp4NoFallback = 1u << 7,
};

enum class EpAwareness : std::uint8_t {
    None = 0,
    Sharded,
    Routed,
    WeightTransfer,
};

struct CommunicationProfile {
    CommunicationKind kind = CommunicationKind::NoComm;
    bool can_overlap_with_compute = false;
    int reduction_priority = 0;
};

struct GroupedSemantics {
    bool is_grouped = false;
    bool routes_tokens = false;
    int expert_dim = -1;
    bool ep_aware = false;
};

struct OpCapabilities {
    std::uint32_t flags = OpCapabilityNone;

    [[nodiscard]] bool has(std::uint32_t flag) const {
        return (flags & flag) == flag;
    }
};

struct EpilogueSupport {
    std::uint32_t flags = EpilogueSupportNone;

    [[nodiscard]] bool has(std::uint32_t flag) const {
        return (flags & flag) == flag;
    }
};

struct StorageCompatibility {
    std::uint32_t flags = StorageCompatibilityGpuResident;

    [[nodiscard]] bool has(std::uint32_t flag) const {
        return (flags & flag) == flag;
    }

    [[nodiscard]] bool supports(StorageTier tier) const {
        switch (tier) {
            case StorageTier::GpuResident: return has(StorageCompatibilityGpuResident);
            case StorageTier::CpuPinnedStream: return has(StorageCompatibilityCpuPinnedStream);
            case StorageTier::CpuPageable: return has(StorageCompatibilityCpuPageable);
            case StorageTier::NvmeOffload: return has(StorageCompatibilityNvmeOffload);
        }
        return false;
    }
};

struct MoECapabilities {
    std::uint32_t flags = MoECapabilityNone;
    StorageCompatibility expert_storage{};
    EpAwareness ep_awareness = EpAwareness::None;

    [[nodiscard]] bool has(std::uint32_t flag) const {
        return (flags & flag) == flag;
    }
};

const char* op_semantic_kind_name(OpSemanticKind kind);
const char* communication_kind_name(CommunicationKind kind);
const char* ep_awareness_name(EpAwareness awareness);
std::string op_capability_flags_string(OpCapabilities caps);
std::string epilogue_support_flags_string(EpilogueSupport support);
std::string storage_compatibility_flags_string(StorageCompatibility compat);
std::string moe_capability_flags_string(MoECapabilities caps);

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H
