// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H
#define SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H

#include <cstdint>

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

const char* op_semantic_kind_name(OpSemanticKind kind);
const char* communication_kind_name(CommunicationKind kind);

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H
