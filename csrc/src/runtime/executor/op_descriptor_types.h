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

const char* op_semantic_kind_name(OpSemanticKind kind);
const char* communication_kind_name(CommunicationKind kind);

}  // namespace dsl

#endif  // SUROGATE_SRC_EXECUTOR_OP_DESCRIPTOR_TYPES_H
