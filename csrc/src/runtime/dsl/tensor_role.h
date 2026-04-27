// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// TensorRole is the Phase-1 structural classification surface. It is shadow
// metadata for now: existing execution still uses legacy routing, while parity
// checks compare role-derived decisions against the old name/slot predicates.

#ifndef SUROGATE_SRC_RUNTIME_DSL_TENSOR_ROLE_H
#define SUROGATE_SRC_RUNTIME_DSL_TENSOR_ROLE_H

#include <cstdint>
#include <string>
#include <string_view>

namespace dsl {

enum class TensorRoleKind : std::uint8_t {
    Unknown = 0,
    Param,
    Activation,
    ParamGrad,
    ActivationGrad,
    Scratch,
    Loss,
    LossInput,
};

enum class TensorOwnership : std::uint8_t {
    Unknown = 0,
    Persistent,
    Stack,
    LoRA,
    MoE,
    EP,
    RopeFreqs,
    Embedding,
};

enum class QuantState : std::uint8_t {
    None = 0,
    FP8Pending,
    FP8Ready,
    FP4Ready,
};

enum class StorageTier : std::uint8_t {
    GpuResident = 0,
    CpuPinnedStream,
    CpuPageable,
    NvmeOffload,
};

enum class DistributionKind : std::uint8_t {
    Replicated = 0,
    ShardedDim,
    ExpertParallel,
    RouterReplicated,
};

struct Distribution {
    DistributionKind kind = DistributionKind::Replicated;
    int shard_dim = -1;
    int num_shards = 1;
    int local_experts = 0;
    int global_experts = 0;
    bool needs_reduce_after = false;
};

struct StreamingHints {
    int prefetch_distance = 0;
    int eviction_policy = 0;
    bool sticky = false;
};

struct TensorRole {
    TensorRoleKind kind = TensorRoleKind::Unknown;
    int block_layer = -1;
    int block_slot = -1;
    TensorOwnership ownership = TensorOwnership::Unknown;
    QuantState quant_state = QuantState::None;
    StorageTier storage = StorageTier::GpuResident;
    StreamingHints streaming{};
    Distribution dist{};

    [[nodiscard]] bool is_moe_owned() const {
        return ownership == TensorOwnership::MoE || ownership == TensorOwnership::EP ||
               dist.kind == DistributionKind::ExpertParallel || dist.kind == DistributionKind::RouterReplicated;
    }

    [[nodiscard]] bool is_rope_freq() const {
        return ownership == TensorOwnership::RopeFreqs;
    }

    [[nodiscard]] bool is_expert_parallel() const {
        return ownership == TensorOwnership::EP || dist.kind == DistributionKind::ExpertParallel;
    }
};

const char* tensor_ownership_name(TensorOwnership ownership);
const char* distribution_kind_name(DistributionKind kind);
const char* quant_state_name(QuantState state);

/// Conservative month-one classifier used for parity checks. It is intentionally
/// broader than the final structural role source because it must mirror existing
/// string/slot behavior without changing execution.
TensorRole infer_tensor_role_from_name(std::string_view name, int block_layer = -1);

/// Legacy-compatible MoE ownership predicate derived from TensorRole.
bool tensor_role_is_moe_name(std::string_view name);

/// Legacy-compatible RoPE/frequency ownership predicate derived from TensorRole.
bool tensor_role_is_rope_name(std::string_view name);

/// Legacy-compatible expert-parallel predicate derived from TensorRole.
bool tensor_role_is_expert_parallel_name(std::string_view name);

/// Legacy-compatible MoE routing/index side-channel predicate derived from TensorRole.
bool tensor_role_is_moe_side_channel_name(std::string_view name);

/// Legacy-compatible router tensor predicate derived from TensorRole.
bool tensor_role_is_router_name(std::string_view name);

/// Legacy-compatible embedding tensor predicate derived from TensorRole.
bool tensor_role_is_embedding_name(std::string_view name);

/// Legacy-compatible shared-expert tensor predicate derived from TensorRole.
bool tensor_role_is_shared_expert_name(std::string_view name);

/// Legacy-compatible expert weight predicate derived from TensorRole.
bool tensor_role_is_expert_weight_name(std::string_view name);

/// Legacy-compatible expert bias predicate derived from TensorRole.
bool tensor_role_is_expert_bias_name(std::string_view name);

/// When SUROGATE_TENSOR_ROLE_PARITY is set, log or abort on mismatches between
/// legacy and role-derived decisions. Set it to "abort" for fatal assertions.
void tensor_role_parity_check(std::string_view name, bool legacy_value, bool role_value, const char* context);

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_TENSOR_ROLE_H
