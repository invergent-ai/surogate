// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// TensorRole is the structural classification surface for execution, save/load,
// optimizer routing, and regression diagnostics.

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
    LMHead,
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

/// Conservative classifier used while role metadata is propagated through the
/// graph and loader surfaces.
TensorRole infer_tensor_role_from_name(std::string_view name, int block_layer = -1);

/// MoE ownership predicate derived from TensorRole.
bool tensor_role_is_moe_name(std::string_view name);

/// RoPE/frequency ownership predicate derived from TensorRole.
bool tensor_role_is_rope_name(std::string_view name);

/// Expert-parallel predicate derived from TensorRole.
bool tensor_role_is_expert_parallel_name(std::string_view name);

/// MoE routing/index side-channel predicate derived from TensorRole.
bool tensor_role_is_moe_side_channel_name(std::string_view name);

/// Router tensor predicate derived from TensorRole.
bool tensor_role_is_router_name(std::string_view name);

/// Embedding tensor predicate derived from TensorRole.
bool tensor_role_is_embedding_name(std::string_view name);

/// LM-head tensor predicate derived from TensorRole.
bool tensor_role_is_lm_head_name(std::string_view name);

/// Standalone gate predicate for optimizer routing; excludes MLP gate weights.
bool tensor_role_is_standalone_gate_name(std::string_view name);

/// Shared-expert tensor predicate derived from TensorRole.
bool tensor_role_is_shared_expert_name(std::string_view name);

/// Expert weight predicate derived from TensorRole.
bool tensor_role_is_expert_weight_name(std::string_view name);

/// Expert projection predicates used by MoE grouped GEMM routing.
bool tensor_role_is_expert_gate_up_name(std::string_view name);
bool tensor_role_is_expert_up_name(std::string_view name);
bool tensor_role_is_expert_down_name(std::string_view name);

/// Expert bias predicate derived from TensorRole.
bool tensor_role_is_expert_bias_name(std::string_view name);

/// Fused projection predicates used by weight-load slice inference.
bool tensor_role_is_fused_qkv_name(std::string_view name);
bool tensor_role_is_fused_mlp_up_name(std::string_view name);

}  // namespace dsl

#endif  // SUROGATE_SRC_RUNTIME_DSL_TENSOR_ROLE_H
