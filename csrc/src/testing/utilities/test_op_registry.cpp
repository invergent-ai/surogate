// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <utility>

#include "runtime/executor/op_registry.h"

namespace dsl {
namespace {

TEST_CASE("op registry descriptor metadata merges without dispatch churn", "[op_registry]") {
    OpDescriptor base;
    base.name = "__test_descriptor_metadata";
    base.type = CompiledOpType::Unknown;
    base.semantic_kind = OpSemanticKind::Dense;
    REQUIRE(OpRegistry::instance().register_op(std::move(base)) == 0);

    OpDescriptor meta;
    meta.name = "__test_descriptor_metadata";
    meta.distribution_kind = DistributionKind::ExpertParallel;
    meta.default_caps.flags = OpCapabilityGroupedMatmul | OpCapabilityFp8Eligible;
    meta.epilogue_support.flags = EpilogueSupportActivation;
    meta.storage_compat.flags = StorageCompatibilityGpuResident | StorageCompatibilityCpuPinnedStream;
    meta.comm_profile.kind = CommunicationKind::ExpertParallelRouted;
    meta.grouped_semantics.is_grouped = true;
    meta.grouped_semantics.expert_dim = 0;
    meta.descriptor_flags = 0x4;
    REQUIRE(OpRegistry::instance().register_op(std::move(meta)) == 0);

    const OpDescriptor* desc = OpRegistry::instance().find_by_name("__test_descriptor_metadata");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(desc->default_caps.has(OpCapabilityGroupedMatmul));
    REQUIRE(desc->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE_FALSE(desc->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(desc->epilogue_support.has(EpilogueSupportActivation));
    REQUIRE(desc->storage_compat.supports(StorageTier::GpuResident));
    REQUIRE(desc->storage_compat.supports(StorageTier::CpuPinnedStream));
    REQUIRE_FALSE(desc->storage_compat.supports(StorageTier::NvmeOffload));
    REQUIRE(desc->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(desc->grouped_semantics.is_grouped);
    REQUIRE(desc->grouped_semantics.expert_dim == 0);
    REQUIRE(desc->descriptor_flags == 0x4);
    REQUIRE(std::string(op_semantic_kind_name(desc->semantic_kind)) == "Dense");
    REQUIRE(std::string(communication_kind_name(desc->comm_profile.kind)) == "ExpertParallelRouted");
}

TEST_CASE("moe ops carry first-month descriptor metadata", "[op_registry]") {
    const OpDescriptor* desc = OpRegistry::instance().find_by_name("moe_grouped_gemm");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::MoE);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(desc->comm_profile.kind == CommunicationKind::ExpertParallelRouted);
    REQUIRE(desc->grouped_semantics.is_grouped);
    REQUIRE_FALSE(desc->grouped_semantics.routes_tokens);
    REQUIRE(desc->grouped_semantics.expert_dim == 0);
    REQUIRE(desc->grouped_semantics.ep_aware);
    REQUIRE(desc->default_caps.has(OpCapabilityGroupedMatmul));
    REQUIRE(desc->default_caps.has(OpCapabilityMoeRouted));
    REQUIRE(desc->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(desc->default_caps.has(OpCapabilityWeightCacheEligible));
    REQUIRE(desc->storage_compat.supports(StorageTier::GpuResident));
    REQUIRE_FALSE(desc->storage_compat.supports(StorageTier::CpuPinnedStream));
}

TEST_CASE("ep ops carry communication profile metadata", "[op_registry]") {
    const OpDescriptor* dispatch = OpRegistry::instance().find_by_name("ep_dispatch");
    REQUIRE(dispatch != nullptr);
    REQUIRE(dispatch->semantic_kind == OpSemanticKind::Collective);
    REQUIRE(dispatch->comm_profile.kind == CommunicationKind::AllToAllIn);
    REQUIRE(dispatch->comm_profile.can_overlap_with_compute);
    REQUIRE(dispatch->grouped_semantics.routes_tokens);
    REQUIRE(dispatch->grouped_semantics.ep_aware);

    const OpDescriptor* combine = OpRegistry::instance().find_by_name("ep_combine");
    REQUIRE(combine != nullptr);
    REQUIRE(combine->comm_profile.kind == CommunicationKind::AllToAllOut);
}

TEST_CASE("mamba and gated-delta ops carry no-comm descriptor metadata", "[op_registry]") {
    const OpDescriptor* mamba_scan = OpRegistry::instance().find_by_name("mamba_ssm_scan");
    REQUIRE(mamba_scan != nullptr);
    REQUIRE(mamba_scan->semantic_kind == OpSemanticKind::Sequence);
    REQUIRE(mamba_scan->distribution_kind == DistributionKind::Replicated);
    REQUIRE(mamba_scan->comm_profile.kind == CommunicationKind::NoComm);

    const OpDescriptor* mamba_out = OpRegistry::instance().find_by_name("mamba_out_proj");
    REQUIRE(mamba_out != nullptr);
    REQUIRE(mamba_out->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(mamba_out->comm_profile.kind == CommunicationKind::NoComm);

    const OpDescriptor* gated_delta = OpRegistry::instance().find_by_name("chunk_gated_delta_rule");
    REQUIRE(gated_delta != nullptr);
    REQUIRE(gated_delta->semantic_kind == OpSemanticKind::Sequence);
    REQUIRE(gated_delta->comm_profile.kind == CommunicationKind::NoComm);
}

TEST_CASE("core transformer ops carry descriptor metadata", "[op_registry]") {
    const OpDescriptor* matmul = OpRegistry::instance().find_by_name("matmul");
    REQUIRE(matmul != nullptr);
    REQUIRE(matmul->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(matmul->comm_profile.kind == CommunicationKind::NoComm);
    REQUIRE(matmul->default_caps.has(OpCapabilityDenseMatmul));
    REQUIRE(matmul->default_caps.has(OpCapabilityFp8Eligible));
    REQUIRE(matmul->default_caps.has(OpCapabilityFp4Eligible));
    REQUIRE(matmul->default_caps.has(OpCapabilityLoRACompatible));
    REQUIRE(matmul->storage_compat.supports(StorageTier::CpuPinnedStream));

    const OpDescriptor* matmul_bias = OpRegistry::instance().find_by_name("matmul_bias");
    REQUIRE(matmul_bias != nullptr);
    REQUIRE(matmul_bias->epilogue_support.has(EpilogueSupportBias));

    const OpDescriptor* rmsnorm = OpRegistry::instance().find_by_name("rmsnorm");
    REQUIRE(rmsnorm != nullptr);
    REQUIRE(rmsnorm->semantic_kind == OpSemanticKind::Normalization);

    const OpDescriptor* flash = OpRegistry::instance().find_by_name("flash_attention");
    REQUIRE(flash != nullptr);
    REQUIRE(flash->semantic_kind == OpSemanticKind::Attention);

    const OpDescriptor* loss = OpRegistry::instance().find_by_name("cross_entropy_loss");
    REQUIRE(loss != nullptr);
    REQUIRE(loss->semantic_kind == OpSemanticKind::Loss);
}

}  // namespace
}  // namespace dsl
