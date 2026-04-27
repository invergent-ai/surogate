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
    meta.comm_profile.kind = CommunicationKind::ExpertParallelRouted;
    meta.grouped_semantics.is_grouped = true;
    meta.grouped_semantics.expert_dim = 0;
    meta.descriptor_flags = 0x4;
    REQUIRE(OpRegistry::instance().register_op(std::move(meta)) == 0);

    const OpDescriptor* desc = OpRegistry::instance().find_by_name("__test_descriptor_metadata");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
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

}  // namespace
}  // namespace dsl
