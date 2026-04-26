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
    meta.descriptor_flags = 0x4;
    REQUIRE(OpRegistry::instance().register_op(std::move(meta)) == 0);

    const OpDescriptor* desc = OpRegistry::instance().find_by_name("__test_descriptor_metadata");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::Dense);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
    REQUIRE(desc->descriptor_flags == 0x4);
    REQUIRE(std::string(op_semantic_kind_name(desc->semantic_kind)) == "Dense");
}

TEST_CASE("moe ops carry first-month descriptor metadata", "[op_registry]") {
    const OpDescriptor* desc = OpRegistry::instance().find_by_name("moe_grouped_gemm");
    REQUIRE(desc != nullptr);
    REQUIRE(desc->semantic_kind == OpSemanticKind::MoE);
    REQUIRE(desc->distribution_kind == DistributionKind::ExpertParallel);
}

}  // namespace
}  // namespace dsl
