// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "runtime/dsl/fusion_rule_registry.h"

namespace dsl {
namespace {

bool dense_no_comm_eligible(const FusionContext& ctx) {
    return ctx.ops.size() == 2 && ctx.all_no_comm() && ctx.all_support_capability(OpCapabilityDenseMatmul);
}

void ensure_test_matmul_rules_registered() {
    FusionRule matmul_bias;
    matmul_bias.name = "__test_fusion_matmul_bias";
    matmul_bias.pattern = {"matmul", "bias_add"};
    matmul_bias.eligible_fn = dense_no_comm_eligible;
    matmul_bias.priority = 20;
    REQUIRE(FusionRuleRegistry::instance().register_rule(std::move(matmul_bias)) == 0);

    FusionRule matmul_activation;
    matmul_activation.name = "__test_fusion_matmul_activation";
    matmul_activation.pattern = {"matmul", "swiglu"};
    matmul_activation.priority = 30;
    REQUIRE(FusionRuleRegistry::instance().register_rule(std::move(matmul_activation)) == 0);
}

TEST_CASE("fusion rule registry stores rules by name and first op", "[fusion_rule]") {
    ensure_test_matmul_rules_registered();
    const FusionRule* found = FusionRuleRegistry::instance().find_by_name("__test_fusion_matmul_bias");
    REQUIRE(found != nullptr);
    REQUIRE(found->pattern.size() == 2);
    REQUIRE(found->pattern[0] == "matmul");
    REQUIRE(found->pattern[1] == "bias_add");
    REQUIRE(found->priority == 20);

    std::vector<const FusionRule*> matmul_rules = FusionRuleRegistry::instance().rules_for_first_op("matmul");
    REQUIRE(matmul_rules.size() >= 2);

    auto first_named = std::find_if(matmul_rules.begin(), matmul_rules.end(), [](const FusionRule* rule) {
        return rule->name == "__test_fusion_matmul_activation";
    });
    auto second_named = std::find_if(matmul_rules.begin(), matmul_rules.end(), [](const FusionRule* rule) {
        return rule->name == "__test_fusion_matmul_bias";
    });
    REQUIRE(first_named != matmul_rules.end());
    REQUIRE(second_named != matmul_rules.end());
    REQUIRE(std::distance(matmul_rules.begin(), first_named) < std::distance(matmul_rules.begin(), second_named));
}

TEST_CASE("fusion rule eligibility can inspect capabilities and communication profile", "[fusion_rule]") {
    ensure_test_matmul_rules_registered();
    const FusionRule* rule = FusionRuleRegistry::instance().find_by_name("__test_fusion_matmul_bias");
    REQUIRE(rule != nullptr);

    FusionContext eligible;
    eligible.ops = {FusionOpView{.name = "matmul",
                                 .semantic_kind = OpSemanticKind::Dense,
                                 .comm_profile = CommunicationProfile{CommunicationKind::NoComm, false, 0},
                                 .caps = OpCapabilities{OpCapabilityDenseMatmul}},
                    FusionOpView{.name = "bias_add",
                                 .semantic_kind = OpSemanticKind::Elementwise,
                                 .comm_profile = CommunicationProfile{CommunicationKind::NoComm, false, 0},
                                 .caps = OpCapabilities{OpCapabilityDenseMatmul}}};
    REQUIRE(rule->eligible(eligible));
    REQUIRE_FALSE(eligible.any_grouped());

    FusionContext collective = eligible;
    collective.ops[1].comm_profile.kind = CommunicationKind::AllReduceAfter;
    REQUIRE_FALSE(rule->eligible(collective));

    FusionContext missing_cap = eligible;
    missing_cap.ops[1].caps = OpCapabilities{OpCapabilityNone};
    REQUIRE_FALSE(rule->eligible(missing_cap));
}

TEST_CASE("fusion rule registry updates duplicate rule names", "[fusion_rule]") {
    FusionRule initial;
    initial.name = "__test_fusion_duplicate";
    initial.pattern = {"a", "b"};
    initial.priority = 1;
    REQUIRE(FusionRuleRegistry::instance().register_rule(std::move(initial)) == 0);

    FusionRule replacement;
    replacement.name = "__test_fusion_duplicate";
    replacement.pattern = {"a", "c"};
    replacement.priority = 7;
    replacement.comm_aware = false;
    REQUIRE(FusionRuleRegistry::instance().register_rule(std::move(replacement)) == 0);

    const FusionRule* found = FusionRuleRegistry::instance().find_by_name("__test_fusion_duplicate");
    REQUIRE(found != nullptr);
    REQUIRE(found->pattern.size() == 2);
    REQUIRE(found->pattern[1] == "c");
    REQUIRE(found->priority == 7);
    REQUIRE_FALSE(found->comm_aware);
}

TEST_CASE("built-in fusion rule declarations are registered inertly", "[fusion_rule]") {
    const FusionRule* matmul_bias = FusionRuleRegistry::instance().find_by_name("matmul_bias");
    REQUIRE(matmul_bias != nullptr);
    REQUIRE(matmul_bias->pattern == std::vector<std::string>{"matmul", "bias_add"});
    REQUIRE(matmul_bias->priority == 100);
    REQUIRE(matmul_bias->comm_aware);

    const FusionRule* moe_permute_quantize = FusionRuleRegistry::instance().find_by_name("moe_permute_quantize");
    REQUIRE(moe_permute_quantize != nullptr);
    REQUIRE(moe_permute_quantize->pattern == std::vector<std::string>{"moe_permute", "moe_grouped_gemm"});
    REQUIRE(moe_permute_quantize->priority == 65);

    std::vector<const FusionRule*> matmul_rules = FusionRuleRegistry::instance().rules_for_first_op("matmul");
    auto found = std::find_if(matmul_rules.begin(), matmul_rules.end(), [](const FusionRule* rule) {
        return rule->name == "matmul_bias";
    });
    REQUIRE(found != matmul_rules.end());
}

}  // namespace
}  // namespace dsl
