// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#ifndef SUROGATE_SRC_RUNTIME_DSL_FUSION_RULE_REGISTRY_H
#define SUROGATE_SRC_RUNTIME_DSL_FUSION_RULE_REGISTRY_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "runtime/executor/op_descriptor_types.h"

namespace dsl {

struct CompiledOp;

struct FusionOpView {
    std::string name;
    OpSemanticKind semantic_kind = OpSemanticKind::Unknown;
    DistributionKind distribution_kind = DistributionKind::Replicated;
    CommunicationProfile comm_profile{};
    GroupedSemantics grouped_semantics{};
    OpCapabilities caps{};
    EpilogueSupport epilogue_support{};
    StorageCompatibility storage_compat{};
};

struct FusionContext {
    std::vector<FusionOpView> ops;

    [[nodiscard]] bool all_no_comm() const;
    [[nodiscard]] bool any_grouped() const;
    [[nodiscard]] bool all_support_capability(std::uint32_t capability) const;
};

using FusionEligibleFn = bool (*)(const FusionContext&);

struct FusionRule {
    std::string name;
    std::vector<std::string> pattern;
    FusionEligibleFn eligible_fn = nullptr;
    int priority = 0;
    bool comm_aware = true;

    [[nodiscard]] bool pattern_matches(const FusionContext& ctx) const;

    [[nodiscard]] bool eligible(const FusionContext& ctx) const {
        return eligible_fn == nullptr || eligible_fn(ctx);
    }

    [[nodiscard]] bool matches(const FusionContext& ctx) const;
};

[[nodiscard]] FusionOpView fusion_op_view_from_compiled(const CompiledOp& op);
[[nodiscard]] FusionContext
make_fusion_context(const std::vector<CompiledOp>& ops, std::size_t start, std::size_t count);

class FusionRuleRegistry {
public:
    static FusionRuleRegistry& instance();

    int register_rule(FusionRule rule);

    const FusionRule* find_by_name(std::string_view name) const;
    std::vector<const FusionRule*> all_rules() const;
    std::vector<const FusionRule*> rules_for_first_op(std::string_view op_name) const;

private:
    FusionRuleRegistry() = default;

    std::unordered_map<std::string, FusionRule> mRulesByName;
};

}  // namespace dsl

#define SUROGATE_FUSION_RULE_REG_CONCAT_(a, b) a##b
#define SUROGATE_FUSION_RULE_REG_CONCAT(a, b) SUROGATE_FUSION_RULE_REG_CONCAT_(a, b)

#define REGISTER_FUSION_RULE(rule_expr)                                                         \
    static const int SUROGATE_FUSION_RULE_REG_CONCAT(_surogate_fusion_rule_reg_, __COUNTER__) = \
        ::dsl::FusionRuleRegistry::instance().register_rule((rule_expr))

#endif  // SUROGATE_SRC_RUNTIME_DSL_FUSION_RULE_REGISTRY_H
