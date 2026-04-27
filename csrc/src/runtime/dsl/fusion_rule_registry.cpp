// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/fusion_rule_registry.h"

#include <algorithm>
#include <utility>

namespace dsl {

bool FusionContext::all_no_comm() const {
    return std::all_of(ops.begin(), ops.end(), [](const FusionOpView& op) {
        return op.comm_profile.kind == CommunicationKind::NoComm;
    });
}

bool FusionContext::any_grouped() const {
    return std::any_of(ops.begin(), ops.end(), [](const FusionOpView& op) { return op.grouped_semantics.is_grouped; });
}

bool FusionContext::all_support_capability(std::uint32_t capability) const {
    return std::all_of(ops.begin(), ops.end(), [capability](const FusionOpView& op) {
        return op.caps.has(capability);
    });
}

FusionRuleRegistry& FusionRuleRegistry::instance() {
    static FusionRuleRegistry registry;
    return registry;
}

int FusionRuleRegistry::register_rule(FusionRule rule) {
    if (rule.name.empty()) {
        return -1;
    }
    mRulesByName[rule.name] = std::move(rule);
    return 0;
}

const FusionRule* FusionRuleRegistry::find_by_name(std::string_view name) const {
    for (const auto& [key, rule] : mRulesByName) {
        if (key == name) {
            return &rule;
        }
    }
    return nullptr;
}

std::vector<const FusionRule*> FusionRuleRegistry::all_rules() const {
    std::vector<const FusionRule*> out;
    out.reserve(mRulesByName.size());
    for (const auto& [_, rule] : mRulesByName) {
        out.push_back(&rule);
    }
    std::sort(out.begin(), out.end(), [](const FusionRule* lhs, const FusionRule* rhs) {
        if (lhs->priority != rhs->priority) {
            return lhs->priority > rhs->priority;
        }
        return lhs->name < rhs->name;
    });
    return out;
}

std::vector<const FusionRule*> FusionRuleRegistry::rules_for_first_op(std::string_view op_name) const {
    std::vector<const FusionRule*> out;
    for (const auto& [_, rule] : mRulesByName) {
        if (!rule.pattern.empty() && rule.pattern.front() == op_name) {
            out.push_back(&rule);
        }
    }
    std::sort(out.begin(), out.end(), [](const FusionRule* lhs, const FusionRule* rhs) {
        if (lhs->priority != rhs->priority) {
            return lhs->priority > rhs->priority;
        }
        return lhs->name < rhs->name;
    });
    return out;
}

}  // namespace dsl
