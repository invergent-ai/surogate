// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/fusion_rule_registry.h"

#include <algorithm>
#include <utility>

#include "runtime/dsl/graph_compiler.h"
#include "runtime/executor/compiled_ops.h"

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

bool FusionRule::pattern_matches(const FusionContext& ctx) const {
    if (ctx.ops.size() != pattern.size()) {
        return false;
    }
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        if (ctx.ops[i].name != pattern[i]) {
            return false;
        }
    }
    return true;
}

bool FusionRule::matches(const FusionContext& ctx) const {
    return pattern_matches(ctx) && eligible(ctx);
}

FusionOpView fusion_op_view_from_compiled(const CompiledOp& op) {
    FusionOpView view;
    view.name = op_type_to_string(op.type);
    view.semantic_kind = op.semantic_kind;
    view.distribution_kind = op.distribution_kind;
    view.comm_profile = op.comm_profile;
    view.grouped_semantics = op.grouped_semantics;
    view.caps = op.default_caps;
    view.epilogue_support = op.epilogue_support;
    view.storage_compat = op.storage_compat;
    return view;
}

FusionContext make_fusion_context(const std::vector<CompiledOp>& ops, std::size_t start, std::size_t count) {
    FusionContext ctx;
    if (start >= ops.size() || count == 0) {
        return ctx;
    }
    const std::size_t end = std::min(ops.size(), start + count);
    ctx.ops.reserve(end - start);
    for (std::size_t i = start; i < end; ++i) {
        ctx.ops.push_back(fusion_op_view_from_compiled(ops[i]));
    }
    return ctx;
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

std::vector<const FusionRule*> FusionRuleRegistry::matching_rules_at(const std::vector<CompiledOp>& ops,
                                                                     std::size_t start) const {
    std::vector<const FusionRule*> out;
    if (start >= ops.size()) {
        return out;
    }
    const char* first_name = op_type_to_string(ops[start].type);
    for (const FusionRule* rule : rules_for_first_op(first_name)) {
        FusionContext ctx = make_fusion_context(ops, start, rule->pattern.size());
        if (rule->matches(ctx)) {
            out.push_back(rule);
        }
    }
    return out;
}

std::size_t FusionRuleRegistry::count_matching_starts(const std::vector<CompiledOp>& ops) const {
    std::size_t count = 0;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (!matching_rules_at(ops, i).empty()) {
            ++count;
        }
    }
    return count;
}

}  // namespace dsl
