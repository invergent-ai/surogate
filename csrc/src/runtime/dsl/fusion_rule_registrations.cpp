// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include "runtime/dsl/fusion_rule_registry.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>

namespace dsl {
namespace {

bool no_collective_comm_fusion(const FusionContext& ctx) {
    return ctx.ops.empty() || ctx.all_no_comm();
}

bool dense_matmul_fusion(const FusionContext& ctx) {
    if (ctx.ops.empty()) {
        return true;
    }
    return ctx.all_no_comm() && ctx.ops.front().caps.has(OpCapabilityDenseMatmul);
}

bool moe_routing_fusion(const FusionContext& ctx) {
    if (ctx.ops.empty()) {
        return true;
    }
    return std::all_of(ctx.ops.begin(), ctx.ops.end(), [](const FusionOpView& op) {
        return op.semantic_kind == OpSemanticKind::MoE && op.comm_profile.kind == CommunicationKind::NoComm;
    });
}

bool moe_grouped_consumer_fusion(const FusionContext& ctx) {
    if (ctx.ops.empty()) {
        return true;
    }
    return std::any_of(ctx.ops.begin(), ctx.ops.end(), [](const FusionOpView& op) {
        return op.grouped_semantics.is_grouped || op.caps.has(OpCapabilityGroupedMatmul);
    });
}

FusionRule make_rule(std::string_view name,
                     std::initializer_list<std::string_view> pattern,
                     FusionEligibleFn eligible,
                     int priority,
                     bool comm_aware = true) {
    FusionRule rule;
    rule.name = std::string{name};
    rule.pattern.reserve(pattern.size());
    for (std::string_view op : pattern) {
        rule.pattern.emplace_back(op);
    }
    rule.eligible_fn = eligible;
    rule.priority = priority;
    rule.comm_aware = comm_aware;
    return rule;
}

}  // namespace

REGISTER_FUSION_RULE(make_rule("matmul_bias", {"matmul", "bias_add"}, dense_matmul_fusion, 100));
REGISTER_FUSION_RULE(make_rule("matmul_swiglu", {"matmul", "swiglu"}, dense_matmul_fusion, 95));
REGISTER_FUSION_RULE(make_rule("qkv_qknorm", {"matmul", "qkv_qk_norm"}, no_collective_comm_fusion, 90));
REGISTER_FUSION_RULE(make_rule("qkv_qknorm_rope", {"matmul", "qkv_qk_norm", "rope"}, no_collective_comm_fusion, 89));
REGISTER_FUSION_RULE(make_rule("residual_rmsnorm", {"add", "rmsnorm"}, no_collective_comm_fusion, 85));
REGISTER_FUSION_RULE(make_rule("lmhead_loss", {"matmul", "cross_entropy_loss"}, dense_matmul_fusion, 80));
REGISTER_FUSION_RULE(
    make_rule("mamba_gated_rmsnorm", {"mamba_ssm_scan", "mamba_gated_rmsnorm"}, no_collective_comm_fusion, 75));
REGISTER_FUSION_RULE(make_rule("moe_routing_topk_softmax", {"moe_softmax", "moe_topk"}, moe_routing_fusion, 70));
REGISTER_FUSION_RULE(
    make_rule("moe_permute_quantize", {"moe_permute", "moe_grouped_gemm"}, moe_grouped_consumer_fusion, 65));

}  // namespace dsl
