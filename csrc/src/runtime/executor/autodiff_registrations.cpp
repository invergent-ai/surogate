// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Phase 2b: centralized autodiff rule registrations.
//
// Every backward (autodiff) rule that was previously in the monolithic
// autodiff_rules.cpp now lives here, registered via the unified
// OpRegistry. Helpers (find_attr / get_string_attr / copy_attrs /
// make_operation / saved_ref / grad_name) moved to autodiff.h and
// autodiff.cpp so per-op files can reuse them when we splinter these
// registrations in 2b.2.
//
// Rule signature is unchanged from autodiff_rules.cpp — only the
// registration mechanism changed (REGISTER_AUTODIFF instead of a
// centralized register_builtin_backward_rules() switch).

#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"

#include <cstdio>
#include <stdexcept>

namespace dsl {

namespace {

// -----------------------------------------------------------------------------
// Attention backward rule
// Forward: out = attention(q, k, v, mask?)
// Backward: dq, dk, dv = attention_backward(d_out, q, k, v, out, softmax_lse, ...)
// -----------------------------------------------------------------------------
std::vector<Operation> attention_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;

    std::string q = fwd.inputs[0];
    std::string k = fwd.inputs[1];
    std::string v = fwd.inputs[2];
    std::string out = fwd.outputs[0];
    // Attention typically also saves softmax_lse
    std::string lse = fwd.outputs.size() > 1 ? fwd.outputs[1] : out + "_lse";

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");  // dq
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");  // dk
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");  // dv

    AttrMap attrs = copy_attrs(fwd.attrs, {"scale", "causal", "window_size"});

    ops.push_back(
        make_operation("attention_backward_" + std::to_string(ctx.op_counter++),
                       "attention_backward",
                       "attention_backward",
                       {ctx.d_output, saved_ref(q), saved_ref(k), saved_ref(v), saved_ref(out), saved_ref(lse)},
                       outputs,
                       attrs));

    return ops;
}

// -----------------------------------------------------------------------------
// Softmax backward rule
// Forward: y = softmax(x)
// Backward: dx = y * (dy - sum(dy * y))
// -----------------------------------------------------------------------------
std::vector<Operation> softmax_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string y = fwd.outputs[0];

        AttrMap attrs = copy_attrs(fwd.attrs, {"dim"});

        ops.push_back(make_operation("softmax_backward_" + std::to_string(ctx.op_counter++),
                                     "softmax_backward",
                                     "softmax_backward",
                                     {ctx.d_output, saved_ref(y)},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// Identity/Copy backward
// Forward: y = x
// Backward: dx = dy
// -----------------------------------------------------------------------------
std::vector<Operation> identity_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        ops.push_back(make_operation("identity_backward_" + std::to_string(ctx.op_counter++),
                                     "identity",
                                     "identity",
                                     {ctx.d_output},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

// -----------------------------------------------------------------------------
// StackedBlocks - compound operation for transformer blocks
// This is a meta-op that doesn't decompose into individual layer backwards
// -----------------------------------------------------------------------------
std::vector<Operation> stacked_blocks_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    // StackedBlocks is handled as a unit - generates StackedBlocksBackward
    std::vector<std::string> outputs;
    for (size_t i = 0; i < ctx.d_inputs.size(); ++i) {
        outputs.push_back(ctx.needs_grad(i) ? ctx.d_inputs[i] : "");
    }

    AttrMap attrs = ctx.fwd_op.attrs;  // Carry all attributes

    ops.push_back(make_operation("StackedBlocksBackward",
                                 "StackedBlocksBackward",
                                 "StackedBlocksBackward",
                                 {ctx.d_output, ctx.d_output},  // d_output for both mlp_down and residual
                                 outputs,
                                 attrs));

    return ops;
}

}  // anonymous namespace
}  // namespace dsl

// -----------------------------------------------------------------------------
// Extras registrations (Phase 2b.2)
// -----------------------------------------------------------------------------
//
// The rules above have no per-op dispatch file under runtime/ops/, so
// they stay here after decentralization. Each is keyed by a string
// name only — CompiledOpType::Unknown — since there's no forward
// dispatch to pair them with.

REGISTER_AUTODIFF("attention", ::dsl::attention_backward);
REGISTER_AUTODIFF("scaled_dot_product_attention", ::dsl::attention_backward);
REGISTER_AUTODIFF("softmax", ::dsl::softmax_backward);
REGISTER_AUTODIFF("identity", ::dsl::identity_backward);
REGISTER_AUTODIFF("copy", ::dsl::identity_backward);
REGISTER_AUTODIFF("StackedBlocks", ::dsl::stacked_blocks_backward);

namespace dsl {

// Compat shim: external code still calls register_builtin_backward_rules()
// on first BackwardRuleRegistry::instance() access. Now a no-op —
// registrations happen via static initializers above.
void register_builtin_backward_rules() {
}

}  // namespace dsl
