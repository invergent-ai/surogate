// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Expert Parallelism dispatch op wiring.
//
// The real dispatch/backward implementations live in
// `runtime/ep/ep_strategy.cpp`. This file keeps only:
//   - the executor-member trampolines that forward to the strategy
//   - the autodiff rule registration
//   - the shape-signature registration

#include "runtime/executor/compiled_ops.h"

#include <vector>

#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/ep/ep_strategy.h"
#include "runtime/executor/op_registry.h"

namespace dsl {

void CompiledExecutor::dispatch_ep_dispatch(const CompiledOp& op) {
    mEpStrategy->dispatch_forward(*this, op);
}

void CompiledExecutor::dispatch_ep_dispatch_backward(const CompiledOp& op) {
    mEpStrategy->dispatch_backward(*this, op);
}

namespace {

// -----------------------------------------------------------------------------
// EP Dispatch backward rule
// Forward: recv_sorted, recv_scatter = ep_dispatch(permuted, routing, scatter, ...)
// Backward: d_permuted = ep_dispatch_backward(d_recv_sorted)
// Only the first input (permuted tokens) is differentiable
// -----------------------------------------------------------------------------
std::vector<Operation> ep_dispatch_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        AttrMap attrs = copy_attrs(fwd.attrs, {"ep_size", "num_experts", "top_k"}, "ep_dispatch_backward");

        ops.push_back(make_operation("ep_dispatch_backward_" + std::to_string(ctx.op_counter++),
                                     "ep_dispatch_backward",
                                     "ep_dispatch_backward",
                                     {ctx.d_outputs[0]},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("ep_dispatch", ::dsl::ep_dispatch_backward_rule);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

const int _ep_dispatch_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "ep_dispatch";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _ep_dispatch_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "ep_dispatch_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto&, const auto&, const AttrMap&, const ShapeEnv&) {
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
