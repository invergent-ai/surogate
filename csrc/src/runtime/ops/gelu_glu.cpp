// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Fused GELU-GLU dispatch: h = gelu_tanh(gate) * up on two separate
// same-shape tensors. The kernel itself lives in csrc/src/kernels/gelu_glu.cu.
//
// Replaces the pair `gate_act = gelu(gate); h = gate_act * up` used in
// Gemma4's non-fuse_gate_up GatedMLP path. Saves two kernel launches + one
// HBM round-trip for the intermediate gate_act buffer per MLP per direction.
//
// Mirrors swiglu.cpp's dispatch structure but for separate-tensor inputs
// (two inputs → one output, matching our `mul` op's shape semantics rather
// than swiglu's packed-[up,gate]-single-input layout).

#include "runtime/executor/compiled_ops.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_gelu_glu(const CompiledOp& op) {
    if (op.inputs.size() < 2) {
        throw std::runtime_error("dispatch_gelu_glu: expected 2 inputs (gate, up)");
    }
    Tensor& gate = resolve_tensor(op.inputs[0]);
    Tensor& up = resolve_tensor(op.inputs[1]);
    if (gate.DType != up.DType) {
        throw std::runtime_error("dispatch_gelu_glu: gate/up dtype mismatch");
    }
    if (gate.nelem() != up.nelem()) {
        throw std::runtime_error("dispatch_gelu_glu: gate/up element count mismatch (" + std::to_string(gate.nelem()) +
                                 " vs " + std::to_string(up.nelem()) + ")");
    }

    std::vector<long> out_shape(gate.Sizes.begin(), gate.Sizes.begin() + gate.Rank);
    Tensor out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.nelem() != gate.nelem() || out.DType != gate.DType) {
        out = mRunState.temp_alloc(gate.DType, out_shape, "gelu_glu_out");
        mTemps.push_back(out);
    }

    const long n = static_cast<long>(gate.nelem());
    if (gate.DType == ETensorDType::BF16) {
        gelu_glu_forward(out.get<nv_bfloat16>(),
                         gate.get<nv_bfloat16>(),
                         up.get<nv_bfloat16>(),
                         n,
                         mRunState.MainStream);
    } else if (gate.DType == ETensorDType::FP32) {
        gelu_glu_forward(out.get<float>(), gate.get<float>(), up.get<float>(), n, mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_gelu_glu: unsupported dtype");
    }
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_gelu_glu_backward(const CompiledOp& op) {
    // Inputs: d_out (dL/dh), gate, up
    // Outputs: d_gate, d_up
    if (op.inputs.size() < 3) {
        throw std::runtime_error("dispatch_gelu_glu_backward: expected 3 inputs (d_out, gate, up)");
    }
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& gate = resolve_tensor(op.inputs[1]);
    Tensor& up = resolve_tensor(op.inputs[2]);
    if (d_out.DType != gate.DType || d_out.DType != up.DType) {
        throw std::runtime_error("dispatch_gelu_glu_backward: dtype mismatch between d_out/gate/up");
    }
    if (d_out.nelem() != gate.nelem() || d_out.nelem() != up.nelem()) {
        throw std::runtime_error("dispatch_gelu_glu_backward: element count mismatch");
    }

    auto allocate_like = [&](std::size_t out_idx, const Tensor& like) -> Tensor {
        if (op.outputs.size() > out_idx && !op.outputs[out_idx].name.empty()) {
            Tensor& out_ref = ensure_output_tensor(op.outputs[out_idx]);
            if (out_ref.nelem() == like.nelem() && out_ref.DType == like.DType) {
                return out_ref;
            }
        }
        std::vector<long> shape(like.Sizes.begin(), like.Sizes.begin() + like.Rank);
        Tensor out = mRunState.temp_alloc(like.DType, shape, "gelu_glu_backward_out");
        mTemps.push_back(out);
        return out;
    };

    Tensor d_gate = allocate_like(0, gate);
    Tensor d_up = allocate_like(1, up);

    const long n = static_cast<long>(gate.nelem());
    if (gate.DType == ETensorDType::BF16) {
        gelu_glu_backward(d_gate.get<nv_bfloat16>(),
                          d_up.get<nv_bfloat16>(),
                          d_out.get<nv_bfloat16>(),
                          gate.get<nv_bfloat16>(),
                          up.get<nv_bfloat16>(),
                          n,
                          mRunState.MainStream);
    } else if (gate.DType == ETensorDType::FP32) {
        gelu_glu_backward(d_gate.get<float>(),
                          d_up.get<float>(),
                          d_out.get<float>(),
                          gate.get<float>(),
                          up.get<float>(),
                          n,
                          mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_gelu_glu_backward: unsupported dtype");
    }

    if (op.outputs.size() > 0 && !op.outputs[0].name.empty()) store_tensor(op.outputs[0], d_gate);
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) store_tensor(op.outputs[1], d_up);
}

namespace {

// -----------------------------------------------------------------------------
// gelu_glu backward rule
// Forward:  h = gelu_tanh(gate) * up       (inputs = [gate, up])
// Backward: d_gate, d_up = gelu_glu_backward(d_h, gate, up)
// -----------------------------------------------------------------------------
std::vector<Operation> gelu_glu_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 2) return ops;

    const std::string& gate = fwd.inputs[0];
    const std::string& up = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    ops.push_back(make_operation("gelu_glu_backward_" + std::to_string(ctx.op_counter++),
                                 "gelu_glu_backward",
                                 "gelu_glu_backward",
                                 {ctx.d_output, saved_ref(gate), saved_ref(up)},
                                 outputs));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("gelu_glu", ::dsl::gelu_glu_backward_rule);

// ---------------------------------------------------------------------------
// Shape signatures
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

const int _gelu_glu_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "gelu_glu";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs,
                       const auto& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        if (inputs.size() < 2 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "gelu_glu requires 2 inputs (gate, up) and 1 output";
            return std::make_optional(err);
        }
        const auto& gate = inputs[0];
        const auto& up = inputs[1];
        const auto& out = outputs[0];
        if (auto err = validators::check_same_numel(gate, up, "gate", "up", "gelu_glu")) return err;
        if (!out.empty()) {
            if (auto err = validators::check_same_numel(gate, out, "gate", "out", "gelu_glu")) return err;
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

const int _gelu_glu_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "gelu_glu_backward";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap&,
                       const ShapeEnv&) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& gate = inputs[1];
        const auto& up = inputs[2];
        if (auto err = validators::check_same_numel(d_out, gate, "d_out", "gate", "gelu_glu_backward")) return err;
        if (auto err = validators::check_same_numel(d_out, up, "d_out", "up", "gelu_glu_backward")) return err;
        if (!outputs.empty() && !outputs[0].empty()) {
            if (auto err = validators::check_same_numel(outputs[0], gate, "d_gate", "gate", "gelu_glu_backward"))
                return err;
        }
        if (outputs.size() > 1 && !outputs[1].empty()) {
            if (auto err = validators::check_same_numel(outputs[1], up, "d_up", "up", "gelu_glu_backward")) return err;
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
