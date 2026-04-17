#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_gpt_oss_moe_act(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    float alpha = (op.attrs.gpt_oss_alpha > 0.0f) ? op.attrs.gpt_oss_alpha : 1.702f;
    float limit = (op.attrs.gpt_oss_limit > 0.0f) ? op.attrs.gpt_oss_limit : 7.0f;

    if (inp.Rank == 2) {
        const long N = inp.Sizes[0];
        const long D = inp.Sizes[1] / 2;
        std::vector<long> out_shape = {N, D};
        Tensor out = mRunState.temp_alloc(inp.DType, out_shape, "gpt_oss_moe_act_out");
        mTemps.push_back(out);

        if (inp.DType == ETensorDType::BF16) {
            gpt_oss_moe_act_forward(out.get<nv_bfloat16>(),
                                    inp.get<nv_bfloat16>(),
                                    static_cast<int>(N),
                                    static_cast<int>(D),
                                    alpha,
                                    limit,
                                    mRunState.MainStream);
        } else if (inp.DType == ETensorDType::FP32) {
            gpt_oss_moe_act_forward(out.get<float>(),
                                    inp.get<float>(),
                                    static_cast<int>(N),
                                    static_cast<int>(D),
                                    alpha,
                                    limit,
                                    mRunState.MainStream);
        } else {
            throw std::logic_error("gpt_oss_moe_act: unsupported input dtype");
        }

        store_tensor(op.outputs[0], out);

        return;
    }

    Tensor& out = ensure_output_tensor(op.outputs[0]);
    const long B = inp.Sizes[0];
    const long T = inp.Sizes[1];
    const long D = inp.Sizes[2] / 2;
    const int N = static_cast<int>(B * T);

    if (inp.DType == ETensorDType::BF16) {
        gpt_oss_moe_act_forward(out.get<nv_bfloat16>(),
                                inp.get<nv_bfloat16>(),
                                N,
                                static_cast<int>(D),
                                alpha,
                                limit,
                                mRunState.MainStream);
    } else if (inp.DType == ETensorDType::FP32) {
        gpt_oss_moe_act_forward(out.get<float>(),
                                inp.get<float>(),
                                N,
                                static_cast<int>(D),
                                alpha,
                                limit,
                                mRunState.MainStream);
    } else {
        throw std::logic_error("gpt_oss_moe_act: unsupported input dtype");
    }
}

void CompiledExecutor::dispatch_gpt_oss_moe_act_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);
    Tensor* d_inp_ptr = nullptr;
    Tensor d_inp_local;

    float alpha = (op.attrs.gpt_oss_alpha > 0.0f) ? op.attrs.gpt_oss_alpha : 1.702f;
    float limit = (op.attrs.gpt_oss_limit > 0.0f) ? op.attrs.gpt_oss_limit : 7.0f;

    if (d_out.Rank == 2) {
        const long N = d_out.Sizes[0];
        const long D = d_out.Sizes[1];
        const long expected_inp = N * D * 2;
        if (inp.nelem() != expected_inp) {
            std::ostringstream oss;
            oss << "gpt_oss_moe_act_backward: shape mismatch: d_out=[" << N << "," << D
                << "] inp_nelem=" << inp.nelem();
            throw std::runtime_error(oss.str());
        }
        d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
        if (static_cast<long>(d_inp_ptr->nelem()) != expected_inp) {
            d_inp_local = mRunState.temp_alloc(inp.DType, {N, D * 2}, "gpt_oss_moe_act_backward_d_inp");
            mTemps.push_back(d_inp_local);
            store_tensor(op.outputs[0], d_inp_local);
            d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
        }
        Tensor& d_inp = *d_inp_ptr;
        if (d_out.DType == ETensorDType::BF16) {
            gpt_oss_moe_act_backward(d_inp.get<nv_bfloat16>(),
                                     d_out.get<nv_bfloat16>(),
                                     inp.get<nv_bfloat16>(),
                                     static_cast<int>(N),
                                     static_cast<int>(D),
                                     alpha,
                                     limit,
                                     mRunState.MainStream);
        } else if (d_out.DType == ETensorDType::FP32) {
            gpt_oss_moe_act_backward(d_inp.get<float>(),
                                     d_out.get<float>(),
                                     inp.get<float>(),
                                     static_cast<int>(N),
                                     static_cast<int>(D),
                                     alpha,
                                     limit,
                                     mRunState.MainStream);
        } else {
            throw std::logic_error("gpt_oss_moe_act_backward: unsupported dtype");
        }
        return;
    }

    const long D = d_out.Sizes[2];
    const int N = static_cast<int>(d_out.Sizes[0] * d_out.Sizes[1]);
    d_inp_ptr = &ensure_output_tensor(op.outputs[0]);
    const long expected_inp = static_cast<long>(N) * D * 2;
    if (static_cast<long>(d_inp_ptr->nelem()) != expected_inp) {
        d_inp_local =
            mRunState.temp_alloc(inp.DType, {d_out.Sizes[0], d_out.Sizes[1], D * 2}, "gpt_oss_moe_act_backward_d_inp");
        mTemps.push_back(d_inp_local);
        store_tensor(op.outputs[0], d_inp_local);
        d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
    }
    Tensor& d_inp = *d_inp_ptr;
    if (d_out.DType == ETensorDType::BF16) {
        gpt_oss_moe_act_backward(d_inp.get<nv_bfloat16>(),
                                 d_out.get<nv_bfloat16>(),
                                 inp.get<nv_bfloat16>(),
                                 N,
                                 static_cast<int>(D),
                                 alpha,
                                 limit,
                                 mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        gpt_oss_moe_act_backward(d_inp.get<float>(),
                                 d_out.get<float>(),
                                 inp.get<float>(),
                                 N,
                                 static_cast<int>(D),
                                 alpha,
                                 limit,
                                 mRunState.MainStream);
    } else {
        throw std::logic_error("gpt_oss_moe_act_backward: unsupported dtype");
    }
}

namespace {

// -----------------------------------------------------------------------------
// GPT-OSS MoE activation backward rule
// Forward: out = gpt_oss_moe_act(inp, alpha, limit)
// Backward: d_inp = gpt_oss_moe_act_backward(d_out, inp)
// -----------------------------------------------------------------------------
std::vector<Operation> gpt_oss_moe_act_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    if (!ctx.needs_grad(0)) {
        return ops;
    }
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.empty()) {
        return ops;
    }
    std::string inp = fwd.inputs[0];
    AttrMap attrs = copy_attrs(fwd.attrs, {"alpha", "limit"});
    ops.push_back(make_operation("gpt_oss_moe_act_backward_" + std::to_string(ctx.op_counter++),
                                 "gpt_oss_moe_act_backward",
                                 "gpt_oss_moe_act_backward",
                                 {ctx.d_output, saved_ref(inp)},
                                 {ctx.d_inputs[0]},
                                 attrs));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("gpt_oss_moe_act", ::dsl::gpt_oss_moe_act_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// GPT-OSS MoE Activation (interleaved gate/up)
// ------------------------------------------------------------------------
const int _gpt_oss_moe_act_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "gpt_oss_moe_act";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        if (inputs.empty() || outputs.empty()) {
            return std::make_optional(ShapeValidationError{"gpt_oss_moe_act requires 1 input and 1 output"});
        }
        const auto& in_shape = inputs[0];
        const auto& out_shape = outputs[0];
        if (in_shape.empty() || out_shape.empty()) {
            return std::optional<ShapeValidationError>();
        }
        if (in_shape.size() != out_shape.size()) {
            ShapeValidationError err;
            err.message = "gpt_oss_moe_act: input and output rank must match";
            return std::make_optional(err);
        }
        if (in_shape.back() != 2 * out_shape.back()) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "gpt_oss_moe_act: input last dim (" << in_shape.back() << ") must be 2x output last dim ("
                << out_shape.back() << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }
        for (size_t i = 0; i + 1 < in_shape.size(); ++i) {
            if (in_shape[i] != out_shape[i]) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "gpt_oss_moe_act: dimension [" << i << "] mismatch: " << in_shape[i] << " vs " << out_shape[i];
                err.message = oss.str();
                return std::make_optional(err);
            }
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// GPT-OSS MoE Activation Backward
// ------------------------------------------------------------------------
const int _gpt_oss_moe_act_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "gpt_oss_moe_act_backward";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& inp = inputs[1];
        const auto& d_inp = outputs[0];

        if (!inp.empty() && !d_out.empty()) {
            if (inp.back() != 2 * d_out.back()) {
                ShapeValidationError err;
                std::ostringstream oss;
                oss << "gpt_oss_moe_act_backward: inp last dim (" << inp.back() << ") must be 2x d_out last dim ("
                    << d_out.back() << ")";
                err.message = oss.str();
                return std::make_optional(err);
            }
        }

        if (auto err = validators::check_same_numel(d_inp, inp, "d_inp", "inp", "gpt_oss_moe_act_backward")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
