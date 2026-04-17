#include "runtime/executor/compiled_ops.h"

#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/dsl/autodiff.h"
#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/executor/op_registry.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

namespace {

// Autodiff rule. Forward: y = factor * x. Backward: d_x = factor * d_y.
std::vector<Operation> scale_backward_rule(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    if (ctx.needs_grad(0)) {
        AttrMap attrs = copy_attrs(ctx.fwd_op.attrs, {"factor"});
        ops.push_back(make_operation("scale_backward_" + std::to_string(ctx.op_counter++),
                                     "scale_backward",
                                     "scale_backward",
                                     {ctx.d_output},
                                     {ctx.d_inputs[0]},
                                     attrs));
    }
    return ops;
}

}  // namespace

// Forward: out = factor * x
void CompiledExecutor::dispatch_scale(const CompiledOp& op) {
    Tensor& x = resolve_tensor(op.inputs[0]);
    const float factor = op.attrs.scale_factor;
    const long n = static_cast<long>(x.nelem());

    std::vector<long> out_shape(x.Sizes.begin(), x.Sizes.begin() + x.Rank);
    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    Tensor out = out_ref;
    if (out.nelem() != x.nelem() || out.DType != x.DType) {
        out = mRunState.temp_alloc(x.DType, out_shape, "scale_out");
        mTemps.push_back(out);
    }

    if (x.DType == ETensorDType::BF16) {
        moe_scale_forward(out.get<nv_bfloat16>(), x.get<nv_bfloat16>(), factor, n, mRunState.MainStream);
    } else if (x.DType == ETensorDType::FP32) {
        moe_scale_forward(out.get<float>(), x.get<float>(), factor, n, mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_scale: unsupported dtype");
    }
    store_tensor(op.outputs[0], out);
}

// Backward: d_x = factor * d_output (same op, same constant)
void CompiledExecutor::dispatch_scale_backward(const CompiledOp& op) {
    // Input: d_output
    // Output: d_x
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    const float factor = op.attrs.scale_factor;
    const long n = static_cast<long>(d_out.nelem());

    Tensor& d_x_ref = ensure_output_tensor(op.outputs[0]);
    Tensor d_x = d_x_ref;
    if (d_x.nelem() != d_out.nelem() || d_x.DType != d_out.DType) {
        std::vector<long> shape(d_out.Sizes.begin(), d_out.Sizes.begin() + d_out.Rank);
        d_x = mRunState.temp_alloc(d_out.DType, shape, "scale_backward_out");
        mTemps.push_back(d_x);
    }

    if (d_out.DType == ETensorDType::BF16) {
        moe_scale_forward(d_x.get<nv_bfloat16>(), d_out.get<nv_bfloat16>(), factor, n, mRunState.MainStream);
    } else if (d_out.DType == ETensorDType::FP32) {
        moe_scale_forward(d_x.get<float>(), d_out.get<float>(), factor, n, mRunState.MainStream);
    } else {
        throw std::runtime_error("dispatch_scale_backward: unsupported dtype");
    }
    store_tensor(op.outputs[0], d_x);
}

}  // namespace dsl

REGISTER_AUTODIFF("scale", ::dsl::scale_backward_rule);
