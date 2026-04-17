#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace dsl {

void CompiledExecutor::dispatch_mask_scatter(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& mask = resolve_tensor(op.inputs[1]);
    Tensor& src = resolve_tensor(op.inputs[2]);

    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const int B = static_cast<int>(mB);
    const int T = static_cast<int>(mT);
    const int C = (inp.Rank >= 1) ? static_cast<int>(inp.Sizes[inp.Rank - 1]) : static_cast<int>(mConfig.HiddenSize);
    const int N = B * T;

    // Visual embeddings are external inputs; sanitize any NaN/Inf to avoid
    // corrupting the residual stream and downstream loss/gradients.
    if (src.DType == ETensorDType::BF16 || src.DType == ETensorDType::FP32) {
        sanitize_non_finite(src, mRunState.MainStream);
    }

    std::size_t temp_bytes = mask_scatter_temp_bytes(N);
    Tensor temp = mRunState.temp_alloc(ETensorDType::BYTE, {static_cast<long>(temp_bytes)}, "mask_scatter_temp");
    mTemps.push_back(temp);
    Tensor prefix = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(N)}, "mask_scatter_prefix");
    mTemps.push_back(prefix);

    mask_scatter_forward(out, inp, mask, src, prefix, temp, B, T, C, mRunState.MainStream);
}

void CompiledExecutor::dispatch_mask_scatter_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& mask = resolve_tensor(op.inputs[1]);

    const bool write_inp = (!op.outputs.empty() && !op.outputs[0].name.empty());
    const bool write_src = (op.outputs.size() > 2 && !op.outputs[2].name.empty());

    Tensor* d_inp = write_inp ? &ensure_output_tensor(op.outputs[0]) : nullptr;
    Tensor* d_src = write_src ? &ensure_output_tensor(op.outputs[2]) : nullptr;
    if (!write_inp && !write_src) {
        return;
    }

    if (write_src && d_src) {
        fill_zero(*d_src, mRunState.MainStream);
    }

    const int B = static_cast<int>(mB);
    const int T = static_cast<int>(mT);
    const int C =
        (d_out.Rank >= 1) ? static_cast<int>(d_out.Sizes[d_out.Rank - 1]) : static_cast<int>(mConfig.HiddenSize);
    const int N = B * T;

    std::size_t temp_bytes = mask_scatter_temp_bytes(N);
    Tensor temp = mRunState.temp_alloc(ETensorDType::BYTE, {static_cast<long>(temp_bytes)}, "mask_scatter_temp");
    mTemps.push_back(temp);
    Tensor prefix = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(N)}, "mask_scatter_prefix");
    mTemps.push_back(prefix);

    Tensor dummy;
    Tensor& d_inp_ref = d_inp ? *d_inp : dummy;
    Tensor& d_src_ref = d_src ? *d_src : dummy;

    mask_scatter_backward(d_inp_ref,
                          d_src_ref,
                          d_out,
                          mask,
                          prefix,
                          temp,
                          B,
                          T,
                          C,
                          mRunState.MainStream,
                          write_inp,
                          write_src);
}

namespace {

// -----------------------------------------------------------------------------
// Masked scatter backward rule
// Forward: out = mask_scatter(x, mask, src)
// Backward: d_x, d_src (mask is non-differentiable)
// -----------------------------------------------------------------------------
std::vector<Operation> mask_scatter_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;
    const auto& fwd = ctx.fwd_op;
    if (fwd.inputs.size() < 2) {
        return ops;
    }
    const std::string& mask = fwd.inputs[1];

    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back("");  // mask has no gradient
    outputs.push_back(ctx.needs_grad(2) ? ctx.d_inputs[2] : "");

    if (outputs[0].empty() && outputs[2].empty()) {
        return ops;
    }

    ops.push_back(make_operation("mask_scatter_backward_" + std::to_string(ctx.op_counter++),
                                 "mask_scatter_backward",
                                 "mask_scatter_backward",
                                 {ctx.d_output, mask},
                                 outputs));
    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("mask_scatter", ::dsl::mask_scatter_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Masked scatter (visual embedding replacement)
// ------------------------------------------------------------------------
const int _mask_scatter_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "mask_scatter";
    sig.min_inputs = 3;
    sig.max_inputs = 3;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const auto& inputs, const auto& outputs, const AttrMap&, const ShapeEnv&) {
        if (inputs.size() < 3 || outputs.empty()) {
            ShapeValidationError err;
            err.message = "mask_scatter requires 3 inputs and 1 output";
            return std::make_optional(err);
        }
        if (auto err = validators::check_rank(inputs[0], 3, "input", "mask_scatter")) return err;
        if (auto err = validators::check_rank(inputs[1], 2, "mask", "mask_scatter")) return err;
        if (auto err = validators::check_rank(inputs[2], 2, "src", "mask_scatter")) return err;

        if (!inputs[0].empty() && !inputs[2].empty()) {
            long C = inputs[0].back();
            if (inputs[2].back() != C) {
                ShapeValidationError err;
                err.message = "mask_scatter: src last dim must match input last dim";
                return std::make_optional(err);
            }
            long N = inputs[0][0] * inputs[0][1];
            if (inputs[2][0] != N) {
                ShapeValidationError err;
                err.message = "mask_scatter: src first dim must equal B*T";
                return std::make_optional(err);
            }
        }

        if (!outputs[0].empty()) {
            return validators::check_same_numel(inputs[0], outputs[0], "input", "output", "mask_scatter");
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
