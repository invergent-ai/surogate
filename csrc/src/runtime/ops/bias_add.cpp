#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/dsl/op_shape_signatures.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_helpers.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_bias_add(const CompiledOp& op) {
    Tensor& x = resolve_tensor(op.inputs[0]);
    Tensor& bias = resolve_tensor(op.inputs[1]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const std::size_t bytes = static_cast<std::size_t>(x.nelem()) * get_dtype_size(x.DType);
    CUDA_CHECK(cudaMemcpyAsync(out.Data, x.Data, bytes, cudaMemcpyDeviceToDevice, mRunState.MainStream));
    add_bias_tensor(out,
                    bias,
                    static_cast<int>(x.Sizes[0]),
                    static_cast<int>(x.Sizes[1]),
                    static_cast<int>(x.Sizes[2]),
                    mRunState.MainStream);
}

void CompiledExecutor::dispatch_bias_add_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);

    // d_input = d_out (pass through)
    if (!op.outputs.empty() && !op.outputs[0].name.empty()) {
        store_tensor(op.outputs[0], d_out);
    }

    // d_bias = sum(d_out, axis=[0,1]) for [B,T,C] or axis=0 for [N,C]
    if (op.outputs.size() > 1 && !op.outputs[1].name.empty()) {
        int Bv = 1, Tv = 1, OC = 1;
        if (d_out.Rank == 2) {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = 1;
            OC = static_cast<int>(d_out.Sizes[1]);
        } else {
            Bv = static_cast<int>(d_out.Sizes[0]);
            Tv = static_cast<int>(d_out.Sizes[1]);
            OC = static_cast<int>(d_out.Sizes[2]);
        }

        Tensor& d_bias = ensure_output_tensor(op.outputs[1]);
        bool accumulate = mAccumulateTensors.count(op.outputs[1].name) > 0;
        if (!accumulate && !op.outputs[1].name.empty() && mCurrentGraph) {
            if (auto base = base_param_from_grad_kind(op.outputs[1].tensor_id, *mCurrentGraph)) {
                accumulate = mAccumulateTensors.count("d_" + *base) > 0;
            }
        }

        // Allocate scratch buffer for bias reduction
        const int scratch_bytes = get_bias_backward_scratch_size(d_out.DType, OC, mRunState.DeviceProp);
        Tensor scratch = mRunState.temp_alloc(ETensorDType::FP32,
                                              {static_cast<long>(scratch_bytes / sizeof(float))},
                                              "bias_add_backward_scratch");
        mTemps.push_back(scratch);

        if (accumulate) {
            // Accumulate into existing gradient: compute to tmp, then add
            Tensor tmp = mRunState.temp_alloc(d_out.DType, {static_cast<long>(OC)}, "bias_add_backward_tmp");
            mTemps.push_back(tmp);
            backward_bias(tmp,
                          d_out,
                          nullptr,
                          nullptr,
                          scratch,
                          Bv,
                          Tv,
                          OC,
                          mRunState.DeviceProp,
                          mRunState.MainStream);
            vector_add_sr(d_bias, d_bias, tmp, 1.0f, static_cast<long>(d_bias.nelem()), 0, mRunState.MainStream);
        } else {
            backward_bias(d_bias,
                          d_out,
                          nullptr,
                          nullptr,
                          scratch,
                          Bv,
                          Tv,
                          OC,
                          mRunState.DeviceProp,
                          mRunState.MainStream);
        }
    }
}

namespace {

// -----------------------------------------------------------------------------
// BiasAdd backward rule
// Forward: y = bias_add(x, bias)
// Backward: dx = dy, d_bias = sum(dy)
// -----------------------------------------------------------------------------
std::vector<Operation> bias_add_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    const auto& fwd = ctx.fwd_op;
    std::vector<std::string> outputs;
    outputs.push_back(ctx.needs_grad(0) ? ctx.d_inputs[0] : "");
    outputs.push_back(ctx.needs_grad(1) ? ctx.d_inputs[1] : "");

    std::vector<std::string> inputs;
    inputs.push_back(ctx.d_output);
    if (fwd.inputs.size() > 1) {
        inputs.push_back(fwd.inputs[1]);
    }

    ops.push_back(make_operation("bias_add_backward_" + std::to_string(ctx.op_counter++),
                                 "bias_add_backward",
                                 "bias_add_backward",
                                 inputs,
                                 outputs));

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("bias_add", ::dsl::bias_add_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// BiasAdd
// ------------------------------------------------------------------------
const int _bias_add_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "bias_add";
    sig.min_inputs = 2;
    sig.max_inputs = 2;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& x = inputs[0];
        const auto& bias = inputs[1];
        const auto& out = outputs[0];

        // Check bias is 1D
        if (auto err = validators::check_rank(bias, 1, "bias", "bias_add")) {
            return err;
        }

        // Check bias dimension matches last dimension of x
        if (!x.empty() && bias[0] != x.back()) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "bias_add: bias dim (" << bias[0] << ") doesn't match input last dim (" << x.back() << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        // Check output matches input
        if (auto err = validators::check_same_numel(out, x, "out", "x", "bias_add")) {
            return err;
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// BiasAddBackward
// ------------------------------------------------------------------------
const int _bias_add_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "bias_add_backward";
    sig.min_inputs = 1;
    sig.max_inputs = 1;
    sig.min_outputs = 2;
    sig.max_outputs = 2;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        const auto& d_out = inputs[0];
        const auto& d_input = outputs[0];
        const auto& d_bias = outputs[1];

        // d_input matches d_out
        if (auto err = validators::check_same_numel(d_input, d_out, "d_input", "d_out", "bias_add_backward")) {
            return err;
        }

        // d_bias is 1D
        if (auto err = validators::check_rank(d_bias, 1, "d_bias", "bias_add_backward")) {
            return err;
        }

        // d_bias dimension matches last dimension of d_out
        if (!d_out.empty() && d_bias[0] != d_out.back()) {
            ShapeValidationError err;
            std::ostringstream oss;
            oss << "bias_add_backward: d_bias dim (" << d_bias[0] << ") doesn't match d_out last dim (" << d_out.back()
                << ")";
            err.message = oss.str();
            return std::make_optional(err);
        }

        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
