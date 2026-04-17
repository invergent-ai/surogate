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
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_zeros(const CompiledOp& op) {
    // Use compiled output shape if available (handles split backward zeros
    // where shape_like reference might not resolve at runtime).
    const auto& shape = op.outputs[0].shape;
    if (!shape.empty()) {
        ETensorDType dtype = (op.outputs[0].dtype == ETensorDType::FP32) ? ETensorDType::FP32 : ETensorDType::BF16;
        Tensor out = mRunState.temp_alloc(dtype, shape, "zeros");
        mTemps.push_back(out);
        fill_zero(out, mRunState.MainStream);
        store_tensor(op.outputs[0], out);
        return;
    }
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    fill_zero(out, mRunState.MainStream);
    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_zeros_backward(const CompiledOp& op) {
    // Zeros backward is a no-op - gradient doesn't flow through zeros initialization
}

namespace {

// -----------------------------------------------------------------------------
// Zeros - no backward (constant has zero gradient)
// -----------------------------------------------------------------------------
std::vector<Operation> zeros_backward(const BackwardRuleContext& ctx) {
    // No operations needed - gradient of a constant is zero
    return {};
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("zeros", ::dsl::zeros_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Zeros
// ------------------------------------------------------------------------
const int _zeros_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "zeros";
    sig.min_inputs = 0;
    sig.max_inputs = 0;
    sig.min_outputs = 1;
    sig.max_outputs = 1;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        // No inputs, output shape determined by allocation
        if (outputs.empty() || outputs[0].empty()) {
            ShapeValidationError err;
            err.message = "zeros: output shape not specified or could not be resolved";

            std::ostringstream hint;
            hint << "The 'zeros' operation requires an explicit output shape. ";
            hint << "This shape should be defined in the IR tensor definition or operation attributes. ";
            hint << "Check that the output tensor is properly declared in the DSL graph with a concrete shape.";
            err.hint = hint.str();

            return std::make_optional(err);
        }
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

// ------------------------------------------------------------------------
// ZerosBackward
// ------------------------------------------------------------------------
const int _zeros_backward_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "zeros_backward";
    sig.min_inputs = 0;
    sig.max_inputs = 0;
    sig.min_outputs = 0;
    sig.max_outputs = 0;
    sig.validator = [](const std::vector<std::vector<long>>& inputs,
                       const std::vector<std::vector<long>>& outputs,
                       const AttrMap& attrs,
                       const ShapeEnv& env) -> std::optional<ShapeValidationError> {
        // No-op, no validation needed
        return std::optional<ShapeValidationError>();
    };
    OpShapeRegistry::instance().register_signature(sig);
    return 0;
}();

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
