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

void CompiledExecutor::dispatch_ones(const CompiledOp& op) {
    // Always allocate from op attrs shape to handle hybrid models where
    // different block types create ones tensors of different sizes.
    const auto& shape = op.attrs.shape;
    if (!shape.empty()) {
        ETensorDType dtype = (op.outputs[0].dtype == ETensorDType::FP32) ? ETensorDType::FP32 : ETensorDType::BF16;
        Tensor out = mRunState.temp_alloc(dtype, shape, "ones");
        mTemps.push_back(out);
        fill_constant(out, 1.0f, static_cast<std::size_t>(out.nelem()), mRunState.MainStream);
        store_tensor(op.outputs[0], out);
    } else {
        Tensor& out = ensure_output_tensor(op.outputs[0]);
        fill_constant(out, 1.0f, static_cast<std::size_t>(out.nelem()), mRunState.MainStream);
    }
}

namespace {

// -----------------------------------------------------------------------------
// Ones - no backward (constant has zero gradient)
// -----------------------------------------------------------------------------
std::vector<Operation> ones_backward(const BackwardRuleContext& ctx) {
    // No operations needed - gradient of a constant is zero
    return {};
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("ones", ::dsl::ones_backward);

// ---------------------------------------------------------------------------
// Shape signatures (Phase 2c)
// ---------------------------------------------------------------------------
namespace dsl {
namespace shape_checker {
namespace {

// ------------------------------------------------------------------------
// Ones
// ------------------------------------------------------------------------
const int _ones_shape_reg = [] {
    OpShapeSignature sig;
    sig.op_name = "ones";
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
            err.message = "ones: output shape not specified or could not be resolved";

            std::ostringstream hint;
            hint << "The 'ones' operation requires an explicit output shape. ";
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

}  // namespace
}  // namespace shape_checker
}  // namespace dsl
