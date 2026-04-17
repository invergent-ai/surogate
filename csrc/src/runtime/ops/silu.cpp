#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "runtime/dsl/autodiff.h"
#include "runtime/executor/op_registry.h"
#include "runtime/executor/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_silu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    silu_forward(out, inp, N, mRunState.MainStream);

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_silu_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);

    // Element-wise backward: output shape matches input shape.
    // Compiled shape may be empty when backward compiler can't track saved tensor shapes.
    Tensor& d_inp = (op.outputs[0].shape.empty() && inp.Rank > 0) ? [&]() -> Tensor& {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        Tensor t = mRunState.temp_alloc(inp.DType, shape, "silu_backward_d_inp");
        fill_zero(t, mRunState.MainStream);
        mTemps.push_back(t);
        store_tensor(op.outputs[0], t);
        return mTensors[op.outputs[0].tensor_id];
    }()
        : ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    // Kernel signature: silu_backward(dinp, inp, dout, n, stream)
    silu_backward(d_inp, inp, d_out, N, mRunState.MainStream);

    store_tensor(op.outputs[0], d_inp);
}

namespace {

// -----------------------------------------------------------------------------
// SiLU backward rule
// Forward: y = silu(x) = x * sigmoid(x)
// Backward: dx = dy * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
//             = dy * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
// -----------------------------------------------------------------------------
std::vector<Operation> silu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation("silu_backward_" + std::to_string(ctx.op_counter++),
                                     "silu_backward",
                                     "silu_backward",
                                     {ctx.d_output, saved_ref(x)},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("silu", ::dsl::silu_backward);
