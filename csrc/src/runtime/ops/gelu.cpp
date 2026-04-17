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

void CompiledExecutor::dispatch_gelu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& out_ref = ensure_output_tensor(op.outputs[0]);
    // If output buffer is wrong size (empty shape at compile time), match input
    Tensor out = out_ref;
    if (out.nelem() != inp.nelem() || !out.Data) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        out = mRunState.temp_alloc(inp.DType, shape, "gelu_out");
        mTemps.push_back(out);
    }

    const long N = static_cast<long>(inp.nelem());
    gelu_forward(out, inp, N, mRunState.MainStream);

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_gelu_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);

    Tensor& d_inp = (op.outputs[0].shape.empty() && inp.Rank > 0) ? [&]() -> Tensor& {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        Tensor t = mRunState.temp_alloc(inp.DType, shape, "gelu_backward_d_inp");
        fill_zero(t, mRunState.MainStream);
        mTemps.push_back(t);
        store_tensor(op.outputs[0], t);
        return mTensors[op.outputs[0].tensor_id];
    }()
        : ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    gelu_backward(d_inp, inp, d_out, N, mRunState.MainStream);

    store_tensor(op.outputs[0], d_inp);
}

namespace {

// -----------------------------------------------------------------------------
// GELU backward rule
// Forward: y = gelu(x)
// Backward: dx = dy * gelu'(x)
// -----------------------------------------------------------------------------
std::vector<Operation> gelu_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation("gelu_backward_" + std::to_string(ctx.op_counter++),
                                     "gelu_backward",
                                     "gelu_backward",
                                     {ctx.d_output, saved_ref(x)},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("gelu", ::dsl::gelu_backward);
