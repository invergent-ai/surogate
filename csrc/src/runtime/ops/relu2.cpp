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

void CompiledExecutor::dispatch_relu2(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Element-wise op: output shape matches input shape.
    // EP-routed MoE intermediates can have dynamic row counts, so the compiled
    // output buffer may be empty or simply the wrong shape for this rank.
    Tensor out;
    auto needs_dynamic = [&](const TensorRef& out_ref, const Tensor& in) -> bool {
        if (out_ref.shape.empty()) return true;
        if (in.Rank <= 0) return false;
        if (static_cast<int>(out_ref.shape.size()) != in.Rank) return true;
        for (int i = 0; i < in.Rank; ++i) {
            if (out_ref.shape[i] != in.Sizes[i]) return true;
        }
        return false;
    };
    if (needs_dynamic(op.outputs[0], inp)) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        out = mRunState.temp_alloc(inp.DType, shape, "relu2_out");
        mTemps.push_back(out);
    } else {
        out = ensure_output_tensor(op.outputs[0]);
    }

    const long N = static_cast<long>(inp.nelem());
    relu2_forward(out, inp, N, mRunState.MainStream);

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_relu2_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);

    // Element-wise backward: output shape matches input shape.
    // EP changes token count dynamically — pre-allocated buffer may be wrong size.
    // Re-allocate if shape is empty OR if element count doesn't match input.
    Tensor& d_inp_ref = ensure_output_tensor(op.outputs[0]);
    Tensor* d_inp_ptr = &d_inp_ref;
    const long expected_nelem = static_cast<long>(inp.nelem());
    if (d_inp_ptr->nelem() != expected_nelem) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        Tensor t = mRunState.temp_alloc(inp.DType, shape, "relu2_backward_d_inp");
        fill_zero(t, mRunState.MainStream);
        mTemps.push_back(t);
        store_tensor(op.outputs[0], t);
        d_inp_ptr = &mTensors[op.outputs[0].tensor_id];
    }

    const long N = expected_nelem;
    // Kernel signature: relu2_backward(dinp, inp, dout, n, stream)
    relu2_backward(*d_inp_ptr, inp, d_out, N, mRunState.MainStream);

    store_tensor(op.outputs[0], *d_inp_ptr);
}

namespace {

// -----------------------------------------------------------------------------
// ReLU² backward rule
// Forward: y = relu2(x) = x * relu(x) = x * max(0, x)
// For x <= 0: y = 0
// For x > 0:  y = x²
// Backward: dx = dy * 2 * relu(x)
// -----------------------------------------------------------------------------
std::vector<Operation> relu2_backward(const BackwardRuleContext& ctx) {
    std::vector<Operation> ops;

    if (ctx.needs_grad(0)) {
        const auto& fwd = ctx.fwd_op;
        std::string x = fwd.inputs[0];

        ops.push_back(make_operation("relu2_backward_" + std::to_string(ctx.op_counter++),
                                     "relu2_backward",
                                     "relu2_backward",
                                     {ctx.d_output, saved_ref(x)},
                                     {ctx.d_inputs[0]}));
    }

    return ops;
}

}  // namespace

}  // namespace dsl

REGISTER_AUTODIFF("relu2", ::dsl::relu2_backward);
