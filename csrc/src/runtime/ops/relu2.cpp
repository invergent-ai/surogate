#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_relu2(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);

    // Element-wise op: output shape matches input shape.
    // Compiled shape may be empty for MoE intermediates, so allocate from input dims.
    Tensor out;
    if (op.outputs[0].shape.empty() && inp.Rank > 0) {
        std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
        out = mRunState.temp_alloc(inp.DType, shape);
        mTemps.push_back(out);
    } else {
        out = ensure_output_tensor(op.outputs[0]);
    }

    const long N = static_cast<long>(inp.nelem());
    relu2_forward(out, inp, N, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = out;
}

void CompiledExecutor::dispatch_relu2_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);

    // Element-wise backward: output shape matches input shape.
    // Compiled shape may be empty when backward compiler can't track saved tensor shapes.
    Tensor& d_inp = (op.outputs[0].shape.empty() && inp.Rank > 0)
        ? [&]() -> Tensor& {
            std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
            Tensor t = mRunState.temp_alloc(inp.DType, shape);
            fill_zero(t, mRunState.MainStream);
            mTemps.push_back(t);
            auto [it, _] = mTensorMap.insert_or_assign(op.outputs[0].name, t);
            return it->second;
        }()
        : ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    // Kernel signature: relu2_backward(dinp, inp, dout, n, stream)
    relu2_backward(d_inp, inp, d_out, N, mRunState.MainStream);

    mTensorMap[op.outputs[0].name] = d_inp;
}


}  // namespace dsl
