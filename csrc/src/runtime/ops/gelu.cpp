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

void CompiledExecutor::dispatch_gelu(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const long N = static_cast<long>(inp.nelem());
    gelu_forward(out, inp, N, mRunState.MainStream);

    store_tensor(op.outputs[0], out);
}

void CompiledExecutor::dispatch_gelu_backward(const CompiledOp& op) {
    Tensor& d_out = resolve_tensor(op.inputs[0]);
    Tensor& inp = resolve_tensor(op.inputs[1]);

    Tensor& d_inp = (op.outputs[0].shape.empty() && inp.Rank > 0)
        ? [&]() -> Tensor& {
            std::vector<long> shape(inp.Sizes.begin(), inp.Sizes.begin() + inp.Rank);
            Tensor t = mRunState.temp_alloc(inp.DType, shape);
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

}  // namespace dsl
