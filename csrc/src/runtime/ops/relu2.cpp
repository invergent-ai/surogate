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
        Tensor t = mRunState.temp_alloc(inp.DType, shape);
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


}  // namespace dsl
