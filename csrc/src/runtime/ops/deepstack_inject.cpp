#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <stdexcept>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "runtime/dsl/graph_executor_utils.h"
#include "kernels/kernels.h"
#include "utilities/tensor.h"

namespace dsl {

void CompiledExecutor::dispatch_deepstack_inject(const CompiledOp& op) {
    Tensor& inp = resolve_tensor(op.inputs[0]);
    Tensor& mask = resolve_tensor(op.inputs[1]);
    Tensor& src = resolve_tensor(op.inputs[2]);

    Tensor& out = ensure_output_tensor(op.outputs[0]);

    const int B = static_cast<int>(mB);
    const int T = static_cast<int>(mT);
    const int C = (inp.Rank >= 1) ? static_cast<int>(inp.Sizes[inp.Rank - 1])
                                 : static_cast<int>(mConfig.HiddenSize);
    const int N = B * T;

    std::size_t temp_bytes = mask_scatter_temp_bytes(N);
    Tensor temp = mRunState.temp_alloc(ETensorDType::BYTE, {static_cast<long>(temp_bytes)});
    mTemps.push_back(temp);
    Tensor prefix = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(N)});
    mTemps.push_back(prefix);

    deepstack_inject_forward(out, inp, mask, src, prefix, temp, B, T, C, mRunState.MainStream);
}

void CompiledExecutor::dispatch_deepstack_inject_backward(const CompiledOp& op) {
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
    const int C = (d_out.Rank >= 1) ? static_cast<int>(d_out.Sizes[d_out.Rank - 1])
                                   : static_cast<int>(mConfig.HiddenSize);
    const int N = B * T;

    std::size_t temp_bytes = mask_scatter_temp_bytes(N);
    Tensor temp = mRunState.temp_alloc(ETensorDType::BYTE, {static_cast<long>(temp_bytes)});
    mTemps.push_back(temp);
    Tensor prefix = mRunState.temp_alloc(ETensorDType::INT32, {static_cast<long>(N)});
    mTemps.push_back(prefix);

    Tensor dummy;
    Tensor& d_inp_ref = d_inp ? *d_inp : dummy;
    Tensor& d_src_ref = d_src ? *d_src : dummy;

    deepstack_inject_backward(d_inp_ref, d_src_ref, d_out, mask, prefix, temp, B, T, C, mRunState.MainStream,
                              write_inp, write_src);
}

}  // namespace dsl

