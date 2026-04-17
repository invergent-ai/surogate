#include "runtime/executor/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/executor/compiled_ops_helpers.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_zeros(const CompiledOp& op) {
    // Use compiled output shape if available (handles split backward zeros
    // where shape_like reference might not resolve at runtime).
    const auto& shape = op.outputs[0].shape;
    if (!shape.empty()) {
        ETensorDType dtype = (op.outputs[0].dtype == ETensorDType::FP32)
            ? ETensorDType::FP32 : ETensorDType::BF16;
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

}  // namespace dsl
