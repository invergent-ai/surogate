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

void CompiledExecutor::dispatch_ones(const CompiledOp& op) {
    // Always allocate from op attrs shape to handle hybrid models where
    // different block types create ones tensors of different sizes.
    const auto& shape = op.attrs.shape;
    if (!shape.empty()) {
        ETensorDType dtype = (op.outputs[0].dtype == ETensorDType::FP32)
            ? ETensorDType::FP32 : ETensorDType::BF16;
        Tensor out = mRunState.temp_alloc(dtype, shape, "ones");
        mTemps.push_back(out);
        fill_constant(out, 1.0f, static_cast<std::size_t>(out.nelem()), mRunState.MainStream);
        store_tensor(op.outputs[0], out);
    } else {
        Tensor& out = ensure_output_tensor(op.outputs[0]);
        fill_constant(out, 1.0f, static_cast<std::size_t>(out.nelem()), mRunState.MainStream);
    }
}

}  // namespace dsl
