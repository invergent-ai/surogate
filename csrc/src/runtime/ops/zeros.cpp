#include "runtime/dsl/compiled_ops.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "runtime/dsl/compiled_ops_helpers.h"
#include "kernels/kernels.h"
#include "utilities/dtype.h"

namespace dsl {

void CompiledExecutor::dispatch_zeros(const CompiledOp& op) {
    Tensor& out = ensure_output_tensor(op.outputs[0]);
    fill_zero(out, mRunState.MainStream);
}

void CompiledExecutor::dispatch_zeros_backward(const CompiledOp& op) {
    // Zeros backward is a no-op - gradient doesn't flow through zeros initialization
}

}  // namespace dsl
