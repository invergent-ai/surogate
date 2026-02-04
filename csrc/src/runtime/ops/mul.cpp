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

void CompiledExecutor::dispatch_mul(const CompiledOp& op) {
    // Element-wise multiplication kernel not yet implemented
    // This is only needed for shared_expert path which is disabled by default
    throw std::runtime_error("CompiledExecutor: element-wise mul operation not yet implemented. "
                             "Set use_shared_expert=False in your model config.");
}

void CompiledExecutor::dispatch_mul_backward(const CompiledOp& op) {
    // Element-wise multiplication backward kernel not yet implemented
    // This is only needed for shared_expert path which is disabled by default
    throw std::runtime_error("CompiledExecutor: element-wise mul_backward operation not yet implemented. "
                             "Set use_shared_expert=False in your model config.");
}

}  // namespace dsl
