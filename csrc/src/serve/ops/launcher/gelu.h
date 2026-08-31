#pragma once

#include "core/tensor.h"
#include "api/ops/gelu.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace sinfer::ops::detail
