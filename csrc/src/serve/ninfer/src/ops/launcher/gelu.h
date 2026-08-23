#pragma once

#include "core/tensor.h"
#include "ninfer/ops/gelu.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gelu_launch(Tensor& x, GeluMode mode, cudaStream_t stream);

} // namespace ninfer::ops::detail
