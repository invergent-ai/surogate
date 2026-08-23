#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void cast_fp32_to_bf16_launch(const Tensor& source, Tensor& destination, cudaStream_t stream);

} // namespace ninfer::ops::detail
