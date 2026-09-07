#pragma once

// sinfer::ops::detail - private launch prototype for logit_softcap.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void logit_softcap_launch(Tensor& x, float cap, cudaStream_t stream);

} // namespace sinfer::ops::detail
