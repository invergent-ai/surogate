#pragma once

// sinfer::ops::detail - private launch prototype for rmsnorm.

#include "api/ops/gated_rmsnorm.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, GatedRmsGate gate, Tensor& out, cudaStream_t stream);
void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream);
void rmsnorm_add_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                        Tensor& out, cudaStream_t stream);

} // namespace sinfer::ops::detail
