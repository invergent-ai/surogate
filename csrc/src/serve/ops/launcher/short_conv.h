#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void short_conv_launch(const Tensor& bcx, const Tensor& taps, Tensor& state, Tensor& out,
                       std::int32_t channels, const Tensor& valid_columns, cudaStream_t stream);

void short_conv_snapshot_launch(const Tensor& bcx, const Tensor& taps, Tensor& conv_states,
                                const Tensor& initial_state_slots,
                                const Tensor& snapshot_base_slots, const Tensor& valid_columns,
                                Tensor& out, std::int32_t channels, cudaStream_t stream);

} // namespace sinfer::ops::detail
