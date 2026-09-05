#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail::kimi_delta_net {

void launch_recurrent(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                      const Tensor& beta, float scale, bool normalize_qk,
                      const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                      cudaStream_t stream);

void launch_recurrent_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                               const Tensor& beta, float scale, bool normalize_qk,
                               Tensor& ssm_states, const Tensor& valid_columns,
                               const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, Tensor& out,
                               cudaStream_t stream);

} // namespace sinfer::ops::detail::kimi_delta_net
