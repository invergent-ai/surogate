#pragma once

#include "api/ops/lora.h"
#include "core/tensor.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace sinfer::ops::detail {

/// out_p[n_p, T] += B_p[id[t]] . (A_p[id[t]] . x[:, t]) for each pair p, in one
/// launch. All pairs read the same x, so k is shared.
void lora_fused_delta_launch(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                             std::int32_t pair_count, const Tensor& ids,
                             const std::int32_t* uniform, cudaStream_t stream);

/// The two-launch flavor of the same site: shrink once into `low` scratch, then
/// expand -- no per-block redundancy, for the geometries where the one-launch
/// kernel's recomputation would exceed its saved launch.
void lora_split_delta_launch(const Tensor& x, const LoraBank* const* banks, Tensor* const* outs,
                             std::int32_t pair_count, const Tensor& ids,
                             const std::int32_t* uniform, Tensor& low, cudaStream_t stream);

} // namespace sinfer::ops::detail
