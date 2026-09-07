#pragma once

#include "api/ops/sampling.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace sinfer::ops::detail {

void sampled_logprob_launch(const Tensor& logits, const Tensor& tokens, Tensor& out,
                            std::int32_t token_domain, const SamplingConfig* configs,
                            cudaStream_t stream);

} // namespace sinfer::ops::detail
