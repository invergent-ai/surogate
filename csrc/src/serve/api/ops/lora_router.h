#pragma once
#include "api/ops/lora.h"
#include "api/ops/sparse_moe.h"

namespace sinfer::ops {
// Saved correction biases affect selection, while path weights use the original scores.
void lora_router_bias(const Tensor& scores, const Tensor& ids, const Tensor& alpha,
                      const float* base_bias, const float* expert_scale,
                      const SparseMoeGeometry& geometry, const LoraBank* banks,
                      const std::int32_t* slots, int slot_stride, cudaStream_t stream);
} // namespace sinfer::ops
