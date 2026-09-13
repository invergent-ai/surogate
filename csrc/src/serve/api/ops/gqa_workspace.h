#pragma once
#include "api/ops/gqa_attention.h"

namespace sinfer::ops {
// Reserve for runtime calls that use different visible-key envelopes within this
// history interval (prompt chunks and decode rounds). Unlike the fixed-envelope
// query, this also covers shorter histories that admit more prompt query tiles.
[[nodiscard]] std::size_t gqa_attention_history_workspace_capacity_bytes(
    std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads, DType cache_dtype,
    GqaExecutionEnvelope envelope, std::int32_t batch_size,
    std::int32_t min_width, std::int32_t max_width);
} // namespace sinfer::ops
