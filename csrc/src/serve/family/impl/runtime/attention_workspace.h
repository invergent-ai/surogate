#pragma once

#include "api/family/text_geometry.h"
#include "api/ops/gqa_workspace.h"

#include <algorithm>

namespace sinfer::family::detail {

inline std::size_t attention_workspace_capacity_bytes(
    const TextGeometry& geometry, DType cache_dtype, ops::GqaExecutionEnvelope envelope,
    std::int32_t batch, std::int32_t min_width, std::int32_t max_width) {
    auto capacity = ops::gqa_attention_history_workspace_capacity_bytes(
        geometry.head_dim, geometry.query_heads, geometry.kv_heads, cache_dtype,
        envelope, batch, min_width, max_width);
    if (geometry.has_global_attention_geometry()) {
        capacity = std::max(capacity, ops::gqa_attention_history_workspace_capacity_bytes(
            geometry.global_head_dim, geometry.query_heads, geometry.global_kv_heads, cache_dtype,
            envelope, batch, min_width, max_width));
    }
    return capacity;
}

} // namespace sinfer::family::detail
