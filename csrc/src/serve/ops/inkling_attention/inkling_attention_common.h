#pragma once

// Host-side validation shared by the Inkling attention entry points (ENGINE-INKLING lane c).

#include "api/ops/inkling_attention.h"

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>

namespace sinfer::ops::inkling_detail {

struct RowsShape {
    std::int32_t width   = 0;
    std::int32_t batch   = 0;
    std::int64_t columns = 0;
    bool masked          = false;
};

[[noreturn]] inline void fail(const char* op, const std::string& what) {
    throw std::invalid_argument(std::string(op) + ": " + what);
}

inline void require_contiguous(const Tensor& t, DType dtype, std::int64_t elements, const char* op,
                               const char* name) {
    if (t.data == nullptr || t.dtype != dtype || !t.is_contiguous() || t.numel() != elements ||
        reinterpret_cast<std::uintptr_t>(t.data) % 16 != 0) {
        fail(op, std::string(name) + " must be a contiguous, 16-byte aligned tensor of " +
                     std::to_string(elements) + " elements of the documented dtype");
    }
}

/// A [inner..., N] view whose inner dims are contiguous and whose column stride is any multiple of
/// `align` bytes (base aligned too). Returns the column stride in elements.
inline std::int64_t column_stride_elements(const Tensor& t, DType dtype,
                                           std::initializer_list<std::int32_t> inner,
                                           std::int64_t columns, int inner_dims, const char* op,
                                           const char* name, int align = 8) {
    const std::int64_t elem = dtype == DType::FP32 ? 4 : 2;
    bool ok                 = t.data != nullptr && t.dtype == dtype &&
              reinterpret_cast<std::uintptr_t>(t.data) % static_cast<std::uintptr_t>(align) == 0;
    std::int64_t expected = elem;
    int d                 = 0;
    for (const std::int32_t extent : inner) {
        ok = ok && t.ne[d] == extent && t.nb[d] == expected;
        expected *= extent;
        ++d;
    }
    ok = ok && d == inner_dims && t.ne[d] == columns && t.nb[d] >= expected && t.nb[d] % align == 0;
    for (int rest = d + 1; rest < 4; ++rest) { ok = ok && t.ne[rest] == 1; }
    if (!ok) {
        fail(op, std::string(name) + ": wrong dtype, shape or strides (inner dims contiguous, " +
                     std::to_string(columns) + " columns, " + std::to_string(align) +
                     "-byte aligned column stride)");
    }
    return t.nb[d] / elem;
}

inline RowsShape check_rows(const InklingAttentionRows& rows, const char* op) {
    RowsShape s;
    const Tensor& pos = rows.positions;
    if (pos.data == nullptr || pos.dtype != DType::I32 || !pos.is_contiguous() || pos.ne[0] <= 0 ||
        pos.ne[1] <= 0 || pos.ne[2] != 1 || pos.ne[3] != 1) {
        fail(op, "positions must be contiguous I32 [W, B]");
    }
    s.width   = pos.ne[0];
    s.batch   = pos.ne[1];
    s.columns = static_cast<std::int64_t>(s.width) * s.batch;
    s.masked  = rows.valid_columns.data != nullptr;
    if (s.masked && (rows.valid_columns.dtype != DType::I32 || !rows.valid_columns.is_contiguous() ||
                     rows.valid_columns.numel() != s.batch)) {
        fail(op, "valid_columns must be empty or contiguous I32 [B]");
    }
    const Tensor& tr = rows.kv_table_rows;
    if (tr.data == nullptr || tr.dtype != DType::I32 || !tr.is_contiguous() || tr.numel() != s.batch) {
        fail(op, "kv_table_rows must be contiguous I32 [B]");
    }
    if (rows.envelope.max_visible_keys > static_cast<std::uint32_t>(1 << 20)) {
        fail(op, "the envelope's max_visible_keys must lie in [0, 1048576]");
    }
    return s;
}

inline void check_kv_cache(const PagedKVBatchLayerView& cache, const char* op) {
    const auto fp8 = [](const Tensor& t) {
        return t.data != nullptr && (t.dtype == DType::FP8_E4M3FN || t.dtype == DType::U8) &&
               reinterpret_cast<std::uintptr_t>(t.data) % 16 == 0;
    };
    if (!fp8(cache.k_pages) || !fp8(cache.v_pages) || cache.dtype != DType::FP8_E4M3FN ||
        cache.head_dim != InklingAttentionGeometry::kHeadDim ||
        cache.num_kv_heads != InklingAttentionGeometry::kKvHeads ||
        cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
        fail(op, "the plane must be unit-scale FP8 e4m3 K and V pages, head dim 128, 8 KV heads, no scales");
    }
    const Tensor& bt = cache.block_tables;
    if (bt.data == nullptr || bt.dtype != DType::I32 || !bt.is_contiguous() || bt.ne[0] <= 0) {
        fail(op, "block_tables must be contiguous I32 [logical_pages, rows]");
    }
}

/// inkling_attention_prologue with one addition: after its programmatic-dependency wait, CTA 0 zeroes `zero_count` ints at
/// `zero` (null: none) -- the fused call's split-route arrival counters, so the attention launched next finds them zero
/// without a memset launch (the counters live in the call's workspace, whose contents are otherwise undefined).
void prologue_launch(const InklingProjections& projections, const InklingAttentionWeights& weights,
                     InklingLayerKind kind, const InklingAttentionRows& rows, const InklingConvStateArgs& conv,
                     const PagedKVBatchLayerView& cache, InklingPrologueOutputs& outputs, cudaStream_t stream,
                     std::int32_t* zero, std::int32_t zero_count);

} // namespace sinfer::ops::inkling_detail
