// sinfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "core/limits.h"
#include "api/ops/batch_invariant.h"
#include "api/ops/gqa_attention.h"
#include "api/ops/gqa_workspace.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gqa_sm90/gqa_fa3.h"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::ops {
namespace {

constexpr std::int32_t kQuantGroup                   = 64;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::int32_t kMaximumBatchSize             = kMaximumBatchColumns;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

// Optimized schedules cover selected query counts; the fallback uses the cache's
// supported head width and KV count with any positive integral query group.
bool supported_attention_shape(std::int32_t dim, std::int32_t queries, std::int32_t kv) {
    return queries > 0 && kv > 0 && queries % kv == 0 &&
           gqa_kv_shape_is_registered(dim, kv);
}

bool optimized_decode_shape(std::int32_t dim, std::int32_t queries, std::int32_t kv, DType dtype) {
    // Keep the new 8q4 shape's existing INT8 support through the prompt kernel:
    // the INT8 decode schedule cannot distribute a query group of two.
    return gqa_shape_is_registered(dim, queries, kv) &&
           !(dtype == DType::I8 && dim == 256 && queries == 8 && kv == 4);
}

// Resolves the served shape against the geometry registry. A query count can be registered
// against more than one KV count (16 queries over 2 or 4; 24 over 4 or 2), which is why the
// caller's KV source picks between them; an unregistered shape throws naming all three numbers.
std::int32_t kv_heads_for_pair(std::int32_t head_dim, std::int32_t q_heads,
                               std::int32_t source_kv_heads) {
    if (!supported_attention_shape(head_dim, q_heads, source_kv_heads)) {
        throw std::invalid_argument("gqa_attention: invalid query/KV head geometry");
    }
    return source_kv_heads;
}

// The head dimension belongs to the shape and comes off the query tensor, which is the only
// place it is stated. It used to be *derived* from the head counts, which held only while no
// two registered shapes shared a pair -- TinyLlama attends with 32 query heads over 4 KV heads
// at width 64 and Qwen3-30B-A3B does the same at 128, so the derivation would have handed one
// of them the other's width and validated every tensor against it.
void require_registered_shape(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                              const char* op) {
    if (!supported_attention_shape(head_dim, q_heads, kv_heads)) {
        throw std::invalid_argument(std::string(op) + ": unregistered head geometry (head dim " +
                                    std::to_string(head_dim) + ", " + std::to_string(q_heads) +
                                    " query heads over " + std::to_string(kv_heads) +
                                    " KV heads)");
    }
}

// The append path holds no query count; the registry answers on the head
// dimension and KV count alone, which is all its kernels read.
void require_kv_heads(std::int32_t head_dim, std::int32_t kv_heads, const char* op) {
    if (!gqa_kv_shape_is_registered(head_dim, kv_heads)) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry (head dim " +
                                    std::to_string(head_dim) + ", " + std::to_string(kv_heads) +
                                    " KV heads)");
    }
}

// The softmax scale, which the kernels do read from the caller: the decode kernels multiply
// each score by it and the prefill kernels fold it into their exp2 as `scale * Log2E`. This
// check used to demand exactly `1/sqrt(head_dim)` on the grounds that the kernels ignored the
// argument, and that had stopped being true.
//
// It is not merely a widening. `1/sqrt(head_dim)` is the convention, not a law: Gemma 4 sets
// its scale to **1.0** and relies on its query/key norm to deliver unit-RMS operands, so a
// check that insisted on the convention refused a model the kernels can serve exactly. What
// remains is what the kernels actually require -- a finite, positive number -- because a
// zero or a NaN here is a caller bug that would otherwise surface as a uniform or an empty
// attention distribution rather than as an error.
void require_scale(float scale, std::int32_t head_dim, const char* op) {
    (void)head_dim;
    if (!std::isfinite(scale) || scale <= 0.0f) {
        throw std::invalid_argument(std::string(op) +
                                    ": softmax scale must be finite and positive, not " +
                                    std::to_string(scale));
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    const std::int32_t head_dim = cache.head_dim;
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8 &&
         cache.dtype != DType::FP8_E4M3FN) ||
        cache.num_kv_heads != kv_heads || !gqa_kv_shape_is_registered(head_dim, kv_heads)) {
        throw std::invalid_argument(std::string(op) +
                                    ": invalid KV cache geometry or dtype (head dim " +
                                    std::to_string(head_dim) + ", " +
                                    std::to_string(cache.num_kv_heads) + " KV heads, expected " +
                                    std::to_string(kv_heads) + ")");
    }
    if (cache.dtype != DType::I8 && cache.quant_group != 0) {
        // e4m3 codes carry no scale plane, so like BF16 they must not name a group.
        throw std::invalid_argument(std::string(op) +
                                    ": unquantized or e4m3 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype = cache.dtype;
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, head_dim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, head_dim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    // Only the grouped int8 cache carries scale planes. BF16 stores values
    // directly and e4m3 stores self-describing codes, so both must arrive
    // without them.
    if (cache.dtype != DType::I8) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) +
                                        ": unquantized or e4m3 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    const std::int32_t groups = head_dim / kQuantGroup;
    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(cache.v_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    const std::int32_t head_dim = cache.head_dim;
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8 &&
         cache.dtype != DType::FP8_E4M3FN) ||
        cache.num_kv_heads != kv_heads || !gqa_kv_shape_is_registered(head_dim, kv_heads)) {
        throw std::invalid_argument(std::string(op) +
                                    ": invalid KV cache geometry or dtype (head dim " +
                                    std::to_string(head_dim) + ", " +
                                    std::to_string(cache.num_kv_heads) + " KV heads, expected " +
                                    std::to_string(kv_heads) + ")");
    }
    if (cache.dtype != DType::I8 && cache.quant_group != 0) {
        // e4m3 codes carry no scale plane, so like BF16 they must not name a group.
        throw std::invalid_argument(std::string(op) +
                                    ": unquantized or e4m3 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype = cache.dtype;
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, head_dim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, head_dim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    // Only the grouped int8 cache carries scale planes. BF16 stores values
    // directly and e4m3 stores self-describing codes, so both must arrive
    // without them.
    if (cache.dtype != DType::I8) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) +
                                        ": unquantized or e4m3 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    const std::int32_t groups = head_dim / kQuantGroup;
    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(cache.v_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(GqaExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
    // Zero means unbounded; a negative window is not a window, and the kernels'
    // predicate would read it as unbounded too, which is the wrong answer that
    // looks right. Refuse it here, where the envelope is first seen.
    if (envelope.sliding_window < 0) {
        throw std::invalid_argument(std::string(op) + ": negative sliding window");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                float scale, const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t kv_heads = kv_heads_for_pair(head_dim, q_heads, cache.num_kv_heads);
    require_registered_shape(head_dim, q_heads, kv_heads, op);
    require_scale(scale, head_dim, op);
    const std::int32_t tokens = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, head_dim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, head_dim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        GqaExecutionEnvelope envelope, float scale,
                                        const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t kv_heads = kv_heads_for_pair(head_dim, q_heads, cache.num_kv_heads);
    require_registered_shape(head_dim, q_heads, kv_heads, op);
    require_scale(scale, head_dim, op);
    const std::int32_t width = q.ne[2];
    const std::int32_t batch = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, head_dim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, head_dim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t head_dim,
                                           std::int32_t q_heads, std::int32_t tokens,
                                           std::int32_t splits, std::int32_t batch_size = 1) {
    return {
        workspace.alloc(DType::FP32, {head_dim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
    };
}

// Query tiles use the same fixed key partitions and FP32 reducer as decode.
// Bound transient memory at long contexts instead of scaling it with prompt length.
int prompt_query_capacity(int dim, int heads, int kv_heads, DType dtype,
                           GqaExecutionEnvelope envelope) {
    const int splits = detail::gqa_attention_split_capacity(dim, heads, kv_heads, 32, dtype, envelope);
    const std::size_t per_query = static_cast<std::size_t>(dim + 2) * heads * splits * sizeof(float);
    const int count = static_cast<int>(std::clamp<std::size_t>((32U << 20) / per_query, 1, 128));
    int tile = 1;
    const int limit = std::min({32, 64 / (heads / kv_heads), count});
    while (tile * 2 <= limit) { tile *= 2; }
    return (count / tile) * tile;
}

int prompt_query_tile_width(int heads, int kv_heads, int queries) {
    const int limit = std::min({32, 64 / (heads / kv_heads), queries});
    int width = 1;
    while (width * 2 <= limit) { width *= 2; }
    return width;
}

template <typename Visit>
void for_each_prompt_tile(int heads, int kv_heads, int tokens, int capacity, Visit&& visit) {
    for (int begin = 0; begin < tokens;) {
        const int remaining = std::min(capacity, tokens - begin);
        const int width = prompt_query_tile_width(heads, kv_heads, remaining);
        const int lanes = remaining / width;
        visit(begin, width, lanes);
        begin += width * lanes;
    }
}

struct PromptTileWorkspace {
    Tensor valid;
    Tensor rows;
    SmallTWorkspace partial;
};

template <typename Allocator>
PromptTileWorkspace allocate_prompt_tile_workspace(Allocator& workspace, int dim, int heads,
                                                    int width, int splits, int lanes) {
    return {workspace.alloc(DType::I32, {lanes}), workspace.alloc(DType::I32, {lanes}),
            allocate_small_t_workspace(workspace, dim, heads, width, splits, lanes)};
}

// FlashAttention-3 (ops/gqa_sm90) takes a prompt segment on Hopper when it is wide enough that
// FA3's 128-row tiles beat the split-KV tiles below, at head dim 64, 128 or 256 over a BF16 or
// e4m3 cache, causal or under a sliding window (FA3's local mask), as vLLM runs it (over e4m3,
// FA3's FP8 kernel on e4m3 queries). At an 8K history on an H100 (head dim 256), FA3 took 0.18 ms
// for 32 and for 64 columns (one CTA per 128 packed rows walks every key); the tiles' ~22 TFLOP/s
// puts them near 0.2 and 0.4 ms. It asks the current device, so workspace planning, like the
// launch, runs with the serving device current.
//
// FA3's arithmetic is not the decode kernels': a query's bits would then depend on whether it was
// decoded, verified or prefilled with others. --batch-invariant (api/ops/batch_invariant.h) keeps
// every width on the split-KV kernels, whose fixed key partitions make them width-invariant.
bool fa3_takes_prompt(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                      DType cache_dtype, std::int32_t width) {
    return !batch_invariant() &&
           (cache_dtype == DType::BF16 || cache_dtype == DType::FP8_E4M3FN) &&
           width >= detail::gqa_fa3::min_columns() &&
           detail::gqa_fa3::supports(head_dim, q_heads, kv_heads) &&
           detail::gqa_fa3::available();
}

struct Fa3PromptWorkspace {
    Tensor metadata;
    DeviceSpan scratch;
};

template <typename Allocator>
Fa3PromptWorkspace allocate_fa3_prompt_workspace(Allocator& workspace, std::int32_t head_dim,
                                                 std::int32_t q_heads, std::int32_t tokens,
                                                 std::int32_t segments, DType cache_dtype) {
    Tensor metadata = workspace.alloc(DType::I32, {detail::gqa_fa3::metadata_ints(segments)});
    return {metadata, workspace.alloc_bytes(detail::gqa_fa3::workspace_bytes(
                          head_dim, q_heads, tokens, segments, cache_dtype == DType::FP8_E4M3FN))};
}

detail::gqa_fa3::PagedLaunch fa3_launch(const Tensor& q, const PagedKVBatchLayerView& cache,
                                         float scale, std::int32_t sliding_window, Tensor& out,
                                         std::int32_t segments, std::int32_t max_q,
                                         const Tensor& metadata) {
    auto* m = static_cast<std::int32_t*>(metadata.data);
    detail::gqa_fa3::PagedLaunch args;
    args.head_dim       = q.ne[0];
    args.q              = q.data;
    args.out            = out.data;
    args.k_pages        = cache.k_pages.data;
    args.v_pages        = cache.v_pages.data;
    args.fp8_cache      = cache.dtype == DType::FP8_E4M3FN;
    args.block_tables   = static_cast<const std::int32_t*>(cache.block_tables.data);
    args.logical_pages  = cache.block_tables.ne[0];
    args.table_rows     = cache.block_tables.ne[1];
    args.physical_pages = cache.k_pages.ne[3];
    args.q_heads        = q.ne[1];
    args.kv_heads       = cache.num_kv_heads;
    args.segments       = segments;
    args.total_q        = q.ne[2] * q.ne[3];
    args.max_q          = max_q;
    // [offsets(segments + 1) | lengths | kv lengths | rows], as gqa_fa3::metadata_ints lays out.
    args.q_offsets  = m;
    args.q_lengths  = m + segments + 1;
    args.kv_lengths = m + 2 * segments + 1;
    args.kv_rows    = m + 3 * segments + 1;
    args.scale      = scale;
    args.sliding_window = sliding_window;
    return args;
}

// One sequence's prompt through FA3: every column is a query, the last ones of its keys, and the
// lengths are derived on the stream from the device positions, so a captured prefill graph
// replays with new ones.
void launch_fa3_prompt(const Tensor& q, const Tensor& positions, const Tensor& parent_row,
                       float scale, std::int32_t sliding_window, const PagedKVBatchLayerView& cache,
                       WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    auto scope              = workspace.scope();
    const std::int32_t tokens = q.ne[2];
    auto [metadata, scratch] = allocate_fa3_prompt_workspace(workspace, q.ne[0], q.ne[1], tokens,
                                                             1, cache.dtype);
    detail::gqa_fa3::prompt_metadata(static_cast<const std::int32_t*>(positions.data), tokens,
                                     static_cast<const std::int32_t*>(parent_row.data),
                                     static_cast<std::int32_t*>(metadata.data), stream);
    detail::gqa_fa3::run(fa3_launch(q, cache, scale, sliding_window, out, 1, tokens, metadata),
                         scratch.data, scratch.bytes, stream);
}

// A geometry with a registered head dim and KV count but no tuned kernels of its own (its query
// count unregistered, as Qwen3-8B's 32 over 8) runs every prompt through the generic kernel, which
// on an H100 took 3.6 ms per layer for a 2,048-token Qwen3-8B prompt. FA3 serves any query group,
// so on Hopper such a prompt takes it under the same conditions as a registered one
// (fa3_takes_prompt): one sequence, every column live, no image block or QSA selection.
bool fa3_takes_generic_prompt(const Tensor& q, const Tensor& valid_columns,
                              const PagedKVBatchLayerView& cache, const GqaBlockMask& selection) {
    return !selection.image_end && !selection.words && q.ne[3] == 1 &&
           valid_columns.data == nullptr &&
           fa3_takes_prompt(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, q.ne[2]);
}

// Decode and verify rows over a BF16 cache take FA3 on Hopper too, one segment a row, as vLLM
// decodes there, for registered geometries and the rest alike. FA3 packs the query group into its
// M tile, so a KV head's keys are read once for all of its query heads; the generic route reads
// them once per query head (Qwen3-8B, 64 rows of ~600 keys on an H100: 157 us a layer, vLLM's
// FA3 ~65). Long histories split over several CTAs as FA3's own scheduler decides on the device,
// with row_splits bounding it from the graph's envelope. An e4m3 cache keeps its rows on the
// split-KV kernels: FA3's FP8 kernel accumulates P.V on FP8 wgmma, whose accumulator keeps far
// fewer bits than FP32, so a long history loses most of its later keys (on an H100, uniform
// attention over 262K keys came out a third low, over 1M keys a sixth of the truth); the tiles
// read the same codes with BF16 queries and FP32 sums. SUROGATE_SERVE_GQA_FA3_ROWS=0 keeps rows
// on the split-KV kernels, as --batch-invariant does (FA3's arithmetic is not theirs).
bool fa3_takes_rows(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                    DType cache_dtype, std::int32_t width, std::int32_t batch,
                    const GqaBlockMask& selection) {
    return !selection.words && !selection.image_end && width <= kMaximumVerifyTokens &&
           !(batch == 1 && width >= detail::gqa_fa3::min_columns()) && !batch_invariant() &&
           cache_dtype == DType::BF16 && detail::gqa_fa3::rows_enabled() &&
           detail::gqa_fa3::supports(head_dim, q_heads, kv_heads) &&
           detail::gqa_fa3::available();
}

detail::gqa_fa3::RowSplits fa3_row_splits(std::int32_t head_dim, std::int32_t q_heads,
                                          std::int32_t kv_heads, std::int32_t width,
                                          std::int32_t batch, GqaExecutionEnvelope envelope) {
    return detail::gqa_fa3::row_splits(head_dim, q_heads, kv_heads, width * batch, width,
                                       static_cast<std::int32_t>(envelope.max_visible_keys),
                                       /*fp8_cache=*/false, envelope.sliding_window);
}

// The rows' scratch, the same for a round as for the plan of its width and batch: the partials'
// reservation grows with the batch whatever split count a round gets (workspace_bytes).
template <typename Allocator>
Fa3PromptWorkspace allocate_fa3_rows_workspace(Allocator& workspace, std::int32_t head_dim,
                                               std::int32_t q_heads, std::int32_t width,
                                               std::int32_t batch,
                                               detail::gqa_fa3::RowSplits splits) {
    Tensor metadata = workspace.alloc(DType::I32, {detail::gqa_fa3::metadata_ints(batch)});
    return {metadata, workspace.alloc_bytes(detail::gqa_fa3::workspace_bytes(
                          head_dim, q_heads, width * batch, batch, /*fp8_cache=*/false,
                          splits.partials))};
}

bool round_metadata_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("SUROGATE_SERVE_GQA_ROUND_METADATA");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

// One input set's part of a GqaRoundMetadata's storage (GqaRoundMetadata::entry_ints): the rows'
// segment arrays, FA3's scheduler metadata over them, and a tile counter per layer that launches
// on them.
std::int32_t round_metadata_offset(std::int32_t rows) {
    return (detail::gqa_fa3::metadata_ints(rows) + 3) / 4 * 4;
}

std::int32_t round_entry_ints(std::int32_t rows) {
    return round_metadata_offset(rows) + detail::gqa_fa3::scheduler_ints(rows) +
           GqaRoundMetadata::kLaunches;
}

// A rows launch's share of its round: where its segment arrays, scheduler metadata and tile
// counter are, and whether it computes the arrays and metadata (the round's first launch over
// these inputs, which also zeroes every layer's counter) or reads them. Empty without a round,
// with the switch off, or when the round has no room for these inputs: the launch then computes
// its own in its scratch, as it always did.
struct RoundShare {
    std::int32_t* metadata  = nullptr;
    std::int32_t* scheduler = nullptr;
    std::int32_t* counters  = nullptr;
    std::int32_t* counter   = nullptr;
    bool compute            = false;
};

RoundShare round_share(GqaRoundMetadata* round, const GqaRoundMetadata::Entry& key) {
    if (round == nullptr || !round_metadata_enabled() || round->storage().data == nullptr ||
        round->rows() <= 0 || key.batch > round->rows() ||
        round_entry_ints(round->rows()) != GqaRoundMetadata::entry_ints(round->rows()) ||
        round->storage().numel() < GqaRoundMetadata::storage_ints(round->rows())) {
        return {};
    }
    const auto same = [&](const GqaRoundMetadata::Entry& e) {
        return e.positions == key.positions && e.valid_columns == key.valid_columns &&
               e.kv_table_rows == key.kv_table_rows && e.width == key.width &&
               e.batch == key.batch && e.head_dim == key.head_dim && e.q_heads == key.q_heads &&
               e.kv_heads == key.kv_heads && e.sliding_window == key.sliding_window &&
               e.splits == key.splits;
    };
    std::int32_t index = 0;
    while (index < round->used() && !same(round->entry(index))) { ++index; }
    if (index == round->used()) {
        if (index == GqaRoundMetadata::kEntries) { return {}; }
        round->add(key);
    }
    GqaRoundMetadata::Entry& entry = round->entry(index);
    if (entry.launches == GqaRoundMetadata::kLaunches) { return {}; }
    auto* base = static_cast<std::int32_t*>(round->storage().data) +
                 static_cast<std::size_t>(index) * round_entry_ints(round->rows());
    RoundShare share;
    share.metadata  = base;
    share.scheduler = base + round_metadata_offset(round->rows());
    share.counters  = share.scheduler + detail::gqa_fa3::scheduler_ints(round->rows());
    share.counter   = share.counters + entry.launches;
    share.compute   = entry.launches == 0;
    ++entry.launches;
    return share;
}

// The rows' keys must already be in the cache: row r is segment r, its valid columns the last of
// its keys. With a round, the round's first launch over these inputs derives the segment arrays
// and FA3's scheduler metadata into the round's storage and the rest of its layers reuse them.
void launch_fa3_rows(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                     const Tensor& kv_table_rows, float scale, GqaExecutionEnvelope envelope,
                     const PagedKVBatchLayerView& cache, WorkspaceArena& workspace, Tensor& out,
                     cudaStream_t stream, GqaRoundMetadata* round = nullptr) {
    auto scope               = workspace.scope();
    const std::int32_t width = q.ne[2];
    const std::int32_t batch = q.ne[3];
    const detail::gqa_fa3::RowSplits splits =
        fa3_row_splits(q.ne[0], q.ne[1], cache.num_kv_heads, width, batch, envelope);
    auto [metadata, scratch] =
        allocate_fa3_rows_workspace(workspace, q.ne[0], q.ne[1], width, batch, splits);
    const RoundShare share = round_share(
        round, {positions.data, valid_columns.data, kv_table_rows.data, width, batch, q.ne[0],
                q.ne[1], cache.num_kv_heads, envelope.sliding_window, splits.splits});
    if (share.metadata != nullptr) {
        metadata = Tensor(share.metadata, DType::I32, {detail::gqa_fa3::metadata_ints(batch)});
    }
    const bool prepare = share.metadata == nullptr || share.compute;
    if (prepare) {
        detail::gqa_fa3::rows_metadata(
            static_cast<const std::int32_t*>(positions.data), width, batch,
            static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<const std::int32_t*>(kv_table_rows.data),
            static_cast<std::int32_t*>(metadata.data), stream, share.counters,
            share.counters != nullptr ? GqaRoundMetadata::kLaunches : 0);
    }
    // FA3 writes only the valid columns; the split-KV kernels' contract is zeros in the rest.
    if (valid_columns.data != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(out.data, 0, out.bytes(), stream));
    }
    detail::gqa_fa3::PagedLaunch args =
        fa3_launch(q, cache, scale, envelope.sliding_window, out, batch, width, metadata);
    args.max_splits   = splits.splits;
    args.scheduler    = share.scheduler;
    args.prepare      = prepare;
    args.tile_counter = share.counter;
    detail::gqa_fa3::run(args, scratch.data, scratch.bytes, stream);
}

// Every other width of such a geometry -- decode and verify rows on any GPU, and prompts where
// FA3 is not available -- takes the generic split-KV route (gqa_attention_generic.cu), save for
// the masks only the prompt kernel applies: a QSA selection and an image's bidirectional block.
bool generic_takes(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                   DType cache_dtype, const GqaBlockMask& selection) {
    return !selection.words && !selection.image_end &&
           !gqa_shape_is_registered(head_dim, q_heads, kv_heads) &&
           detail::gqa_generic_attention_serves(head_dim, cache_dtype);
}

// How many CTAs share a query's history on the generic route. A width FA3 takes for one sequence
// is never split: the plan books FA3's scratch for it, and a masked sequence of that width that
// lands here instead must not need more.
int generic_splits(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                   DType cache_dtype, std::int32_t batch, std::int32_t width) {
    if (batch == 1 && fa3_takes_prompt(head_dim, q_heads, kv_heads, cache_dtype, width)) {
        return 1;
    }
    return detail::gqa_generic_attention_splits(q_heads, batch * width,
                                                kGqaAttentionMaximumVisibleKeys);
}

DeviceSpan allocate_generic_workspace(WorkspaceArena& workspace, std::int32_t head_dim,
                                      std::int32_t q_heads, std::int32_t columns, int splits) {
    const std::size_t bytes =
        detail::gqa_generic_attention_workspace_bytes(head_dim, q_heads, columns, splits);
    return bytes == 0 ? DeviceSpan{} : workspace.alloc_bytes(bytes);
}

std::int32_t prompt_kernel_min_columns() {
    static const std::int32_t value = [] {
        const char* text = std::getenv("SUROGATE_SERVE_GQA_PROMPT_KERNEL_MIN_COLUMNS");
        if (text == nullptr || *text == '\0') { return 64; }
        char* end         = nullptr;
        const long parsed = std::strtol(text, &end, 10);
        return (end != nullptr && *end == '\0' && parsed >= 0 && parsed <= 1 << 20)
                   ? static_cast<std::int32_t>(parsed) : 64;
    }();
    return value;
}

// The tensor-core prompt kernel (gqa_attention_prefill_bf16.cuh, FlashAttention-2's forward: one
// CTA per 64 query rows of a query head, walking the history 32 keys at a time on mma.sync) takes
// a prompt segment FA3 does not, which is every prompt off Hopper, from
// SUROGATE_SERVE_GQA_PROMPT_KERNEL_MIN_COLUMNS columns on (default 64; 0 keeps prompts on the
// routes it replaces). Those are the split-KV tiles for a registered geometry, which walk each
// tile of a few queries through the decode kernels' fixed key partitions, FP32 partials and
// reducer (in 12 s of eight users sending 2,048-token prompts to Qwen3.6-35B-A3B on a DGX Spark,
// they ran 1.5 s where vLLM's FlashAttention-2 ran 0.34 s), and the generic CUDA-core kernel for
// a query group the registry does not carry (Qwen3-8B's 32 over 8: 60% of the GPU time in the
// same load on the Spark). Its arithmetic is not the split-KV kernels', so --batch-invariant
// keeps every width on those; a 512-wide head keeps them too, its 16-key variant being
// unmeasured against them.
bool prompt_kernel_takes(std::int32_t head_dim, std::int32_t q_heads, std::int32_t kv_heads,
                         DType cache_dtype, std::int32_t width) {
    const std::int32_t min_columns = prompt_kernel_min_columns();
    return min_columns > 0 && width >= min_columns && head_dim <= 256 && !batch_invariant() &&
           (cache_dtype == DType::BF16 || cache_dtype == DType::FP8_E4M3FN) &&
           !fa3_takes_prompt(head_dim, q_heads, kv_heads, cache_dtype, width);
}

// One sequence's prompt of a geometry the generic route would take (generic_takes), wide enough
// for the prompt kernel.
bool prompt_kernel_takes_generic(const Tensor& q, const PagedKVBatchLayerView& cache,
                                 const GqaBlockMask& selection) {
    return q.ne[3] == 1 &&
           generic_takes(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, selection) &&
           prompt_kernel_takes(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, q.ne[2]);
}

void launch_cached_prompt_tiles(const Tensor& q, const Tensor& positions,
                                 const Tensor& parent_valid, const Tensor& parent_row,
                                 float scale, PagedKVBatchLayerView cache,
                                 GqaExecutionEnvelope envelope, WorkspaceArena& workspace,
                                 Tensor& out, cudaStream_t stream) {
    if (parent_valid.data == nullptr &&
        fa3_takes_prompt(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, q.ne[2])) {
        launch_fa3_prompt(q, positions, parent_row, scale, envelope.sliding_window, cache,
                          workspace, out, stream);
        return;
    }
    if (prompt_kernel_takes(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, q.ne[2])) {
        detail::gqa_attention_prompt_cached_launch(q, positions, parent_valid, parent_row, scale,
                                                   cache, out, stream, envelope.sliding_window);
        return;
    }
    // Public prompt invocations have one sequence; only the private launcher sees the
    // independent query tiles as lanes, all selecting that sequence's table row.
    const int capacity = prompt_query_capacity(q.ne[0], q.ne[1], cache.num_kv_heads,
                                               cache.dtype, envelope);
    for_each_prompt_tile(q.ne[1], cache.num_kv_heads, q.ne[2], capacity,
                          [&](int begin, int width, int lanes) {
        auto scope = workspace.scope();
        const int count = lanes * width;
        const int splits = detail::gqa_attention_split_capacity(q.ne[0], q.ne[1],
            cache.num_kv_heads, width, cache.dtype, envelope);
        auto [valid, rows, partial] = allocate_prompt_tile_workspace(workspace,
            q.ne[0], q.ne[1], width, splits, lanes);
        detail::gqa_query_tile_metadata(parent_valid, parent_row, q.ne[2], begin,
                                         width, valid, rows, stream);
        Tensor queries = q.slice(2, begin, count).view({q.ne[0], q.ne[1], width, lanes});
        Tensor pos = positions.slice(0, begin, count).view({width, lanes});
        Tensor result = out.slice(2, begin, count).view({q.ne[0], q.ne[1], width, lanes});
        detail::gqa_attention_cached_batch_small_t_launch(queries, pos, valid, rows, scale,
            cache, envelope, 0, width, partial.acc, partial.m, partial.l, result, stream);
    });
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            std::int32_t kv_heads, DType cache_dtype,
                            GqaExecutionEnvelope envelope, Tensor& out, Launch&& launch) {
    const std::int32_t step = detail::gqa_attention_small_t_max_width(q.ne[1], kv_heads);
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += step) {
        const std::int32_t count = std::min(step, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[0], q.ne[1], kv_heads, count,
                                                 cache_dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], count, splits);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                            cudaStream_t stream, GqaBlockMask selection = {}) {
    const std::int32_t step = detail::gqa_attention_small_t_max_width(q.ne[1], k.ne[1]);
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += step) {
        const std::int32_t count = std::min(step, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[0], q.ne[1], k.ne[1], count, cache.dtype,
                                                 envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], count, splits, q.ne[3]);
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, envelope, begin, count, partial.acc, partial.m,
                                             partial.l, out, stream, selection);
    }
}

void launch_cached_chunked_batch_small_t(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& table_rows,
                                        float scale, PagedKVBatchLayerView cache,
                                        GqaExecutionEnvelope envelope, WorkspaceArena& workspace,
                                        Tensor& out, cudaStream_t stream,
                                        GqaBlockMask selection = {}) {
    const std::int32_t width  = q.ne[2];
    const std::int32_t step   = detail::gqa_attention_small_t_max_width(q.ne[1], cache.num_kv_heads);
    const std::int32_t count  = std::min(width, step);
    const std::int32_t splits = detail::gqa_attention_split_capacity(
        q.ne[0], q.ne[1], cache.num_kv_heads, count, cache.dtype, envelope);
    SmallTWorkspace partial =
        allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], count, splits, q.ne[3]);
    for (std::int32_t begin = 0; begin < width; begin += count) {
        const std::int32_t chunk = std::min(count, width - begin);
        detail::gqa_attention_cached_batch_small_t_launch(
            q, positions, valid_columns, table_rows, scale, cache, envelope, begin, chunk,
            partial.acc, partial.m, partial.l, out, stream, selection);
    }
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                                   GqaBlockMask selection = {}) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.num_kv_heads, cache.dtype, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::gqa_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                        envelope, partial.acc, partial.m, partial.l,
                                                        out_chunk, stream, selection);
        });
}

} // namespace

namespace detail {

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t kv_heads,
                                              std::int32_t width, std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope) {
    const std::int32_t step = gqa_attention_small_t_max_width(q_heads, kv_heads);
    if (width >= 1 && width <= step) { return GqaAttentionRoute::SmallT; }
    if (batch_size > 1) { return GqaAttentionRoute::ChunkedSmallT; }
    const std::uint32_t prompt_visible_keys =
        width <= 2 * step ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    return GqaAttentionRoute::Prompt;
}

const char* gqa_attention_route_name(GqaAttentionRoute route) {
    switch (route) {
    case GqaAttentionRoute::SmallT:
        return "small_t";
    case GqaAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case GqaAttentionRoute::Prompt:
        return "prompt";
    }
    return "unknown";
}

} // namespace detail

std::size_t gqa_attention_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
                                                   std::int32_t kv_heads, DType cache_dtype,
                                                   GqaExecutionEnvelope envelope,
                                                   std::int32_t batch_size, std::int32_t min_width,
                                                   std::int32_t max_width) {
    if (!supported_attention_shape(head_dim, q_heads, kv_heads)) {
        throw std::invalid_argument("gqa_attention workspace: unsupported head geometry");
    }
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8 &&
         cache_dtype != DType::FP8_E4M3FN) ||
        batch_size <= 0 ||
        batch_size > kMaximumBatchSize || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_width)) {
        throw std::invalid_argument("gqa_attention workspace: invalid profile or interval");
    }

    // FA3's rows (fa3_takes_rows): their scratch grows with the batch, so the plan's batch
    // covers every smaller one. The other routes stay booked for the same widths: a round with
    // a QSA selection or an image block, or a lone sequence at a prompt width, still takes them.
    std::size_t rows_bytes = 0;
    for (std::int32_t width = min_width; width <= std::min(max_width, kMaximumVerifyTokens); ++width) {
        if (!fa3_takes_rows(head_dim, q_heads, kv_heads, cache_dtype, width, batch_size, {})) {
            continue;
        }
        WorkspaceLayoutBuilder layout;
        (void)allocate_fa3_rows_workspace(
            layout, head_dim, q_heads, width, batch_size,
            fa3_row_splits(head_dim, q_heads, kv_heads, width, batch_size, envelope));
        rows_bytes = std::max(rows_bytes, layout.peak_bytes(1));
    }

    if (!optimized_decode_shape(head_dim, q_heads, kv_heads, cache_dtype)) {
        // FA3 (fa3_takes_generic_prompt) needs its scratch, which grows with the width; the
        // generic split-KV route (generic_takes) its partials while it splits, which it does
        // only for a few columns.
        const auto fa3_width = [&](std::int32_t width) {
            return batch_size == 1 && fa3_takes_prompt(head_dim, q_heads, kv_heads, cache_dtype, width);
        };
        std::size_t bytes = 0;
        if (fa3_width(max_width)) {
            WorkspaceLayoutBuilder layout;
            (void)allocate_fa3_prompt_workspace(layout, head_dim, q_heads, max_width, 1, cache_dtype);
            bytes = layout.peak_bytes(1);
        }
        if (generic_takes(head_dim, q_heads, kv_heads, cache_dtype, {})) {
            // Splits never grow with the width, so the first unsplit width ends the walk, as
            // does the first one the prompt kernel takes (prompt_kernel_takes_generic), which
            // needs none and takes every wider one FA3 does not.
            for (std::int32_t width = min_width; width <= max_width; ++width) {
                if (batch_size == 1 &&
                    prompt_kernel_takes(head_dim, q_heads, kv_heads, cache_dtype, width)) {
                    break;
                }
                const std::int32_t columns = batch_size * width;
                const int splits =
                    generic_splits(head_dim, q_heads, kv_heads, cache_dtype, batch_size, width);
                if (splits <= 1) { break; }
                WorkspaceLayoutBuilder layout;
                (void)layout.alloc_bytes(detail::gqa_generic_attention_workspace_bytes(
                    head_dim, q_heads, columns, splits));
                bytes = std::max(bytes, layout.peak_bytes(1));
            }
        }
        return std::max(bytes, rows_bytes);
    }

    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(head_dim, q_heads, kv_heads, width,
                                                 cache_dtype, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, head_dim, q_heads, width, splits, batch_size);
        return layout.peak_bytes(1);
    };
    const int prompt_capacity = cache_dtype == DType::I8 ? 0 :
        prompt_query_capacity(head_dim, q_heads, kv_heads, cache_dtype, envelope);
    const auto exact_prompt_capacity = [&](int width) {
        WorkspaceLayoutBuilder layout;
        // Complete stripes repeat the same allocation; visit one plus the tail.
        const int representative = width > prompt_capacity
            ? prompt_capacity + width % prompt_capacity : width;
        for_each_prompt_tile(q_heads, kv_heads, representative, prompt_capacity,
                              [&](int, int tile, int lanes) {
            auto scope = layout.scope();
            const int splits = detail::gqa_attention_split_capacity(head_dim, q_heads,
                kv_heads, tile, cache_dtype, envelope);
            (void)allocate_prompt_tile_workspace(layout, head_dim, q_heads, tile, splits, lanes);
        });
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::GqaAttentionRoute route =
            detail::gqa_attention_resolve_route(q_heads, kv_heads, width, batch_size, envelope);
        if (route == detail::GqaAttentionRoute::Prompt) {
            if (cache_dtype == DType::I8) { return std::size_t{0}; }
            if (batch_size == 1 &&
                fa3_takes_prompt(head_dim, q_heads, kv_heads, cache_dtype, width)) {
                WorkspaceLayoutBuilder layout;
                (void)allocate_fa3_prompt_workspace(layout, head_dim, q_heads, width, 1, cache_dtype);
                return layout.peak_bytes(1);
            }
            // The prompt kernel (launch_cached_prompt_tiles) needs no workspace.
            if (prompt_kernel_takes(head_dim, q_heads, kv_heads, cache_dtype, width)) {
                return std::size_t{0};
            }
            return exact_prompt_capacity(width);
        }
        if (route == detail::GqaAttentionRoute::SmallT) { return chunk_capacity(width); }
        const std::int32_t step = detail::gqa_attention_small_t_max_width(q_heads, kv_heads);
        std::size_t maximum     = 0;
        for (std::int32_t begin = 0; begin < width; begin += step) {
            maximum = std::max(maximum, chunk_capacity(std::min(step, width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    const int last = std::min(max_width, std::max(kMaximumVerifyTokens, prompt_capacity));
    for (int width = min_width; width <= last; ++width) {
        maximum = std::max(maximum, exact_capacity(width));
    }
    if (max_width > last) { maximum = std::max(maximum, exact_capacity(max_width)); }

    return std::max(maximum, rows_bytes);
}

std::size_t gqa_attention_history_workspace_capacity_bytes(std::int32_t head_dim, std::int32_t q_heads,
    std::int32_t kv_heads, DType cache_dtype, GqaExecutionEnvelope envelope,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    std::size_t maximum = gqa_attention_workspace_capacity_bytes(head_dim, q_heads, kv_heads,
        cache_dtype, envelope, batch_size, min_width, max_width);
    if (!optimized_decode_shape(head_dim, q_heads, kv_heads, cache_dtype)) { return maximum; }
    // Shorter histories can fit more query tiles under the prompt's 32 MiB limit.
    // Their rounded allocation can exceed the one at max_visible_keys, even though
    // each individual query needs fewer split buffers. Cover the entire history
    // interval, including alignment of three partial planes and two metadata planes.
    // The tiles take only the prompts narrower than the prompt kernel's first width.
    const std::int32_t tiled_max =
        prompt_kernel_takes(head_dim, q_heads, kv_heads, cache_dtype, max_width)
            ? std::min(max_width, prompt_kernel_min_columns() - 1) : max_width;
    if (batch_size == 1 && cache_dtype != DType::I8 && tiled_max >= min_width &&
        envelope.min_visible_keys < envelope.max_visible_keys &&
        detail::gqa_attention_resolve_route(q_heads, kv_heads, tiled_max, 1,
            {envelope.min_visible_keys, envelope.min_visible_keys, envelope.sliding_window}) ==
            detail::GqaAttentionRoute::Prompt) {
        const int splits = detail::gqa_attention_split_capacity(head_dim, q_heads, kv_heads,
            32, cache_dtype, envelope);
        const std::size_t per_query = static_cast<std::size_t>(head_dim + 2) * q_heads * splits * 4;
        const std::size_t payload = std::max(per_query,
            std::min<std::size_t>(32U << 20, per_query * std::min(tiled_max, 128)));
        maximum = std::max(maximum, payload + 2048);
    }
    return maximum;
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                   GqaBlockMask selection, GqaRoundMetadata* round) {
    constexpr const char* op = "gqa_attention";
    if (selection.image_begin < 0 || selection.image_end < selection.image_begin ||
        static_cast<std::uint32_t>(selection.image_end) > envelope.max_visible_keys ||
        (selection.image_end && q.ne[3] != 1)) { throw std::invalid_argument("invalid image attention block"); }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t kv_heads = kv_heads_for_pair(head_dim, q.ne[1], k.ne[1]);
    require_registered_shape(head_dim, q.ne[1], kv_heads, op);
    require_shape(k, head_dim, kv_heads, width, batch, op, "k");
    require_shape(v, head_dim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    if (fa3_takes_rows(head_dim, q.ne[1], kv_heads, cache.dtype, width, batch, selection)) {
        detail::gqa_generic_kv_append_launch(k, v, positions, valid_columns, kv_table_rows, cache,
                                             stream);
        launch_fa3_rows(q, positions, valid_columns, kv_table_rows, scale, envelope, cache,
                        workspace, out, stream, round);
        return;
    }
    if (selection.image_end || !optimized_decode_shape(head_dim, q.ne[1], kv_heads, cache.dtype)) {
        if (fa3_takes_generic_prompt(q, valid_columns, cache, selection)) {
            detail::gqa_kv_append_batch_launch(k, v, positions, valid_columns, kv_table_rows, cache,
                                               stream);
            launch_fa3_prompt(q, positions, kv_table_rows, scale, envelope.sliding_window, cache,
                              workspace, out, stream);
            return;
        }
        if (prompt_kernel_takes_generic(q, cache, selection)) {
            detail::gqa_kv_append_batch_launch(k, v, positions, valid_columns, kv_table_rows, cache,
                                               stream);
            detail::gqa_attention_prompt_cached_launch(q, positions, valid_columns, kv_table_rows,
                                                       scale, cache, out, stream,
                                                       envelope.sliding_window);
            return;
        }
        if (generic_takes(head_dim, q.ne[1], kv_heads, cache.dtype, selection)) {
            auto scope = workspace.scope();
            detail::gqa_generic_kv_append_launch(k, v, positions, valid_columns, kv_table_rows,
                                                 cache, stream);
            const int splits =
                generic_splits(head_dim, q.ne[1], kv_heads, cache.dtype, batch, width);
            const DeviceSpan partials =
                allocate_generic_workspace(workspace, head_dim, q.ne[1], width * batch, splits);
            detail::gqa_generic_attention_launch(q, positions, valid_columns, kv_table_rows, scale,
                                                 cache, envelope.sliding_window, splits, partials,
                                                 out, stream);
            return;
        }
        detail::gqa_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                            scale, cache, out, stream, envelope.sliding_window,
                                            selection);
        return;
    }
    auto scope = workspace.scope();
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], kv_heads, width, batch, envelope);
    if (route == detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream, selection);
        return;
    }
    if (route == detail::GqaAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[0], q.ne[1], kv_heads, width,
                                                 cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], width, splits, batch);
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, envelope, 0, width, partial.acc,
                                             partial.m, partial.l, out, stream, selection);
        return;
    }
    if (cache.dtype != DType::I8 && !selection.words) {
        detail::gqa_kv_append_batch_launch(k, v, positions, valid_columns, kv_table_rows, cache, stream);
        launch_cached_prompt_tiles(q, positions, valid_columns, kv_table_rows, scale, cache,
                                     envelope, workspace, out, stream);
        return;
    }
    detail::gqa_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                        cache, out, stream, envelope.sliding_window, selection);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    const std::int32_t head_dim = k.ne[0];
    require_kv_heads(head_dim, kv_heads, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    require_shape(k, head_dim, kv_heads, tokens, 1, op, "k");
    require_shape(v, head_dim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    const std::uint32_t capacity = validate_cache(cache, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KV cache capacity");
    }
    detail::gqa_kv_append_launch(k, v, positions, cache, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, const Tensor& valid_columns,
                          const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView cache,
                          GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                          cudaStream_t stream, GqaBlockMask selection, GqaRoundMetadata* round) {
    constexpr const char* op = "gqa_attention_cached";
    if (selection.image_begin < 0 || selection.image_end < selection.image_begin ||
        static_cast<std::uint32_t>(selection.image_end) > envelope.max_visible_keys ||
        (selection.image_end && q.ne[3] != 1)) { throw std::invalid_argument("invalid image attention block"); }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       envelope, scale, op);
    const std::int32_t width = q.ne[2];
    const std::int32_t batch = q.ne[3];
    require_registered_shape(q.ne[0], q.ne[1], cache.num_kv_heads, op);

    if (fa3_takes_rows(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, width, batch, selection)) {
        launch_fa3_rows(q, positions, valid_columns, kv_table_rows, scale, envelope, cache,
                        workspace, out, stream, round);
        return;
    }
    if (selection.image_end || !optimized_decode_shape(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype)) {
        if (fa3_takes_generic_prompt(q, valid_columns, cache, selection)) {
            launch_fa3_prompt(q, positions, kv_table_rows, scale, envelope.sliding_window, cache,
                              workspace, out, stream);
            return;
        }
        if (prompt_kernel_takes_generic(q, cache, selection)) {
            detail::gqa_attention_prompt_cached_launch(q, positions, valid_columns, kv_table_rows,
                                                       scale, cache, out, stream,
                                                       envelope.sliding_window);
            return;
        }
        if (generic_takes(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, selection)) {
            auto scope = workspace.scope();
            const int splits =
                generic_splits(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, batch, width);
            const DeviceSpan partials =
                allocate_generic_workspace(workspace, q.ne[0], q.ne[1], width * batch, splits);
            detail::gqa_generic_attention_launch(q, positions, valid_columns, kv_table_rows, scale,
                                                 cache, envelope.sliding_window, splits, partials,
                                                 out, stream);
            return;
        }
        detail::gqa_attention_prompt_cached_launch(q, positions, valid_columns, kv_table_rows,
                                                   scale, cache, out, stream,
                                                   envelope.sliding_window, selection);
        return;
    }
    auto scope = workspace.scope();
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], cache.num_kv_heads, width, batch, envelope);
    if (route == detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_batch_small_t(q, positions, valid_columns, kv_table_rows, scale,
                                            cache, envelope, workspace, out, stream, selection);
        return;
    }
    if (route == detail::GqaAttentionRoute::SmallT) {
        const std::int32_t splits = detail::gqa_attention_split_capacity(
            q.ne[0], q.ne[1], cache.num_kv_heads, width, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], width, splits, batch);
        detail::gqa_attention_cached_batch_small_t_launch(
            q, positions, valid_columns, kv_table_rows, scale, cache, envelope, 0, width,
            partial.acc, partial.m, partial.l, out, stream, selection);
        return;
    }
    if (cache.dtype != DType::I8 && !selection.words) {
        launch_cached_prompt_tiles(q, positions, valid_columns, kv_table_rows, scale, cache,
                                     envelope, workspace, out, stream);
        return;
    }
    detail::gqa_attention_prompt_cached_launch(q, positions, valid_columns, kv_table_rows, scale,
                                               cache, out, stream, envelope.sliding_window,
                                               selection);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                          GqaBlockMask selection) {
    constexpr const char* op = "gqa_attention_cached";
    if (selection.image_begin < 0 || selection.image_end < selection.image_begin ||
        static_cast<std::uint32_t>(selection.image_end) > envelope.max_visible_keys ||
        (selection.image_end && q.ne[3] != 1)) { throw std::invalid_argument("invalid image attention block"); }
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);
    const PagedKVBatchLayerView batch_cache{
        .k_pages = cache.k_pages, .v_pages = cache.v_pages,
        .k_scale_pages = cache.k_scale_pages, .v_scale_pages = cache.v_scale_pages,
        .indexer_pages = cache.indexer_pages, .block_tables = cache.block_table,
        .head_dim = cache.head_dim, .num_kv_heads = cache.num_kv_heads,
        .dtype = cache.dtype, .quant_group = cache.quant_group,
    };

    if (fa3_takes_rows(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, q.ne[2], 1, selection)) {
        launch_fa3_rows(q, positions, Tensor{}, Tensor{}, scale, envelope, batch_cache, workspace,
                        out, stream);
        return;
    }
    if (selection.image_end || !optimized_decode_shape(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype)) {
        if (fa3_takes_generic_prompt(q, Tensor{}, batch_cache, selection)) {
            launch_fa3_prompt(q, positions, Tensor{}, scale, envelope.sliding_window, batch_cache,
                              workspace, out, stream);
            return;
        }
        if (prompt_kernel_takes_generic(q, batch_cache, selection)) {
            detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream,
                                                          envelope.sliding_window);
            return;
        }
        if (generic_takes(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, selection)) {
            auto scope = workspace.scope();
            const int splits =
                generic_splits(q.ne[0], q.ne[1], cache.num_kv_heads, cache.dtype, 1, q.ne[2]);
            const DeviceSpan partials =
                allocate_generic_workspace(workspace, q.ne[0], q.ne[1], q.ne[2], splits);
            detail::gqa_generic_attention_launch(q, positions, scale, cache,
                                                 envelope.sliding_window, splits, partials, out,
                                                 stream);
            return;
        }
        detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream,
                                                      envelope.sliding_window, selection);
        return;
    }
    auto scope = workspace.scope();
    if (detail::gqa_attention_resolve_route(q.ne[1], cache.num_kv_heads, q.ne[2], 1, envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream,
                                      selection);
        return;
    }
    if (q.ne[2] >= 1 &&
        q.ne[2] <= detail::gqa_attention_small_t_max_width(q.ne[1], cache.num_kv_heads)) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[0], q.ne[1], cache.num_kv_heads, q.ne[2],
                                                 cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[0], q.ne[1], q.ne[2], splits);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                    partial.acc, partial.m, partial.l, out, stream,
                                                    selection);
        return;
    }
    if (cache.dtype != DType::I8 && !selection.words) {
        launch_cached_prompt_tiles(q, positions, {}, {}, scale, batch_cache,
                                     envelope, workspace, out, stream);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream,
                                                  envelope.sliding_window, selection);
}

bool gqa_attention_packs_prompt(std::int32_t head_dim, std::int32_t q_heads,
                                std::int32_t kv_heads, DType cache_dtype, std::int32_t width,
                                GqaExecutionEnvelope envelope) {
    // A width FA3 takes as a row runs alone, so its bits are the ones its own call gives.
    return (cache_dtype == DType::BF16 || cache_dtype == DType::FP8_E4M3FN) && width > 0 &&
           supported_attention_shape(head_dim, q_heads, kv_heads) &&
           optimized_decode_shape(head_dim, q_heads, kv_heads, cache_dtype) &&
           !fa3_takes_rows(head_dim, q_heads, kv_heads, cache_dtype, width, 1, {}) &&
           detail::gqa_attention_resolve_route(q_heads, kv_heads, width, 1, envelope) ==
               detail::GqaAttentionRoute::Prompt;
}

void gqa_attention_packed_prompts(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& positions,
                                  const std::vector<GqaPackedSegment>& segments, float scale,
                                  PagedKVBatchLayerView cache, WorkspaceArena& workspace,
                                  Tensor& out, cudaStream_t stream,
                                  std::int32_t max_lanes_per_launch) {
    constexpr const char* op = "gqa_attention_packed_prompts";
    if (segments.empty()) { return; }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        throw std::logic_error(std::string(op) + ": its tile metadata is copied from the host; not for capture");
    }
    const std::int32_t head_dim = q.ne[0];
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = cache.num_kv_heads;
    const std::int32_t columns  = q.ne[2];
    const bool append           = k.data != nullptr || v.data != nullptr;
    require_registered_shape(head_dim, q_heads, kv_heads, op);
    if (q.ne[3] != 1 || out.ne[2] != columns || out.ne[3] != 1 || positions.ne[0] < columns) {
        throw std::invalid_argument(std::string(op) + ": q, positions and out must share columns");
    }
    if (append) {
        require_shape(k, head_dim, kv_heads, columns, 1, op, "k");
        require_shape(v, head_dim, kv_heads, columns, 1, op, "v");
        require_contiguous_nonnull(k, op, "k");
        require_contiguous_nonnull(v, op, "v");
        if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
            throw std::invalid_argument(std::string(op) + ": k/v must be BF16");
        }
    }
    const auto segment_count = static_cast<std::int32_t>(segments.size());
    const std::int32_t window = segments.front().envelope.sliding_window;

    // Every segment is checked before anything is written.
    {
        auto check_scope = workspace.scope();
        std::vector<std::int32_t> rows(segments.size());
        for (std::size_t i = 0; i < segments.size(); ++i) { rows[i] = segments[i].table_row; }
        Tensor row_table = workspace.alloc(DType::I32, {segment_count});
        if (append) { // only the appends read it; the checks below need its shape
            CUDA_CHECK(cudaMemcpyAsync(row_table.data, rows.data(), rows.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
        }
        for (std::int32_t i = 0; i < segment_count; ++i) {
            const GqaPackedSegment& segment = segments[i];
            if (segment.column < 0 || segment.width <= 0 || segment.column + segment.width > columns ||
                segment.table_row < 0 || segment.table_row >= cache.block_tables.ne[1] ||
                segment.envelope.sliding_window != window ||
                !gqa_attention_packs_prompt(head_dim, q_heads, kv_heads, cache.dtype, segment.width,
                                            segment.envelope)) {
                throw std::invalid_argument(std::string(op) + ": segment " + std::to_string(i) +
                                            " is not a packable prompt");
            }
            Tensor q_part   = q.slice(2, segment.column, segment.width);
            Tensor out_part = out.slice(2, segment.column, segment.width);
            validate_batched_attention_tensors(q_part, positions.slice(0, segment.column, segment.width),
                                               Tensor{}, row_table.slice(0, i, 1), out_part, cache,
                                               segment.envelope, scale, op);
        }
        // The prompt route appends before it attends; every tile below reads these keys. The row
        // table's memory goes back when this scope closes: later work on the stream runs after
        // these appends.
        if (append) {
            for (std::int32_t i = 0; i < segment_count; ++i) {
                const GqaPackedSegment& segment = segments[i];
                detail::gqa_kv_append_batch_launch(
                    k.slice(2, segment.column, segment.width), v.slice(2, segment.column, segment.width),
                    positions.slice(0, segment.column, segment.width), Tensor{}, row_table.slice(0, i, 1),
                    cache, stream);
            }
        }
    }

    // The segments FA3 takes on their own (launch_cached_prompt_tiles) go in one varlen launch;
    // it computes each segment's tiles exactly as a one-segment launch does.
    // Those the prompt kernel takes (launch_cached_prompt_tiles) run their own launches.
    std::vector<const GqaPackedSegment*> tiled;
    std::vector<const GqaPackedSegment*> flash;
    std::vector<const GqaPackedSegment*> kernel;
    for (const GqaPackedSegment& segment : segments) {
        if (fa3_takes_prompt(head_dim, q_heads, kv_heads, cache.dtype, segment.width)) {
            flash.push_back(&segment);
        } else if (prompt_kernel_takes(head_dim, q_heads, kv_heads, cache.dtype, segment.width)) {
            kernel.push_back(&segment);
        } else {
            tiled.push_back(&segment);
        }
    }
    if (!kernel.empty()) {
        auto kernel_scope = workspace.scope();
        std::vector<std::int32_t> rows(kernel.size());
        for (std::size_t i = 0; i < kernel.size(); ++i) { rows[i] = kernel[i]->table_row; }
        Tensor row_table = workspace.alloc(DType::I32, {static_cast<std::int32_t>(rows.size())});
        CUDA_CHECK(cudaMemcpyAsync(row_table.data, rows.data(), rows.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        for (std::size_t i = 0; i < kernel.size(); ++i) {
            const GqaPackedSegment& segment = *kernel[i];
            Tensor result = out.slice(2, segment.column, segment.width);
            detail::gqa_attention_prompt_cached_launch(
                q.slice(2, segment.column, segment.width),
                positions.slice(0, segment.column, segment.width), Tensor{},
                row_table.slice(0, static_cast<std::int32_t>(i), 1), scale, cache, result, stream,
                window);
        }
    }
    if (!flash.empty()) {
        // The arena is planned for one prompt of the round's width through FA3 (or its far larger
        // tiles); a round of many segments needs a few KiB more for their metadata.
        WorkspaceLayoutBuilder layout;
        (void)allocate_fa3_prompt_workspace(layout, head_dim, q_heads, columns,
                                            static_cast<std::int32_t>(flash.size()), cache.dtype);
        if (workspace.capacity() - workspace.used() < layout.peak_bytes(1) + 256) {
            tiled.insert(tiled.end(), flash.begin(), flash.end());
            flash.clear();
        }
    }
    if (!flash.empty()) {
        auto flash_scope = workspace.scope();
        const auto n = static_cast<std::int32_t>(flash.size());
        std::vector<std::int32_t> metadata(static_cast<std::size_t>(detail::gqa_fa3::metadata_ints(n)), 0);
        std::int32_t max_q = 0;
        for (std::int32_t i = 0; i < n; ++i) {
            metadata[i]                 = flash[i]->column;
            metadata[n + 1 + i]         = flash[i]->width;
            metadata[3 * n + 1 + i]     = flash[i]->table_row;
            max_q = std::max(max_q, flash[i]->width);
        }
        metadata[n] = columns;
        auto [device_metadata, scratch] =
            allocate_fa3_prompt_workspace(workspace, head_dim, q_heads, columns, n, cache.dtype);
        auto* m = static_cast<std::int32_t*>(device_metadata.data);
        CUDA_CHECK(cudaMemcpyAsync(m, metadata.data(), metadata.size() * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, stream));
        detail::gqa_fa3::segment_kv_lengths(static_cast<const std::int32_t*>(positions.data), m,
                                            m + n + 1, n, m + 2 * n + 1, stream);
        detail::gqa_fa3::run(fa3_launch(q, cache, scale, window, out, n, max_q, device_metadata),
                             scratch.data, scratch.bytes, stream);
    }

    // Every other segment's tiles exactly as its own call cuts them (launch_cached_prompt_tiles),
    // filed by width.
    struct Lane {
        std::int32_t column = 0;
        std::int32_t row    = 0;
        GqaExecutionEnvelope envelope{};
        std::int32_t splits = 0;
    };
    std::vector<std::vector<Lane>> by_width(33);
    for (const GqaPackedSegment* tiled_segment : tiled) {
        const GqaPackedSegment& segment = *tiled_segment;
        const int capacity = prompt_query_capacity(head_dim, q_heads, kv_heads, cache.dtype,
                                                   segment.envelope);
        for_each_prompt_tile(q_heads, kv_heads, segment.width, capacity,
                             [&](int begin, int width, int lanes) {
            const std::int32_t splits = detail::gqa_attention_split_capacity(
                head_dim, q_heads, kv_heads, width, cache.dtype, segment.envelope);
            for (int lane = 0; lane < lanes; ++lane) {
                by_width.at(width).push_back(Lane{segment.column + begin + lane * width,
                                                  segment.table_row, segment.envelope, splits});
            }
        });
    }

    // A launch takes as many lanes as the workspace's free room holds, up to kPackedBudget -- the
    // arena is sized for the per-prompt path, which needs a tile group's partials at once, so a
    // lone lane always fits where its own call fit -- and never more lanes than a grid holds.
    constexpr std::size_t kPackedBudget = std::size_t{64} << 20;
    for (std::int32_t width = 1; width <= 32; ++width) {
        std::vector<Lane>& lanes = by_width[width];
        if (lanes.empty()) { continue; }
        // A launch's partial buffers are sized for its longest history, so lanes that need fewer
        // key partitions go together. Which lanes share a launch changes no lane's result.
        std::stable_sort(lanes.begin(), lanes.end(),
                         [](const Lane& a, const Lane& b) { return a.splits < b.splits; });
        const auto lane_bytes = [&](std::int32_t splits) {
            return static_cast<std::size_t>(head_dim + 2) * q_heads * width * splits * sizeof(float);
        };
        std::size_t lane_limit = std::size_t{65535} / static_cast<std::size_t>(width);
        if (max_lanes_per_launch > 0) {
            lane_limit = std::min(lane_limit, static_cast<std::size_t>(max_lanes_per_launch));
        }
        for (std::size_t start = 0; start < lanes.size();) {
            auto launch_scope = workspace.scope();
            // Four allocations below, each padded to 256 bytes at most.
            constexpr std::size_t kSlack = 4 * 256 + 256;
            const std::size_t room   = workspace.capacity() - workspace.used();
            const std::size_t free   = room > kSlack ? room - kSlack : 0;
            const std::size_t budget = std::min(free, kPackedBudget);
            std::size_t count   = 0;
            std::int32_t splits = 0;
            while (start + count < lanes.size() && count < lane_limit) {
                const std::int32_t next = std::max(splits, lanes[start + count].splits);
                const std::size_t need  = lane_bytes(next) * (count + 1) + 3 * sizeof(std::int32_t) * (count + 1);
                if (count != 0 && need > budget) { break; }
                splits = next;
                ++count;
            }
            GqaExecutionEnvelope envelope = lanes[start].envelope;
            const auto n = static_cast<std::int32_t>(count);
            // One upload per launch: valid columns, table rows and lane columns, back to back.
            std::vector<std::int32_t> metadata(3 * count);
            for (std::size_t i = 0; i < count; ++i) {
                const Lane& lane = lanes[start + i];
                envelope.min_visible_keys = std::min(envelope.min_visible_keys, lane.envelope.min_visible_keys);
                envelope.max_visible_keys = std::max(envelope.max_visible_keys, lane.envelope.max_visible_keys);
                metadata[i]             = width;
                metadata[count + i]     = lane.row;
                metadata[2 * count + i] = lane.column;
            }
            Tensor lane_data = workspace.alloc(DType::I32, {3 * n});
            CUDA_CHECK(cudaMemcpyAsync(lane_data.data, metadata.data(), metadata.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, stream));
            Tensor valid_columns = lane_data.slice(0, 0, n);
            Tensor table_rows    = lane_data.slice(0, n, n);
            Tensor lane_columns  = lane_data.slice(0, 2 * n, n);
            // The launch sizes its splits for the longest history among its lanes; every lane still
            // walks only its own key partitions (they are anchored to absolute positions).
            const std::int32_t launch_splits = detail::gqa_attention_split_capacity(
                head_dim, q_heads, kv_heads, width, cache.dtype, envelope);
            SmallTWorkspace partial =
                allocate_small_t_workspace(workspace, head_dim, q_heads, width, launch_splits, n);
            if (count == 1) {
                // A lone lane: the single-sequence form of the same kernel.
                const std::int32_t column = metadata[2];
                const Tensor queries = q.slice(2, column, width).view({head_dim, q_heads, width, 1});
                const Tensor pos     = positions.slice(0, column, width).view({width, 1});
                Tensor result = out.slice(2, column, width).view({head_dim, q_heads, width, 1});
                detail::gqa_attention_cached_batch_small_t_launch(
                    queries, pos, valid_columns, table_rows, scale, cache, envelope, 0, width,
                    partial.acc, partial.m, partial.l, result, stream);
            } else {
                detail::gqa_attention_cached_lanes_small_t_launch(
                    q, positions, valid_columns, table_rows, lane_columns, scale, cache, envelope,
                    width, partial.acc, partial.m, partial.l, out, stream);
            }
            start += count;
        }
    }
}

} // namespace sinfer::ops
