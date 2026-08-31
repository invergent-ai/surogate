#pragma once

// Grouped-query head geometries served by the decode and prefill kernels.
//
// A geometry is a SHAPE, not a model. Head dimension, query-head count and
// KV-head count are all compile-time parameters, and registration below names
// the shape rather than the checkpoint that first needed it — Qwen, Gemma, GLM
// and Kimi do not share a head dimension, so anything that bakes one in stops
// at the first non-Qwen architecture. Adding an architecture means adding a
// registration line whose shape is not already present; it must never mean
// editing a kernel.
//
// An unregistered shape is a hard error, never a silent fallback onto another
// shape's kernel: that kernel bakes the head counts into its addressing, so
// serving one shape with another's kernel reads the wrong strides — wrong
// numbers and out-of-bounds loads, not merely a slower path. Registering a
// shape whose decode translation unit is missing is a link error; serving a
// shape nobody registered throws at the dispatcher.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {

template <int HeadDimValue, int QHeadsValue, int KVHeadsValue, int DecodeSplitScaleValue>
struct GqaGeometry {
    static_assert(HeadDimValue > 0 && HeadDimValue % 16 == 0,
                  "head dimension must be a positive multiple of 16");
    static_assert(QHeadsValue > 0 && KVHeadsValue > 0);
    static_assert(QHeadsValue % KVHeadsValue == 0,
                  "query heads must group evenly onto KV heads");
    static_assert(DecodeSplitScaleValue > 0);

    static constexpr int HeadDim          = HeadDimValue;
    static constexpr int QHeads           = QHeadsValue;
    static constexpr int KVHeads          = KVHeadsValue;
    static constexpr int GroupSize        = QHeads / KVHeads;
    static constexpr int DecodeSplitScale = DecodeSplitScaleValue;
    static constexpr int DecodeSplits     = 85 * DecodeSplitScale;
};

// ---- Registered shapes ------------------------------------------------------
//
// Named <head dim>_<query heads>q<KV heads>. The comment records which
// checkpoints happen to use the shape today; the name does not depend on it.

using Gqa256_24q4  = GqaGeometry<256, 24, 4, 1>; // qwen3.8-27b
using Gqa256_16q2  = GqaGeometry<256, 16, 2, 2>; // qwen3.6-35b-a3b
using Gqa256_8q2   = GqaGeometry<256, 8, 2, 2>;  // qwen3.5-0.8b
using Gqa256_16q4  = GqaGeometry<256, 16, 4, 2>; // qwen3.5-4b, qwen3.5-2b
using Gqa256_24q2  = GqaGeometry<256, 24, 2, 1>; // qwen3.8-flash-next (group of twelve)

// The registry. Every dispatcher below and in the launchers is generated from
// this list, so a registration line is the whole of adding a shape — with the
// one exception the linker enforces: the small-T decode launcher is instantiated
// one geometry per translation unit, so a new entry also needs its
// ops/launcher/gqa_attention_decode_<name>.cu and a CMake source line. Omitting
// them is an undefined reference at link time, which is the intended failure.
#define SINFER_GQA_FOR_EACH_GEOMETRY(X)                                                            \
    X(Gqa256_24q4)                                                                                 \
    X(Gqa256_16q2)                                                                                 \
    X(Gqa256_8q2)                                                                                  \
    X(Gqa256_16q4)                                                                                 \
    X(Gqa256_24q2)

namespace detail {

// Sentinel closing the comma-separated pack the registry expands into. Its
// zero head counts match no registered shape.
struct GqaGeometryListEnd {
    static constexpr int HeadDim = 0;
    static constexpr int QHeads  = 0;
    static constexpr int KVHeads = 0;
};

// Registered shapes must be distinct in (QHeads, KVHeads): the dispatchers that
// see only the head counts — workspace sizing and split capacity, whose public
// signatures carry no head dimension — rely on that pair naming one shape, and
// picking the wrong one there under-sizes the split buffers the kernel writes.
template <typename... Geometries>
constexpr bool gqa_head_counts_are_distinct() {
    constexpr std::size_t kCount = sizeof...(Geometries);
    const int q_heads[kCount]    = {Geometries::QHeads...};
    const int kv_heads[kCount]   = {Geometries::KVHeads...};
    for (std::size_t i = 0; i < kCount; ++i) {
        for (std::size_t j = i + 1; j < kCount; ++j) {
            if (q_heads[i] == q_heads[j] && kv_heads[i] == kv_heads[j]) { return false; }
        }
    }
    return true;
}

#define SINFER_GQA_LIST_GEOMETRY(Name) Name,
static_assert(gqa_head_counts_are_distinct<SINFER_GQA_FOR_EACH_GEOMETRY(
                  SINFER_GQA_LIST_GEOMETRY) GqaGeometryListEnd>(),
              "two registered geometries share a (query heads, KV heads) pair");
#undef SINFER_GQA_LIST_GEOMETRY

// A dispatcher that knows only some of the triple passes 0 for the rest, which
// prints as "any": the shape is unregistered whatever the missing field held.
[[noreturn]] inline void throw_unregistered_geometry(std::int64_t head_dim, std::int64_t q_heads,
                                                     std::int64_t kv_heads) {
    const auto field = [](std::int64_t value) {
        return value > 0 ? std::to_string(value) : std::string("any");
    };
    throw std::invalid_argument("gqa_attention: unregistered head geometry (head dim " +
                                field(head_dim) + ", " + field(q_heads) + " query heads over " +
                                field(kv_heads) +
                                " KV heads); register the shape in gqa_attention_geometry.cuh");
}

} // namespace detail

// Invokes `visitor.template operator()<Geometry>()` for the registered shape
// matching (head_dim, q_heads, kv_heads) exactly. At most one arm can match, so
// the order of the registry does not affect the result; an unregistered shape
// throws rather than reaching a kernel built for a different shape.
template <typename Visitor>
decltype(auto) gqa_dispatch_geometry(std::int64_t head_dim, std::int64_t q_heads,
                                     std::int64_t kv_heads, Visitor&& visitor) {
#define SINFER_GQA_DISPATCH_ARM(Name)                                                              \
    if (head_dim == Name::HeadDim && q_heads == Name::QHeads && kv_heads == Name::KVHeads) {       \
        return visitor.template operator()<Name>();                                                \
    }
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_DISPATCH_ARM)
#undef SINFER_GQA_DISPATCH_ARM
    detail::throw_unregistered_geometry(head_dim, q_heads, kv_heads);
}

// Head-count-only dispatch, for the capacity and workspace paths whose public
// signatures carry no head dimension. The pair is unique across the registry
// (asserted above), so this resolves the same shape the launcher will pick.
template <typename Visitor>
decltype(auto) gqa_dispatch_geometry(std::int64_t q_heads, std::int64_t kv_heads,
                                     Visitor&& visitor) {
#define SINFER_GQA_DISPATCH_ARM(Name)                                                              \
    if (q_heads == Name::QHeads && kv_heads == Name::KVHeads) {                                    \
        return visitor.template operator()<Name>();                                                \
    }
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_DISPATCH_ARM)
#undef SINFER_GQA_DISPATCH_ARM
    detail::throw_unregistered_geometry(0, q_heads, kv_heads);
}

// KV-only dispatch, for the cache-append kernels: they touch the head dimension
// and the KV head count and nothing else, so every registered shape carrying
// that pair generates identical code and the first match is taken. A pair no
// registered shape carries throws rather than being appended with another
// shape's strides.
template <typename Visitor>
decltype(auto) gqa_dispatch_kv_geometry(std::int64_t head_dim, std::int64_t kv_heads,
                                        Visitor&& visitor) {
#define SINFER_GQA_DISPATCH_KV_ARM(Name)                                                           \
    if (head_dim == Name::HeadDim && kv_heads == Name::KVHeads) {                                  \
        return visitor.template operator()<Name>();                                                \
    }
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_DISPATCH_KV_ARM)
#undef SINFER_GQA_DISPATCH_KV_ARM
    detail::throw_unregistered_geometry(head_dim, 0, kv_heads);
}

// Whether any registered shape carries this (head dim, KV head count) pair, for
// callers validating a KV tensor ahead of the append dispatch above.
inline bool gqa_kv_shape_is_registered(std::int64_t head_dim, std::int64_t kv_heads) {
#define SINFER_GQA_KV_MATCH(Name)                                                                  \
    if (head_dim == Name::HeadDim && kv_heads == Name::KVHeads) { return true; }
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_KV_MATCH)
#undef SINFER_GQA_KV_MATCH
    return false;
}

// The KV-head count of the registered shape serving `q_heads`. A query count can
// be registered against more than one KV count (16 queries over 2 or 4 KV heads;
// 24 over 4 or 2), which is why callers pass the head count of whatever KV
// source they hold; pass 0 when holding none, which resolves only a query count
// that is unique in the registry. Unresolvable pairs throw.
inline std::int64_t gqa_registered_kv_heads(std::int64_t q_heads, std::int64_t source_kv_heads) {
    std::int64_t sole_match = 0;
    int matches             = 0;
#define SINFER_GQA_KV_ARM(Name)                                                                    \
    if (q_heads == Name::QHeads) {                                                                 \
        if (source_kv_heads == Name::KVHeads) { return Name::KVHeads; }                            \
        sole_match = Name::KVHeads;                                                                \
        ++matches;                                                                                 \
    }
    SINFER_GQA_FOR_EACH_GEOMETRY(SINFER_GQA_KV_ARM)
#undef SINFER_GQA_KV_ARM
    if (matches == 1) { return sole_match; }
    detail::throw_unregistered_geometry(0, q_heads, source_kv_heads);
}

} // namespace sinfer::ops
