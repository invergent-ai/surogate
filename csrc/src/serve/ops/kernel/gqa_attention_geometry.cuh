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
// An unregistered shape is a build error at the dispatcher, never a silent
// fallback to a slower path.

#include <cstdint>

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

// Compatibility aliases for call sites not yet migrated. New code uses the
// shape names above; these disappear once the last dispatcher is converted.
using Gqa27Geometry = Gqa256_24q4;
using Gqa35Geometry = Gqa256_16q2;
using Gqa08Geometry = Gqa256_8q2;
using Gqa4BGeometry = Gqa256_16q4;

} // namespace sinfer::ops
