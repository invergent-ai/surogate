#pragma once

#include "api/family/decoder_state.h"
#include "api/family/text_geometry.h"

#include <stdexcept>

namespace sinfer::family::detail {

// Keep global cache IDs stable while storing only the KV owners in this stage.
// Shared-KV consumers stay with their owners through pipeline_partition.
inline KvLayerRange stage_kv_layers(const TextGeometry& geometry, int first, int last) {
    if (last == 0 && first == 0) { last = geometry.layers; }
    if (first < 0 || last <= first || last > geometry.layers) {
        throw std::invalid_argument("Invalid pipeline cache layer range");
    }
    KvLayerRange range;
    for (int layer = 0; layer < last; ++layer) {
        if (!geometry.layer_attends(layer) || !geometry.layer_owns_kv(layer)) { continue; }
        if (layer < first) { ++range.first; }
        ++range.last;
    }
    return range;
}

} // namespace sinfer::family::detail
