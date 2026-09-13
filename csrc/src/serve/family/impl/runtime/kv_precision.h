#pragma once

#include <api/family/text_geometry.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sinfer::family::detail {

// Public precision exclusions name model layers; the cache stores only KV owners.
// A shared layer protects the same planes as the earlier owner it attends over.
inline std::vector<std::uint32_t> kv_precision_layers(
    const TextGeometry& geometry, const std::vector<std::uint32_t>& model_layers) {
    std::vector<std::uint32_t> result;
    for (const auto layer : model_layers) {
        if (layer >= static_cast<std::uint32_t>(geometry.layers) || !geometry.layer_attends(layer)) {
            throw std::invalid_argument("--kv-cache-dtype-skip-layers must name attention layers in this model");
        }
        auto owner = static_cast<std::int32_t>(layer);
        if (!geometry.layer_owns_kv(owner)) {
            do { --owner; } while (owner >= 0 &&
                (!geometry.layer_attends(owner) || !geometry.layer_owns_kv(owner) ||
                 geometry.layer_is_windowed(owner) != geometry.layer_is_windowed(layer)));
            if (owner < 0) {
                throw std::invalid_argument("cache precision exclusion names a shared layer with no KV owner");
            }
        }
        std::uint32_t plane = 0;
        for (std::int32_t earlier = 0; earlier < owner; ++earlier) {
            plane += geometry.layer_attends(earlier) && geometry.layer_owns_kv(earlier);
        }
        result.push_back(plane);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace sinfer::family::detail
