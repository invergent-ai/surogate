#pragma once

#include "api/family/text_geometry.h"

#include <algorithm>
#include <string>

namespace sinfer::family {
inline const LinearStorage& require_linear_storage(const TextGeometry& geometry, std::string_view name) {
    const auto it = geometry.linear_storage.find(name);
    if (it == geometry.linear_storage.end() || it->second.formats.empty()) {
        throw std::invalid_argument(std::string(name) + ": matrix storage is required for workspace planning");
    }
    return it->second;
}

template<class Capacity>
std::size_t matrix_workspace(const TextGeometry& geometry, std::string_view name, Capacity capacity) {
    const auto& matrix = require_linear_storage(geometry, name);
    std::size_t peak = 0;
    for (const auto format : matrix.formats) {
        peak = std::max(peak, capacity(format, matrix.rows, matrix.columns));
    }
    return peak;
}

template<class Capacity>
std::size_t matrix_pair_workspace(const TextGeometry& geometry, std::string_view first,
                                  std::string_view second, Capacity capacity) {
    const auto& a = require_linear_storage(geometry, first);
    const auto& b = require_linear_storage(geometry, second);
    if (a.columns != b.columns) {
        throw std::invalid_argument(std::string(first) + ": paired projections disagree on input width");
    }
    std::size_t peak = 0;
    for (const auto a_format : a.formats) {
        for (const auto b_format : b.formats) {
            peak = std::max(peak, capacity(a_format, b_format, a.rows, b.rows, a.columns));
        }
    }
    return peak;
}

enum class WorkspaceLayers { All, Attention, Linear };

template<class Capacity>
std::size_t text_layers_workspace(const TextGeometry& geometry, WorkspaceLayers selection,
                                   Capacity capacity) {
    std::size_t peak = 0;
    for (std::int32_t layer = 0; layer < geometry.layers; ++layer) {
        if ((selection == WorkspaceLayers::Attention && !geometry.layer_attends(layer)) ||
            (selection == WorkspaceLayers::Linear && geometry.layer_attends(layer))) {
            continue;
        }
        peak = std::max(peak, capacity("text/layers/" + std::to_string(layer) + "/"));
    }
    return peak;
}
} // namespace sinfer::family
