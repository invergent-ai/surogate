#pragma once

#include "api/family/text_geometry.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>

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

/// The largest workspace any text layer's stored `role` matrix asks of `capacity`, 0 when the
/// artifact's storage was not resolved. What a planner adds for formats an artifact carries
/// beyond its profile's own: a GGUF's K-quants, an FP8 export's block-scaled codes.
template<class Capacity>
std::size_t stored_role_workspace(const TextGeometry& geometry, std::string_view role,
                                  Capacity capacity) {
    std::size_t peak = 0;
    for (std::int32_t layer = 0; layer < geometry.layers && !geometry.linear_storage.empty(); ++layer) {
        const std::string name = "text/layers/" + std::to_string(layer) + "/" + std::string(role);
        if (geometry.linear_storage.contains(name)) {
            peak = std::max(peak, matrix_workspace(geometry, name, capacity));
        }
    }
    return peak;
}

/// `capacity(type)` when the artifact stores any text matrix as `type`, else 0: how a planner
/// that sizes for its profile's format and a GGUF's K-quants also covers an FP8 export's codes.
template<class Capacity>
std::size_t stored_format_workspace(const TextGeometry& geometry, QType type, Capacity capacity) {
    for (const auto& [name, storage] : geometry.linear_storage) {
        if (std::find(storage.formats.begin(), storage.formats.end(), type) != storage.formats.end()) {
            return capacity(type);
        }
    }
    return 0;
}

/// `capacity(type)` when some text layer stores one of `roles` as `type`, else 0. The form above
/// answers for the whole artifact, which is right only when one format covers every projection:
/// an export that quantises some roles and not others -- NVIDIA's Gemma 4 31B keeps its
/// attention BF16 (stored W8) and its feed-forward NVFP4 -- would otherwise have its attention
/// shapes planned for a format they never take, and NVFP4 refuses the 31B's global widths.
template<class Capacity>
std::size_t stored_roles_format_workspace(const TextGeometry& geometry,
                                          std::initializer_list<std::string_view> roles,
                                          QType type, Capacity capacity) {
    constexpr std::string_view kLayers = "text/layers/";
    for (const auto& [name, storage] : geometry.linear_storage) {
        const std::string_view key = name;
        if (!key.starts_with(kLayers) ||
            std::find(storage.formats.begin(), storage.formats.end(), type) == storage.formats.end()) {
            continue;
        }
        const auto slash = key.find('/', kLayers.size());
        if (slash == std::string_view::npos) {
            continue;
        }
        const std::string_view role = key.substr(slash + 1);
        if (std::find(roles.begin(), roles.end(), role) != roles.end()) {
            return capacity(type);
        }
    }
    return 0;
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
