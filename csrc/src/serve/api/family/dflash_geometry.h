#pragma once

#include <api/family/geometry_validation.h>
#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <limits>

namespace sinfer::family {

struct DFlashGeometry {
#define SINFER_GEOMETRY_INT(name) std::int32_t name = 0;
#define SINFER_GEOMETRY_FLOAT(name) float name = 0.0F;
#include "dflash_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
    std::array<int, 256> target_feature_layers{};
    [[nodiscard]] constexpr int query_size() const { return query_heads * head_dim; }
    [[nodiscard]] constexpr int kv_size() const { return kv_heads * head_dim; }
    [[nodiscard]] std::span<const int> target_layers() const {
        return {target_feature_layers.data(), static_cast<std::size_t>(feature_layers)};
    }
    [[nodiscard]] static DFlashGeometry resolved(const std::map<std::string, double>& values,
                                                 std::span<const std::int32_t> targets,
                                                 int target_hidden, int target_layers, int target_vocab) {
        DFlashGeometry g;
        struct IntMember { std::string_view name; std::int32_t DFlashGeometry::* value; };
        static constexpr IntMember ints[] = {
#define SINFER_GEOMETRY_INT(name) {#name, &DFlashGeometry::name},
#define SINFER_GEOMETRY_FLOAT(name)
#include "dflash_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        struct FloatMember { std::string_view name; float DFlashGeometry::* value; };
        static constexpr FloatMember floats[] = {
#define SINFER_GEOMETRY_FLOAT(name) {#name, &DFlashGeometry::name},
#define SINFER_GEOMETRY_INT(name)
#include "dflash_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        detail::apply_geometry(g, values, ints, floats, "dflash_geometry");
        for (const auto& entry : ints) {
            if (!values.contains(std::string(entry.name)) ||
                (entry.name != "mask_token" && g.*(entry.value) <= 0)) {
                throw std::invalid_argument("missing or invalid dflash_geometry." + std::string(entry.name));
            }
        }
        for (const auto& entry : floats) {
            if (!values.contains(std::string(entry.name)) || g.*(entry.value) <= 0) {
                throw std::invalid_argument("missing or invalid dflash_geometry." + std::string(entry.name));
            }
        }
        if (g.layers < 2 || g.layers > 256 || g.local_layers != g.layers - 1 ||
            g.hidden != target_hidden || g.mask_token >= target_vocab ||
            g.query_heads % g.kv_heads || g.head_dim % 2 || g.block_size < 2 || g.block_size > 16 ||
            g.feature_layers != targets.size() || targets.size() > g.target_feature_layers.size() ||
            std::int64_t(g.feature_layers) * g.hidden != g.feature_rows ||
            (std::int64_t(g.query_heads) + 2LL * g.kv_heads) * g.head_dim > INT32_MAX ||
            std::int64_t(2) * g.intermediate > INT32_MAX) {
            throw std::invalid_argument("inconsistent DFlash checkpoint geometry");
        }
        std::array<bool, 256> seen{};
        for (std::size_t i = 0; i < targets.size(); ++i) {
            const auto layer = targets[i];
            if (layer < 0 || layer >= target_layers || layer >= 256 || seen[layer]) {
                throw std::invalid_argument("DFlash target layers must be distinct valid target layer indices");
            }
            g.target_feature_layers[i] = layer;
            seen[layer] = true;
        }
        return g;
    }
    [[nodiscard]] constexpr bool operator==(const DFlashGeometry&) const noexcept = default;
};
} // namespace sinfer::family
