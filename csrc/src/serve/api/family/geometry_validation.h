#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::family::detail {

template <class Geometry, class IntMember, std::size_t IntCount,
          class FloatMember, std::size_t FloatCount>
void apply_geometry(Geometry& geometry, const std::map<std::string, double>& declared,
                    const IntMember (&ints)[IntCount], const FloatMember (&floats)[FloatCount],
                    std::string_view scope) {
    // Validate the whole declaration before modifying anything. Unknown fields must not
    // silently leave a compiled checkpoint's value in effect.
    for (const auto& [name, value] : declared) {
        const auto integer = std::find_if(std::begin(ints), std::end(ints),
                                         [&](const auto& field) { return field.name == name; });
        const auto floating = std::find_if(std::begin(floats), std::end(floats),
                                          [&](const auto& field) { return field.name == name; });
        const std::string field = std::string(scope) + "." + name;
        if (integer == std::end(ints) && floating == std::end(floats)) {
            throw std::invalid_argument("unknown " + field + "; rebuild the serving cache");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument(field + " must be a finite number");
        }
        if (integer != std::end(ints)) {
            if (value < 0 || value > std::numeric_limits<std::int32_t>::max() ||
                std::trunc(value) != value) {
                throw std::invalid_argument(field + " must be a nonnegative int32");
            }
        } else if (value < 0 || value > std::numeric_limits<float>::max()) {
            throw std::invalid_argument(field + " must be a nonnegative float32");
        }
    }
    for (const auto& field : ints) {
        if (const auto it = declared.find(std::string(field.name)); it != declared.end()) {
            geometry.*(field.value) = static_cast<std::int32_t>(it->second);
        }
    }
    for (const auto& field : floats) {
        if (const auto it = declared.find(std::string(field.name)); it != declared.end()) {
            geometry.*(field.value) = static_cast<float>(it->second);
        }
    }
}

} // namespace sinfer::family::detail
