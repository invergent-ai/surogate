#pragma once

#include "api/family/text_geometry.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"

#include <algorithm>
#include <limits>

namespace sinfer::artifact {
inline void resolve_linear_storage(const Reader& reader, family::TextGeometry& geometry) {
    for (const auto& object : reader.objects()) {
        const auto* tensor = std::get_if<TensorDescriptor>(&object);
        if (!tensor || tensor->shape.size() != 2 ||
            !(tensor->name.starts_with("text/") || tensor->name.starts_with("mtp/"))) {
            continue;
        }
        for (const auto dimension : tensor->shape) {
            if (dimension == 0 || dimension > std::numeric_limits<std::int32_t>::max()) {
                throw ArtifactError(tensor->name + ": matrix dimension is outside the runtime domain");
            }
        }
        family::LinearStorage storage{
            .rows = static_cast<std::int32_t>(tensor->shape[0]),
            .columns = static_cast<std::int32_t>(tensor->shape[1]),
            .formats = {qtype_for(tensor->format)},
        };
        for (const auto& segment : tensor->segments) {
            const auto format = qtype_for(segment.format);
            if (std::find(storage.formats.begin(), storage.formats.end(), format) == storage.formats.end()) {
                storage.formats.push_back(format);
            }
        }
        geometry.linear_storage.emplace(tensor->name, std::move(storage));
    }
}
} // namespace sinfer::artifact
