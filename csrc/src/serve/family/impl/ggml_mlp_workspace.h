#pragma once

#include "api/family/text_geometry.h"
#include "family/impl/storage_workspace.h"
#include "family/impl/mlp_swiglu.h"
#include "ops/linear/ggml/ggml_dispatch.h"

#include <algorithm>

namespace sinfer::family {

// Size the actual dense MLP stored in the artifact. Adapters can require both
// halves or the complete parent; keep that reservation when a bank is enabled.
inline std::size_t stored_mlp_workspace_capacity_bytes(const TextGeometry& geometry,
                                                       std::string_view prefix,
                                                       ops::LinearPolicy policy, std::int32_t first,
                                                       std::int32_t last, bool lora_enabled) {
    const auto& down = require_linear_storage(geometry, std::string(prefix) + "mlp/down");
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {geometry.intermediate, last});
    {
        auto scope = layout.scope();
        const bool separate = geometry.linear_storage.contains(std::string(prefix) + "mlp/gate");
        if (separate) {
            auto pair_scope = layout.scope();
            // Some checkpoints store separate gate/up matrices. Their operation
            // always materializes both outputs, with or without adapters.
            (void)layout.alloc(DType::BF16, {geometry.intermediate, last});
            (void)layout.alloc(DType::BF16, {geometry.intermediate, last});
            std::size_t bytes = 0;
            for (auto role : {"mlp/gate", "mlp/up"}) {
                const auto& weight = require_linear_storage(geometry, std::string(prefix) + role);
                for (auto type : weight.formats) {
                    bytes = std::max(bytes, ops::linear_workspace_capacity_bytes(
                                                type, geometry.intermediate, geometry.hidden,
                                                policy, first, last));
                }
            }
            (void)layout.alloc_bytes(bytes);
        }
        // If both representations are present, retain enough space for either
        // family binding, including the fused parent's full adapter route.
        if (!separate || geometry.linear_storage.contains(std::string(prefix) + "mlp/gate_up")) {
            auto parent_scope = layout.scope();
            const auto& parent =
                require_linear_storage(geometry, std::string(prefix) + "mlp/gate_up");
            const bool ggml =
                std::all_of(parent.formats.begin(), parent.formats.end(),
                            [](QType type) { return ops::detail::ggml::is_ggml_qtype(type); });
            if (!lora_enabled && ggml) {
                std::size_t bytes = 0;
                for (auto gate : parent.formats)
                    for (auto up : parent.formats) {
                        bytes = std::max(
                            bytes, ops::linear_swiglu_workspace_capacity_bytes(
                                       gate, up, parent.rows, parent.columns, policy, first, last));
                    }
                (void)layout.alloc_bytes(bytes);
            } else {
                for (auto type : parent.formats) {
                    swiglu_mlp_layout(layout, geometry.intermediate, geometry.hidden, type, policy,
                                      first, last);
                }
            }
        }
    }
    for (auto type : down.formats) {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            type, geometry.hidden, geometry.intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace sinfer::family
