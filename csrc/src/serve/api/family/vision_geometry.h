#pragma once

#include <api/family/geometry_validation.h>

#include <cstdint>
#include <map>
#include <limits>
#include <string>
#include <string_view>

namespace sinfer::family {

/// Tower dimensions and execution settings resolved from the checkpoint.
struct VisionGeometry {
    std::int32_t gemma_version = 0;
    std::int32_t gemma_pad_token = 0;
    std::int32_t encoder_free = 0;
    std::int32_t clipped_linears = 0;
    std::int32_t standardize = 0;
    std::int32_t attention_mode = 0;
    std::int32_t max_image_tokens = 0;
    std::int32_t deepstack_layers = 0;
    std::int32_t siglip2 = 0;
    std::int32_t projector_hidden = 0;
    std::int32_t projector_norm = 0;
    std::int32_t layers              = 0;
    std::int32_t hidden              = 0;
    std::int32_t intermediate        = 0;
    std::int32_t heads               = 0;
    std::int32_t patch_dim           = 0;
    /// The side of the square block of patches one visual token is merged from.
    std::int32_t merge               = 0;
    std::int32_t position_embeddings = 0;
    /// How many lanes of a head RoPE turns. Every tower shipped so far rotates the whole head,
    /// so a declaration that moves `hidden` or `heads` has to name this too -- it is primary
    /// here precisely because a family may one day rotate less than the head.
    std::int32_t rotary_dim          = 0;
    /// The width the merger projects into: the text model's hidden state, since merged visual
    /// tokens join the residual stream. It belongs to the tower's output contract rather than
    /// to the tower, which is why the family's shared config leaves it unstated.
    std::int32_t output_hidden       = 0;
    float rope_theta                 = 0.0F;
    float norm_epsilon               = 0.0F;

    [[nodiscard]] constexpr std::int32_t head_dim() const noexcept {
        return heads == 0 ? 0 : hidden / heads;
    }
    [[nodiscard]] constexpr std::int32_t merge_unit() const noexcept { return merge * merge; }
    [[nodiscard]] constexpr std::int32_t merger_hidden() const noexcept {
        return hidden * merge_unit();
    }
    [[nodiscard]] constexpr std::int32_t projector_width() const noexcept {
        return siglip2 ? projector_hidden : merger_hidden();
    }

    [[nodiscard]] static VisionGeometry resolved(const std::map<std::string, double>& values) {
        VisionGeometry g;
        g.override_from(values);
#define SINFER_GEOMETRY_INT(name) \
        if (std::string_view(#name) != "gemma_pad_token" && std::string_view(#name) != "gemma_version" && std::string_view(#name) != "encoder_free" && \
            std::string_view(#name) != "clipped_linears" && std::string_view(#name) != "standardize" && \
            std::string_view(#name) != "attention_mode" && std::string_view(#name) != "max_image_tokens" && \
            !(g.encoder_free && std::string_view(#name) == "layers") && \
            std::string_view(#name) != "siglip2" && std::string_view(#name) != "projector_hidden" && \
            std::string_view(#name) != "projector_norm" && std::string_view(#name) != "deepstack_layers" && \
             !((g.siglip2 || g.gemma_version == 3 || g.encoder_free) && (std::string_view(#name) == "rotary_dim" || std::string_view(#name) == "rope_theta")) && \
            (!values.contains(#name) || g.name <= 0)) { \
            throw std::invalid_argument("missing or invalid vision_geometry." #name); \
        }
#define SINFER_GEOMETRY_FLOAT(name) SINFER_GEOMETRY_INT(name)
#include "vision_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        if ((g.gemma_version != 0 && g.gemma_version != 3 && g.gemma_version != 4) ||
            g.encoder_free > 1 || g.clipped_linears > 1 || g.standardize > 1 || g.attention_mode > 2 ||
            (g.gemma_version && (g.siglip2 || g.deepstack_layers || g.max_image_tokens <= 0))) {
            throw std::invalid_argument("invalid Gemma vision geometry");
        }
        if (g.siglip2 && (g.siglip2 != 1 || g.projector_hidden <= 0 || g.projector_norm > 1 ||
                         g.rotary_dim != 0 || g.merge != 2)) {
            throw std::invalid_argument("invalid SigLIP2 projector geometry");
        }
        if (g.deepstack_layers > g.layers || (g.siglip2 && g.deepstack_layers)) {
            throw std::invalid_argument("invalid vision deepstack count");
        }
        if (g.hidden % g.heads || g.head_dim() % 4 || g.rotary_dim > g.head_dim() || g.rotary_dim % 4 ||
            std::int64_t(g.merge) * g.merge > std::numeric_limits<std::int32_t>::max() / g.hidden ||
            std::int64_t(3) * g.hidden > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("invalid vision checkpoint geometry");
        }
        return g;
    }

    /// Lay `declared` over this: a key present replaces the member of that name.
    void override_from(const std::map<std::string, double>& declared) {
        struct IntMember { std::string_view name; std::int32_t VisionGeometry::* value; };
        static constexpr IntMember kInts[] = {
#define SINFER_GEOMETRY_INT(name) {#name, &VisionGeometry::name},
#define SINFER_GEOMETRY_FLOAT(name)
#include "vision_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        struct FloatMember { std::string_view name; float VisionGeometry::* value; };
        static constexpr FloatMember kFloats[] = {
#define SINFER_GEOMETRY_FLOAT(name) {#name, &VisionGeometry::name},
#define SINFER_GEOMETRY_INT(name)
#include "vision_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        detail::apply_geometry(*this, declared, kInts, kFloats, "vision_geometry");
    }

    [[nodiscard]] constexpr bool operator==(const VisionGeometry&) const noexcept = default;
};

} // namespace sinfer::family
