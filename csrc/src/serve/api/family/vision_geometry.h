#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace sinfer::family {

/// The dimensions of a vision tower, as data.
///
/// The tower a checkpoint ships is per-model, not per-family: the 2B and 4B carry 24 layers of
/// 1024, the 27B/35B and Flash-Next 27 of 1152. Compiling those numbers into a `VisionConfig`
/// is why a target serves exactly one tower. This is the same set of numbers as a value: a
/// target fills it from its compiled config, then lets the artifact override whatever it
/// declares, and everything that plans a buffer or checks a shape reads it from here.
///
/// Only primary dimensions are members. Everything a tower derives from them (`head_dim`, the
/// merge unit, the merger's width) is a function here, so an artifact that declares `hidden`
/// cannot leave a stale `merger_hidden` behind.
///
/// The artifact's `vision_geometry` member is a flat object of numbers keyed by these member
/// names. Absent keys keep the compiled value, so an artifact written before the member existed
/// loads exactly as it did; unknown keys are ignored, so an artifact may name a dimension a
/// future engine reads and this one does not.
struct VisionGeometry {
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

    /// The compiled config as a value: every primary member the target's tower config declares.
    /// A config that leaves one unstated -- the family's shared tower cannot name the text width
    /// it projects into -- leaves it at zero, which is what "not stated here" means.
    template <class Config>
    [[nodiscard]] static constexpr VisionGeometry compiled() {
        VisionGeometry g;
#define SINFER_VISION_GEOMETRY_TAKE(member) \
        if constexpr (requires { Config::member; }) { g.member = Config::member; }
        SINFER_VISION_GEOMETRY_TAKE(layers)
        SINFER_VISION_GEOMETRY_TAKE(hidden)
        SINFER_VISION_GEOMETRY_TAKE(intermediate)
        SINFER_VISION_GEOMETRY_TAKE(heads)
        SINFER_VISION_GEOMETRY_TAKE(patch_dim)
        SINFER_VISION_GEOMETRY_TAKE(merge)
        SINFER_VISION_GEOMETRY_TAKE(position_embeddings)
        SINFER_VISION_GEOMETRY_TAKE(rotary_dim)
        SINFER_VISION_GEOMETRY_TAKE(output_hidden)
        SINFER_VISION_GEOMETRY_TAKE(rope_theta)
        SINFER_VISION_GEOMETRY_TAKE(norm_epsilon)
#undef SINFER_VISION_GEOMETRY_TAKE
        return g;
    }

    /// The compiled config with the artifact's declaration laid over it.
    template <class Config>
    [[nodiscard]] static VisionGeometry declared(const std::map<std::string, double>& declared) {
        VisionGeometry g = compiled<Config>();
        g.override_from(declared);
        return g;
    }

    /// Lay `declared` over this: a key present replaces the member of that name.
    void override_from(const std::map<std::string, double>& declared) {
        struct IntMember { std::string_view name; std::int32_t VisionGeometry::* value; };
        static constexpr IntMember kInts[] = {
            {"layers", &VisionGeometry::layers},
            {"hidden", &VisionGeometry::hidden},
            {"intermediate", &VisionGeometry::intermediate},
            {"heads", &VisionGeometry::heads},
            {"patch_dim", &VisionGeometry::patch_dim},
            {"merge", &VisionGeometry::merge},
            {"position_embeddings", &VisionGeometry::position_embeddings},
            {"rotary_dim", &VisionGeometry::rotary_dim},
            {"output_hidden", &VisionGeometry::output_hidden},
        };
        struct FloatMember { std::string_view name; float VisionGeometry::* value; };
        static constexpr FloatMember kFloats[] = {
            {"rope_theta", &VisionGeometry::rope_theta},
            {"norm_epsilon", &VisionGeometry::norm_epsilon},
        };
        for (const IntMember& m : kInts) {
            if (const auto it = declared.find(std::string(m.name)); it != declared.end()) {
                this->*(m.value) = static_cast<std::int32_t>(it->second);
            }
        }
        for (const FloatMember& m : kFloats) {
            if (const auto it = declared.find(std::string(m.name)); it != declared.end()) {
                this->*(m.value) = static_cast<float>(it->second);
            }
        }
    }

    [[nodiscard]] constexpr bool operator==(const VisionGeometry&) const noexcept = default;
};

} // namespace sinfer::family
