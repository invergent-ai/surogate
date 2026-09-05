#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::family {

/// The dimensions of a text model, as data.
///
/// Every target compiles one `TextConfig` of static constants, which is why the engine serves
/// one size of each family. This is the same set of numbers as a value: a target fills it from
/// its compiled config, then lets the artifact override whatever it declares, and everything
/// that plans a buffer or checks a shape reads it from here. What a variant fuses stays compiled.
/// So does the schedule, for every family that repeats a fixed interval -- but not for one whose
/// checkpoint chooses its own, which is why `attention_layer_mask` is below.
///
/// Only primary dimensions are members. Everything a target derives from them (`key_dim`,
/// `convolution_dim`, the MTP row counts) is a function here, so an artifact that declares
/// `gdn_key_heads` cannot leave a stale `key_dim` behind.
///
/// The artifact's `geometry` member is a flat object of numbers keyed by these member names.
/// Absent keys keep the compiled value, so an artifact written before the member existed loads
/// exactly as it did; unknown keys are ignored, so an artifact may name a dimension a future
/// engine reads and this one does not.
struct TextGeometry {
    std::int32_t hidden             = 0;
    /// The width of the residual stream, which is `hidden` unless the family widens it.
    std::int32_t residual           = 0;
    std::int32_t layers             = 0;
    std::int32_t intermediate       = 0;
    std::int32_t output_rows        = 0;
    std::int32_t token_domain       = 0;
    std::int32_t query_heads        = 0;
    std::int32_t kv_heads           = 0;
    std::int32_t head_dim           = 0;
    std::int32_t rotary_dim         = 0;
    std::int32_t gdn_conv_kernel    = 0;
    std::int32_t gdn_key_heads      = 0;
    std::int32_t gdn_key_head_dim   = 0;
    std::int32_t gdn_value_heads    = 0;
    std::int32_t gdn_value_head_dim = 0;
    std::int32_t mtp_layers         = 0;
    std::int32_t sliding_window     = 0;
    /// Multi-head latent attention: the ranks the query and the key/value are compressed to
    /// before being expanded per head. Zero for an attention that projects q, k and v directly.
    std::int32_t q_lora_rank        = 0;
    std::int32_t kv_lora_rank       = 0;
    /// The rank the linear mixer's forget and output gates pass through, zero for a mixer
    /// whose gates are full-rank projections.
    std::int32_t kda_gate_rank      = 0;
    /// How many copies of the model width the residual carries, for a hyper-connected stack.
    /// One for every other family, and `residual` is then `hidden`.
    std::int32_t hc_streams         = 1;
    /// The feed-forward width of the layers that are dense, where a mixture model has some.
    /// `intermediate` is the routed experts' width, which is what the post-mixer is sized from.
    std::int32_t dense_intermediate = 0;
    float rms_epsilon               = 0.0F;
    float rope_theta                = 0.0F;

    /// Which layers attend, for a family whose checkpoint chooses its own schedule instead of
    /// repeating a fixed interval. LFM2 attends at layers 2, 5, 8, 10, 12 and 14 of sixteen and
    /// convolves at the rest -- a list, not a period, and a different list at each size. Left
    /// undeclared it means nothing at all and the target's compiled schedule answers, which is
    /// what every family here did before it existed.
    ///
    /// A bitmask because the question is asked once per layer per round and has to answer in
    /// constant time, and four words because nothing this engine serves has more than 256
    /// layers. Deliberately not a member of the numeric override map below: a 64-bit mask does
    /// not survive a double, and a target that has a schedule builds it from the artifact's own
    /// blocks rather than from a number somebody would then have to keep in step by hand.
    std::array<std::uint64_t, 4> attention_layer_mask{};
    bool attention_schedule_declared = false;

    /// Record that `layer` attends. A layer outside the mask's reach is refused rather than
    /// folded onto another word: a schedule that is quietly wrong runs the wrong mixer at every
    /// round and says nothing about it.
    constexpr void declare_attention_layer(std::int32_t layer) {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= attention_layer_mask.size()) {
            throw std::out_of_range("TextGeometry attention schedule layer is out of range");
        }
        attention_layer_mask[word] |= std::uint64_t{1} << (static_cast<unsigned>(layer) % 64U);
        attention_schedule_declared = true;
    }

    /// Whether `layer` attends by the declared schedule. Meaningful only where one is declared;
    /// the runtime is what chooses between this and the compiled predicate.
    [[nodiscard]] constexpr bool layer_attends(std::int32_t layer) const noexcept {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= attention_layer_mask.size()) { return false; }
        return ((attention_layer_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    [[nodiscard]] constexpr std::int32_t query_size() const noexcept { return query_heads * head_dim; }
    [[nodiscard]] constexpr std::int32_t kv_size() const noexcept { return kv_heads * head_dim; }
    [[nodiscard]] constexpr std::int32_t query_projection_rows() const noexcept { return 2 * query_size(); }
    [[nodiscard]] constexpr std::int32_t gdn_conv_state_width() const noexcept { return gdn_conv_kernel - 1; }
    [[nodiscard]] constexpr std::int32_t key_dim() const noexcept { return gdn_key_heads * gdn_key_head_dim; }
    [[nodiscard]] constexpr std::int32_t value_dim() const noexcept { return gdn_value_heads * gdn_value_head_dim; }
    [[nodiscard]] constexpr std::int32_t convolution_dim() const noexcept { return 2 * key_dim() + value_dim(); }
    [[nodiscard]] constexpr std::int32_t mtp_input_rows() const noexcept { return 2 * hidden; }
    [[nodiscard]] constexpr std::int32_t mtp_attention_input_rows() const noexcept {
        return 2 * query_size() + 2 * kv_size();
    }
    [[nodiscard]] constexpr std::int32_t mtp_mlp_gate_up_rows() const noexcept { return 2 * intermediate; }
    /// The latent expansion's per-head halves, for an attention that compresses its key/value.
    [[nodiscard]] constexpr std::int32_t latent_key_rows() const noexcept {
        return query_heads * head_dim;
    }
    [[nodiscard]] constexpr std::int32_t hyper_connection_mix_rows() const noexcept {
        return (2 + hc_streams) * hc_streams;
    }

    /// The compiled config as a value: every primary member the target's `TextConfig` declares.
    /// A config without GDN, MTP or a window leaves those at zero, which is what "none" means.
    template <class Config>
    [[nodiscard]] static constexpr TextGeometry compiled() {
        TextGeometry g;
#define SINFER_TEXT_GEOMETRY_TAKE(member) \
        if constexpr (requires { Config::member; }) { g.member = Config::member; }
        SINFER_TEXT_GEOMETRY_TAKE(hidden)
        if constexpr (requires { Config::residual; }) {
            g.residual = Config::residual;
        } else {
            g.residual = Config::hidden;
        }
        SINFER_TEXT_GEOMETRY_TAKE(layers)
        SINFER_TEXT_GEOMETRY_TAKE(intermediate)
        SINFER_TEXT_GEOMETRY_TAKE(output_rows)
        SINFER_TEXT_GEOMETRY_TAKE(token_domain)
        SINFER_TEXT_GEOMETRY_TAKE(query_heads)
        SINFER_TEXT_GEOMETRY_TAKE(kv_heads)
        SINFER_TEXT_GEOMETRY_TAKE(head_dim)
        SINFER_TEXT_GEOMETRY_TAKE(rotary_dim)
        SINFER_TEXT_GEOMETRY_TAKE(gdn_conv_kernel)
        SINFER_TEXT_GEOMETRY_TAKE(gdn_key_heads)
        SINFER_TEXT_GEOMETRY_TAKE(gdn_key_head_dim)
        SINFER_TEXT_GEOMETRY_TAKE(gdn_value_heads)
        SINFER_TEXT_GEOMETRY_TAKE(gdn_value_head_dim)
        SINFER_TEXT_GEOMETRY_TAKE(mtp_layers)
        SINFER_TEXT_GEOMETRY_TAKE(sliding_window)
        SINFER_TEXT_GEOMETRY_TAKE(q_lora_rank)
        SINFER_TEXT_GEOMETRY_TAKE(kv_lora_rank)
        SINFER_TEXT_GEOMETRY_TAKE(kda_gate_rank)
        SINFER_TEXT_GEOMETRY_TAKE(hc_streams)
        SINFER_TEXT_GEOMETRY_TAKE(dense_intermediate)
        SINFER_TEXT_GEOMETRY_TAKE(rms_epsilon)
        SINFER_TEXT_GEOMETRY_TAKE(rope_theta)
#undef SINFER_TEXT_GEOMETRY_TAKE
        return g;
    }

    /// The compiled config with the artifact's declaration laid over it.
    template <class Config>
    [[nodiscard]] static TextGeometry declared(const std::map<std::string, double>& declared) {
        TextGeometry g = compiled<Config>();
        g.override_from(declared);
        // A family that does not widen its residual stream has one exactly as wide as its
        // hidden state, at every size. Deriving it here rather than making every artifact
        // restate it keeps a declaration that names `hidden` alone from leaving the residual
        // at the compiled width -- which is a mismatch the first embedding lookup finds.
        if constexpr (!requires { Config::residual; }) { g.residual = g.hidden; }
        return g;
    }

    /// Lay `declared` over this: a key present replaces the member of that name.
    void override_from(const std::map<std::string, double>& declared) {
        struct IntMember { std::string_view name; std::int32_t TextGeometry::* value; };
        static constexpr IntMember kInts[] = {
            {"hidden", &TextGeometry::hidden},
            {"residual", &TextGeometry::residual},
            {"layers", &TextGeometry::layers},
            {"intermediate", &TextGeometry::intermediate},
            {"output_rows", &TextGeometry::output_rows},
            {"token_domain", &TextGeometry::token_domain},
            {"query_heads", &TextGeometry::query_heads},
            {"kv_heads", &TextGeometry::kv_heads},
            {"head_dim", &TextGeometry::head_dim},
            {"rotary_dim", &TextGeometry::rotary_dim},
            {"gdn_conv_kernel", &TextGeometry::gdn_conv_kernel},
            {"gdn_key_heads", &TextGeometry::gdn_key_heads},
            {"gdn_key_head_dim", &TextGeometry::gdn_key_head_dim},
            {"gdn_value_heads", &TextGeometry::gdn_value_heads},
            {"gdn_value_head_dim", &TextGeometry::gdn_value_head_dim},
            {"mtp_layers", &TextGeometry::mtp_layers},
            {"sliding_window", &TextGeometry::sliding_window},
            {"q_lora_rank", &TextGeometry::q_lora_rank},
            {"kv_lora_rank", &TextGeometry::kv_lora_rank},
            {"kda_gate_rank", &TextGeometry::kda_gate_rank},
            {"hc_streams", &TextGeometry::hc_streams},
            {"dense_intermediate", &TextGeometry::dense_intermediate},
        };
        struct FloatMember { std::string_view name; float TextGeometry::* value; };
        static constexpr FloatMember kFloats[] = {
            {"rms_epsilon", &TextGeometry::rms_epsilon},
            {"rope_theta", &TextGeometry::rope_theta},
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

    [[nodiscard]] constexpr bool operator==(const TextGeometry&) const noexcept = default;
};

} // namespace sinfer::family
