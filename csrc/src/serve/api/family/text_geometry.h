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
    /// The *second* attention geometry, for a family whose global layers are shaped
    /// differently from its windowed ones. Gemma 4 attends through its window with 8 key/value
    /// heads of 256 and over the whole context with 1 head of 512; the members above describe
    /// the windowed layers, these the global ones.
    ///
    /// Zero means "not stated", and every family that attends one way leaves them zero, which
    /// is what makes `head_dim_for` answer with the single geometry for all of them. Read
    /// these through the `*_for(windowed)` accessors rather than directly: a site that reads
    /// `head_dim` where it meant "this layer's head dim" is the failure this exists to
    /// prevent, and it is silent -- a global layer's cache sized for a windowed one.
    std::int32_t global_head_dim    = 0;
    std::int32_t global_kv_heads    = 0;
    /// How many of a global head's angle pairs carry a non-zero rope frequency. Gemma 4's
    /// proportional rope rotates 64 of a 512-wide head's 256 pairs and leaves the rest at
    /// zero, which is the identity -- so the head rotates over its whole width with an
    /// inert tail, not over a contiguous prefix. Zero means every pair rotates.
    std::int32_t global_rotary_angles = 0;
    /// The per-token, per-layer input a family mixes into every block: how wide one layer's
    /// slice is, and the vocabulary of the table it is looked up in. Zero for a family that
    /// has none, which is every one here but Gemma 4's E-series.
    std::int32_t per_layer_input_dim  = 0;
    std::int32_t per_layer_vocab      = 0;
    /// The feed-forward width of the layers that share their key/value planes, where a family
    /// widens exactly those. Gemma 4's E2B holds 12,288 there against 6,144 elsewhere; its E4B
    /// leaves the flag off and the two coincide. Zero means the model has one width.
    std::int32_t shared_kv_intermediate = 0;
    float rms_epsilon               = 0.0F;
    float rope_theta                = 0.0F;
    /// What the embedding lookup is multiplied by before the first block, where a family
    /// scales it. Gemma's is `sqrt(hidden)` rounded to bf16 as the reference rounds it, which
    /// is 62.0 at hidden 3840 and 73.5 at 5376 -- so a target serving two sizes cannot compile
    /// it, and reads it from here. Zero means the target's own compiled value stands.
    float embedding_scale           = 0.0F;
    /// The bound logits are squashed to, `tanh(x / c) * c`, for a family that caps them.
    /// Gemma 4 caps at 30. Zero means no cap, which is every other family here.
    float logit_softcap             = 0.0F;
    /// The rope base the *windowed* layers rotate at, where a family rotates its two kinds of
    /// layer at different bases -- Gemma's windowed layers at 1e4 against 1e6 global, which is
    /// `rope_theta` above. Zero means one base for every layer.
    float sliding_rope_theta        = 0.0F;

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

    /// Which attending layers look through the window rather than at the whole context.
    ///
    /// A second mask, and for a reason the first one does not cover: Gemma 3 compiles this
    /// schedule because its target serves one size, but a target serving two sizes cannot --
    /// Gemma 4's 12B has 48 layers where its 31B has 60, and one compiled array cannot be
    /// both. So the target reads it off the artifact, the way LFM2 reads which layers attend,
    /// and for the same reason: the checkpoint states it and a period that had to be guessed
    /// would be guessed wrong in silence.
    ///
    /// Not in the numeric override map, exactly as `attention_layer_mask` is not: a mask does
    /// not survive a double, and the target builds this from the artifact's own objects
    /// rather than from a number somebody would have to keep in step by hand.
    std::array<std::uint64_t, 4> windowed_layer_mask{};
    bool windowed_schedule_declared = false;

    constexpr void declare_windowed_layer(std::int32_t layer) {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= windowed_layer_mask.size()) {
            throw std::out_of_range("TextGeometry window schedule layer is out of range");
        }
        windowed_layer_mask[word] |= std::uint64_t{1} << (static_cast<unsigned>(layer) % 64U);
        windowed_schedule_declared = true;
    }

    /// Whether `layer` looks through the window by the declared schedule. Meaningful only
    /// where one is declared; the runtime chooses between this and the compiled predicate.
    [[nodiscard]] constexpr bool layer_is_windowed(std::int32_t layer) const noexcept {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= windowed_layer_mask.size()) { return false; }
        return ((windowed_layer_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    /// Which layers hold key and value planes of their own.
    ///
    /// Gemma 4's E-series ends in a run of layers that project a query and nothing else: they
    /// attend over an *earlier* layer's keys and values, the last one before the run that
    /// attends the same way they do. E2B shares 20 of its 35 layers. Every other family here
    /// gives every attending layer its own planes and leaves this undeclared, which is what
    /// makes `layer_owns_kv` answer true for all of them.
    ///
    /// A mask rather than a count because the run is not the only thing that matters -- which
    /// layer a sharer reads depends on the window schedule, so the two masks are read together.
    std::array<std::uint64_t, 4> kv_owner_mask{};
    bool kv_sharing_declared = false;

    constexpr void declare_kv_owner(std::int32_t layer) {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= kv_owner_mask.size()) {
            throw std::out_of_range("TextGeometry key/value owner layer is out of range");
        }
        kv_owner_mask[word] |= std::uint64_t{1} << (static_cast<unsigned>(layer) % 64U);
    }

    /// Record that the model shares key/value planes at all. Separate from marking an owner,
    /// because a model where *every* layer owns its planes still has to say so -- otherwise
    /// an all-ones mask is indistinguishable from an undeclared one.
    constexpr void declare_kv_sharing() { kv_sharing_declared = true; }

    /// Whether `layer` writes its own keys and values. True everywhere no sharing is declared.
    [[nodiscard]] constexpr bool layer_owns_kv(std::int32_t layer) const noexcept {
        if (!kv_sharing_declared) { return true; }
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= kv_owner_mask.size()) { return false; }
        return ((kv_owner_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    [[nodiscard]] constexpr std::int32_t query_size() const noexcept { return query_heads * head_dim; }
    [[nodiscard]] constexpr std::int32_t kv_size() const noexcept { return kv_heads * head_dim; }

    /// Whether this model attends at two different head geometries. False for every family
    /// but Gemma 4, and false for a Gemma 4 whose two geometries happen to coincide -- the
    /// question a caller asks is "must I distinguish?", not "is a second one declared?".
    [[nodiscard]] constexpr bool has_global_attention_geometry() const noexcept {
        return global_head_dim > 0 && global_kv_heads > 0
               && (global_head_dim != head_dim || global_kv_heads != kv_heads);
    }

    /// This layer's attention geometry, by whether the layer attends through the window.
    /// A model that states one geometry answers with it either way, so a caller need not
    /// know whether the family it is serving has two.
    [[nodiscard]] constexpr std::int32_t head_dim_for(bool windowed) const noexcept {
        return (windowed || global_head_dim <= 0) ? head_dim : global_head_dim;
    }
    [[nodiscard]] constexpr std::int32_t kv_heads_for(bool windowed) const noexcept {
        return (windowed || global_kv_heads <= 0) ? kv_heads : global_kv_heads;
    }
    [[nodiscard]] constexpr std::int32_t query_size_for(bool windowed) const noexcept {
        return query_heads * head_dim_for(windowed);
    }
    [[nodiscard]] constexpr std::int32_t kv_size_for(bool windowed) const noexcept {
        return kv_heads_for(windowed) * head_dim_for(windowed);
    }

    /// This layer's feed-forward width. Wider on a layer that shares its key/value planes,
    /// where the family widens those; the same everywhere else.
    [[nodiscard]] constexpr std::int32_t intermediate_for(bool owns_kv) const noexcept {
        return (owns_kv || shared_kv_intermediate <= 0) ? intermediate : shared_kv_intermediate;
    }
    /// The widest feed-forward any layer holds, which a shared plane must be sized for.
    [[nodiscard]] constexpr std::int32_t maximum_intermediate() const noexcept {
        return shared_kv_intermediate > intermediate ? shared_kv_intermediate : intermediate;
    }
    /// The widest of this model's attention geometries, which is what a buffer shared by
    /// every layer has to be sized for.
    [[nodiscard]] constexpr std::int32_t maximum_query_size() const noexcept {
        return query_size_for(true) > query_size_for(false) ? query_size_for(true)
                                                            : query_size_for(false);
    }
    [[nodiscard]] constexpr std::int32_t maximum_kv_size() const noexcept {
        return kv_size_for(true) > kv_size_for(false) ? kv_size_for(true) : kv_size_for(false);
    }
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
        SINFER_TEXT_GEOMETRY_TAKE(global_head_dim)
        SINFER_TEXT_GEOMETRY_TAKE(global_kv_heads)
        SINFER_TEXT_GEOMETRY_TAKE(global_rotary_angles)
        SINFER_TEXT_GEOMETRY_TAKE(per_layer_input_dim)
        SINFER_TEXT_GEOMETRY_TAKE(per_layer_vocab)
        SINFER_TEXT_GEOMETRY_TAKE(shared_kv_intermediate)
        SINFER_TEXT_GEOMETRY_TAKE(rms_epsilon)
        SINFER_TEXT_GEOMETRY_TAKE(rope_theta)
        SINFER_TEXT_GEOMETRY_TAKE(embedding_scale)
        SINFER_TEXT_GEOMETRY_TAKE(logit_softcap)
        SINFER_TEXT_GEOMETRY_TAKE(sliding_rope_theta)
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
            {"global_head_dim", &TextGeometry::global_head_dim},
            {"global_kv_heads", &TextGeometry::global_kv_heads},
            {"global_rotary_angles", &TextGeometry::global_rotary_angles},
            {"per_layer_input_dim", &TextGeometry::per_layer_input_dim},
            {"per_layer_vocab", &TextGeometry::per_layer_vocab},
            {"shared_kv_intermediate", &TextGeometry::shared_kv_intermediate},
        };
        struct FloatMember { std::string_view name; float TextGeometry::* value; };
        static constexpr FloatMember kFloats[] = {
            {"rms_epsilon", &TextGeometry::rms_epsilon},
            {"rope_theta", &TextGeometry::rope_theta},
            {"embedding_scale", &TextGeometry::embedding_scale},
            {"logit_softcap", &TextGeometry::logit_softcap},
            {"sliding_rope_theta", &TextGeometry::sliding_rope_theta},
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
