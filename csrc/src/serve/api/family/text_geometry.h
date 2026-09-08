#pragma once

#include "core/dtype.h"
#include <api/family/dflash_geometry.h>

#include <api/family/geometry_validation.h>
#include <api/family/linear_storage.h>

#include <array>
#include <cstdint>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sinfer::family {

/// Validated checkpoint dimensions and execution settings, shared by loading and planning.
/// Derived widths are computed from primary fields. Required fields and the explicit layer
/// schedule are checked by the target resolver before this value reaches the runtime.
struct TextGeometry {
    DFlashGeometry dflash;
    /// Storage-dependent planning uses the artifact's matrices, independently of its profile ID.
    std::map<std::string, LinearStorage, std::less<>> linear_storage;
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
    // Optional contiguous rotary prefix for windowed layers; zero inherits rotary_dim.
    std::int32_t sliding_rotary_dim = 0;
    std::int32_t residual_fp32 = 0;
    [[nodiscard]] DType residual_dtype() const noexcept {
        return residual_fp32 ? DType::FP32 : DType::BF16;
    }
    std::int32_t gdn_conv_kernel    = 0;
    std::int32_t gdn_key_heads      = 0;
    std::int32_t gdn_key_head_dim   = 0;
    std::int32_t gdn_value_heads    = 0;
    std::int32_t gdn_value_head_dim = 0;
    std::int32_t mtp_layers         = 0;
    std::int32_t draft_vocab        = 0;
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
    std::int32_t experts               = 0;
    std::int32_t experts_per_token     = 0;
    std::int32_t shared_intermediate   = 0;
    std::int32_t qk_head_dim           = 0;
    std::int32_t v_head_dim            = 0;
    std::int32_t max_context           = 0;
    std::int32_t kv_shared_layers      = 0;
    std::int32_t attention_k_eq_v      = 0;
    std::int32_t leading_dense_layers  = 0;
    std::int32_t hc_sinkhorn_iterations = 0;
    std::int32_t hc_low_rank = 0;
    std::int32_t indexer_heads = 0;
    std::int32_t indexer_head_dim = 0;
    std::int32_t indexer_top_k = 0;
    std::int32_t indexer_block = 0;
    std::int32_t ple_layer = 0;
    std::int32_t ple_ngram = 0;
    std::int32_t ple_heads_per_ngram = 0;
    std::int32_t ple_head_dim = 0;
    std::int32_t ple_conv_kernel = 0;
    std::int32_t ple_table_rows = 0;
    std::int32_t ple_eos_token = 0;
    std::int32_t ple_image_token = 0;
    float hc_epsilon                  = 0.0F;
    /// Positive magnitude of a KDA gate's negative lower bound.
    float kda_gate_bound              = 0.0F;
    float attention_scale             = 0.0F;
    float gdn_scale                   = 0.0F;
    float routed_scale                = 0.0F;
    float swiglu_limit                = 0.0F;
    float rms_epsilon               = 0.0F;
    float rope_theta                = 0.0F;
    /// What the embedding lookup is multiplied by before the first block, where a family
    /// scales it. Gemma's is `sqrt(hidden)` rounded to bf16 as the reference rounds it, which
    /// is 62.0 at hidden 3840 and 73.5 at 5376 -- so a target serving two sizes cannot compile
    /// it, and reads it from here. Zero means no embedding scaling.
    float embedding_scale           = 0.0F;
    /// The bound logits are squashed to, `tanh(x / c) * c`, for a family that caps them.
    /// Gemma 4 caps at 30. Zero means no cap, which is every other family here.
    float logit_softcap             = 0.0F;
    /// The rope base the *windowed* layers rotate at, where a family rotates its two kinds of
    /// layer at different bases -- Gemma's windowed layers at 1e4 against 1e6 global, which is
    /// `rope_theta` above. Zero means one base for every layer.
    float sliding_rope_theta        = 0.0F;

    /// The explicit layer schedule from the checkpoint. Resolvers require it before planning
    /// or execution; a missing schedule cannot select a target preset.
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
        return global_head_dim > 0 && global_kv_heads > 0;
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
    [[nodiscard]] constexpr std::int32_t linear_layer_count() const noexcept {
        std::int32_t count = 0;
        for (std::int32_t layer = 0; layer < layers; ++layer) { count += !layer_attends(layer); }
        return count;
    }
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

    /// Read a complete checkpoint specification without a compiled fallback.
    [[nodiscard]] static TextGeometry resolved(const std::map<std::string, double>& values,
                                               std::span<const std::string> layer_types) {
        TextGeometry g;
        g.override_from(values);
        constexpr const char* required[] = {
#define SINFER_GEOMETRY_REQUIRED(name) #name,
#include "required_text_geometry_fields.inc"
#undef SINFER_GEOMETRY_REQUIRED
        };
        for (const char* name : required) {
            const auto found = values.find(name);
            if (found == values.end()) {
                throw std::invalid_argument(std::string("missing geometry.") + name +
                                            "; rebuild the serving cache");
            }
            if (std::string_view(name) != "rotary_dim" && std::string_view(name) != "rope_theta" &&
                found->second <= 0.0) {
                throw std::invalid_argument(std::string("geometry.") + name + " must be positive");
            }
        }
        if (g.token_domain > g.output_rows || g.query_heads % g.kv_heads != 0 ||
            g.rotary_dim > g.head_dim || g.rotary_dim % 2 != 0 ||
            (g.rotary_dim > 0 && g.rope_theta <= 0.0F)) {
            throw std::invalid_argument("inconsistent checkpoint text geometry");
        }
        if (g.residual_fp32 != 0 && g.residual_fp32 != 1) {
            throw std::invalid_argument("residual_fp32 must be 0 or 1");
        }
        if (g.sliding_rotary_dim > g.head_dim || g.sliding_rotary_dim % 2 ||
            (g.sliding_rotary_dim > 0 && (g.sliding_window <= 0 || g.sliding_rope_theta <= 0))) {
            throw std::invalid_argument("invalid windowed rotary geometry");
        }
        if (g.layers > 256) { throw std::invalid_argument("serving supports at most 256 layers"); }
        for (const auto heads : {g.query_heads, g.kv_heads}) {
            if (static_cast<std::int64_t>(heads) * g.head_dim > INT32_MAX) {
                throw std::invalid_argument("attention projection width exceeds int32");
            }
        }
        g.apply_layer_types(layer_types);
        return g;
    }

    void apply_layer_types(std::span<const std::string> layer_types) {
        if (layers <= 0 || layers > 256) {
            throw std::invalid_argument("layer_types requires geometry.layers in [1,256]");
        }
        if (layer_types.size() != static_cast<std::size_t>(layers)) {
            throw std::invalid_argument("layer_types must contain one entry per geometry.layers");
        }
        for (const auto& kind : layer_types) {
            if (kind != "full_attention" && kind != "sliding_attention" && kind != "linear_attention") {
                throw std::invalid_argument("layer_types contains an unsupported layer type");
            }
            if (kind == "sliding_attention" && sliding_window <= 0) {
                throw std::invalid_argument("sliding_attention requires a positive geometry.sliding_window");
            }
        }
        attention_layer_mask = {};
        windowed_layer_mask = {};
        attention_schedule_declared = true;
        windowed_schedule_declared = true;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            if (layer_types[layer] != "linear_attention") { declare_attention_layer(layer); }
            if (layer_types[layer] == "sliding_attention") { declare_windowed_layer(layer); }
        }
    }

    [[nodiscard]] static TextGeometry resolved_hybrid(const std::map<std::string, double>& values,
                                                     std::span<const std::string> layer_types,
                                                     bool mixture = false, bool hyper_connected = false) {
        auto g = mixture ? resolved_moe(values, layer_types) : resolved(values, layer_types);
        for (const auto key : {"gdn_key_heads", "gdn_key_head_dim", "gdn_value_heads",
                               "gdn_value_head_dim", "gdn_conv_kernel", "gdn_scale", "draft_vocab"}) {
            if (!values.contains(key) || (values.at(key) <= 0 && !(hyper_connected && std::string_view(key) == "draft_vocab"))) {
                throw std::invalid_argument(std::string("missing or invalid geometry.") + key);
            }
        }
        if (!values.contains("mtp_layers") || g.mtp_layers > 1 || g.draft_vocab > g.token_domain ||
            g.gdn_value_heads % g.gdn_key_heads || g.rotary_dim != 64 || (!hyper_connected && g.residual != g.hidden)) {
            throw std::invalid_argument("unsupported hybrid checkpoint geometry");
        }
        if (g.gdn_conv_kernel != 4) {
            throw std::invalid_argument("hybrid GDN convolution currently requires four taps");
        }
        if (mixture && (!values.contains("shared_intermediate") || g.shared_intermediate <= 0 ||
                        !values.contains("routed_scale") || g.routed_scale != 1.0F)) {
            throw std::invalid_argument("missing or unsupported hybrid shared expert geometry");
        }
        for (const auto& kind : layer_types) {
            if (kind == "sliding_attention") { throw std::invalid_argument("hybrid target does not implement sliding attention"); }
        }
        if (2LL * g.query_size() + 2LL * g.kv_size() > INT32_MAX || 2LL * g.shared_intermediate > INT32_MAX) {
            throw std::invalid_argument("hybrid projection rows exceed int32");
        }
        const auto key_width = std::int64_t(g.gdn_key_heads) * g.gdn_key_head_dim;
        const auto value_width = std::int64_t(g.gdn_value_heads) * g.gdn_value_head_dim;
        const auto max_rows = std::numeric_limits<std::int32_t>::max();
        if (key_width > max_rows / 2 || value_width > max_rows / 2 ||
            2 * key_width + 2 * value_width > max_rows) {
            throw std::invalid_argument("hybrid projection rows exceed int32");
        }
        return g;
    }

    [[nodiscard]] std::int32_t ple_heads() const { return ple_ngram ? (ple_ngram - 1) * ple_heads_per_ngram : 0; }
    [[nodiscard]] std::int32_t ple_embed() const { return ple_heads() * ple_head_dim; }
    [[nodiscard]] std::int32_t ple_conv_history() const { return ple_ngram * (ple_conv_kernel - 1); }
    [[nodiscard]] std::int32_t ple_table_row_bytes() const { return ple_head_dim / 32 * 18; }

    [[nodiscard]] static TextGeometry resolved_qwen4exp(
        const std::map<std::string, double>& values, std::span<const std::string> layer_types) {
        auto g = resolved_hybrid(values, layer_types, true, true);
        for (const auto name : {"hc_streams", "hc_low_rank", "indexer_heads", "indexer_head_dim",
                                "indexer_top_k", "indexer_block"}) {
            if (!values.contains(name) || values.at(name) <= 0) {
                throw std::invalid_argument(std::string("missing or invalid geometry.") + name);
            }
        }
        for (const auto name : {"ple_layer", "ple_ngram", "ple_heads_per_ngram", "ple_head_dim",
                                "ple_conv_kernel", "ple_table_rows", "ple_eos_token", "ple_image_token"}) {
            if (!values.contains(name)) {
                throw std::invalid_argument(std::string("missing geometry.") + name);
            }
        }
        if (g.hc_streams > 8 || g.hidden % 8 || g.hc_low_rank % 8 ||
            std::int64_t(g.hidden) * g.hc_streams != g.residual) {
            throw std::invalid_argument("unsupported hyper-connection geometry");
        }
        // These are kernel capabilities, not checkpoint dimension defaults.
        if (g.indexer_head_dim != 128 || g.indexer_block != 4 ||
            (g.indexer_heads != 4 && g.indexer_heads != 8) ||
            std::int64_t(g.indexer_top_k) + g.indexer_block - 1 > INT32_MAX ||
            std::int64_t(g.max_context) + g.indexer_block - 1 > INT32_MAX) {
            throw std::invalid_argument("unsupported QSA indexer geometry");
        }
        if (g.draft_vocab != 0) { throw std::invalid_argument("qwen4exp does not use a shortlist head"); }
        if (g.ple_ngram) {
            if (g.ple_ngram < 2 || g.ple_ngram > 3 || g.ple_heads_per_ngram <= 0 ||
                std::int64_t(g.ple_ngram - 1) * g.ple_heads_per_ngram > 16 ||
                g.ple_layer >= g.layers || g.ple_head_dim <= 0 || g.ple_head_dim % 32 ||
                g.ple_conv_kernel <= 0 || g.ple_table_rows <= 0 ||
                g.ple_eos_token >= g.output_rows || g.ple_image_token >= g.output_rows ||
                std::int64_t(g.ple_heads()) * g.ple_head_dim > INT32_MAX ||
                std::int64_t(g.ple_ngram) * (g.ple_conv_kernel - 1) > INT32_MAX) {
                throw std::invalid_argument("unsupported PLE geometry");
            }
        } else if (g.ple_layer || g.ple_heads_per_ngram || g.ple_head_dim || g.ple_conv_kernel ||
                   g.ple_table_rows || g.ple_eos_token || g.ple_image_token) {
            throw std::invalid_argument("PLE dimensions require an active PLE layer");
        }
        return g;
    }

    [[nodiscard]] static TextGeometry resolved_moe(const std::map<std::string, double>& values,
                                                   std::span<const std::string> layer_types) {
        auto g = resolved(values, layer_types);
        if (g.experts <= 0 || g.experts_per_token <= 0 || g.experts_per_token > g.experts) {
            throw std::invalid_argument("missing or invalid expert geometry; rebuild the serving cache");
        }
        if (static_cast<std::int64_t>(g.experts) * 2 * g.intermediate > INT32_MAX ||
            static_cast<std::int64_t>(g.experts) * g.hidden > INT32_MAX) {
            throw std::invalid_argument("routed expert projection width exceeds int32");
        }
        return g;
    }

    [[nodiscard]] static TextGeometry resolved_gemma3(
        const std::map<std::string, double>& values, std::span<const std::string> layer_types) {
        auto g = resolved(values, layer_types);
        for (const auto name : {"sliding_window", "sliding_rope_theta", "embedding_scale"}) {
            if (!values.contains(name) || values.at(name) <= 0) {
                throw std::invalid_argument(std::string("missing or invalid geometry.") + name);
            }
        }
        for (const auto& kind : layer_types) {
            if (kind == "linear_attention") { throw std::invalid_argument("Gemma3 requires attention at every layer"); }
        }
        return g;
    }

    [[nodiscard]] static TextGeometry resolved_gemma4(
        const std::map<std::string, double>& values, std::span<const std::string> layer_types,
        bool per_layer_inputs = false, bool mixture = false) {
        auto g = mixture ? resolved_moe(values, layer_types) : resolved(values, layer_types);
        const auto require = [&](const char* name, bool positive = true) {
            const auto found = values.find(name);
            if (found == values.end() || (positive && found->second <= 0.0)) {
                throw std::invalid_argument(std::string("missing or invalid geometry.") + name +
                                            "; rebuild the serving cache");
            }
        };
        for (const char* name : {"global_head_dim", "global_kv_heads", "sliding_rope_theta",
                                 "embedding_scale", "sliding_window", "global_rotary_angles"}) { require(name); }
        for (const char* name : {"attention_k_eq_v", "logit_softcap"}) {
            require(name, false);
        }
        if (g.attention_scale != 1.0F || g.attention_k_eq_v > 1 ||
            g.query_heads % g.global_kv_heads || g.global_head_dim % 2 ||
            g.rotary_dim != g.head_dim || g.global_rotary_angles > g.global_head_dim / 2) {
            throw std::invalid_argument("invalid Gemma 4 attention configuration");
        }
        for (const auto heads : {g.query_heads, g.global_kv_heads}) {
            if (static_cast<std::int64_t>(heads) * g.global_head_dim > INT32_MAX) {
                throw std::invalid_argument("global attention projection width exceeds int32");
            }
        }
        for (std::int32_t layer = 0; layer < g.layers; ++layer) {
            if (!g.layer_attends(layer)) {
                throw std::invalid_argument("Gemma 4 does not support linear attention layers");
            }
        }
        if (mixture) { require("dense_intermediate"); }
        if (per_layer_inputs) {
            for (const char* name : {"per_layer_input_dim", "per_layer_vocab",
                                     "shared_kv_intermediate"}) { require(name); }
            require("kv_shared_layers", false);
            if (g.kv_shared_layers >= g.layers || g.per_layer_vocab < g.token_domain ||
                static_cast<std::int64_t>(g.layers) * g.per_layer_input_dim > INT32_MAX) {
                throw std::invalid_argument("invalid Gemma 4 per-layer input configuration");
            }
            g.declare_kv_sharing();
            bool owner_types[2] = {false, false};
            const auto first_shared = g.layers - g.kv_shared_layers;
            for (std::int32_t layer = 0; layer < g.layers; ++layer) {
                const auto kind = g.layer_is_windowed(layer) ? 1 : 0;
                if (layer < first_shared) {
                    g.declare_kv_owner(layer);
                    owner_types[kind] = true;
                } else if (!owner_types[kind]) {
                    throw std::invalid_argument("shared KV layer has no owner of its attention type");
                }
            }
        }
        return g;
    }

    /// Lay `declared` over this: a key present replaces the member of that name.
    void override_from(const std::map<std::string, double>& declared) {
        struct IntMember { std::string_view name; std::int32_t TextGeometry::* value; };
        static constexpr IntMember kInts[] = {
#define SINFER_GEOMETRY_INT(name) {#name, &TextGeometry::name},
#define SINFER_GEOMETRY_FLOAT(name)
#include "text_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        struct FloatMember { std::string_view name; float TextGeometry::* value; };
        static constexpr FloatMember kFloats[] = {
#define SINFER_GEOMETRY_FLOAT(name) {#name, &TextGeometry::name},
#define SINFER_GEOMETRY_INT(name)
#include "text_geometry_fields.inc"
#undef SINFER_GEOMETRY_INT
#undef SINFER_GEOMETRY_FLOAT
        };
        detail::apply_geometry(*this, declared, kInts, kFloats, "geometry");
    }

    [[nodiscard]] bool operator==(const TextGeometry&) const noexcept = default;
};

} // namespace sinfer::family
