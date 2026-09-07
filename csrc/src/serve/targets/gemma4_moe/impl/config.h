#pragma once

#include <api/family/frontend.h>
#include <api/family/hybrid_topology.h>
#include <api/family/vision.h>
#include "api/ops/sparse_moe.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace sinfer::targets::gemma4_moe::detail {

/// The mixture Gemma 4: `google/gemma-4-26B-A4B` and its instruction tune.
///
/// The attention is the dense target's, exactly: two head geometries -- 16 query heads over
/// 8 key/value heads of 256 through the window, over 2 of 512 globally -- with
/// `attention_k_eq_v` dropping the value projection from the global layers. Read that
/// target's `config.h` for what those numbers mean; nothing about them is different here.
///
/// The feed-forward is what makes this a target of its own. **Every** layer runs a
/// 2,112-wide dense feed-forward *and* 128 routed experts of 704 over the same input, and
/// sums them under three norms a dense Gemma 4 layer does not have. That is not the shape
/// the other routed targets have, where a layer is dense *or* routed, so `intermediate`
/// below is the routed experts' width (the family's convention for a mixture) and
/// `dense_intermediate` the branch beside them.
inline constexpr ops::SparseMoeGeometry kMoeGeometry = ops::kSparseMoeGemma4Geometry;

struct TextConfig {
    static constexpr int hidden       = kMoeGeometry.hidden;
    static constexpr int layers       = 30;
    /// The routed experts' width.
    static constexpr int intermediate = kMoeGeometry.intermediate;
    /// The dense feed-forward that runs beside them, on every layer.
    static constexpr int dense_intermediate = 2112;

    static constexpr int output_rows  = 262144;
    static constexpr int token_domain = 262144;

    // No linear mixer; declared because the shared runtime sizes an empty state from them.
    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    /// The windowed layers' geometry.
    static constexpr int query_heads = 16;
    static constexpr int kv_heads    = 8;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 256;

    /// The global layers'. Two key/value heads here where the dense 12B has one.
    static constexpr int global_head_dim      = 512;
    static constexpr int global_kv_heads      = 2;
    /// `partial_rotary_factor` 0.25 of a 512-wide head: 64 of its 256 angle pairs rotate.
    static constexpr int global_rotary_angles = 64;

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e6F;

    static constexpr int sliding_window       = 1024;
    static constexpr float sliding_rope_theta = 1.0e4F;

    /// The 26B-A4B's schedule: every sixth layer global, the last one included. An
    /// artifact's own schedule wins, and every Gemma 4 artifact declares one; this stands
    /// for one that does not.
    static constexpr std::array<bool, layers> kWindowedAttention{
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
    };

    [[nodiscard]] static constexpr bool is_windowed_attention(int layer) {
        return kWindowedAttention[static_cast<std::size_t>(layer)];
    }

    [[nodiscard]] static constexpr float layer_rope_theta(int layer) {
        return is_windowed_attention(layer) ? sliding_rope_theta : rope_theta;
    }

    /// `sqrt(hidden)` rounded to bf16, as the reference rounds it: 53.0 at hidden 2,816.
    /// The artifact states it too; this is what stands for an artifact that does not.
    static constexpr float embedding_scale = 5.3e1F;

    /// Logits are squashed to `tanh(x / c) * c`. Gemma 4 caps at 30.
    static constexpr float logit_softcap = 3.0e1F;

    static constexpr int key_dim               = 0;
    static constexpr int value_dim             = 0;
    static constexpr int convolution_dim       = 0;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    /// The mixture, named as the other routed targets name it.
    static constexpr int experts           = kMoeGeometry.experts;
    static constexpr int experts_per_token = kMoeGeometry.experts_per_token;
    static constexpr int router_rows       = kMoeGeometry.router_rows();
    /// Layers with routed experts: every one of them, which is what the expert slot cache is
    /// sized against.
    static constexpr int expert_layers     = layers;

    /// Every layer attends -- "full attention" is the *mixer* axis, and both a windowed and a
    /// global layer are ordinary attention on it.
    [[nodiscard]] static constexpr bool is_full_attention(int) { return true; }
    [[nodiscard]] static constexpr int full_attention_layers() { return layers; }
    [[nodiscard]] static constexpr int gdn_layers() { return 0; }
    [[nodiscard]] static constexpr int full_attention_index(int layer) { return layer; }
    [[nodiscard]] static constexpr int gdn_index(int) { return -1; }
};

static_assert(TextConfig::full_attention_layers() == 30);
static_assert(TextConfig::router_rows == TextConfig::experts,
              "this mixture routes every token; a router with a shared-expert gate row would "
              "be read one row past its end");
static_assert(!kMoeGeometry.has_shared(),
              "the dense feed-forward beside the experts is the layer's own projection, not "
              "this mixture's always-on expert -- an op that thought otherwise would add it "
              "twice and read a router row that is not there");
static_assert(kMoeGeometry.activation == ops::GatedActivation::GeluTanh,
              "Gemma 4's experts are GELU-gated like the rest of the model; served through "
              "SiLU they compute a different function and nothing would raise");
static_assert(!TextConfig::kWindowedAttention[TextConfig::layers - 1],
              "the last layer is global in every published Gemma 4; `transformers` forces it "
              "and a schedule that ended windowed would cap the model's context at its window");

struct VisionConfig : family::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

struct DFlashConfig {
    static constexpr bool supported     = false;
    static constexpr int local_layers   = 0;
    static constexpr int local_capacity = 0;
    static constexpr int kv_heads       = 0;
    static constexpr int head_dim       = 0;
    static constexpr int feature_rows   = 0;
    static constexpr int hidden         = 0;
    static constexpr int intermediate   = 0;
    static constexpr int query_size     = 0;
    static constexpr int kv_size        = 0;
};

/// **One, not `1/sqrt(head_dim)`.** As on the dense target: Gemma 4 sets `scaling = 1.0` and
/// relies on its QK-norm for unit-RMS queries and keys.
inline constexpr float kAttentionScale                   = 1.0F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 262144;

/// What the router multiplies its normalised, per-channel-scaled input by before the
/// projection: `hidden ** -0.5`, which `Gemma4TextRouter` calls `scalar_root_size`.
///
/// Derived rather than declared by the artifact: it is a function of the width, and the width
/// the artifact states is the one this is computed from at bind time.
[[nodiscard]] constexpr float router_input_scale(int hidden) {
    // `constexpr` needs a square root it can evaluate; Newton on a positive double converges
    // in a handful of steps and this runs once per bind.
    double guess = static_cast<double>(hidden);
    for (int step = 0; step < 40; ++step) { guess = 0.5 * (guess + static_cast<double>(hidden) / guess); }
    return static_cast<float>(1.0 / guess);
}

} // namespace sinfer::targets::gemma4_moe::detail
