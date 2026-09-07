#pragma once

#include <api/family/frontend.h>
#include <api/family/hybrid_topology.h>
#include <api/family/vision.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace sinfer::targets::gemma4::detail {

/// The dense half of the Gemma 4 family: `google/gemma-4-12B` and
/// `google/gemma-4-31B-it`. Compiled for the 12B; the 31B is served by laying its
/// artifact's declared geometry over these, which is why the namespace is the
/// architecture and not a size (`gemma3_270m` is the size, because that target
/// serves one).
///
/// **This model attends at two geometries.** Its windowed layers run 16 query heads
/// over 8 key/value heads of 256; its global layers run the same 16 query heads over
/// *one* key/value head of 512. The members below name the windowed geometry, as the
/// family's do, and the `global_*` members the other one -- `family::TextGeometry`
/// carries both and `head_dim_for(windowed)` is how a caller asks for a layer's.
///
/// Which layers are which is *not* compiled, unlike Gemma 3: this target serves two
/// sizes and their schedules differ in length (48 layers against 60). `kWindowedAttention`
/// below is the 12B's, and it stands only for an artifact that declares no schedule of
/// its own; a Gemma 4 artifact always does, derived in `bindings.cpp` from the width of
/// each layer's query norm.
struct TextConfig {
    static constexpr int hidden       = 3840;
    static constexpr int layers       = 48;
    static constexpr int intermediate = 15360;

    // Gemma 4 pads neither: the head has one row per tokenizer-addressable id.
    static constexpr int output_rows  = 262144;
    static constexpr int token_domain = 262144;

    // No linear mixer. These stay declared because the shared runtime reads them when it
    // sizes the linear-attention state, which is empty here.
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

    /// The global layers': a wider head, and far fewer of them on the key/value side.
    /// A global layer's cache is paid for over the whole context where a windowed
    /// layer's is bounded by its window, which is what buys the extra width.
    static constexpr int global_head_dim  = 512;
    static constexpr int global_kv_heads  = 1;
    /// How many of a global head's 256 angle pairs carry a non-zero rope frequency.
    /// `partial_rotary_factor 0.25` of a 512-wide head: 64 rotate and 192 do not.
    ///
    /// The 192 are *appended*, not skipped -- `_compute_proportional_rope_parameters`
    /// concatenates zero frequencies and `apply_rotary_pos_emb` then rotates the whole
    /// head, so the untouched channels are the tail of each half rather than a
    /// contiguous suffix of the head. A rotary_dim of 128 would rotate the wrong
    /// channels and would look almost right.
    static constexpr int global_rotary_angles = 64;

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = 1.0e-6F;
    /// The global layers' rope base. The windowed layers rotate at their own, below.
    static constexpr float rope_theta            = 1.0e6F;

    static constexpr int sliding_window          = 1024;
    static constexpr float sliding_rope_theta    = 1.0e4F;

    /// The 12B's schedule: every sixth layer global, the last one included. Read the
    /// class comment before relying on it -- an artifact's own schedule wins, and every
    /// Gemma 4 artifact declares one.
    static constexpr std::array<bool, layers> kWindowedAttention{
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
        true, true, true, true, true, false,
    };

    /// True when this layer attends through the sliding window, false for a global layer
    /// that sees the whole context.
    [[nodiscard]] static constexpr bool is_windowed_attention(int layer) {
        return kWindowedAttention[static_cast<std::size_t>(layer)];
    }

    /// The rope base this layer rotates at.
    [[nodiscard]] static constexpr float layer_rope_theta(int layer) {
        return is_windowed_attention(layer) ? sliding_rope_theta : rope_theta;
    }

    /// Applied to the embedding lookup before the first block: `sqrt(hidden)`, rounded to
    /// bf16 as the reference rounds it before multiplying. 62.0 at hidden 3840 -- and 73.5
    /// at the 31B's 5376, which is why the artifact declares it rather than inheriting this.
    static constexpr float embedding_scale       = 6.2e1F;

    /// Logits are squashed to `tanh(x / c) * c` before sampling. Gemma 4 caps at 30;
    /// no other target here caps at all.
    static constexpr float logit_softcap         = 3.0e1F;

    static constexpr int key_dim               = 0;
    static constexpr int value_dim             = 0;
    static constexpr int convolution_dim       = 0;
    static constexpr int query_size            = query_heads * head_dim;
    static constexpr int kv_size               = kv_heads * head_dim;
    // Ungated: the projection carries query rows only, unlike the hybrid family.
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    /// Every layer attends -- "full attention" is the *mixer* axis here, and both a
    /// windowed and a global layer are ordinary attention on it. The other axis, which
    /// is the one this model varies, is `is_windowed_attention` above.
    [[nodiscard]] static constexpr bool is_full_attention(int) { return true; }

    [[nodiscard]] static constexpr int full_attention_layers() { return layers; }

    [[nodiscard]] static constexpr int gdn_layers() { return 0; }

    [[nodiscard]] static constexpr int full_attention_index(int layer) { return layer; }

    [[nodiscard]] static constexpr int gdn_index(int) { return -1; }
};

static_assert(TextConfig::full_attention_layers() == 48);
static_assert(TextConfig::gdn_layers() == 0);
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

/// **One, not `1/sqrt(head_dim)`.** Gemma 4 sets `self.scaling = 1.0` and relies on its
/// QK-norm to deliver unit-RMS queries and keys. Inheriting the kernel's default would
/// divide a global layer's scores by 22.6 and a windowed layer's by 16, which is not a
/// subtle error but is a silent one -- the model would still emit tokens.
inline constexpr float kAttentionScale                   = 1.0F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 262144;

} // namespace sinfer::targets::gemma4::detail
