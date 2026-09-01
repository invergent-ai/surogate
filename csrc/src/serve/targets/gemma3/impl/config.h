#pragma once

#include <api/family/frontend.h>
#include <api/family/hybrid_topology.h>
#include <api/family/vision.h>

#include <cstdint>

namespace sinfer::targets::gemma3_270m::detail {

struct TextConfig {
    static constexpr int hidden       = 640;
    static constexpr int layers       = 18;
    static constexpr int intermediate = 2048;

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 262144;
    static constexpr int token_domain = 262144;

    // No linear mixer. These stay declared because the shared runtime reads them
    // when it sizes the linear-attention state, which is empty here.
    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    static constexpr int query_heads = 4;
    static constexpr int kv_heads    = 1;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 256;

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e6F;

    // Causal sliding-window attention. Zero is a model whose every layer sees the
    // whole context; otherwise a query at position i admits keys j with
    // `i - j < sliding_window`, i.e. exactly `sliding_window` keys including its
    // own. `sliding_window_period` is how often a layer escapes the window: with
    // 6, every 6th layer is global. Windowed layers rotate at their own base.
    static constexpr int sliding_window          = 512;
    static constexpr int sliding_window_period   = 6;
    static constexpr float sliding_rope_theta    = 1.0e4F;

    // Applied to the embedding lookup before the first block. Zero means none.
    static constexpr float embedding_scale       = 2.529822e1F;

    /// True when this layer attends through the sliding window, false for a global
    /// layer that sees the whole context. A model with no window has
    /// every layer global; otherwise the period says which escape it. Gemma 3
    /// counts from the end -- its last layer is global -- which is what
    /// `(layer + 1) % period == 0` expresses.
    [[nodiscard]] static constexpr bool is_windowed_attention(int layer) {
        if (sliding_window <= 0) { return false; }
        if (sliding_window_period <= 0) { return true; }
        return ((layer + 1) % sliding_window_period) != 0;
    }

    /// The rope base this layer rotates at.
    [[nodiscard]] static constexpr float layer_rope_theta(int layer) {
        return is_windowed_attention(layer) ? sliding_rope_theta : rope_theta;
    }

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

    [[nodiscard]] static constexpr bool is_full_attention(int) { return true; }

    [[nodiscard]] static constexpr int full_attention_layers() { return layers; }

    [[nodiscard]] static constexpr int gdn_layers() { return 0; }

    [[nodiscard]] static constexpr int full_attention_index(int layer) { return layer; }

    [[nodiscard]] static constexpr int gdn_index(int) { return -1; }
};

static_assert(TextConfig::full_attention_layers() == 18);
static_assert(TextConfig::gdn_layers() == 0);

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

inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 32768;

} // namespace sinfer::targets::gemma3_270m::detail
