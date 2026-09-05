#pragma once

#include <api/family/frontend.h>
#include <api/family/vision.h>

#include <cstdint>

namespace sinfer::targets::lfm2::detail {

struct TextConfig {
    static constexpr int hidden       = 2048;
    static constexpr int layers       = 16;
    static constexpr int intermediate = 8192;

    // LFM2's vocabulary is not padded for the kernels: every id the tokenizer can produce is a
    // row of the head, so the two are the same number.
    static constexpr int output_rows  = 65536;
    static constexpr int token_domain = output_rows;

    // The short-convolution mixer's kernel. It occupies the geometry member the family named
    // for the linear mixer's convolution, because that is what it is -- the mixer's
    // convolution -- and there is only ever one of them in a checkpoint.
    static constexpr int gdn_conv_kernel      = 3;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    // No recurrent state at all: a short convolution's whole memory is the K-1 columns behind
    // the round. Zero here is what the state pool reads as "convolution only".
    static constexpr int gdn_key_heads      = 0;
    static constexpr int gdn_key_head_dim   = 0;
    static constexpr int gdn_value_heads    = 0;
    static constexpr int gdn_value_head_dim = 0;

    static constexpr int query_heads = 32;
    static constexpr int kv_heads    = 8;
    static constexpr int head_dim    = 64;
    static constexpr int rotary_dim  = head_dim;

    static constexpr float rms_epsilon = 1.0e-5F;
    static constexpr float rope_theta  = 1.0e6F;

    static constexpr int key_dim   = 0;
    static constexpr int value_dim = 0;
    // The mixer convolves the residual stream itself, not a fused q|k|v, so the family's
    // `convolution_dim` -- which is that fusion's width -- is zero here and the state pool
    // takes its channel count from `hidden` instead.
    static constexpr int convolution_dim = 0;

    static constexpr int query_size = query_heads * head_dim;
    static constexpr int kv_size    = kv_heads * head_dim;
    /// Ungated: the projection carries query rows only.
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    /// Which layers attend, compiled for the 1.2B.
    ///
    /// LFM2 names its attention layers outright -- `full_attn_idxs` is a list, not a period --
    /// and the list differs at every size, so this is a default rather than the schedule. Every
    /// artifact declares its own and the runtime reads that instead; this answers only for a
    /// checkpoint that declared none, and for the static assertions below.
    static constexpr int kAttentionLayers[]      = {2, 5, 8, 10, 12, 14};
    static constexpr int full_attention_interval = 0;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        for (const int attending : kAttentionLayers) {
            if (attending == layer) { return true; }
        }
        return false;
    }

    [[nodiscard]] static constexpr int full_attention_layers() {
        int count = 0;
        for (int layer = 0; layer < layers; ++layer) { count += is_full_attention(layer) ? 1 : 0; }
        return count;
    }

    [[nodiscard]] static constexpr int gdn_layers() { return layers - full_attention_layers(); }

    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        int index = 0;
        for (int earlier = 0; earlier < layer; ++earlier) {
            index += is_full_attention(earlier) ? 1 : 0;
        }
        return index;
    }

    [[nodiscard]] static constexpr int gdn_index(int layer) {
        int index = 0;
        for (int earlier = 0; earlier < layer; ++earlier) {
            index += is_full_attention(earlier) ? 0 : 1;
        }
        return index;
    }
};

static_assert(TextConfig::full_attention_layers() == 6);
static_assert(TextConfig::gdn_layers() == 10);
static_assert(TextConfig::query_size == TextConfig::hidden,
              "LFM2's attention output projection is [hidden, query_size]");

/// No vision tower. Declared because the shared ModelView names one; never bound.
struct VisionConfig {
    static constexpr int layers              = 0;
    static constexpr int hidden              = 0;
    static constexpr int intermediate        = 0;
    static constexpr int heads               = 0;
    static constexpr int head_dim            = 0;
    static constexpr int patch_dim           = 0;
    static constexpr int merge               = 1;
    static constexpr int merge_unit          = 1;
    static constexpr int merger_hidden       = 0;
    static constexpr int position_embeddings = 0;
    static constexpr int rotary_dim          = 0;
    static constexpr float rope_theta        = 0.0F;
    static constexpr float norm_epsilon      = 0.0F;
    static constexpr int output_hidden       = TextConfig::hidden;
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

/// 1 / sqrt(head_dim), with head_dim 64.
inline constexpr float kAttentionScale                   = 0.125F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 128000;

} // namespace sinfer::targets::lfm2::detail
