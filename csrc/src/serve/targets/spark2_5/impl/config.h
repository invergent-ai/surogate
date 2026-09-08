#pragma once

#include <api/family/frontend.h>
#include <api/family/hybrid_topology.h>
#include <api/family/vision.h>

#include <cstdint>

namespace sinfer::targets::spark2_5::detail {

// Runtime dimensions and execution parameters are required artifact metadata.
// These zero-valued members exist only for the shared template interface.
struct TextConfig {
    static constexpr int hidden       = 0;
    static constexpr int layers       = 0;
    static constexpr int intermediate = 0;

    static constexpr int output_rows  = 0;
    static constexpr int token_domain = 0;

    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    static constexpr int query_heads = 0;
    static constexpr int kv_heads    = 0;
    static constexpr int head_dim    = 0;
    static constexpr int rotary_dim  = 0;

    static constexpr int full_attention_interval = 0;
    static constexpr float rms_epsilon           = 0;
    static constexpr float rope_theta            = 0;

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

    [[nodiscard]] static constexpr bool is_full_attention(int) { return true; }

    [[nodiscard]] static constexpr int full_attention_layers() { return layers; }

    [[nodiscard]] static constexpr int gdn_layers() { return 0; }

    [[nodiscard]] static constexpr int full_attention_index(int layer) { return layer; }

    [[nodiscard]] static constexpr int gdn_index(int) { return -1; }
};


static_assert(TextConfig::gdn_layers() == 0);

struct VisionConfig : family::VisionBackboneConfig {
    static constexpr int output_hidden = 0;
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

inline constexpr float kAttentionScale                   = 0.0F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 0;

} // namespace sinfer::targets::spark2_5::detail
