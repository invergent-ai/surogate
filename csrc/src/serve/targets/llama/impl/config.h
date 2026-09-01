#pragma once

#include <api/targets/qwen3_6/frontend.h>
#include <api/targets/qwen3_6/hybrid_topology.h>
#include <api/targets/qwen3_6/vision.h>

#include <cstdint>

namespace sinfer::targets::llama::detail {

struct TextConfig {
    static constexpr int hidden       = 2048;
    static constexpr int layers       = 22;
    static constexpr int intermediate = 5632;

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 32000;
    static constexpr int token_domain = 32000;

    // No linear mixer. These stay declared because the shared runtime reads them
    // when it sizes the linear-attention state, which is empty here.
    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    static constexpr int query_heads = 32;
    static constexpr int kv_heads    = 4;
    static constexpr int head_dim    = 64;
    static constexpr int rotary_dim  = 64;

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = 1.0e-5F;
    static constexpr float rope_theta            = 1.0e4F;

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

static_assert(TextConfig::full_attention_layers() == 22);
static_assert(TextConfig::gdn_layers() == 0);

struct VisionConfig : qwen3_6::VisionBackboneConfig {
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

inline constexpr float kAttentionScale                   = 0.125F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 2048;

} // namespace sinfer::targets::llama::detail

