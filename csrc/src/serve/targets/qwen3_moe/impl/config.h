#pragma once

#include <api/family/frontend.h>
#include <api/family/vision.h>
#include "api/ops/sparse_moe.h"

#include <cstdint>

namespace sinfer::targets::qwen3_moe::detail {

/// The mixture this target serves, compiled for the 30B-A3B. The op is closed over its
/// registered geometries, so this names one of them rather than restating its numbers.
inline constexpr ops::SparseMoeGeometry kMoeGeometry = ops::kSparseMoeQwen3MoeGeometry;

struct TextConfig {
    static constexpr int hidden       = kMoeGeometry.hidden;
    static constexpr int layers       = 48;
    /// The routed experts' FFN width. There is no dense MLP on any layer of this architecture.
    static constexpr int intermediate = kMoeGeometry.intermediate;

    // The output matrix is padded for the selected kernels. Only token IDs in
    // [0, token_domain) are tokenizer-addressable and valid sampling results.
    static constexpr int output_rows  = 151936;
    static constexpr int token_domain = 151669;

    // No linear mixer: every layer attends. These stay declared because the shared runtime
    // reads them when it sizes the linear-attention state, which is empty here.
    static constexpr int gdn_conv_kernel      = 0;
    static constexpr int gdn_conv_state_width = 0;
    static constexpr int gdn_key_heads        = 0;
    static constexpr int gdn_key_head_dim     = 0;
    static constexpr int gdn_value_heads      = 0;
    static constexpr int gdn_value_head_dim   = 0;

    static constexpr int query_heads = 32;
    static constexpr int kv_heads    = 4;
    static constexpr int head_dim    = 128;
    static constexpr int rotary_dim  = 128;

    static constexpr int full_attention_interval = 1;
    static constexpr float rms_epsilon           = 1.0e-6F;
    static constexpr float rope_theta            = 1.0e6F;

    static constexpr int key_dim         = 0;
    static constexpr int value_dim       = 0;
    static constexpr int convolution_dim = 0;
    static constexpr int query_size      = query_heads * head_dim;
    static constexpr int kv_size         = kv_heads * head_dim;
    /// Ungated: the projection carries query rows only, unlike the hybrid family.
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    /// The mixture. `router_rows` is the expert count and nothing else: this architecture has
    /// no always-on expert, so there is no gate row to fuse onto the router.
    static constexpr int experts           = kMoeGeometry.experts;
    static constexpr int experts_per_token = kMoeGeometry.experts_per_token;
    static constexpr int router_rows       = kMoeGeometry.router_rows();
    /// Layers with routed experts: every one of them, which is what the expert slot cache is
    /// sized against.
    static constexpr int expert_layers     = layers;

    [[nodiscard]] static constexpr bool is_full_attention(int) { return true; }
    [[nodiscard]] static constexpr int full_attention_layers() { return layers; }
    [[nodiscard]] static constexpr int gdn_layers() { return 0; }
    [[nodiscard]] static constexpr int full_attention_index(int layer) { return layer; }
    [[nodiscard]] static constexpr int gdn_index(int) { return 0; }
};

static_assert(TextConfig::router_rows == TextConfig::experts,
              "this mixture routes every token; a router with a shared-expert gate row would "
              "be read one row past its end");
static_assert(!kMoeGeometry.has_shared());

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

/// 1 / sqrt(head_dim), with head_dim 128.
inline constexpr float kAttentionScale                   = 0.08838834764831845F;
inline constexpr float kGdnScale                         = 0.0F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
inline constexpr std::uint32_t kNativeContext            = 40960;

} // namespace sinfer::targets::qwen3_moe::detail
