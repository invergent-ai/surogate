#pragma once

// Qwen3.8-Flash-Next (`qwen4exp`). Geometry from the GGUF metadata and HF config; the
// forward-pass contract is design/serve-engine-flash-next.md §5.

#include <api/targets/qwen3_6/frontend.h>
#include <api/targets/qwen3_6/hybrid_topology.h>
#include <api/targets/qwen3_6/vision.h>

#include <array>
#include <cstdint>

namespace ninfer::targets::qwen4exp::detail {

struct TextConfig {
    static constexpr int hidden       = 2560;
    static constexpr int layers       = 48;
    static constexpr int intermediate = 640; // routed expert FFN width

    static constexpr int output_rows  = 248320;
    static constexpr int eos_token    = 248044; // also cuts the n-gram context
    static constexpr int token_domain = static_cast<int>(qwen3_6::kTokenDomain);

    // Hyper-connections: the residual is `hc_count` streams of `hidden`, mixed through a
    // low-rank bottleneck before every block and scattered back with per-stream weights.
    static constexpr int hc_count    = 4;
    static constexpr int hc_width    = hc_count * hidden; // 10240
    static constexpr int hc_low_rank = 320;
    // The family sizes its residual planes from this: four streams side by side.
    static constexpr int residual    = hc_width;

    // Gated delta net (the Qwen3.5/3.6 mixer with a sigmoid output gate).
    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 16;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 48;
    static constexpr int gdn_value_head_dim   = 128;

    // Full attention: gated (q|gate interleaved per head), 2 kv heads, partial rope.
    static constexpr int query_heads = 24;
    static constexpr int kv_heads    = 2;
    static constexpr int head_dim    = 256;
    static constexpr int rotary_dim  = 64;

    // QSA indexer on the full-attention layers (phase 2; the artifact carries its weights).
    static constexpr int indexer_heads    = 4;
    static constexpr int indexer_head_dim = 128;
    static constexpr int indexer_top_k    = 2048;
    static constexpr int indexer_block    = 4;
    // Below this many cached tokens the indexer selects every cell, so dense attention is
    // exact: min(n_kv, top_k + block - 1).
    static constexpr int dense_exact_context = indexer_top_k + indexer_block - 1; // 2051

    // Sparse MoE: softmax router, top-10 renormalised, plus a sigmoid-gated shared expert.
    static constexpr int experts             = 512;
    static constexpr int experts_per_token   = 10;
    static constexpr int shared_intermediate = 640;
    static constexpr int router_rows         = experts + 1; // + the shared-expert gate row

    // n-gram PLE memory: bigram and trigram hashes, 8 heads each, 160-wide rows, one layer.
    static constexpr int ple_layer          = 1;
    static constexpr int ple_ngram          = 3;
    static constexpr int ple_heads_per_gram = 8;
    static constexpr int ple_heads          = (ple_ngram - 1) * ple_heads_per_gram; // 16
    static constexpr int ple_head_dim       = 160;
    static constexpr int ple_embed          = ple_heads * ple_head_dim; // 2560
    static constexpr int ple_conv_kernel    = 4;
    static constexpr int ple_conv_dilation  = ple_ngram;
    static constexpr int ple_conv_history   = (ple_conv_kernel - 1) * ple_conv_dilation; // 9
    static constexpr std::int64_t ple_table_rows = 320001536;
    static constexpr int ple_table_row_bytes     = 90; // IQ4_NL: 5 blocks of 18 bytes

    static constexpr float rms_epsilon = 1.0e-6F;
    static constexpr float rope_theta  = 1.0e7F;

    static constexpr int full_attention_interval = 4;

    // No MTP block is served (the GGUF carries none); the family's MTP planes are sized by
    // these but never materialised, so they mirror the text block's shapes.
    static constexpr int mtp_layers               = 1;
    static constexpr int mtp_input_rows           = 2 * hidden;
    static constexpr int mtp_attention_input_rows = 2 * (query_heads * head_dim) + 2 * (kv_heads * head_dim);
    static constexpr int mtp_mlp_gate_up_rows     = 2 * intermediate;

    static constexpr int key_dim               = gdn_key_heads * gdn_key_head_dim;     // 2048
    static constexpr int value_dim             = gdn_value_heads * gdn_value_head_dim; // 6144
    static constexpr int convolution_dim       = 2 * key_dim + value_dim;              // 10240
    static constexpr int query_size            = query_heads * head_dim;               // 6144
    static constexpr int kv_size               = kv_heads * head_dim;                  // 512
    static constexpr int query_projection_rows = 2 * query_size + 2 * kv_size;         // 13312
    static constexpr int gdn_projection_rows   = convolution_dim + value_dim;          // 16384

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return (layer + 1) % full_attention_interval == 0;
    }
    [[nodiscard]] static constexpr int full_attention_layers() {
        return layers / full_attention_interval;
    }
    [[nodiscard]] static constexpr int gdn_layers() { return layers - full_attention_layers(); }
    [[nodiscard]] static constexpr int full_attention_index(int layer) {
        return (layer + 1) / full_attention_interval - 1;
    }
    [[nodiscard]] static constexpr int gdn_index(int layer) {
        return layer - (layer + 1) / full_attention_interval;
    }
};

static_assert(TextConfig::full_attention_layers() == 12);
static_assert(TextConfig::gdn_layers() == 36);
static_assert(TextConfig::is_full_attention(3) && !TextConfig::is_full_attention(2));
static_assert(TextConfig::full_attention_index(3) == 0 && TextConfig::full_attention_index(47) == 11);
static_assert(TextConfig::gdn_index(0) == 0 && TextConfig::gdn_index(4) == 3 &&
              TextConfig::gdn_index(46) == 35);
static_assert(TextConfig::ple_embed == TextConfig::hidden);
static_assert(TextConfig::gdn_projection_rows == 16384 && TextConfig::query_projection_rows == 13312);

// The family's vision context is instantiated but never enabled for this target.
struct VisionConfig : qwen3_6::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

struct DFlashConfig {
    static constexpr bool supported     = false;
    static constexpr int layers         = 1;
    static constexpr int local_layers   = 1;
    static constexpr int feature_layers = 1;
    static constexpr int feature_rows   = feature_layers * TextConfig::hidden;
    static constexpr int hidden         = TextConfig::hidden;
    static constexpr int intermediate   = TextConfig::intermediate;
    static constexpr int query_heads    = TextConfig::query_heads;
    static constexpr int kv_heads       = TextConfig::kv_heads;
    static constexpr int head_dim       = TextConfig::head_dim;
    static constexpr int query_size     = query_heads * head_dim;
    static constexpr int kv_size        = kv_heads * head_dim;
    static constexpr int local_capacity = 1;
    static constexpr int mask_token     = 0;
    static constexpr float rms_epsilon  = TextConfig::rms_epsilon;
    static constexpr float rope_theta   = TextConfig::rope_theta;
    static constexpr float attention_scale = 0.0625F;
    static constexpr std::array<int, feature_layers> target_feature_layers{0};
};

// 1/sqrt(head_dim) for both mixers (head 256 and GDN head 128).
inline constexpr float kAttentionScale                   = 0.0625F;
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 15;
// Dense attention is exact only below the indexer's budget; longer contexts wait for the
// QSA indexer.
inline constexpr std::uint32_t kNativeContext            = TextConfig::dense_exact_context;

} // namespace ninfer::targets::qwen4exp::detail
