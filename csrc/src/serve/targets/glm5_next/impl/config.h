#pragma once

// GLM-5.3-Flash (`glm5next`). Every dimension here is the released 45-layer checkpoint's; the
// artifact declares its own and the runtime binds against those, so this is the default and the
// subject of the static assertions below rather than the schedule.

#include <api/family/frontend.h>
#include <api/family/runtime.h>
#include <api/family/vision.h>

#include <cstdint>

namespace sinfer::targets::glm5_next::detail {

struct TextConfig {
    static constexpr int hidden = 4096;
    static constexpr int layers = 45;
    /// The routed experts' FFN width, which is what the family's post-mixer is sized from.
    static constexpr int intermediate = 2048;
    static constexpr int dense_intermediate = 12288;

    static constexpr int output_rows  = 154880;
    static constexpr int token_domain = output_rows;

    // Manifold-constrained hyper-connections: the residual is `hc_streams` copies of the model
    // width, collapsed to one before every block and recombined after it by a stream x stream
    // matrix on the doubly-stochastic manifold.
    static constexpr int hc_streams = 4;
    static constexpr int hc_width   = hc_streams * hidden;              // 16384
    static constexpr int hc_mix     = (2 + hc_streams) * hc_streams;    // 24
    static constexpr int hc_sinkhorn_iterations = 20;
    static constexpr float hc_epsilon           = 1.0e-6F;
    /// The family sizes its residual planes from this.
    static constexpr int residual = hc_width;

    // Kimi Delta Attention. Its q, k and v are all 64 heads of 128, so llama.cpp stores no
    // permutation and the engine reads the heads as they lie.
    static constexpr int gdn_conv_kernel      = 4;
    static constexpr int gdn_conv_state_width = gdn_conv_kernel - 1;
    static constexpr int gdn_key_heads        = 64;
    static constexpr int gdn_key_head_dim     = 128;
    static constexpr int gdn_value_heads      = 64;
    static constexpr int gdn_value_head_dim   = 128;
    /// The forget and output gates are low-rank: hidden -> this -> the full width.
    static constexpr int kda_gate_rank = 128;
    /// The bound the forget gate's logistic is scaled by (`kda.gate_lower_bound`). A decay of
    /// exp(g) then lies in [exp(-5), 1): the state neither grows nor is erased outright.
    static constexpr float kda_gate_lower_bound = -5.0F;

    // Multi-head latent attention, NoPE, served in its absorbed form. The query is folded
    // through the key half of the expansion into the 512-wide latent and attends over the
    // latent itself: one key/value head as wide as the latent, sixty-four query heads over it,
    // and the value half unfolds the attended latent per head afterwards. The cache then holds
    // 1 KB per layer per token where the expanded form -- sixty-four heads of 256, each its own
    // key and value -- held 64 KB; on a 200 GB checkpoint split over eight cards that is the
    // difference between sixteen lanes fitting and not.
    static constexpr int query_heads   = 64;
    static constexpr int kv_heads      = 1;
    static constexpr int q_lora_rank   = 1536;
    static constexpr int kv_lora_rank  = 512;
    /// The head the attention kernels see: the latent.
    static constexpr int head_dim      = kv_lora_rank;
    static constexpr int rotary_dim    = 0;
    /// The per-head widths of the query and value projections -- what the checkpoint was
    /// trained with, and what its softmax scale is over. The artifact does not declare these;
    /// a checkpoint with others fails the shape check at bind, which is the honest failure.
    static constexpr int qk_head_dim   = 256;
    static constexpr int v_head_dim    = 256;

    // The sparse indexer, which this target does not run. It selects `index_top_k / index_pool`
    // pools of `index_pool` tokens and, because `index_kpool_always_select_tail` is set, the up
    // to `index_pool - 1` newest tokens that do not yet fill a pool; that is 2,051 cells, the
    // `n_select` llama.cpp gates its own indexer on. Below that many cached tokens every visible
    // one is selected, so dense attention is exactly what the indexer would have asked for;
    // above it the two differ and the engine refuses rather than attending to more than the
    // model was trained to.
    static constexpr int index_top_k         = 2048;
    static constexpr int index_pool          = 4;
    static constexpr int dense_exact_context = index_top_k + index_pool - 1;

    // The mixture: a sigmoid-plus-bias router over 288 experts, top-8 renormalised and scaled,
    // plus an always-on expert added with weight one. Its router therefore has one row per
    // expert and no gate row.
    static constexpr int experts             = 288;
    static constexpr int experts_per_token   = 8;
    static constexpr int shared_intermediate = 2048;
    static constexpr int router_rows         = experts;
    static constexpr float routed_scale      = 2.5F;
    static constexpr float swiglu_limit      = 10.0F;
    static constexpr int leading_dense_layers = 3;

    static constexpr float rms_epsilon = 1.0e-5F;
    /// NoPE: no rotary at all. Declared because the family's geometry names it.
    static constexpr float rope_theta = 1.0e4F;

    static constexpr int key_dim   = gdn_key_heads * gdn_key_head_dim;     // 8192
    static constexpr int value_dim = gdn_value_heads * gdn_value_head_dim; // 8192
    /// The mixer convolves a fused q|k|v, so this is its width.
    static constexpr int convolution_dim = 2 * key_dim + value_dim;        // 24576

    static constexpr int query_size = query_heads * head_dim;              // 32768: the absorbed query
    static constexpr int kv_size    = kv_heads * head_dim;                 // 512: the latent
    /// What `query_b` produces and what the output projection reads: the heads at their own
    /// widths, on either side of the absorbed attention.
    static constexpr int query_rows = query_heads * qk_head_dim;           // 16384
    static constexpr int value_rows = query_heads * v_head_dim;            // 16384
    /// Ungated: the projection carries query rows only.
    static constexpr int query_projection_rows = query_size;

    static constexpr int mtp_layers               = 0;
    static constexpr int mtp_input_rows           = 0;
    static constexpr int mtp_attention_input_rows = 0;
    static constexpr int mtp_mlp_gate_up_rows     = 0;

    /// Which layers attend, compiled for the released checkpoint. Every fourth from the third,
    /// which is what `attention.head_count_kv` says layer by layer; an artifact declares its
    /// own schedule and the runtime reads that instead.
    static constexpr int full_attention_interval = 4;

    [[nodiscard]] static constexpr bool is_full_attention(int layer) {
        return layer % full_attention_interval == full_attention_interval - 1;
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
    /// Whether this layer's feed-forward is the mixture. The first few are dense.
    [[nodiscard]] static constexpr bool is_sparse(int layer) { return layer >= leading_dense_layers; }
};

static_assert(TextConfig::full_attention_layers() == 11);
static_assert(TextConfig::gdn_layers() == 34);
static_assert(TextConfig::is_full_attention(3) && !TextConfig::is_full_attention(2));
static_assert(TextConfig::full_attention_index(3) == 0 &&
              TextConfig::full_attention_index(43) == 10);
static_assert(TextConfig::gdn_index(0) == 0 && TextConfig::gdn_index(44) == 33);
static_assert(TextConfig::hc_mix == 24 && TextConfig::hc_width == 16384);
static_assert(TextConfig::dense_exact_context == 2051);
static_assert(TextConfig::head_dim == TextConfig::kv_lora_rank && TextConfig::kv_heads == 1,
              "absorbed MLA attends over the latent itself");

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

/// The attention kernels apply 1/sqrt(head_dim), and the head they see is the 512-wide latent,
/// so this is 1/sqrt(512). The model's scale is 1/sqrt(256), over the 256-wide query head it
/// was trained with; the ratio, sqrt 2, is folded into the absorbed query by `kAbsorbScale`.
inline constexpr float kAttentionScale                   = 0.044194173824159216F;
inline constexpr float kAbsorbScale                      = 1.4142135623730951F;
/// 1/sqrt(128) for the delta recurrence.
inline constexpr float kGdnScale                         = 0.08838834764831845F;
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
/// The NextN draft head proposes up to this many tokens a round, the family's bound.
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 5;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 0;
/// What this target serves, not what the checkpoint was trained for (1,048,576). The sparse
/// indexer is not bound, and below its budget full attention is exactly what it would have
/// selected; above it they are different models.
inline constexpr std::uint32_t kNativeContext            = TextConfig::dense_exact_context;

} // namespace sinfer::targets::glm5_next::detail
