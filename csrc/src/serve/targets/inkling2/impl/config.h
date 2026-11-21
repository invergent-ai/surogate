#pragma once

// Inkling-2-Mini compile-time constants (ref/config.json text_config; ARTIFACT.md 2-3). The served geometry
// is the 48-visit one (DESIGN-target.md 0.2): `TextConfig::layers` counts VISITS; the physical layers are
// `TextConfig::physical_layers`.

#include <api/family/vision.h>

#include <cstdint>

namespace sinfer::targets::inkling2::detail {

struct TextConfig {
    static constexpr int hidden          = 2304;
    static constexpr int physical_layers = 24;
    static constexpr int core_start      = 6;
    static constexpr int core_end        = 18;
    static constexpr int passes          = 3;
    static constexpr int layers          = physical_layers + (passes - 1) * (core_end - core_start); // 48 visits
    static constexpr int block_layers    = 6;  ///< n_layers_per_attn_res_block
    static constexpr int stack_slots     = 7;  ///< AttnRes stack slots per column (DESIGN-target.md 0.3)
    static constexpr int residual        = stack_slots * hidden;

    static constexpr int output_rows  = 201024; ///< padded vocabulary (LM head rows)
    static constexpr int token_domain = 200058;

    static constexpr int query_heads = 24;
    static constexpr int kv_heads    = 8;
    static constexpr int head_dim    = 128;
    static constexpr int d_rel       = 16;
    static constexpr int window      = 1024;
    static constexpr int conv_kernel = 4;

    static constexpr int experts       = 128;
    static constexpr int experts_per_token = 8;
    static constexpr int intermediate  = 1152;
    static constexpr int gate_rank     = 64;

    static constexpr int qkvr_rows      = 8600; ///< [wq 3072 | wr 384 | gate 3072 | sink 24 | wk 1024 | wv 1024]
    static constexpr int qkvr_query_end = 6552;
    static constexpr int qkvr_k_begin   = 6552;
    static constexpr int qkvr_v_begin   = 7576;
    static constexpr int attn_width     = query_heads * head_dim; // 3072

    static constexpr float rms_epsilon   = 1.0e-6F;
    static constexpr float logit_softcap = 30.0F;

    [[nodiscard]] static constexpr bool is_global(int physical_layer) {
        return physical_layer == 5 || physical_layer == 11 || physical_layer == 17 || physical_layer == 23;
    }
};

static_assert(TextConfig::layers == 48);
static_assert(TextConfig::residual == 16128);

struct VisionConfig : family::VisionBackboneConfig {
    static constexpr int output_hidden = TextConfig::hidden;
};

/// The DSpark drafter rides the DFlash runtime (lane e2, family/impl/runtime/dspark_round.h): `--speculative dflash`
/// runs it, k <= 7 drafts per round (block 8).
struct DFlashConfig {
    static constexpr bool supported = true;
};

inline constexpr float kAttentionScale                   = 0.08838834764831845F; // 128^-0.5
inline constexpr float kGdnScale                         = 0.0F;                 // no linear-attention layers
inline constexpr std::uint32_t kPrefillChunkAlignment    = 128;
inline constexpr std::uint32_t kMaximumMtpDraftTokens    = 0;
inline constexpr std::uint32_t kMaximumDFlashDraftTokens = 7;
inline constexpr std::uint32_t kNativeContext            = 1048576;

} // namespace sinfer::targets::inkling2::detail
