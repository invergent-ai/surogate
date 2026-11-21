#pragma once
// Inkling-2 (inkling2_loop) weight bindings: lane b's half of the target (ARTIFACT.md, interface I-1).
//
// Two steps, as every target binds:
//   bind_artifact()  -- requires every object of the artifact with its format and shape, chooses a placement
//                       per object (the BF16 attention or one FP8 variant, the drafter or not), parses the
//                       `inkling2/config.json` resource into `ArtifactConfig`. Host only: no device is touched.
//   load_weights()   -- after materialisation: the physical weight sets as Weights / Tensors / raw pointers.
//
// The 24 PHYSICAL layers are bound once. The 48 visits (lane e1's served geometry) point at these structs; a core
// layer's three passes share one weight set (and so one LoRA bank: the Weights alias the same bytes).
// Every value is the checkpoint's word: the AttnRes / cross-pass queries are RAW (the consumer applies the
// reference's 2304^-0.5), the NVFP4 per-expert global is the raw amax + denominator, never folded.

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "core/tensor.h"

#include <api/family/frontend_resources.h>

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::load {

enum class ExpertFormat : std::uint8_t { Nvfp4, Mxfp8 };

/// The attention projection weights to serve. Bf16 = the faithful oracle path (always in the artifact);
/// Fp8 = the artifact's `attn/fp8-<scheme>/...` objects (owner ruling 06:58: the default once DESIGN 10.1 has
/// picked the scheme).
struct AttentionChoice {
    bool fp8 = false;
    std::string scheme; ///< "tensor" | "channel" | "g1x128" | "b128" when fp8
};

struct DraftConfig {
    std::int32_t layers = 0, hidden = 0, query_heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0;
    std::int32_t sliding_window = 0, block_size = 0, markov_rank = 0;
    std::int64_t max_position = 0;
    double rope_theta = 0.0, rms_eps = 0.0;
    std::vector<std::int32_t> local_layers;
    std::vector<std::int32_t> target_visits; ///< last = visit count (the final-norm output)
};

/// `inkling2/config.json` (schema "inkling2-artifact/1"), the dimensions the root `geometry` cannot state.
struct ArtifactConfig {
    std::int32_t hidden = 0, layers = 0, vocab_padded = 0, vocab = 0;
    std::int32_t query_heads = 0, kv_heads = 0, head_dim = 0, d_rel = 0, rel_extent = 0;
    std::int32_t sliding_window = 0, conv_kernel = 0;
    std::int32_t experts = 0, experts_per_token = 0, expert_intermediate = 0;
    std::int32_t log_scaling_n_floor = 0, attn_res_block_layers = 0, eos_token_id = 0;
    std::int64_t max_context = 0;
    double rms_eps = 0.0, qk_l2_clip = 0.0, rel_l2_clip = 0.0, log_scaling_alpha = 0.0, logit_softcap = 0.0;
    double gate_softcap = 0.0, up_softcap = 0.0, route_scale = 0.0;
    std::int32_t core_start = 0, core_end = 0, passes = 0, gate_rank = 0;
    std::vector<bool> global;                    ///< per physical layer
    std::vector<ExpertFormat> expert_format;     ///< per physical layer
    std::vector<std::array<std::int32_t, 2>> visits; ///< (layer, pass) per visit, execution order
    bool gate_up_interleaved = true;             ///< w13 rows: 2j = gate j, 2j+1 = up j (else [gate; up])
    std::vector<std::string> fp8_schemes;        ///< FP8 attention variants the artifact carries
    std::optional<DraftConfig> dspark;

    [[nodiscard]] bool is_core(std::int32_t layer) const noexcept { return layer >= core_start && layer < core_end; }
    [[nodiscard]] std::int32_t q_rows() const noexcept { return query_heads * head_dim; }
    [[nodiscard]] std::int32_t kv_rows() const noexcept { return kv_heads * head_dim; }
    /// Rows of the fused projection: [wq 3072 | wr 384 | gate 3072 | sink 24 | wk 1024 | wv 1024].
    [[nodiscard]] std::int32_t qkvr_rows() const noexcept {
        return 2 * q_rows() + query_heads * d_rel + query_heads + 2 * kv_rows();
    }
    /// The query side (q | r | gate | sink) that a 3-input visit projects from x_q.
    [[nodiscard]] std::int32_t qkvr_query_side_rows() const noexcept {
        return 2 * q_rows() + query_heads * d_rel + query_heads;
    }
};

/// Parses and checks the resource; throws artifact::ArtifactError on anything unexpected.
[[nodiscard]] ArtifactConfig parse_artifact_config(std::span<const std::byte> json);

// -------------------------------------------------------------------------------------------- binding plan

struct ExpertBankPlan {
    artifact::ObjectHandle bank;                            ///< NVFP4 or MXFP8 [experts * rows, columns]
    std::optional<artifact::ObjectHandle> amax;             ///< NVFP4: FP32 [experts]
    std::optional<artifact::ObjectHandle> outer_scale_denominator; ///< NVFP4: FP32 [1]
    float outer_scale_denominator_value = 0.0F;            ///< NVFP4: read at bind (host)
};

struct LayerPlan {
    bool resident = true;
    bool global   = false;
    bool core     = false;
    ExpertFormat expert_format = ExpertFormat::Nvfp4;
    artifact::ObjectHandle attn_input_norm;                 ///< BF16 [1 | 3, D] (global: q / k / v rows)
    artifact::LinearBinding qkvr;                           ///< BF16 [qkvr_rows, D] (oracle path)
    artifact::LinearBinding output;                         ///< BF16 [D, 3072] (wo_ud)
    std::optional<artifact::LinearBinding> qkvr_fp8;        ///< the chosen FP8 variant, when one is served
    std::optional<artifact::LinearBinding> output_fp8;
    artifact::ObjectHandle q_scale, r_scale, rel_proj;      ///< BF16 [24, 128], [24, 16], [16, 1024]
    artifact::ObjectHandle k_conv, v_conv;                  ///< local: BF16 [1024, 4]
    artifact::ObjectHandle k_conv_static, v_conv_static;    ///< global: BF16 [4, 1024]
    std::optional<artifact::LinearBinding> k_conv_proj, v_conv_proj; ///< global: BF16 [32, D]
    artifact::ObjectHandle attn_out_conv, mlp_out_conv;     ///< BF16 [D, 4]
    artifact::ObjectHandle moe_input_norm;                  ///< BF16 [D]
    artifact::LinearBinding router;                         ///< BF16 [128, D]
    artifact::ObjectHandle router_bias;                     ///< FP32 [128]
    ExpertBankPlan w13, w2;
    artifact::ObjectHandle loop_down, loop_gain_forget;     ///< core: BF16 [4, 64, 2D], [4, 2D, 64]
};

/// Where each group of globals is placed (resolved from BindOptions at bind time).
struct GlobalsPlacement {
    artifact::TensorPlacement embedding     = artifact::TensorPlacement::Device; ///< embedding + embedding_norm
    artifact::TensorPlacement head          = artifact::TensorPlacement::Device; ///< output_head + final_norm
    artifact::TensorPlacement small_globals = artifact::TensorPlacement::Device; ///< the AttnRes / cross-pass queries
};

struct GlobalsPlan {
    GlobalsPlacement placement;
    artifact::LinearBinding embedding;                      ///< BF16 [vocab_padded, D]
    artifact::ObjectHandle embedding_norm, final_norm;      ///< BF16 [D]
    artifact::LinearBinding output_head;                    ///< BF16 [vocab_padded, D]
    std::vector<artifact::ObjectHandle> attn_query;         ///< index = layer (1..23; [0] unused): FP32 [1|3, D]
    std::vector<artifact::ObjectHandle> mlp_query;          ///< index = layer: FP32 [1, D]
    artifact::ObjectHandle output_query;                    ///< FP32 [1, D]
    artifact::ObjectHandle cross_pass_queries;              ///< FP32 [2 * core, D] (attn / mlp per core layer)
    artifact::ObjectHandle cross_pass_kv_queries;           ///< FP32 [2 * core, D] (k / v per core layer)
};

struct DraftLayerPlan {
    artifact::ObjectHandle feature_norm, input_norm, post_attention_norm; ///< BF16 [D]
    artifact::LinearBinding q, k, v, o, k_ctx, v_ctx;
    artifact::ObjectHandle q_norm, k_norm, k_ctx_norm;      ///< BF16 [head_dim]
    artifact::LinearBinding gate, up, down;
};

struct DraftPlan {
    artifact::TensorPlacement placement = artifact::TensorPlacement::ValidateOnly; ///< resolved at bind
    artifact::ObjectHandle fuse_weight;                     ///< BF16 [layers + 1, features]
    artifact::ObjectHandle mask_embed, final_norm;          ///< BF16 [D]
    std::vector<DraftLayerPlan> layers;
    artifact::ObjectHandle markov_feature_norm;             ///< BF16 [D]
    artifact::LinearBinding markov_hidden_out;              ///< BF16 [D, rank]
    artifact::ObjectHandle markov_hidden_out_bias;          ///< BF16 [D]
    artifact::LinearBinding markov_w1;                      ///< BF16 [vocab_padded, rank] (a token table)
    artifact::LinearBinding markov_rnn_proj;                ///< BF16 [3 rank, 2 rank + 2 D]
    artifact::ObjectHandle markov_rnn_proj_bias;            ///< BF16 [3 rank]
};

struct BindOptions {
    AttentionChoice attention;
    /// Bind the DSpark drafter to the device (MTP on); otherwise its objects are validated only.
    bool draft = false;
    /// The drafter's dense linears (q k v o k_ctx v_ctx gate up down) in a mapped host bank instead of the device: the
    /// target re-quantises them to FP8 on the device at load (lane e2, DSpark FP8 drafter weights).
    bool draft_linears_host = false;
    /// Placement of one physical layer's objects (a pipeline stage keeps its own visits' layers; a core layer
    /// is needed by every stage that runs one of its passes). Unset: the placement in force for every layer.
    std::function<artifact::TensorPlacement(std::int32_t layer)> layer_placement;
    /// Placement of the embedding / norms / head / queries. Unset: the placement in force.
    std::optional<artifact::TensorPlacement> globals_placement;
    /// A pipeline stage's share of the globals (the 2-card path, lanes/e1/DESIGN-2card.md); each unset one takes
    /// `globals_placement`, so a 1-card engine places exactly what it placed before.
    ///  embedding:     `text/embedding` + `text/embedding_norm` -- the stage that embeds (the first). A DSpark
    ///                 drafter reads the anchor's normalized embedding too: its stage needs the rows (or this group).
    ///  head:          `text/output_head` + `text/final_norm` -- the stage that runs the head (the last).
    ///  small_globals: the RAW AttnRes attn / mlp / output queries and the cross-pass (kv) queries (~1 MB in all):
    ///                 every stage by default (each stage's visits, its derived message queries and any cut use them).
    std::optional<artifact::TensorPlacement> embedding_placement;
    std::optional<artifact::TensorPlacement> head_placement;
    std::optional<artifact::TensorPlacement> small_globals_placement;
    /// The DSpark drafter's objects (`mtp/*`) when `draft` is set. Unset: `head_placement` (it reads the LM head).
    std::optional<artifact::TensorPlacement> draft_placement;
};

struct BindingPlan {
    ArtifactConfig config;
    BindOptions options;
    family::FrontendResourcePlan frontend;
    artifact::ObjectHandle special_tokens_map, config_resource, source_config, hf_quant_config;
    GlobalsPlan globals;
    std::vector<LayerPlan> layers;
    bool has_draft = false;
    DraftPlan draft;
};

/// Binds every object of an Inkling-2 artifact (and fails on any it does not know): after it, `binder.finish()`
/// must succeed. Host only.
[[nodiscard]] BindingPlan bind_artifact(artifact::Binder& binder, const BindOptions& options);

// -------------------------------------------------------------------------------------------- loaded weights

/// One routed bank as stored: rows = expert * rows_per_expert + row.
///  NVFP4: `codes` = E2M1 pairs [rows][columns / 2] (low nibble = even column); `scales` = E4M3 block scales per 16
///         in the layout's 128x4 tiles (blockscale-k16-m128x4-v1, the TRT-LLM / CUTLASS order); value =
///         e2m1 * e4m3 * fp32(amax[e] / outer_scale_denominator); `weight` = the routed NVFP4 Weight (identity
///         divisors) for kernels that take one.
///  MXFP8: `codes` = E4M3 [rows][columns]; `scales` = E8M0 [rows][columns / 32], natural order;
///         value = e4m3 * 2^(s - 127). No Weight (no QType exists for it).
struct ExpertBank {
    ExpertFormat format = ExpertFormat::Nvfp4;
    std::int32_t experts = 0, rows_per_expert = 0, columns = 0;
    const std::uint8_t* codes  = nullptr;
    const std::uint8_t* scales = nullptr;
    const float* amax                     = nullptr; ///< NVFP4: device FP32 [experts]
    const float* outer_scale_denominator  = nullptr; ///< NVFP4: device FP32 [1]
    float outer_scale_denominator_value   = 0.0F;    ///< NVFP4: the same word, host
    bool gate_up_interleaved              = true;    ///< w13 only
    std::optional<Weight> weight;                    ///< NVFP4 only
};

struct InklingLayerWeights {
    std::int32_t layer = 0;
    bool global = false, core = false;
    Tensor attn_input_norm;           ///< BF16 ne {D, 1|3}
    Weight qkvr;                      ///< the SERVED attention input projection (BF16 or the chosen FP8)
    Weight output;                    ///< the SERVED wo_ud
    bool fp8_attention = false;
    Tensor q_scale, r_scale, rel_proj;  ///< BF16 ne {128, 24}, {16, 24}, {1024, 16}
    Tensor k_conv, v_conv;              ///< local: BF16 ne {4, 1024}
    Tensor k_conv_static, v_conv_static; ///< global: BF16 ne {1024, 4}
    std::optional<Weight> k_conv_proj, v_conv_proj; ///< global: BF16 [32, D]
    Tensor attn_out_conv, mlp_out_conv; ///< BF16 ne {4, D}
    Tensor moe_input_norm;              ///< BF16 ne {D}
    Weight router;                      ///< BF16 [128, D]
    const float* router_bias = nullptr; ///< device FP32 [128]
    ExpertBank w13, w2;
    Tensor loop_down, loop_gain_forget; ///< core: BF16 ne {2D, 64, 4}, {64, 2D, 4}
};

struct InklingGlobals {
    /// Which groups this stage holds (BindOptions *_placement): a group on another stage is left empty.
    bool has_embedding = false, has_head = false, has_small_globals = false;
    Weight embedding, output_head;
    Tensor embedding_norm, final_norm;
    std::vector<Tensor> attn_query;    ///< FP32 ne {D, 1|3}, RAW; index = layer ([0] empty)
    std::vector<Tensor> mlp_query;     ///< FP32 ne {D, 1}, RAW
    Tensor output_query;               ///< FP32 ne {D, 1}, RAW
    Tensor cross_pass_queries, cross_pass_kv_queries; ///< FP32 ne {D, 2 * core}, RAW
};

struct InklingDraftLayerWeights {
    Tensor feature_norm, input_norm, post_attention_norm;
    Weight q, k, v, o, k_ctx, v_ctx;
    Tensor q_norm, k_norm, k_ctx_norm;
    Weight gate, up, down;
};

struct InklingDraftWeights {
    Tensor fuse_weight, mask_embed, final_norm;
    std::vector<InklingDraftLayerWeights> layers;
    Tensor markov_feature_norm, markov_hidden_out_bias, markov_rnn_proj_bias;
    Weight markov_hidden_out, markov_w1, markov_rnn_proj;
};

struct InklingWeights {
    ArtifactConfig config;
    InklingGlobals globals;
    std::vector<InklingLayerWeights> layers; ///< 24 physical layers; a non-resident one is left empty
    bool has_draft = false;
    InklingDraftWeights draft;
};

/// The materialised weights of a plan. Only objects bound to the device (or host bank) are read.
[[nodiscard]] InklingWeights load_weights(const BindingPlan& plan, const artifact::MaterializedArtifact& materialized);

} // namespace sinfer::targets::inkling2::load
