#pragma once
#include "family/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "family/impl/runtime/linear_state_slots.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "api/ops/sampling.h"
#include "api/ops/gqa_attention.h"
#include "api/ops/qsa_indexer.h"
#include <api/family/text_geometry.h>
#include <api/family/decoder_state.h>
#include <api/family/prepared_prompt.h>
#include <api/family/round_state.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include "family/impl/runtime/residual_policy.h"

namespace sinfer::family::detail {
class PrefillGraphFamily;
} // namespace sinfer::family::detail

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule {

// Target-private compatibility vocabulary for the mechanically preserved fixed schedule. It is
// data-only: TextContext is constructed on the stack for one schedule recording/execution and owns
// neither weights nor device state.
struct ModelConfig {
    // The geometry, as data. Every member still defaults to the target's compiled `TextConfig`,
    // so nothing about a registered model changes; what changes is that the runtime now reads
    // these off an object it could have been handed instead. That is the whole of the
    // difference between serving one size of a family and serving the family: of the 326 reads
    // in this runtime, not one needs the value at compile time.
    int hidden              = TextConfig::hidden;
    int residual            = residual_width<TextConfig>();
    int n_layers            = TextConfig::layers;
    int intermediate        = TextConfig::intermediate;
    int vocab               = TextConfig::output_rows;
    int token_domain        = TextConfig::token_domain;
    int gdn_k_heads         = TextConfig::gdn_key_heads;
    int gdn_k_dim           = TextConfig::gdn_key_head_dim;
    int gdn_v_heads         = TextConfig::gdn_value_heads;
    int gdn_v_dim           = TextConfig::gdn_value_head_dim;
    int n_q                 = TextConfig::query_heads;
    int n_kv                = TextConfig::kv_heads;
    int head_dim            = TextConfig::head_dim;
    int rotary_dim          = TextConfig::rotary_dim;
    int key_dim             = TextConfig::key_dim;
    int value_dim           = TextConfig::value_dim;
    int conv_dim            = TextConfig::convolution_dim;
    int q_size              = TextConfig::query_size;
    int kv_size             = TextConfig::kv_size;
    int mtp_fc_in           = TextConfig::mtp_input_rows;
    int mtp_attn_in         = TextConfig::mtp_attention_input_rows;
    int mtp_mlp_gateup_rows = TextConfig::mtp_mlp_gate_up_rows;
    float rms_eps           = TextConfig::rms_epsilon;
    float rope_theta        = TextConfig::rope_theta;
    int mtp_layers          = TextConfig::mtp_layers;

    [[nodiscard]] static constexpr bool is_full(int layer) {
        return TextConfig::is_full_attention(layer);
    }

    /// How many layers attend and how many are linear. Which *kind* a layer is stays
    /// compiled -- that is the family's schedule, not a dimension -- but how many there
    /// are follows the layer count, which the artifact may declare.
    [[nodiscard]] constexpr int n_full() const {
        int count = 0;
        for (int layer = 0; layer < n_layers; ++layer) { count += is_full(layer) ? 1 : 0; }
        return count;
    }

    [[nodiscard]] constexpr int n_gdn() const { return n_layers - n_full(); }

    [[nodiscard]] static constexpr int full_idx(int layer) {
        return TextConfig::full_attention_index(layer);
    }

    [[nodiscard]] static constexpr int gdn_idx(int layer) { return TextConfig::gdn_index(layer); }

    /// The compiled defaults, which is what a target without a declared geometry gets.
    ModelConfig() = default;

    /// The dimensions the weights were bound against. Everything derived is derived here
    /// rather than copied, so a geometry that declares `gdn_key_heads` cannot disagree
    /// with its own `key_dim`.
    explicit ModelConfig(const family::TextGeometry& geometry)
        : hidden(geometry.hidden),
          residual(geometry.residual),
          n_layers(geometry.layers),
          intermediate(geometry.intermediate),
          vocab(geometry.output_rows),
          token_domain(geometry.token_domain),
          gdn_k_heads(geometry.gdn_key_heads),
          gdn_k_dim(geometry.gdn_key_head_dim),
          gdn_v_heads(geometry.gdn_value_heads),
          gdn_v_dim(geometry.gdn_value_head_dim),
          n_q(geometry.query_heads),
          n_kv(geometry.kv_heads),
          head_dim(geometry.head_dim),
          rotary_dim(geometry.rotary_dim),
          key_dim(geometry.key_dim()),
          value_dim(geometry.value_dim()),
          conv_dim(geometry.convolution_dim()),
          q_size(geometry.query_size()),
          kv_size(geometry.kv_size()),
          mtp_fc_in(geometry.mtp_input_rows()),
          mtp_attn_in(geometry.mtp_attention_input_rows()),
          mtp_mlp_gateup_rows(geometry.mtp_mlp_gate_up_rows()),
          rms_eps(geometry.rms_epsilon),
          rope_theta(geometry.rope_theta),
          mtp_layers(geometry.mtp_layers) {}
};

inline const ModelConfig kCfg{};
using Hooks = ResidualHooks<Variant>;
inline constexpr float kAttnScale                     = kAttentionScale;
// False only for a target whose attention has no output gate (a dense GQA
// stack); every hybrid target in the family leaves it at the default.
inline constexpr bool kAttentionOutputGate            = family::detail::attention_output_gate<Variant>();
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

struct MlpW {
    const MlpWeights* payload = nullptr;
};

struct FullLayerW {
    const Tensor* input_norm                         = nullptr;
    const FullAttentionProjectionWeights* projection = nullptr;
    const Weight* o_proj                             = nullptr;
    const Tensor* q_norm                             = nullptr;
    const Tensor* k_norm                             = nullptr;
    const Tensor* post_attn_norm                     = nullptr;
    MlpW mlp;
};

struct GdnLayerW {
    const Tensor* input_norm               = nullptr;
    const GdnProjectionWeights* projection = nullptr;
    const Tensor* conv1d                   = nullptr;
    const Tensor* gdn_norm                 = nullptr;
    const Weight* out_proj                 = nullptr;
    const Tensor* post_attn_norm           = nullptr;
    MlpW mlp;
};

struct MtpW {
    const MtpWeights* payload           = nullptr;
    const Weight* fc                    = nullptr;
    const Tensor* pre_fc_norm_embedding = nullptr;
    const Tensor* pre_fc_norm_hidden    = nullptr;
    const Tensor* input_norm            = nullptr;
    const Tensor* q_norm                = nullptr;
    const Tensor* k_norm                = nullptr;
    const Weight* o_proj                = nullptr;
    const Tensor* post_attn_norm        = nullptr;
    const Tensor* norm                  = nullptr;
};

using Phase = family::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const int> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;

/// Pipeline stage: the layer range this program runs, and the pinned buffers the residual
/// crosses at the boundaries (import when first > 0, export when last < layers). Whole-model
/// programs leave the defaults.
struct StageSpan {
    int first                 = 0;
    int last                  = -1;      // -1: through the last layer
    const void* import_pinned = nullptr; // [residual, columns] BF16, written by the previous stage
    void* export_pinned       = nullptr; // [residual, columns] BF16, read by the next stage
    std::int32_t columns      = 0;       // capacity of both buffers
};

class TextContext {
public:
    TextContext(DeviceContext& ctx, const LoadedModelData& weights, WorkspaceArena& work,
                family::PagedKVCacheView kv, LinearAttentionStatePool& state,
                family::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                family::PagedKVCacheView mtp_kv           = family::PagedKVCacheView(),
                const family::PagedKVCache* batch_text_kv = nullptr,
                const family::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    /// Pipeline stage: restricts the layer loop to [first, last) and swaps the embedding for a
    /// residual import (first > 0) and the finish/head for a residual export (last < layers).
    void set_stage(const StageSpan& stage);

    // QSA sparse selection for one full-attention layer (design/INFERENCE.md, phase 4): projects
    // the layer's indexer keys and queries out of the attention input, folds the completed blocks
    // into the cache's indexer plane, and scores the blocks for every query column. Returns an
    // empty mask when the target has no indexer or the history still fits the budget — the dense
    // attention below `dense_exact_context` is then bit-identical to what it was.
    template <class V = Variant>
    [[nodiscard]] ops::GqaBlockMask text_indexer_selection(const FullLayerW& w, const Tensor& hidden,
                                                           std::int32_t tokens,
                                                           const Tensor& cache_positions,
                                                           const Tensor& rope_positions,
                                                           const Tensor& table_rows,
                                                           std::int32_t columns_per_row,
                                                           std::int32_t keys,
                                                           PagedKVBatchLayerView cache);
    [[nodiscard]] bool stage_embeds() const noexcept { return stage_first_ == 0; }
    [[nodiscard]] bool stage_finishes() const noexcept { return stage_last_ == cfg_.n_layers; }
    void set_proposal_head(const Weight* weight, const std::int32_t* ids, int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_rewrite_checkpoint_frontier(std::int64_t position) noexcept {
        prefill_rewrite_checkpoint_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t current_slot, std::int32_t rewrite_checkpoint_slot);
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);
    /// The per-slot state a layer prologue keeps (null when the target has none).
    void set_ple_state(NgramPleStatePool* pool) noexcept { ple_state_ = pool; }
    // Column facts for the layer prologue, staged by each forward entry before its layers.
    PrologueColumns prologue_{};
    NgramPleStatePool* ple_state_ = nullptr;

    // Prefill CUDA graphs (PATCHES.md #27): non-null routes eligible prefill
    // chunks through bucket-captured graph bodies; null keeps the eager body.
    void set_prefill_graph_family(PrefillGraphFamily* family) noexcept {
        prefill_graph_family_ = family;
    }

    // Captures every bucket up to the effective chunk at load time so first
    // requests replay instead of paying capture (PATCHES.md #27). Requires a
    // family; stops early if the family dies.
    void precapture_prefill_graphs(std::int32_t effective_chunk);

    [[nodiscard]] const Weight* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    // Mixed-token round (PATCHES.md #30): one forward over concatenated
    // [prefill-chunk | decode-batch] columns. GEMM/fused ops run once over
    // all columns; the mixers split per slice (prefill kernels over the
    // chunk columns, batch forms over the decode columns).
    struct MixedDecodeSlice {
        Tensor ids;                // I32 [B]
        Tensor cache_positions;    // I32 [B]
        Tensor rope_positions;     // I32 [B]
        Tensor kv_table_rows;      // I32 [B]
        Tensor linear_state_slots; // I32 [B]
        ops::GqaExecutionEnvelope envelope{};
        Tensor hidden;             // BF16 [hidden, B] out
        Tensor logits;             // BF16 [vocab, B] out
    };
    [[nodiscard]] PrefillChunkResult
    mixed_chunk(std::span<const int> full_ids, std::uint32_t begin, std::uint32_t nominal_length,
                bool finalize_at_end, const MixedDecodeSlice& decode);

    // Multi-prompt prefill (#80): several prompts' chunks share one round. The GEMMs and
    // fused ops already run over concatenated columns, so only the two mixers split per
    // segment — each carries its own KV row, GDN/conv state slot and context base, and the
    // segments that finish sample together through the batched sampler. One segment is
    // exactly the single-prompt round above.
    struct MixedPrefillSegment {
        std::span<const int> ids;  // this prompt's chunk
        std::int32_t kv_base;      // tokens already resident for this sequence
        std::int32_t kv_table_row; // its paged-KV row; negative keeps the row the caller staged
        std::int32_t state_slot;   // its GDN/conv state slot
        bool finalize;             // sample after this chunk
    };
    // Staging for the segments that finish in this round: their last hidden columns are
    // gathered into `hidden`, one lm_head produces `logits`, and the batched sampler writes
    // `tokens`. Sized for the number of finalizing segments the caller allows.
    struct MixedPrefillFinalize {
        Tensor hidden;    // BF16 [hidden, F]
        Tensor logits;    // BF16 [vocab, F]
        Tensor positions;      // I32  [F] logical positions for the sampler
        Tensor rope_positions; // I32  [F] optional; the single-prompt path keeps io_.rope_pos
        Tensor tokens;    // I32  [F] out
        const ops::SamplingConfig* sampling = nullptr; // device array [F]
    };
    [[nodiscard]] PrefillChunkResult
    mixed_chunk_multi(std::span<const MixedPrefillSegment> segments,
                      const MixedDecodeSlice& decode, const MixedPrefillFinalize& finalize);

    // Mixed-round CUDA graphs (PATCHES.md #30): the mixed body captured at a
    // (chunk bucket, batch bucket) pair. The decode slice must be sliced to
    // the batch bucket, with pad rows staged as duplicates of a live row.
    [[nodiscard]] bool try_mixed_graph_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                             std::uint32_t nominal, const MixedDecodeSlice& decode,
                                             std::int32_t batch_bucket,
                                             std::int32_t band);
    void mixed_graph_window(std::int32_t chunk_bucket, std::int32_t batch_bucket);

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const family::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_slots, ops::GqaExecutionEnvelope envelope,
                               Tensor& hidden, Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                             ops::GqaExecutionEnvelope envelope, Tensor& hidden, Tensor& logits,
                             Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::GqaExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    void bind();

    /// The width of the hidden that crosses a round boundary: the wide residual when a
    /// trunk-block draft head is running, the model width otherwise.
    [[nodiscard]] std::int32_t round_hidden_width() const noexcept {
        // The model's frozen features, not this card's KV binding: the buffers were sized once
        // at startup, and a card that never binds MTP KV still writes into them.
        if constexpr (mtp_block_is_trunk_layer<Variant>()) {
            if (weights_.features.mtp()) { return weights_.geometry.residual; }
        }
        return cfg_.hidden;
    }

    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    [[nodiscard]] const MtpW& mtp_weights() const;
    // `index` selects this layer's KV plane (the full-attention index); `layer` is
    // the absolute layer index, which is what the per-layer rope base and sliding
    // window are declared over. They coincide only in a stack that is all attention.
    /// Which KV plane a mixer writes into. The draft head keeps its own, one plane deep, and
    /// attends densely -- the trunk's QSA only prunes past a 2,048-token budget, so dense is a
    /// numerical superset and the target verifies the draft either way.
    enum class KvPlane { Text, Mtp };
    void attn_mix(const FullLayerW& weights, Tensor& x, int index, int layer, Phase phase,
                  KvPlane plane = KvPlane::Text);
    void gdn_mix(const GdnLayerW& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const Tensor* post_norm, const MlpW& weights, Tensor& x, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows, const Tensor& linear_state_slots,
                                  ops::GqaExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    /// Fills `wide` -- the hidden that crosses the round boundary -- and returns the tensor
    /// the LM head reads. The same tensor unless a trunk-block draft head widened the
    /// boundary, in which case `wide` takes the residual and the collapse goes to a scratch.
    template <class V = Variant>
    Tensor finish_prefill(Tensor& wide, const Tensor& x, cudaStream_t stream);
    /// The LM head's view of a stored boundary hidden. Identity unless a trunk-block
    /// draft head widened it, in which case the trunk's output mixer collapses it.
    template <class V = Variant>
    Tensor lm_head_view(const Tensor& stored, cudaStream_t stream);
    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden);
    /// A draft head whose block is a trunk block: fold, then the trunk's own mixer and
    /// post-mixer over the wide residual, which is what the next draft step reads back.
    /// Templated on the Variant so the body is discarded for the targets that keep the fixed
    /// draft tail: their Variants have none of the hooks it names.
    template <class V = Variant>
    void mtp_forward_trunk_block(const Tensor& ids, const Tensor& hidden,
                                 const Tensor* input_embeddings, const Tensor& positions,
                                 const Tensor& rope_positions,
                                 ops::GqaExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions, ops::GqaExecutionEnvelope envelope,
                          Tensor& mtp_hidden, const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::GqaExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    /// `hidden` is the residual the draft head produced: the model width for a fixed-tail
    /// head, the wide stream for a trunk-block one, which this collapses.
    template <class V = Variant>
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    // Prefill CUDA graphs (PATCHES.md #27).
    [[nodiscard]] bool try_prefill_graph_chunk(std::span<const int> ids, int t0, int len,
                                               int base_i, bool is_last, int checkpoint_rel);
    void prefill_graph_window(std::int32_t bucket);



    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const LoadedModelData& weights_;
    WorkspaceArena& work_;
    family::PagedKVCacheView kv_;
    family::PagedKVCacheView mtp_kv_;
    const family::PagedKVCache* batch_text_kv_ = nullptr;
    const family::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    family::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                 = nullptr;
    const Tensor* active_rope_positions_                  = nullptr;
    const Tensor* active_kv_table_rows_                   = nullptr;
    const Tensor* active_linear_state_slots_              = nullptr;
    const Tensor* active_valid_columns_                   = nullptr;
    const Tensor* active_backend_kv_table_rows_           = nullptr;
    const ops::GqaExecutionEnvelope* active_gqa_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                   = 0;
    std::int32_t active_sequence_width_                   = 0;
    std::int32_t rope_delta_                              = 0;
    std::int32_t linear_state_current_slot_               = 0;
    std::int32_t linear_state_rewrite_checkpoint_slot_    = 0;
    GdnStateAction gdn_state_action_                      = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_               = nullptr;
    std::int64_t prefill_rewrite_checkpoint_frontier_     = -1;
    PrefillGraphFamily* prefill_graph_family_             = nullptr;
    Tensor graph_pad_valid_storage_;
    MixedDecodeSlice mixed_graph_decode_{};
    const Tensor* graph_pad_valid_                        = nullptr;
    Tensor* rewrite_checkpoint_hidden_output_             = nullptr;
    std::uint32_t mtp_proposal_extent_                    = 0;

    int stage_first_                            = 0;
    /// The geometry this context runs. It defaults to the target's compiled `TextConfig`, so a
    /// registered model is unchanged; holding it per context rather than reading a namespace
    /// constant is what lets two engines of different sizes share one process.
    ModelConfig cfg_{};
    /// The same dimensions as the value the workspace recipe is shaped by.
    [[nodiscard]] const family::TextGeometry& cfg_geometry() const noexcept {
        return weights_.geometry;
    }
    int stage_last_                             = cfg_.n_layers;
    StageSpan stage_{};
    void stage_import(Tensor& x, cudaStream_t stream);
    void stage_export(const Tensor& x, cudaStream_t stream);

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const Weight* lm_head_                      = nullptr;
    const Weight* proposal_head_                = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    MtpW mtp_;
    // Sized from the bound geometry rather than by the type: a checkpoint of this family
    // may have a different number of layers, and so a different split between them.
    std::vector<FullLayerW> full_{};
    std::vector<GdnLayerW> gdn_{};
    std::vector<Weight> gdn_in_a_{};
    std::vector<Weight> gdn_in_b_{};
    std::vector<Tensor> gdn_conv1d_views_{};
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS::schedule
