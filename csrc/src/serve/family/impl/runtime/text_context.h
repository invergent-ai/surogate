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
#include "api/ops/short_conv.h"
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

struct ModelConfig {
    // Values are initialized only from the geometry bound to the loaded weights.
    int hidden              = 0;
    int residual            = 0;
    int n_layers            = 0;
    int intermediate        = 0;
    int vocab               = 0;
    int token_domain        = 0;
    int gdn_k_heads         = 0;
    int gdn_k_dim           = 0;
    int gdn_v_heads         = 0;
    int gdn_v_dim           = 0;
    int n_q                 = 0;
    int n_kv                = 0;
    int head_dim            = 0;
    int rotary_dim          = 0;
    int sliding_rotary_dim  = 0;
    int key_dim             = 0;
    int value_dim           = 0;
    int conv_dim            = 0;
    int q_size              = 0;
    int kv_size             = 0;
    int mtp_fc_in           = 0;
    int mtp_attn_in         = 0;
    int mtp_mlp_gateup_rows = 0;
    float rms_eps           = 0;
    float rope_theta        = 0;
    float attention_scale   = 0;
    float gdn_scale         = 0;
    float logit_softcap     = 0;
    int mtp_layers          = 0;

    std::array<std::uint64_t, 4> attention_mask{};
    bool schedule_declared = false;

    /// The *second* attention geometry, and which layers attend at it.
    ///
    /// Gemma 4 attends through its window with 8 key/value heads of 256 and over the whole
    /// context with 1 head of 512, so `n_kv`, `head_dim`, `q_size` and `kv_size` above
    /// describe only half its layers. Every other family leaves these zero and the
    /// `layer_*` accessors below then answer with the single geometry, unchanged.
    ///
    /// The window schedule is a second mask for the same reason the first one exists, and a
    /// stronger one: a target serving two sizes cannot compile it, because Gemma 4's 12B has
    /// 48 layers and its 31B has 60.
    int global_n_kv      = 0;
    int global_head_dim  = 0;
    /// Which layers hold key/value planes of their own, for a model whose tail shares them.
    /// Undeclared means every attending layer owns its planes, which is every family but the
    /// Gemma 4 E-series.
    std::array<std::uint64_t, 4> kv_owner_mask{};
    bool kv_sharing_declared = false;
    /// How many of a global head's pairs carry a real rope angle. Zero means all of them.
    int global_rotary_angles = 0;
    std::array<std::uint64_t, 4> windowed_mask{};
    bool window_declared = false;

    /// Whether `layer` looks through the window. Layers of a model that declares no window
    /// schedule are all global, which is what makes the accessors below answer with the one
    /// geometry for every family but Gemma 4.
    [[nodiscard]] bool layer_windowed(int layer) const noexcept {
        if (!window_declared) { return false; }
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= windowed_mask.size()) { return false; }
        return ((windowed_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    /// This layer's attention geometry. A model with one geometry answers the same for every
    /// layer, so a caller need not know whether the family it serves has two.
    [[nodiscard]] int layer_head_dim(int layer) const noexcept {
        return (global_head_dim <= 0 || layer_windowed(layer)) ? head_dim : global_head_dim;
    }
    [[nodiscard]] int layer_n_kv(int layer) const noexcept {
        return (global_n_kv <= 0 || layer_windowed(layer)) ? n_kv : global_n_kv;
    }
    [[nodiscard]] int layer_q_size(int layer) const noexcept { return n_q * layer_head_dim(layer); }
    [[nodiscard]] int layer_kv_size(int layer) const noexcept {
        return layer_n_kv(layer) * layer_head_dim(layer);
    }

    /// Whether this layer writes its own keys and values, rather than reading an earlier
    /// layer's.
    [[nodiscard]] bool layer_owns_kv(int layer) const noexcept {
        if (!kv_sharing_declared) { return true; }
        const auto word = static_cast<std::size_t>(layer) / 64U;
        if (layer < 0 || word >= kv_owner_mask.size()) { return false; }
        return ((kv_owner_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    /// The key/value plane this layer attends over.
    ///
    /// Its own where it has one. Where it does not, the plane of the last *owning* layer
    /// before it that attends the same way -- windowed layers share a windowed layer's keys
    /// and global layers a global one's, because the two see different spans and a shared
    /// layer applies its own mask over what it reads. That is the reference's rule
    /// (`store_full_length_kv` marks the last non-sharing layer of each `layer_type`), and
    /// getting it wrong is a layer attending over the wrong history rather than an error.
    [[nodiscard]] int kv_plane_index(int layer) const {
        if (!kv_sharing_declared) { return full_idx(layer); }
        int source = layer;
        if (!layer_owns_kv(layer)) {
            const bool windowed = layer_windowed(layer);
            source              = -1;
            for (int earlier = layer - 1; earlier >= 0; --earlier) {
                if (layer_owns_kv(earlier) && is_full(earlier) &&
                    layer_windowed(earlier) == windowed) {
                    source = earlier;
                    break;
                }
            }
            if (source < 0) {
                throw std::logic_error(
                    "a layer shares key/value planes but no earlier layer of its own "
                    "attention kind owns any");
            }
        }
        int index = 0;
        for (int earlier = 0; earlier < source; ++earlier) {
            index += (is_full(earlier) && layer_owns_kv(earlier)) ? 1 : 0;
        }
        return index;
    }

    /// How many key/value planes the cache holds: one per owning attention layer.
    [[nodiscard]] int n_kv_planes() const {
        if (!kv_sharing_declared) { return n_full(); }
        int count = 0;
        for (int layer = 0; layer < n_layers; ++layer) {
            count += (is_full(layer) && layer_owns_kv(layer)) ? 1 : 0;
        }
        return count;
    }

    /// The rotation this layer applies: how wide it is, and how many of its pairs are real.
    ///
    /// A global Gemma 4 head rotates over its **whole** 512 width with only its first 64
    /// pairs carrying an angle -- the pairs are (i, i + 256) either way, and narrowing the
    /// rotation instead would pair each channel with a different partner.
    [[nodiscard]] int layer_rotary_dim(int layer) const noexcept {
        if (layer_windowed(layer) && sliding_rotary_dim > 0) { return sliding_rotary_dim; }
        return (global_head_dim <= 0 || layer_windowed(layer)) ? rotary_dim : global_head_dim;
    }
    [[nodiscard]] int layer_rotary_pairs(int layer) const noexcept {
        if (global_head_dim <= 0 || layer_windowed(layer)) { return layer_rotary_dim(layer) / 2; }
        return global_rotary_angles > 0 ? global_rotary_angles : global_head_dim / 2;
    }

    /// The widest of the model's geometries, which is what a plane every layer shares has to
    /// be sized for. Equal to the single geometry wherever there is only one.
    [[nodiscard]] int max_head_dim() const noexcept {
        return global_head_dim > head_dim ? global_head_dim : head_dim;
    }
    [[nodiscard]] int max_n_kv() const noexcept { return global_n_kv > n_kv ? global_n_kv : n_kv; }
    [[nodiscard]] int max_q_size() const noexcept { return n_q * max_head_dim(); }
    [[nodiscard]] int max_kv_size() const noexcept {
        const int windowed = n_kv * head_dim;
        const int global   = global_n_kv * global_head_dim;
        return global > windowed ? global : windowed;
    }

    /// Whether `layer` attends. No longer static: for a family whose checkpoint picks its own
    /// layer kinds -- LFM2 attends at six irregular layers of sixteen, and at different ones
    /// per size -- the answer is a property of the model that was loaded, not of the target
    /// that was compiled.
    [[nodiscard]] constexpr bool is_full(int layer) const {
        const auto word = static_cast<std::size_t>(layer) / 64U;
        return word < attention_mask.size() &&
               ((attention_mask[word] >> (static_cast<unsigned>(layer) % 64U)) & 1U) != 0U;
    }

    /// How many layers attend and how many are linear.
    [[nodiscard]] constexpr int n_full() const {
        int count = 0;
        for (int layer = 0; layer < n_layers; ++layer) { count += is_full(layer) ? 1 : 0; }
        return count;
    }

    [[nodiscard]] constexpr int n_gdn() const { return n_layers - n_full(); }

    /// A layer's index among its own kind. Counted rather than computed when the schedule is
    /// declared, because an irregular one has no closed form; the loop is over layers already
    /// walked, on the launch path, and costs nothing measurable next to the round it launches.
    [[nodiscard]] constexpr int full_idx(int layer) const {
        return count_before(layer, true);
    }

    [[nodiscard]] constexpr int gdn_idx(int layer) const {
        return count_before(layer, false);
    }

    [[nodiscard]] constexpr int count_before(int layer, bool attending) const {
        int index = 0;
        for (int earlier = 0; earlier < layer; ++earlier) {
            index += is_full(earlier) == attending ? 1 : 0;
        }
        return index;
    }

    ModelConfig() = delete;

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
          sliding_rotary_dim(geometry.sliding_rotary_dim),
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
          attention_scale(geometry.attention_scale),
          gdn_scale(geometry.gdn_scale),
          logit_softcap(geometry.logit_softcap),
          mtp_layers(geometry.mtp_layers) {
        if (!geometry.attention_schedule_declared || !geometry.windowed_schedule_declared) {
            throw std::invalid_argument("runtime geometry requires a declared layer schedule");
        }
        if (geometry.attention_schedule_declared) {
            attention_mask    = geometry.attention_layer_mask;
            schedule_declared = true;
        }
        if (geometry.has_global_attention_geometry()) {
            global_n_kv          = geometry.global_kv_heads;
            global_head_dim      = geometry.global_head_dim;
            global_rotary_angles = geometry.global_rotary_angles;
        }
        if (geometry.windowed_schedule_declared) {
            windowed_mask   = geometry.windowed_layer_mask;
            window_declared = true;
        }
        if (geometry.kv_sharing_declared) {
            kv_owner_mask       = geometry.kv_owner_mask;
            kv_sharing_declared = true;
        }
    }
};

using Hooks = ResidualHooks<Variant>;
// False only for a target whose attention has no output gate (a dense GQA
// stack); every hybrid target in the family leaves it at the default.
inline constexpr bool kAttentionOutputGate            = family::detail::attention_output_gate<Variant>();
// Which mixer the non-attending layers run. A compile-time constant, so the branch that is not
// this target's costs nothing and the leaves it would have called are never reached.
inline constexpr family::LinearMixer kLinearMixer      = family::detail::linear_mixer<Variant>();
inline constexpr std::uint32_t kPrefillChunkAlignment = 128;

struct MlpW {
    const MlpWeights* payload = nullptr;
};

/// One prefill segment of a mixed round, as the short-convolution mixer needs it: where its
/// columns start, how many there are, and which state slot continues its history.
struct ShortConvSegment {
    std::int32_t offset     = 0;
    std::int32_t columns    = 0;
    std::int32_t state_slot = 0;
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

/// Pipeline stage: the layer range this program runs, and the pinned buffers the residual
/// crosses at the boundaries (import when first > 0, export when last < layers). Whole-model
/// programs leave the defaults.
struct StageSpan {
    int first                 = 0;
    int last                  = -1;      // -1: through the last layer
    const void* import_pinned = nullptr; // [residual, columns] BF16, written by the previous stage
    void* export_pinned       = nullptr; // [residual, columns] BF16, read by the next stage
    std::int32_t columns      = 0;       // capacity of both buffers
    std::size_t column_bytes = 0;
    std::size_t residual_bytes = 0;
    Tensor* features = nullptr; // compact DFlash feature columns carried beside the residual

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
    StageSpan stage;
    cudaStream_t stream = nullptr;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

class VisionPrefillSession;


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
    /// This round's token ids. A per-layer input is an embedding lookup, so the block's
    /// epilogue needs them, and nothing else inside the layer loop carries them: `prologue_`
    /// is built only for a target that declares a *layer prologue*, which is a different
    /// feature and a different set of targets.
    Tensor active_ids_{};
    /// The model's embedded input, as it stood before the first block.
    ///
    /// Gemma 4's E-series projects *this* into every layer's per-layer input -- once, from the
    /// token embedding -- not the running residual, which is what the reference's
    /// `project_per_layer_inputs(inputs_embeds, ...)` takes. A copy, because the layers write
    /// the residual in place.
    Tensor active_embedded_{};
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
        // The draft head, aligned over this chunk's columns on the stage that holds it: the
        // head's paged-KV row (a trunk-block head addresses the batch view by it), the
        // segment's own per-sequence head view (a fixed-tail head appends through it), and
        // the prompt's next token per column (the ids shifted by one). A negative row leaves
        // the head out of the round.
        std::int32_t mtp_kv_table_row = -1;
        family::PagedKVCacheView mtp_kv{};
        std::span<const int> mtp_shifted_ids{};
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
    /// Project a stored round boundary through the model's output transform and head.
    void logits_from_hidden(const Tensor& hidden, Tensor& logits);
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
    /// The short-convolution mixer, for a family whose non-attending layers run one. Same slot
    /// in the same schedule as `gdn_mix`, and the same three phases; a different mixer.
    void short_conv_mix(const GdnLayerW& weights, Tensor& x, int index, Phase phase);
    void short_conv_mix_mixed(const GdnLayerW& weights, Tensor& x, int index,
                              std::span<const ShortConvSegment> segments,
                              std::int32_t prefill_columns, std::int32_t batch,
                              const Tensor& valid, const Tensor& decode_slots);
    /// Keep the embedded input for a family whose blocks read it. A no-op for the rest.
    void capture_per_layer_source(const Tensor& x, cudaStream_t stream);

    void mlp_tail(const Tensor* post_norm, const MlpW& weights, Tensor& x, int layer, Phase phase);
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap, const Tensor* deepstack = nullptr,
                    std::span<const std::int32_t> visual_indices = {});
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
    /// The draft head's attended heads to the model width: the target's own leaf where it has
    /// one (an absorbed attention unfolds per head first), the family's one linear otherwise.
    void mtp_attention_output(const Tensor& attention, Tensor& out);
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
    /// The geometry bound to this context's loaded weights.
    ModelConfig cfg_;
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
