#pragma once
#include "family/impl/adaptive_dflash.h"
#include "runtime/contract/constraint.h"
#include "family/impl/runtime/speculative_constraint.h"
#include <api/family/text_geometry.h>

#include "runtime/contract/round_lifecycle.h"
#include "family/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "api/ops/sampling.h"
#include "core/decode_graph.h"
#include <api/family/prepared_prompt.h>

#include "family/impl/runtime/layouts.h"
#include "family/impl/runtime/prefill_graph.h"
#include "family/impl/runtime/dflash_context.h"
#include "family/impl/runtime/linear_state_slots.h"
#include "family/impl/runtime/prefix_identity.h"
#include "family/impl/archive_storage.h"
#include "family/impl/radix_prefix_cache.h"
#include "family/impl/runtime/text_context.h"
#include "family/impl/runtime/vision_context.h"
#include "family/impl/runtime/vision_prefill.h"

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

using PreparedPromptData    = family::PreparedPromptData;
struct ArchivedSequence;
struct GpuPrefix;
struct GpuPrefixStorage;
struct SharedPrefix;
using RewriteCheckpointKind = family::RewriteCheckpointKind;
using RewriteCheckpointSpec = family::RewriteCheckpointSpec;

using ReusePath = sinfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS

namespace sinfer::family::detail {

template <>
struct RequestBasePlanImpl<SINFER_FAMILY_VARIANT> {
    runtime::RequestPlanSummary summary;
    // One-step readout for finite-choice classification. Values are raw logits,
    // in this token order, independent of sampling filters and penalties.
    std::vector<TokenId> next_token_candidates;
    bool cache_prompt = false;
    bool target_only = false;
    std::shared_ptr<const GpuPrefixKey> gpu_prefix;
    std::shared_ptr<const GpuPrefixKey> save_gpu_prefix;
    std::shared_ptr<SINFER_FAMILY_RUNTIME_NS::GpuPrefixStorage> gpu_storage;
    int prompt_logprobs = -1;
    int top_logprobs = -1;
    ops::SamplingConfig sampling;
    std::unordered_map<TokenId, float> logit_bias;
    std::shared_ptr<const CompiledTokenConstraint> constraint;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const family::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<family::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
    /// Chained hashes of the prompt's whole 64-token pages before its last token, for the
    /// shared-prefix cache; null when the request cannot use it.
    std::shared_ptr<const std::vector<std::uint64_t>> page_hashes;
    /// The adapter slot the request selected; copied into RequestControl at admit.
    std::int32_t lora_slot = -1;
    /// A minimum length and the stop ids barred until it is reached; both copied
    /// into RequestControl at admit, where the per-round staging reads them.
    std::uint32_t min_tokens         = 0;
    std::uint32_t stop_barrier_count = 0;
};

template <>
struct RequestPlanImpl<SINFER_FAMILY_VARIANT> {
    runtime::RequestPlanSummary summary;
    SINFER_FAMILY_RUNTIME_NS::ReusePath reuse = SINFER_FAMILY_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    SINFER_FAMILY_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        SINFER_FAMILY_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    bool retain_prefix = true;
    std::shared_ptr<const SINFER_FAMILY_RUNTIME_NS::ArchivedSequence> archived;
    std::shared_ptr<const SINFER_FAMILY_RUNTIME_NS::GpuPrefix> device_prefix;
    std::shared_ptr<SINFER_FAMILY_RUNTIME_NS::SharedPrefix> shared_prefix;
    std::shared_ptr<const std::vector<std::uint64_t>> page_hashes;
    std::optional<SINFER_FAMILY_RUNTIME_NS::VisionPrefillPlan> vision;
    SINFER_FAMILY_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        SINFER_FAMILY_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<family::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    std::uint32_t reusable_scores = 0;
    // One-step readout for finite-choice classification. Values are raw logits,
    // in this token order, independent of sampling filters and penalties.
    std::vector<TokenId> next_token_candidates;
    bool cache_prompt = false;
    bool target_only = false;
    std::shared_ptr<const GpuPrefixKey> gpu_prefix;
    std::shared_ptr<const GpuPrefixKey> save_gpu_prefix;
    std::shared_ptr<SINFER_FAMILY_RUNTIME_NS::GpuPrefixStorage> gpu_storage;
    int prompt_logprobs = -1;
    int top_logprobs = -1;
    ops::SamplingConfig sampling;
    std::unordered_map<TokenId, float> logit_bias;
    std::shared_ptr<const CompiledTokenConstraint> constraint;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    /// The adapter slot the request selected; read once at admission.
    std::int32_t lora_slot = -1;
    std::uint32_t min_tokens         = 0;
    std::uint32_t stop_barrier_count = 0;
};

} // namespace sinfer::family::detail

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

using RequestPlanImpl     = family::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = family::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
    /// A speculative round that ran narrow: one column, its recurrent state already updated
    /// in place, nothing recorded -- the resolve folds zero columns for this lane.
    bool in_place_state = false;
};

/// What one speculative round decided for one lane, copied out of the round's host egress
/// the moment the round is consumed. The egress is one buffer per program, and the next
/// round's copy into it is enqueued as soon as that round launches -- so a lane whose
/// resolution comes after another round's launch (a pipeline keeping several groups in
/// flight) would read the wrong round's decision from it. This is the lane's own copy, and
/// it is also the record a stage without the head adopts from the stage that has it.
struct SpeculativeOutcome {
    std::int32_t licensed_count  = 0;
    std::int32_t accepted_drafts = 0;
    std::int32_t next_extent     = 0;
    std::array<TokenId, family::kDFlashDecodeMaximumWidth> licensed_tokens{};
    std::array<TokenId, family::kMtpDecodeMaximumDrafts> next_drafts{};
};
static_assert(std::is_trivially_copyable_v<SpeculativeOutcome>);

/// A lane's draft state after a prefill: what the head proposed for the first round. On a
/// pipeline only the stage with the head proposes anything; the others adopt this.
struct LaneDraftState {
    std::uint32_t count = 0;
    std::array<TokenId, family::kMtpDecodeMaximumDrafts> drafts{};
};
static_assert(std::is_trivially_copyable_v<LaneDraftState>);

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

/// A decode graph family and its base-round twin.
///
/// A graph records the launches its capture saw, and a base round -- one none of whose rows
/// selects an adapter, run under ops::ScopedLoraBaseRound -- launches the base routes where
/// any other round of an adapter-serving engine launches the adapter-capable ones. They are
/// different graphs over the same profiles, so an engine that carries adapters captures both
/// and each round replays the one of its own flavor. `rounds` is the family every engine has;
/// `base_rounds` stays empty unless the engine's rounds can be base rounds.
struct DecodeGraphFlavors {
    DecodeGraphFamily rounds;
    DecodeGraphFamily base_rounds;
    [[nodiscard]] DecodeGraphFamily& of(bool base_round) noexcept {
        return base_round ? base_rounds : rounds;
    }
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    std::vector<TokenScore> cached_scores;
    family::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, family::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    bool cacheable                = true;
    bool target_only              = false;
    RewriteCheckpoint rewrite_checkpoint;
    void copy_metadata(const SequenceState& source) {
        execution_frontier = source.execution_frontier; ledger_frontier = source.ledger_frontier;
        ledger = source.ledger; cached_scores = source.cached_scores;
        prefix_identity = source.prefix_identity; rope_delta = source.rope_delta;
        text_kv_valid = source.text_kv_valid; mtp_kv_valid = source.mtp_kv_valid;
        dflash_context_frontier = source.dflash_context_frontier;
        mtp_drafts = source.mtp_drafts; mtp_draft_count = source.mtp_draft_count;
        tail_hidden_valid = source.tail_hidden_valid; retained = source.retained;
        cacheable = source.cacheable; target_only = source.target_only;
        rewrite_checkpoint = source.rewrite_checkpoint;
    }
};

struct GpuPrefixStorage {
    DeviceArena memory;
    std::size_t stride;
    std::vector<std::uint32_t> free;
    GpuPrefixStorage(std::size_t bytes, std::uint32_t count) : memory(bytes * count), stride(bytes) {
        free.reserve(count);
        for (std::uint32_t i = 0; i < count; ++i) free.push_back(count - i - 1);
    }
};
struct GpuPrefix {
    std::weak_ptr<const GpuPrefixKey> key;
    SequenceState state;
    std::shared_ptr<SequenceKVBundle> pages;
    std::shared_ptr<GpuPrefixStorage> storage;
    std::uint32_t slot = 0;
    std::vector<Tensor> current;
    ~GpuPrefix() { if (storage) storage->free.push_back(slot); }
};

// A prompt prefix several conversations share (a system prompt, tool definitions), kept on
// the GPU at a page boundary: its full KV pages, which forks borrow in place, and a copy of
// the lane's recurrent state there. `state` is the sequence a fork resumes from.
struct SharedPrefix {
    std::uint64_t hash = 0;
    std::uint32_t tokens = 0;
    SequenceState state;
    std::shared_ptr<const PagedKVAllocation> pages;
    std::shared_ptr<GpuPrefixStorage> storage;
    std::uint32_t slot = 0;
    std::vector<Tensor> current;
    std::uint64_t last_use = 0;
    std::uint32_t hits = 0;
    ~SharedPrefix() { if (storage) storage->free.push_back(slot); }
};

/// A finished lane's prefix kept in host memory after its lane went to another request. Every
/// byte lives in `storage`; the views below are its tensors and KV planes.
struct ArchivedSequence {
    SequenceState state;
    family::detail::ArchiveBlock storage;
    std::uint32_t text_pages = 0, backend_pages = 0;
    std::vector<std::byte*> text, backend;
    std::vector<std::span<std::byte>> current, checkpoint;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    /// Valid while `pending.kind == Speculative`: the round's decision for this lane.
    SpeculativeOutcome outcome;
    // One-step readout for finite-choice classification. Values are raw logits,
    // in this token order, independent of sampling filters and penalties.
    std::vector<TokenId> next_token_candidates;
    bool cache_prompt = false;
    bool target_only = false;
    std::shared_ptr<const GpuPrefixKey> gpu_prefix;
    std::shared_ptr<const GpuPrefixKey> save_gpu_prefix;
    std::shared_ptr<GpuPrefixStorage> gpu_storage;
    int prompt_logprobs = -1;
    int top_logprobs = -1;
    std::vector<TokenScore> prompt_scores;
    std::vector<TokenScore> completion_scores;
    std::vector<float> next_token_logits;
    TokenId readout_token = 0;
    float readout_logprob = std::numeric_limits<float>::quiet_NaN();
    ops::SamplingConfig sampling_host;
    std::vector<float> logit_bias_host;
    std::unique_ptr<TokenConstraintState> constraint;
    std::vector<std::int32_t> token_bitmask_host;
    std::size_t constraint_frontier = 0;
    /// The adapter slot this request selected; staged per lane every round.
    std::int32_t lora_slot = -1;
    /// A minimum length, and what it takes to honour it: the stop ids stay barred
    /// in `sampling_host` until the request has produced this many tokens. Zero
    /// means the request asked for no minimum and nothing is ever barred.
    std::uint32_t min_tokens         = 0;
    std::uint32_t stop_barrier_count = 0;
    std::uint32_t prompt_tokens      = 0;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        // Recurrent prefill keeps its chunk boundaries when snapshot storage is unavailable.
        std::optional<std::uint32_t> recurrent_boundary;
        // A shared-prefix capture: the chunk stops here and the lane's state is kept.
        std::optional<std::uint32_t> shared_capture;
        std::uint64_t shared_capture_hash = 0;
        bool shared_restore = false; // resumed from a shared prefix (reported, not a path)
        [[nodiscard]] std::optional<std::uint32_t> chunk_boundary() const noexcept {
            std::optional<std::uint32_t> boundary;
            if (recurrent_boundary) { boundary = recurrent_boundary; }
            else if (rewrite_checkpoint_capture) { boundary = rewrite_checkpoint_capture->frontier; }
            if (shared_capture && cursor < *shared_capture &&
                (!boundary || cursor >= *boundary || *shared_capture < *boundary)) {
                boundary = shared_capture;
            }
            return boundary;
        }
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        bool use_graph                   = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);

    /// The dimensions this program runs at: the weights' own, which is the target's compiled
    /// config with the artifact's declaration laid over it.
    family::TextGeometry cfg;
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient,
                                                                bool defer_first_chunk = false);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::MixedRoundResult
    advance_prefill_mixed(std::span<const std::uint32_t> prefill_lanes, std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] bool mixed_round_supported(std::uint32_t prefill_lane, std::uint32_t decode_rows) const noexcept;
    /// --batch-invariant (api/ops/batch_invariant.h): prompts are cut only at multiples of this
    /// many tokens from position zero, whatever else shares their rounds. Zero when off.
    [[nodiscard]] std::uint32_t invariant_prefill_chunk() const noexcept;
    /// The prompt tokens a staged prefill advances by next under --batch-invariant: up to its
    /// next invariant cut, or to its end.
    [[nodiscard]] std::uint32_t invariant_prefill_piece(const RequestControl::Prefill& staged) const noexcept;
    [[nodiscard]] std::string last_mixed_round_description(std::size_t row) const;
    void set_round_burst_limit(std::uint32_t limit) noexcept { round_burst_limit = limit; }
    [[nodiscard]] std::uint32_t reusable_append_frontier(const SequenceState& sequence) const noexcept;
    [[nodiscard]] bool acquire_rewrite_checkpoint(SequenceState& sequence);
    [[nodiscard]] std::uint64_t prefix_cache_revision(std::uint32_t lane) const noexcept {
        return decoder->checkpoint_revision() + archived_prefixes.revision() + shared_prefix_revision +
               (lane < checkpoint_revisions.size() ? checkpoint_revisions[lane] : 0);
    }
    static void burst_egress_copy_host(void* user) noexcept;
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    /// Applies the lanes' deferred folds now, for a round that reads their state without the
    /// replay-record verify that would apply them (an ordinary, narrow or one-column round).
    void settle_deferred_folds(std::span<const std::uint32_t> lanes);
    /// Brings the device's deferred fold counts up to date before a round can read them.
    void flush_deferred_fold_counts();
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    void evict_archived_prefixes() noexcept;
    TokenScoreDelta logprob_delta(std::uint32_t lane, std::size_t first, std::size_t end, bool prompt) const;
    void cache_logprobs(std::uint32_t lane, const GenerationResult& result);
    void collect_logprobs(std::uint32_t lane, GenerationResult& result);
    std::function<void(const Tensor&, int, bool)> score_observer(std::uint32_t lane);
    void score_prefill_hidden(std::uint32_t lane, const Tensor& hidden, int base);
    void append_completion_score(std::uint32_t lane, const RawTokenScores& score);
    void score_completion(std::uint32_t lane, const Tensor& logits, TokenId token);
    void finish_readout_prefills(std::span<const std::uint32_t> lanes, runtime::MixedRoundResult& result);
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    /// Main KV pool occupancy. Cheap and allocation-free: the executor samples it every
    /// time it publishes runtime stats.
    [[nodiscard]] PagedKVOccupancy kv_occupancy() const noexcept;
    /// Blocks until an elastic Main pool has no map or unmap work pending (no-op otherwise).
    void kv_settle() noexcept;
    /// An elastic Main pool on a device whose gate recently refused an entitlement.
    [[nodiscard]] bool kv_under_pressure() const noexcept;
    /// Round boundary: perform a reserve release another engine asked for, report pressure.
    bool kv_service_pressure() noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    DeviceContext& device;
    schedule::StageSpan stage{}; // pipeline stage of this program (whole model by default)
    // Pinned export buffer of a stage before the last ([residual, boundary columns] BF16).
    struct PinnedBoundary {
        void* data = nullptr;
        ~PinnedBoundary() { if (data != nullptr) { cudaFreeHost(data); } }
    } stage_export, stage_import;
    std::size_t stage_boundary_bytes_ = 0;
    [[nodiscard]] const void* stage_export_buffer() const noexcept { return stage_export.data; }
    [[nodiscard]] void* stage_import_buffer() const noexcept { return stage_import.data; }
    [[nodiscard]] std::size_t stage_boundary_bytes() const noexcept { return stage_boundary_bytes_; }
    [[nodiscard]] std::int32_t stage_boundary_columns() const noexcept { return stage.columns; }
    /// True for a program that runs only part of the model (a pipeline stage).
    [[nodiscard]] bool pipeline_stage() const noexcept {
        return stage.first > 0 || (stage.last >= 0 && stage.last < cfg.layers);
    }
    /// True where the logits are: a whole-model program, or the pipeline stage that runs the
    /// last layer. A speculative round is decided here and adopted everywhere else.
    [[nodiscard]] bool stage_holds_head() const noexcept {
        return !(stage.last >= 0 && stage.last < cfg.layers);
    }
    /// The pipeline chooses one draft count and applies it to every stage of the flight.
    [[nodiscard]] std::uint32_t select_dflash_draft_window(std::span<const std::uint32_t> lanes);

    void set_dflash_draft_window(std::uint32_t drafts) { pipeline_dflash_window = drafts; }
    /// Maximum columns per lane; pipeline storage remains sized for the configured ceiling.
    [[nodiscard]] std::uint32_t speculative_round_width() const noexcept {
        return speculative_backend != SpeculativeBackend::None ? draft_window + 1U : 1U;
    }
    /// How many decode lanes the executor has in flight this round, across every group of a
    /// pipeline: the MTP round verifies drafts only while that is within
    /// `speculative_max_lanes`, and runs its narrow round otherwise. A program that is never
    /// told sees its own batch.
    void set_round_width_hint(std::uint32_t lanes) noexcept { round_width_hint_ = lanes; }
    [[nodiscard]] bool narrow_round_for(std::size_t lanes) const noexcept {
        // SUROGATE_SERVE_MTP_FORCE_NARROW=1: every MTP round narrow, whatever the width -- the
        // diagnostic that isolates the narrow round from the switch into it.
        static const bool force_narrow = std::getenv("SUROGATE_SERVE_MTP_FORCE_NARROW") != nullptr;
        if (force_narrow) { return true; }
        if (speculative_max_lanes == kSpeculateAtAnyWidth) { return false; }
        return std::max<std::uint32_t>(round_width_hint_, static_cast<std::uint32_t>(lanes)) >
               speculative_max_lanes;
    }
    /// The decision of the round this program last consumed, one `SpeculativeOutcome` per
    /// row in the round's lane order, for the pipeline driver to hand to the stages without
    /// the head. Valid until the next consume.
    [[nodiscard]] std::span<const std::byte> speculative_outcome() const noexcept {
        return std::span<const std::byte>(outcome_export_.data(), outcome_export_.size());
    }
    /// A stage without the head: takes the head stage's decision for these lanes, whose
    /// rounds it ran headless and left pending with nothing produced. After this the lanes
    /// resolve exactly as they do on the head stage.
    void adopt_speculative_outcome(std::span<const std::uint32_t> lanes,
                                   std::span<const std::byte> outcome);
    /// The lane's draft state (`LaneDraftState`) after its prefill completed, and its
    /// adoption on a stage whose own prefill proposed nothing.
    [[nodiscard]] std::span<const std::byte> lane_draft_state(std::uint32_t lane) const;
    void adopt_lane_draft_state(std::uint32_t lane, std::span<const std::byte> state);
    void adopt_pipeline_prefill_features(std::uint32_t lane, std::span<const std::byte> packet,
                                          std::uint32_t tokens, std::int32_t mixed_base = -1);
    void adopt_pipeline_decode_features(std::span<const std::uint32_t> lanes,
                                         std::span<const std::byte> packet);

    void configure_stage(const SequencePlanImpl& plan);
    /// Pipeline stages without the head record a placeholder token per round; the driver
    /// replaces each lane's last ledger entry with the token the last stage sampled.
    void replace_pending_tokens(std::span<const std::uint32_t> lanes, std::span<const TokenId> tokens);
    /// Pipeline driver access to the decode round's two halves, whichever round the backend
    /// runs: the ordinary round, or the MTP verify round.
    [[nodiscard]] runtime::RoundHandle launch_decode_round(std::span<const std::uint32_t> lanes,
                                                           std::span<const runtime::RoundBudget> budgets) {
        if (speculative_backend == SpeculativeBackend::Mtp) { return launch_mtp_round(lanes, budgets); }
        if (speculative_backend == SpeculativeBackend::DFlash) {
            return launch_dflash_round(lanes, budgets);
        }
        return launch_ordinary_round(lanes, budgets);
    }
    [[nodiscard]] runtime::BatchedGeneratedRound consume_decode_round(runtime::RoundHandle handle) {
        if (speculative_backend == SpeculativeBackend::Mtp) { return consume_mtp_round(handle); }
        if (speculative_backend == SpeculativeBackend::DFlash) { return consume_dflash_round(handle); }
        return consume_ordinary_round(handle);
    }
    [[nodiscard]] runtime::RoundHandle launch_mixed_round(std::span<const std::uint32_t> prefill_lanes,
                                                          std::span<const std::uint32_t> lanes,
                                                          std::span<const runtime::RoundBudget> budgets,
                                                          schedule::TargetVerifyFrameView* verify = nullptr);
    [[nodiscard]] runtime::MixedRoundResult consume_mixed_round(runtime::RoundHandle handle);
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t batch_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    std::optional<AdaptiveDFlash> adaptive_dflash;
    std::uint32_t active_dflash_window = 0;
    std::optional<std::uint32_t> pipeline_dflash_window;
    std::optional<GdnReplayRecords> dflash_record_storage;
    std::chrono::steady_clock::time_point dflash_measurement_started{};
    bool dflash_measurement_mixed = false;
    const std::uint32_t speculative_max_lanes;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    std::uint64_t checkpoint_clock = 0;
    std::vector<std::uint64_t> checkpoint_last_use;
    std::vector<std::uint64_t> checkpoint_revisions;
    // Shape of the most recent mixed round, kept for the corruption
    // attribution line: which band the graph was captured for, the batch's
    // maximum frontier, and each row's own frontier.
    struct LastMixedRound {
        std::uint32_t maximum_frontier = 0;
        std::int32_t band              = -1;
        bool graph_hit                 = false;
        std::array<std::uint32_t, kMaximumBatchColumns> row_frontiers{};
    };
    LastMixedRound last_mixed_round{};
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    std::size_t graph_observed_bytes = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<family::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    /// Each lane's deferred GDN fold (replay_records->defers_fold()): the transitions its last
    /// round accepted and its next verify applies, as the device's pending_columns holds them
    /// once `deferred_fold_counts_dirty` is flushed. Zero for a lane with nothing pending.
    std::vector<std::int32_t> deferred_fold_columns;
    bool deferred_fold_counts_dirty = false;
    std::optional<DFlashPersistentState> dflash;
    family::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor logit_bias;
    Tensor token_bitmask;
    std::unique_ptr<family::detail::SpeculativeConstraintRound> speculative_constraints;
    // One stable readout workspace per program, reused by every field/branch.
    DeviceArena candidate_readout_storage{256 * (sizeof(TokenId) + sizeof(float))};
    std::array<float, 256> candidate_readout_host{};
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::vector<SequenceState> sequences;
    std::vector<RequestControl> requests;
    // Host snapshots retain complete continuation state without reserving another active
    // lane or increasing the GPU KV pool. Each pipeline stage has a bounded local cache.
    static constexpr std::size_t kArchivedPrefixBytes = 512ULL << 20;
    family::detail::RadixPrefixCache<ArchivedSequence> archived_prefixes{kArchivedPrefixBytes};
    /// Page-locked storage for `archived_prefixes`, made at the first archive; null if the host
    /// refused it (`archive_arena_refused`), and images then use the heap.
    std::shared_ptr<family::detail::PinnedArchiveArena> archive_arena;
    bool archive_arena_refused = false;
    std::unordered_map<const GpuPrefixKey*, std::shared_ptr<GpuPrefix>> gpu_prefixes;
    void prune_gpu_prefixes();
    void capture_gpu_prefix(SequenceState& sequence, const std::shared_ptr<const GpuPrefixKey>& key,
                             const std::shared_ptr<GpuPrefixStorage>& storage);
    void restore_gpu_prefix(SequenceState& sequence, const RequestPlanImpl& plan);

    // Shared-prefix cache (SUROGATE_SERVE_SHARED_PREFIX_SLOTS, 0 disables): a page-aligned
    // prompt prefix seen in an earlier request is captured once, mid-prefill, and later
    // requests on any lane fork it instead of prefilling it again.
    std::uint32_t shared_prefix_slots = 0;
    std::uint32_t shared_prefix_page_budget = 0;
    std::unordered_map<std::uint64_t, std::shared_ptr<SharedPrefix>> shared_prefixes;
    // Page hashes earlier requests carried, in two generations so the set stays bounded.
    std::array<std::unordered_set<std::uint64_t>, 2> shared_prefix_seen;
    std::shared_ptr<GpuPrefixStorage> shared_prefix_storage;
    std::uint64_t shared_prefix_clock = 0;
    std::uint64_t shared_prefix_revision = 0;
    [[nodiscard]] std::shared_ptr<const std::vector<std::uint64_t>>
    shared_prefix_hashes(const PreparedPromptData& prompt, std::int32_t lora_slot) const;
    [[nodiscard]] std::shared_ptr<SharedPrefix> find_shared_prefix(const PreparedPromptData& prompt,
        const std::vector<std::uint64_t>& hashes, std::int32_t lora_slot) const;
    void plan_shared_prefix_capture(RequestControl::Prefill& staged,
                                    const std::vector<std::uint64_t>& hashes);
    void capture_shared_prefix(SequenceState& sequence, RequestControl::Prefill& staged);
    void restore_shared_prefix(SequenceState& sequence, const RequestPlanImpl& plan);
    [[nodiscard]] std::uint32_t shared_prefix_pages() const noexcept;
    void drop_shared_prefixes() noexcept;
    [[nodiscard]] static ReusePath reported_reuse(const RequestControl::Prefill& staged) noexcept {
        return staged.shared_restore ? ReusePath::SharedPrefix : staged.reuse;
    }

    DecodeGraphFlavors ordinary_graphs;
    // Round chaining (PATCHES.md #32): the chained flavor of every ordinary
    // profile, plus the device 1-scalar its in-graph increments read and the
    // host-side burst plumbing (per-round egress copies via stream host
    // functions, row-major token assembly for the ragged round result).
    DecodeGraphFlavors ordinary_chained_graphs;
    void* chain_one_storage = nullptr;
    Tensor chain_one;
    static constexpr std::uint32_t kChainBurstLimit = 8;
    struct BurstEgressCopy {
        TokenId* destination           = nullptr;
        const TokenId* source          = nullptr;
        float* logprob_destination     = nullptr;
        const float* logprob_source    = nullptr;
        RawTokenScores* score_destination = nullptr;
        const RawTokenScores* score_source = nullptr;
        std::int32_t count             = 0;
    };
        // Round lifecycle state (runtime/contract/round_lifecycle.h): what the
    // launch half hands the consume half. Depth is one today; overlap means
    // rotating these per in-flight round.
    struct InFlightRound {
        std::uint64_t id    = 0;
        std::uint32_t rows  = 0;
        std::uint32_t burst = 0;
        std::chrono::steady_clock::time_point start{};
        std::array<std::uint32_t, kMaximumBatchColumns> lanes{};
        /// The MTP round checks what it licensed against the budget when it consumes.
        std::array<runtime::RoundBudget, kMaximumBatchColumns> budgets{};
        /// An MTP round that verified nothing: one column per lane, tokens at stride one.
        bool narrow = false;
    };
    InFlightRound in_flight_{};
    std::uint32_t round_width_hint_ = 0;
    /// A narrow round licenses exactly one token per lane.
    std::array<std::int32_t, kMaximumBatchColumns> narrow_counts_{};
    /// `speculative_outcome()`: the last consumed MTP round's decisions, row-major.
    std::vector<std::byte> outcome_export_;
    /// `lane_draft_state()`: one record, rewritten per call.
    mutable std::array<std::byte, sizeof(LaneDraftState)> draft_state_export_{};
    /// A headless stage's MTP round licenses nothing itself; its result carries these.
    std::array<std::int32_t, kMaximumBatchColumns> headless_counts_{};
    std::uint64_t in_flight_counter_ = 0;
    // A mixed round between launch and consume (the pipeline driver's seam).
    struct MixedInFlight {
        bool valid          = false;
        std::uint64_t id    = 0;
        std::chrono::steady_clock::time_point start{};
        std::uint32_t rows  = 0;
        std::array<std::uint32_t, kMaximumBatchColumns> lanes{};
        std::uint32_t prefill_lane_count = 0;
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> prefill_lanes{};
        std::size_t staged_count = 0;
        bool graph_hit           = false;
        schedule::PrefillChunkResult chunk{};
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> nominals{};
        /// Per prompt, its first segment's length when it brought two (0: one segment).
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> splits{};
        /// The decode lanes verified their drafts in this round (an MTP round carrying the
        /// prompts): consume reads them through consume_mtp_round.
        bool mtp_verify = false;
    };
    MixedInFlight mixed_in_flight_{};
    std::uint64_t mixed_in_flight_counter_ = 0;

    std::array<TokenId, kMaximumBatchColumns * kChainBurstLimit> burst_rounds{};
    std::array<TokenId, kMaximumBatchColumns * kChainBurstLimit> burst_tokens{};
    std::array<float, kMaximumBatchColumns * kChainBurstLimit> burst_logprobs{};
    std::array<float, kMaximumBatchColumns * kChainBurstLimit> burst_rounds_logprobs{};
    std::array<RawTokenScores, kMaximumBatchColumns * kChainBurstLimit> burst_rounds_scores{};
    std::array<std::int32_t, kMaximumBatchColumns> burst_counts{};
    std::array<BurstEgressCopy, kChainBurstLimit> burst_copy_ctx{};
    std::uint32_t round_burst_limit = 1;
    // Prefill CUDA graphs (PATCHES.md #27); engaged in prepare_graphs when the
    // backend is plain decode and SUROGATE_SERVE_PREFILL_GRAPH != 0.
    std::optional<PrefillGraphFamily> prefill_graphs;
    DecodeGraphFlavors mtp_graphs;
    /// The narrow round's graphs, captured only when a width limit makes them reachable.
    DecodeGraphFlavors mtp_narrow_graphs;
    std::array<DecodeGraphFlavors, 16> dflash_graphs;
    /// Whether this program runs base rounds (ops::ScopedLoraBaseRound): it carries adapters,
    /// so a round that selects none has a cheaper flavor to take. Decided once, before the
    /// graphs are captured, because it decides which graphs exist.
    ///
    /// Not under --batch-invariant: the two flavors are the same model through different
    /// kernels, so a base request's bits would depend on whether an adapter request shared
    /// its round -- exactly what that mode promises they do not. Not on a pipeline stage,
    /// where every stage must take the same flavor for a round and that has not been proven.
    /// And not with in-place Marlin residency opted into (SUROGATE_SERVE_MARLIN_FP8): a base
    /// round's fused MLP would adopt a weight the adapter route then could not read.
    bool base_rounds_ = false;
    /// Whether a round over these lanes is a base round: none of their requests selects an
    /// adapter. `more` is a mixed round's other half (its staged prompts beside its decoders).
    [[nodiscard]] bool base_round_for(std::span<const std::uint32_t> lanes,
                                      std::span<const std::uint32_t> more = {}) const noexcept;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
/// The step token's log-probability, pinned beside it in `round_host`.
float* host_token_logprob = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    family::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    family::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    family::MtpDecodeIngress* mtp_host_ingress = nullptr;
    family::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    family::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    family::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    [[nodiscard]] RequestPlan plan_request_for_sequence(std::uint32_t lane,
        const SequenceState& sequence, const PreparedPromptData& prompt,
        const RequestBasePlan& base, bool archived = false);
    [[nodiscard]] std::vector<Tensor> prefix_state_tensors(const SequenceState& sequence,
                                                          bool checkpoint, bool draft = true,
                                                          bool hidden = true) const;
    void archive_sequence(const SequenceState& sequence);
    void restore_archived_sequence(SequenceState& sequence, const RequestPlanImpl& plan);
    void prepare_graphs();
    void bind_dflash_window(std::uint32_t drafts);
    [[nodiscard]] ops::SamplingConfig staged_sampling(RequestControl& request,
                                                      const SequenceState& sequence) const;
    void update_constraint(const SequenceState& sequence, RequestControl& request) const;
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::RoundHandle
    launch_ordinary_round(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    consume_ordinary_round(runtime::RoundHandle handle);
    [[nodiscard]] static constexpr std::uint32_t maximum_rounds_in_flight() noexcept {
        return 1;
    }
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    /// The MTP round's two halves (the ordinary round's seam, for the pipeline): launch
    /// stages the ingress and enqueues the round without synchronising; consume waits, reads
    /// the egress into each lane's `SpeculativeOutcome` and records the pending candidate.
    /// A stage without the head runs the verify forward only and leaves the lanes pending
    /// with nothing produced, for `adopt_speculative_outcome`.
    /// With `prefill_lanes`, the staged prompts' chunks ride the verify forward as a mixed
    /// round does (launch_mixed_round routes there), so the lanes keep verifying drafts.
    [[nodiscard]] runtime::RoundHandle
    launch_mtp_round(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets,
                     std::span<const std::uint32_t> prefill_lanes = {});
    [[nodiscard]] runtime::BatchedGeneratedRound consume_mtp_round(runtime::RoundHandle handle);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::RoundHandle
    launch_dflash_round(std::span<const std::uint32_t> lanes,
                         std::span<const runtime::RoundBudget> budgets,
                         std::span<const std::uint32_t> prefill_lanes = {});
    [[nodiscard]] runtime::BatchedGeneratedRound consume_dflash_round(runtime::RoundHandle handle);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    // Maps the pages a graph chunk starting at `cursor` will write: its whole
    // 128-rounded bucket, pad columns included, which admission (mapped to the
    // prompt only) does not cover. A window past capacity is left unmapped --
    // the graph layer refuses such a chunk and the eager body writes only the
    // real tokens, which admission already mapped.
    void materialize_graph_chunk_window(SequenceState& sequence, std::uint32_t cursor,
                                        std::uint32_t nominal);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] family::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const family::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] family::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] family::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
};

} // namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS

namespace sinfer::family::detail {

template <>
class ProgramImpl<SINFER_FAMILY_VARIANT> final : public SINFER_FAMILY_RUNTIME_NS::ProgramImplCore {
public:
    using SINFER_FAMILY_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace sinfer::family::detail
