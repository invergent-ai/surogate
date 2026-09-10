#pragma once
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
#include <vector>

namespace sinfer::family::detail::SINFER_FAMILY_RUNTIME_NS {

using PreparedPromptData    = family::PreparedPromptData;
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
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const family::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<family::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
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
    std::optional<SINFER_FAMILY_RUNTIME_NS::VisionPrefillPlan> vision;
    SINFER_FAMILY_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        SINFER_FAMILY_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<family::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
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
    std::array<TokenId, family::kMtpDecodeMaximumWidth> licensed_tokens{};
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
    family::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, family::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RewriteCheckpoint rewrite_checkpoint;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    /// Valid while `pending.kind == Speculative`: the round's decision for this lane.
    SpeculativeOutcome outcome;
    ops::SamplingConfig sampling_host;
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
    [[nodiscard]] bool mixed_round_supported(std::uint32_t prefill_lane) const noexcept;
    [[nodiscard]] std::string last_mixed_round_description(std::size_t row) const;
    void set_round_burst_limit(std::uint32_t limit) noexcept { round_burst_limit = limit; }
    static void burst_egress_copy_host(void* user) noexcept;
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
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
    /// Columns a decode round licenses per lane at most: the draft window plus the anchor
    /// under MTP, one otherwise. The pipeline driver sizes its token and residual carries by it.
    [[nodiscard]] std::uint32_t speculative_round_width() const noexcept {
        return speculative_backend == SpeculativeBackend::Mtp ? draft_window + 1U : 1U;
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
            throw std::logic_error("the DFlash round has no launch and consume halves");
        }
        return launch_ordinary_round(lanes, budgets);
    }
    [[nodiscard]] runtime::BatchedGeneratedRound consume_decode_round(runtime::RoundHandle handle) {
        if (speculative_backend == SpeculativeBackend::Mtp) { return consume_mtp_round(handle); }
        return consume_ordinary_round(handle);
    }
    [[nodiscard]] runtime::RoundHandle launch_mixed_round(std::span<const std::uint32_t> prefill_lanes,
                                                          std::span<const std::uint32_t> lanes,
                                                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::MixedRoundResult consume_mixed_round(runtime::RoundHandle handle);
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t batch_capacity;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const std::uint32_t speculative_max_lanes;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const bool rewrite_checkpoints;
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
    std::optional<DFlashPersistentState> dflash;
    family::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    std::vector<SequenceState> sequences;
    std::vector<RequestControl> requests;

    DecodeGraphFamily ordinary_graphs;
    // Round chaining (PATCHES.md #32): the chained flavor of every ordinary
    // profile, plus the device 1-scalar its in-graph increments read and the
    // host-side burst plumbing (per-round egress copies via stream host
    // functions, row-major token assembly for the ragged round result).
    DecodeGraphFamily ordinary_chained_graphs;
    void* chain_one_storage = nullptr;
    Tensor chain_one;
    static constexpr std::uint32_t kChainBurstLimit = 8;
    struct BurstEgressCopy {
        TokenId* destination           = nullptr;
        const TokenId* source          = nullptr;
        float* logprob_destination     = nullptr;
        const float* logprob_source    = nullptr;
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
    };
    MixedInFlight mixed_in_flight_{};
    std::uint64_t mixed_in_flight_counter_ = 0;

    std::array<TokenId, kMaximumBatchColumns * kChainBurstLimit> burst_rounds{};
    std::array<TokenId, kMaximumBatchColumns * kChainBurstLimit> burst_tokens{};
    std::array<float, kMaximumBatchColumns * kChainBurstLimit> burst_logprobs{};
    std::array<float, kMaximumBatchColumns * kChainBurstLimit> burst_rounds_logprobs{};
    std::array<std::int32_t, kMaximumBatchColumns> burst_counts{};
    std::array<BurstEgressCopy, kChainBurstLimit> burst_copy_ctx{};
    std::uint32_t round_burst_limit = 1;
    // Prefill CUDA graphs (PATCHES.md #27); engaged in prepare_graphs when the
    // backend is plain decode and SUROGATE_SERVE_PREFILL_GRAPH != 0.
    std::optional<PrefillGraphFamily> prefill_graphs;
    DecodeGraphFamily mtp_graphs;
    /// The narrow round's graphs, captured only when a width limit makes them reachable.
    DecodeGraphFamily mtp_narrow_graphs;
    DecodeGraphFamily dflash_graphs;

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
    void prepare_graphs();
    [[nodiscard]] ops::SamplingConfig staged_sampling(const RequestControl& request,
                                                      const SequenceState& sequence) const;
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
    [[nodiscard]] runtime::RoundHandle
    launch_mtp_round(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound consume_mtp_round(runtime::RoundHandle handle);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
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
