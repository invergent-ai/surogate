#pragma once

// Pipeline parallelism (design/INFERENCE.md, phase 3): N stage instances of one target, each a
// whole-model program on its own device that executes only its layer range, driven in lockstep
// behind the unchanged executor. The residual crosses a stage boundary through the previous
// stage's pinned export buffer (device -> pinned -> device, no P2P). Every stage keeps its own
// per-lane bookkeeping (KV rows, state slots, ledgers), so every program call the executor
// makes is replayed on every stage in order, and the sampled tokens come from the last stage.
//
// Status (2026-08-28): lockstep — stage s+1 starts a round when stage s has finished it; the
// overlap of several micro-batches across stages is step C.
//
// Speculative rounds (2026-09-07): a decode round may license up to `width` tokens per lane
// (the draft window plus one). Every stage runs the verify forward for its own layers; the
// stage with the head decides -- accept, propose -- and its decision, as bytes the family
// defines, is adopted by the other stages before the executor resolves the round and every
// stage folds its recurrent state on the same integers. A prefill's first drafts cross the
// same way. The driver never reads the bytes.

#include "api/types.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "runtime/contract/round_lifecycle.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "runtime/engine/request_memory.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace sinfer::runtime {

template <class Stage>
class PipelineInstance;

/// The executor's request-memory surface over all stages: activation and release apply to
/// every stage's transient region; the region handed out is stage 0's (the pipeline program
/// substitutes each stage's own region when it starts a prefill).
template <class Stage>
class PipelineRequestMemory {
public:
    explicit PipelineRequestMemory(std::vector<Stage*> stages) : stages_(std::move(stages)) {}
    void activate(std::size_t bytes, std::size_t alignment) {
        for (Stage* stage : stages_) { stage->request_memory.activate(bytes, alignment); }
    }
    void deactivate() noexcept {
        for (Stage* stage : stages_) { stage->request_memory.deactivate(); }
    }
    [[nodiscard]] TransientRegion region() const noexcept { return stages_.front()->request_memory.region(); }
    [[nodiscard]] ArenaMemorySummary summary() const noexcept { return stages_.front()->request_memory.summary(); }
    void reset_peak() noexcept {
        for (Stage* stage : stages_) { stage->request_memory.reset_peak(); }
    }

private:
    std::vector<Stage*> stages_;
};

/// One program interface over the stages (the executor's `Package::Program`).
template <class Stage>
class PipelineProgram {
public:
    using StagePackage  = typename Stage::Package;
    using PreparedPrompt = typename StagePackage::PreparedPrompt;
    using StageBasePlan = typename StagePackage::RequestBasePlan;
    using StagePlan     = typename StagePackage::RequestPlan;

    struct RequestBasePlan {
        std::vector<StageBasePlan> stages;
        [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return stages.front().summary(); }
    };
    struct RequestPlan {
        std::vector<StagePlan> stages;
        [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return stages.front().summary(); }
    };

    PipelineProgram(std::vector<Stage*> stages, std::vector<int> devices)
        : stages_(std::move(stages)), devices_(std::move(devices)) {
        // Micro-batch groups: lanes are partitioned by lane % groups and the groups flow
        // through the stages as a software pipeline (SUROGATE_SERVE_PIPELINE_GROUPS overrides;
        // 1 = lockstep).
        groups_ = static_cast<std::uint32_t>(stages_.size());
        if (const char* raw = std::getenv("SUROGATE_SERVE_PIPELINE_GROUPS"); raw != nullptr && *raw != '\0') {
            const long parsed = std::strtol(raw, nullptr, 10);
            if (parsed >= 1 && parsed <= 64) { groups_ = static_cast<std::uint32_t>(parsed); }
        }
        boundary_bytes_ = stages_.front()->program->stage_boundary_bytes();
        {
            const std::int32_t columns = stages_.front()->program->stage_boundary_columns();
            column_bytes_ = columns > 0 ? boundary_bytes_ / static_cast<std::size_t>(columns) : boundary_bytes_;
        }
        trace_ = std::getenv("SUROGATE_SERVE_PIPELINE_TRACE") != nullptr;
        // A group narrower than this is not worth a round of its own (every round pays the
        // fixed costs on every stage): the group count follows the round's width.
        if (const char* raw = std::getenv("SUROGATE_SERVE_PIPELINE_MIN_LANES"); raw != nullptr && *raw != '\0') {
            const long parsed = std::strtol(raw, nullptr, 10);
            if (parsed >= 1 && parsed <= 128) { min_lanes_per_group_ = static_cast<std::uint32_t>(parsed); }
        }
        // One host staging slot per (boundary, group): a stage's export of one group is parked
        // there while the stage moves on to the next group.
        slots_.resize(stages_.size() > 1 ? stages_.size() - 1 : 0);
        for (auto& boundary : slots_) {
            boundary.resize(groups_);
            for (auto& slot : boundary) { slot.resize(boundary_bytes_); }
        }
        width_ = std::max<std::uint32_t>(1, stages_.front()->program->speculative_round_width());
        assembled_tokens_.resize(static_cast<std::size_t>(kMaximumBatchColumns) * width_);
        assembled_counts_.resize(kMaximumBatchColumns);
        assembled_prefill_tokens_.fill(0);
        flights_.resize(groups_);
        stage_owner_.assign(stages_.size(), -1);
        held_by_.assign(stages_.size(), -1);
    }

    [[nodiscard]] RequestBasePlan plan_request_base(const PreparedPrompt& prompt,
                                                    const ResolvedExecutionOptions& options) {
        RequestBasePlan out;
        out.stages.reserve(stages_.size());
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            out.stages.push_back(stages_[s]->program->plan_request_base(prompt, options));
        }
        return out;
    }
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane, const PreparedPrompt& prompt,
                                                    const RequestBasePlan& base) {
        RequestPlan out;
        out.stages.reserve(stages_.size());
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            out.stages.push_back(stages_[s]->program->plan_request_for_lane(lane, prompt, base.stages[s]));
        }
        return out;
    }
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            if (!stages_[s]->program->can_admit_lane(lane, plan.stages[s])) { return false; }
        }
        return true;
    }
    [[nodiscard]] bool can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                                              const RequestPlan& plan) const noexcept {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            if (!stages_[s]->program->can_admit_lane_after_retained_eviction(lane, plan.stages[s])) { return false; }
        }
        return true;
    }
    [[nodiscard]] AdmissionResources admission_capacity() const noexcept {
        AdmissionResources out = stages_.front()->program->admission_capacity();
        for (std::size_t s = 1; s < stages_.size(); ++s) {
            const AdmissionResources other = stages_[s]->program->admission_capacity();
            out.active_lanes     = std::min(out.active_lanes, other.active_lanes);
            out.main_kv_pages    = std::min(out.main_kv_pages, other.main_kv_pages);
            out.backend_kv_pages = std::min(out.backend_kv_pages, other.backend_kv_pages);
        }
        return out;
    }
    [[nodiscard]] PrefillStepResult start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                                       RequestPlan&& plan, TransientRegion /*transient*/,
                                                       bool defer_first_chunk = false) {
        PrefillStepResult result{};
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            // A program honours the deferral only for the shapes a mixed round can advance; a
            // draft-head prompt, a vision prompt or a bridged reuse runs its first chunk right
            // here, on every stage in turn -- so the residual crosses between them exactly as it
            // does in advance_prefill_lane. Nothing processed on the previous stage means
            // nothing to carry.
            if (s > 0 && result.processed_prompt_tokens > 0) {
                std::memcpy(stages_[s]->program->stage_import_buffer(),
                            stages_[s - 1]->program->stage_export_buffer(), boundary_bytes_);
            }
            PreparedPrompt copy = (s + 1 == stages_.size()) ? std::move(prompt) : prompt.clone();
            trace("start_prefill_lane", s, defer_first_chunk ? 0 : 1);
            result = stages_[s]->program->start_prefill_lane(lane, std::move(copy), std::move(plan.stages[s]),
                                                            stages_[s]->request_memory.region(),
                                                            defer_first_chunk);
        }
        // A prompt that completed inside its first chunk sampled its token on the last stage
        // only; the others recorded placeholders (and, under a draft head, proposed nothing).
        propagate_prefill_features(lane, result);
        if (result.complete) { propagate_prefill_token(lane, result); }
        return result;
    }
    [[nodiscard]] PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        PrefillStepResult result{};
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            if (s > 0 && result.processed_prompt_tokens > 0) {
                std::memcpy(stages_[s]->program->stage_import_buffer(),
                            stages_[s - 1]->program->stage_export_buffer(), boundary_bytes_);
            }
            trace("advance_prefill_lane", s, 1);
            result = stages_[s]->program->advance_prefill_lane(lane);
        }
        propagate_prefill_features(lane, result);
        if (result.complete) { propagate_prefill_token(lane, result); }
        return result;
    }
    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const RoundBudget> budgets) {
        if (stages_.size() == 1) {
            select(0);
            return stages_[0]->program->decode_batch(lanes, budgets);
        }
        run_grouped_round({}, lanes, budgets);
        BatchedGeneratedRound result{
            .tokens     = std::span<const TokenId>(assembled_tokens_.data(), lanes.size() * width_),
            .row_counts = std::span<const std::int32_t>(assembled_counts_.data(), lanes.size()),
            .row_stride = width_};
        propagate_round_tokens(lanes, result, assembled_outcome_);
        return result;
    }
    [[nodiscard]] MixedRoundResult advance_prefill_mixed(std::span<const std::uint32_t> prefill_lanes,
                                                         std::span<const std::uint32_t> lanes,
                                                         std::span<const RoundBudget> budgets) {
        if (stages_.size() == 1) {
            select(0);
            return stages_[0]->program->advance_prefill_mixed(prefill_lanes, lanes, budgets);
        }
        run_grouped_round(prefill_lanes, lanes, budgets);
        MixedRoundResult result = assembled_mixed_;
        result.round            = BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(assembled_tokens_.data(), lanes.size() * width_),
            .row_counts = std::span<const std::int32_t>(assembled_counts_.data(), lanes.size()),
            .row_stride = width_};
        std::uint32_t feature_columns = 0;
        for (std::size_t i = 0; i < result.prefill_count; ++i) { feature_columns += result.prefill_at(i).processed_prompt_tokens; }
        propagate_round_tokens(lanes, result.round, assembled_outcome_, feature_columns);
        for (std::size_t i = 0; i < result.prefill_count && i < prefill_lanes.size(); ++i) {
            const PrefillStepResult& step = result.prefill_at(i);
            propagate_prefill_features(prefill_lanes[i], step);
            if (step.complete) { propagate_prefill_token(prefill_lanes[i], step); }
        }
        return result;
    }
    [[nodiscard]] bool mixed_round_supported(std::uint32_t prefill_lane, std::uint32_t decode_rows) const noexcept {
        for (Stage* stage : stages_) {
            if (!stage->program->mixed_round_supported(prefill_lane, decode_rows)) { return false; }
        }
        return true;
    }
    [[nodiscard]] std::string last_mixed_round_description(std::size_t row) const {
        return stages_.back()->program->last_mixed_round_description(row);
    }
    void set_round_burst_limit(std::uint32_t limit) noexcept {
        for (Stage* stage : stages_) { stage->program->set_round_burst_limit(limit); }
    }
    /// The round width the executor reports is taken once, when a flight begins, and pinned
    /// onto each stage right before that stage launches the flight's round: a program decides
    /// the shape of its MTP round from the hint, and the stages of one round must all decide
    /// alike -- a round stage 0 launched wide that a later stage launched narrow, after the
    /// hint moved, crosses a boundary whose two sides disagree on how many columns it holds.
    void set_round_width_hint(std::uint32_t lanes) noexcept { width_hint_ = lanes; }
    void resolve_prefill_lane(std::uint32_t lane, bool terminal) {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            stages_[s]->program->resolve_prefill_lane(lane, terminal);
        }
    }
    void resolve_pending_batch(std::span<const std::uint32_t> lanes, std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal, std::span<const std::uint8_t> cancelled) {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            stages_[s]->program->resolve_pending_batch(lanes, accepted_tokens, terminal, cancelled);
        }
        // The fold these lanes' round left for every stage has now been enqueued: the stages a
        // verifying flight of theirs held are free for the next one.
        for (std::size_t s = 0; s < held_by_.size(); ++s) {
            if (held_by_[s] < 0) { continue; }
            const Flight& holder = flights_[static_cast<std::size_t>(held_by_[s])];
            if (same_lanes(holder.lanes, lanes)) { held_by_[s] = -1; }
        }
    }
    void abort_lane(std::uint32_t lane) noexcept {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            stages_[s]->program->abort_lane(lane);
        }
    }
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept {
        return stages_.front()->program->has_retained_lane(lane);
    }
    void evict_retained_lane(std::uint32_t lane) noexcept {
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            stages_[s]->program->evict_retained_lane(lane);
        }
    }
    TokenScoreDelta logprob_delta(std::uint32_t lane, std::size_t first, std::size_t end, bool prompt) const {
        return stages_.back()->program->logprob_delta(lane, first, end, prompt);
    }
    void collect_logprobs(std::uint32_t lane, GenerationResult& result) const {
        stages_.back()->program->collect_logprobs(lane, result);
        for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
            stages_[s]->program->cache_logprobs(lane, result);
        }
    }
    /// A request's compute time is the sum of its stages'. Every stage runs its layers in turn
    /// for every round, so the last stage's clock alone is a fraction of the round: on eight
    /// stages it reported a decode rate eight times the wall clock's. The request-level fields
    /// (preparation, first token, total) are the last stage's, which owns the request.
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept {
        GenerationTimings out = stages_.back()->program->generation_timings_lane(lane);
        out.vision_seconds    = 0.0;
        out.prefill_seconds   = 0.0;
        out.decode_seconds    = 0.0;
        for (const auto& stage : stages_) {
            const GenerationTimings timings = stage->program->generation_timings_lane(lane);
            out.vision_seconds += timings.vision_seconds;
            out.prefill_seconds += timings.prefill_seconds;
            out.decode_seconds += timings.decode_seconds;
        }
        return out;
    }
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept {
        return stages_.back()->program->speculative_stats_lane(lane);
    }
    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        return stages_.front()->program->memory_summary();
    }
    /// Every stage carries the full-attention layers of its own span, and the pool geometry is
    /// planned per stage; the first stage's is the one memory_summary() already reports.
    [[nodiscard]] PagedKVOccupancy kv_occupancy() const noexcept {
        return stages_.front()->program->kv_occupancy();
    }
    void kv_settle() noexcept {
        for (Stage* stage : stages_) { stage->program->kv_settle(); }
    }
    [[nodiscard]] bool kv_under_pressure() const noexcept {
        for (Stage* stage : stages_) {
            if (stage->program->kv_under_pressure()) { return true; }
        }
        return false;
    }
    bool kv_service_pressure() noexcept {
        bool pressure = false;
        for (Stage* stage : stages_) { pressure = stage->program->kv_service_pressure() || pressure; }
        return pressure;
    }
    void reset_memory_peaks() noexcept {
        for (Stage* stage : stages_) { stage->program->reset_memory_peaks(); }
    }

    // ---- Steady-state pipeline (step C3): groups are independent rounds that flow through the
    // stages while other groups occupy other stages; the executor launches a group when it has
    // nothing in flight and processes it when its last stage completes.
    enum class FlightKind : std::uint8_t { Decode, Mixed, Prefill };
    struct GroupResult {
        FlightKind kind = FlightKind::Decode;
        BatchedGeneratedRound round{};
        MixedRoundResult mixed{};
        PrefillStepResult prefill{};
    };
    [[nodiscard]] std::uint32_t group_count() const noexcept { return groups_; }
    [[nodiscard]] bool group_in_flight(std::uint32_t g) const noexcept { return flights_.at(g).active; }
    [[nodiscard]] bool has_finished_pending() const noexcept { return !pending_finished_.empty(); }
    /// A group whose flight completed inside another group's launch (a lone prefill parked
    /// mid-pipeline runs on whenever a stage frees up) waits here for the next tick to hand it
    /// back. It is not in flight, but it is not free either: its lane's result has not been
    /// resolved, and launching it again would run the same prefill step twice.
    [[nodiscard]] bool group_finished_pending(std::uint32_t g) const noexcept {
        return std::find(pending_finished_.begin(), pending_finished_.end(), g) != pending_finished_.end();
    }
    [[nodiscard]] bool any_in_flight() const noexcept {
        for (const auto& f : flights_) { if (f.active) { return true; } }
        return false;
    }
    void launch_group_decode(std::uint32_t g, std::span<const std::uint32_t> lanes, std::span<const RoundBudget> budgets) {
        begin_flight(g, FlightKind::Decode, {}, lanes, budgets, 0);
    }
    void launch_group_mixed(std::uint32_t g, std::span<const std::uint32_t> prefill_lanes,
                            std::span<const std::uint32_t> lanes, std::span<const RoundBudget> budgets) {
        begin_flight(g, FlightKind::Mixed, prefill_lanes, lanes, budgets, 0);
    }
    void launch_group_prefill(std::uint32_t g, std::uint32_t lane) {
        begin_flight(g, FlightKind::Prefill, {}, {}, {}, lane);
    }
    /// Advances the pipeline: consumes the stage round that has been running longest (the only
    /// blocking wait), moves its group to the next stage or finishes it, and launches every
    /// parked group whose next stage is free. Returns the groups that finished.
    std::vector<std::uint32_t> tick() {
        // Groups that completed inside a launch (a lone prefill runs its stages synchronously)
        // are handed back here.
        std::vector<std::uint32_t> finished = std::move(pending_finished_);
        pending_finished_.clear();
        int oldest = -1;
        for (std::size_t g = 0; g < flights_.size(); ++g) {
            const Flight& f = flights_[g];
            if (!f.active || f.between) { continue; }
            if (oldest < 0 || f.stage_sequence < flights_[static_cast<std::size_t>(oldest)].stage_sequence) {
                oldest = static_cast<int>(g);
            }
        }
        if (oldest >= 0) {
            Flight& f = flights_[static_cast<std::size_t>(oldest)];
            const std::size_t s = f.stage;
            select(s);
            trace("consume", s, static_cast<std::size_t>(oldest));
            if (f.kind == FlightKind::Mixed) {
                MixedRoundResult part = stages_[s]->program->consume_mixed_round(f.handle);
                if (s + 1 == stages_.size()) {
                    f.result.kind  = FlightKind::Mixed;
                    f.result.mixed = part;
                    store_round(f, part.round);
                    f.result.mixed.round = f.result.round;
                }
            } else {
                const BatchedGeneratedRound part = stages_[s]->program->consume_decode_round(f.handle);
                if (s + 1 == stages_.size()) {
                    f.result.kind = FlightKind::Decode;
                    store_round(f, part);
                }
            }
            f.handle = RoundHandle{};
            stage_owner_[s] = -1;
            finish_stage(static_cast<std::uint32_t>(oldest), s, finished);
        }
        advance_parked(finished);
        return finished;
    }
    [[nodiscard]] const GroupResult& group_result(std::uint32_t g) const { return flights_.at(g).result; }

private:
    struct Flight {
        bool active   = false;
        bool between  = false; // parked: export copied out, waiting for `stage` to be free
        FlightKind kind = FlightKind::Decode;
        std::size_t stage = 0;
        std::uint64_t stage_sequence = 0;
        RoundHandle handle{};
        std::vector<std::uint32_t> lanes;
        std::vector<RoundBudget> budgets;
        std::vector<std::uint32_t> prefill_lanes;
        std::uint32_t prefill_lane = 0;
        std::vector<TokenId> tokens;
        std::vector<std::int32_t> counts;
        // A speculative round's decision, copied out of the last stage at consume time for the
        // same reason as the tokens below.
        std::vector<std::byte> outcome;
        // The round licensed one token per lane at stride one: a narrow MTP round.
        bool narrow = false;
        // Decided when the flight began, from the pinned width: a verifying (wide) MTP round
        // records every stage's recurrent state for a fold that runs at resolve, into records
        // indexed by batch row and shared by every round on the stage. Until that fold has
        // been enqueued, no other verifying flight may run on a stage this one ran on -- its
        // verify would overwrite the records the fold is about to read, and one lane would
        // fold another's state. Narrow rounds and prefills write no records and pass freely.
        bool verifying = false;
        // A finished prompt's first sampled token, copied out of the last stage's egress buffer:
        // the executor reads the result after `tick()` returns, by which time another round may
        // have been launched on that stage and overwritten the buffer the span pointed into.
        std::array<TokenId, runtime::kMaximumMixedPrefills> prefill_tokens{};
        TokenId lone_prefill_token = 0;
        std::vector<std::byte> park;
        std::size_t carry_bytes = 0; // bytes of residual this flight carries across a boundary
        std::uint32_t dflash_window = 0;
        std::uint32_t width_hint = 0; // the round width the flight was launched under
        GroupResult result{};
    };
    std::vector<Flight> flights_;
    std::vector<int> stage_owner_;
    std::uint64_t stage_sequence_counter_ = 0;

    void begin_flight(std::uint32_t g, FlightKind kind, std::span<const std::uint32_t> prefill_lanes,
                      std::span<const std::uint32_t> lanes, std::span<const RoundBudget> budgets, std::uint32_t prefill_lane) {
        Flight& f = flights_.at(g);
        if (f.active) { throw std::logic_error("pipeline group already in flight"); }
        f.active  = true;
        f.between = true;
        f.kind    = kind;
        f.stage   = 0;
        f.lanes.assign(lanes.begin(), lanes.end());
        f.budgets.assign(budgets.begin(), budgets.end());
        f.prefill_lanes.assign(prefill_lanes.begin(), prefill_lanes.end());
        f.prefill_lane = prefill_lane;
        f.width_hint   = width_hint_;
        f.verifying    = false;
        if ((kind == FlightKind::Decode || kind == FlightKind::Mixed) && width_ > 1 && !lanes.empty()) {
            select(0);
            stages_[0]->program->set_round_width_hint(f.width_hint);
            f.verifying = kind == FlightKind::Decode ? !stages_[0]->program->speculative_round_is_narrow(lanes.size())
                : !stages_[0]->program->speculative_round_is_narrow(0);
            f.dflash_window = stages_[0]->program->select_dflash_draft_window(lanes);
        }
        f.result       = GroupResult{};
        // A decode round's residual is exactly `width` columns per lane (one, or the verify's
        // draft window plus one); mixed and prefill rounds carry the full boundary (their
        // graphs pad to buckets).
        f.carry_bytes = kind == FlightKind::Decode
                            ? std::min(boundary_bytes_, column_bytes_ * lanes.size() * width_)
                            : boundary_bytes_;
        advance_parked(pending_finished_);
    }
    std::vector<std::uint32_t> pending_finished_;
    void store_round(Flight& f, const BatchedGeneratedRound& part) {
        const std::size_t stride = part.row_stride > 0 ? static_cast<std::size_t>(part.row_stride) : 1;
        f.tokens.assign(f.lanes.size() * width_, 0);
        f.counts.resize(f.lanes.size());
        for (std::size_t i = 0; i < f.lanes.size(); ++i) {
            const std::int32_t count = part.row_counts.empty() ? 1 : part.row_counts[i];
            if (count < 0 || static_cast<std::uint32_t>(count) > width_ ||
                (count > 0 && i * stride + static_cast<std::size_t>(count) > part.tokens.size())) {
                throw std::logic_error("pipeline round licensed more tokens than its width");
            }
            f.counts[i] = count;
            for (std::size_t j = 0; j < static_cast<std::size_t>(count); ++j) {
                f.tokens[i * width_ + j] = part.tokens[i * stride + j];
            }
        }
        if (width_ > 1) {
            const std::span<const std::byte> outcome = stages_.back()->program->speculative_outcome();
            f.outcome.assign(outcome.begin(), outcome.end());
        }
        f.narrow = width_ > 1 && stride == 1;
        f.result.round = BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(f.tokens.data(), f.lanes.size() * width_),
            .row_counts = std::span<const std::int32_t>(f.counts.data(), f.lanes.size()),
            .row_stride = width_};
    }
    // The group finished stage s (consumed, or a synchronous prefill step): park it for the
    // next stage or complete it.
    void finish_stage(std::uint32_t g, std::size_t s, std::vector<std::uint32_t>& finished) {
        Flight& f = flights_[g];
        if (f.kind == FlightKind::Prefill && f.result.prefill.processed_prompt_tokens == 0) {
            f.carry_bytes = 0; // An encoder step has no text residual to transfer.
        }
        trace("finished-stage", s, g);
        if (f.verifying) { held_by_[s] = static_cast<int>(g); }
        if (s + 1 < stages_.size()) {
            f.park.resize(boundary_bytes_);
            std::memcpy(f.park.data(), stages_[s]->program->stage_export_buffer(), f.carry_bytes);
            park_checksum(g, s, f);
            f.between = true;
            f.stage   = s + 1;
            return;
        }
        // Last stage: commit tokens on the head-less stages and hand the group back. Every
        // token the caller will read is copied into the flight first — the program's egress
        // buffers belong to the next round on that stage.
        if (f.kind == FlightKind::Prefill) {
            propagate_prefill_features(f.prefill_lane, f.result.prefill);
            if (f.result.prefill.complete) {
                store_prefill_token(f.result.prefill, f.lone_prefill_token);
                propagate_prefill_token(f.prefill_lane, f.result.prefill);
            }
        } else {
            std::uint32_t feature_columns = 0;
            if (f.kind == FlightKind::Mixed) {
                for (std::size_t i = 0; i < f.result.mixed.prefill_count; ++i) {
                    feature_columns += f.result.mixed.prefill_at(i).processed_prompt_tokens;
                }
            }
            propagate_round_tokens(f.lanes, f.result.round, f.outcome, feature_columns);
            if (f.kind == FlightKind::Mixed) {
                for (std::size_t i = 0; i < f.result.mixed.prefill_count && i < f.prefill_lanes.size(); ++i) {
                    PrefillStepResult& step = f.result.mixed.prefills[i];
                    propagate_prefill_features(f.prefill_lanes[i], step);
                    if (!step.complete) { continue; }
                    store_prefill_token(step, f.prefill_tokens[i]);
                    propagate_prefill_token(f.prefill_lanes[i], step);
                }
            }
        }
        f.active  = false;
        f.between = false;
        finished.push_back(g);
    }
    // Launch every parked group whose next stage is free, oldest first (a prefill flight runs
    // its stage synchronously and moves on at once).
    void advance_parked(std::vector<std::uint32_t>& finished) {
        for (;;) {
            int pick = -1;
            for (std::size_t g = 0; g < flights_.size(); ++g) {
                const Flight& f = flights_[g];
                if (!f.active || !f.between || stage_owner_[f.stage] >= 0) { continue; }
                if (f.verifying && held_by_[f.stage] >= 0 && held_by_[f.stage] != static_cast<int>(g)) { continue; }
                if (pick < 0 || f.stage_sequence < flights_[static_cast<std::size_t>(pick)].stage_sequence) {
                    pick = static_cast<int>(g);
                }
            }
            if (pick < 0) { return; }
            Flight& f           = flights_[static_cast<std::size_t>(pick)];
            const std::size_t s = f.stage;
            select(s);
            if (s > 0 && f.carry_bytes > 0) {
                std::memcpy(stages_[s]->program->stage_import_buffer(), f.park.data(), f.carry_bytes);
            }
            f.between        = false;
            f.stage_sequence = ++stage_sequence_counter_;
            if (f.kind == FlightKind::Prefill) {
                trace("prefill-step", s, static_cast<std::size_t>(pick));
                f.result.kind    = FlightKind::Prefill;
                f.result.prefill = stages_[s]->program->advance_prefill_lane(f.prefill_lane);
                finish_stage(static_cast<std::uint32_t>(pick), s, finished);
                continue;
            }
            stage_owner_[s] = pick;
            stages_[s]->program->set_round_width_hint(f.width_hint);
            stages_[s]->program->set_dflash_draft_window(f.dflash_window);
            if (f.kind == FlightKind::Mixed) {
                trace("launch_mixed_round", s, f.prefill_lanes.size() * 1000 + f.lanes.size());
                f.handle = stages_[s]->program->launch_mixed_round(f.prefill_lanes, f.lanes, f.budgets);
            } else {
                trace("launch_decode_round", s, f.lanes.size());
                f.handle = stages_[s]->program->launch_decode_round(f.lanes, f.budgets);
            }
        }
    }

    void select(std::size_t stage) const noexcept { (void)cudaSetDevice(devices_[stage]); }

    /// SUROGATE_SERVE_PIPELINE_CHECKSUM=N: for the first N parked residuals, one line per
    /// column of the carry -- sum and absmax of the BF16 values -- so a graph replay's crossing
    /// can be read on the host, column by column, where the runtime's own checksum (which runs
    /// inside a body) cannot see a replay.
    void park_checksum(std::uint32_t g, std::size_t s, const Flight& f) {
        static const long budget = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_PIPELINE_CHECKSUM");
            return raw != nullptr ? std::strtol(raw, nullptr, 10) : 0L;
        }();
        static long printed = 0;
        if (budget <= 0 || printed >= budget) { return; }
        ++printed;
        const std::size_t column_values = column_bytes_ / sizeof(std::uint16_t);
        const std::size_t columns       = column_values == 0 ? 0 : f.carry_bytes / column_bytes_;
        const auto* values              = reinterpret_cast<const std::uint16_t*>(f.park.data());
        std::string line = "pipeline-checksum: group " + std::to_string(g) + " stage " + std::to_string(s) +
                           " kind " + std::to_string(static_cast<int>(f.kind)) + " columns " + std::to_string(columns);
        for (std::size_t c = 0; c < columns && c < 8; ++c) {
            double sum = 0.0, absmax = 0.0;
            for (std::size_t i = 0; i < column_values; ++i) {
                const std::uint32_t bits = static_cast<std::uint32_t>(values[c * column_values + i]) << 16;
                float v;
                std::memcpy(&v, &bits, sizeof(v));
                sum += v;
                absmax = std::max(absmax, static_cast<double>(std::fabs(v)));
            }
            char buf[96];
            std::snprintf(buf, sizeof(buf), " | c%zu sum %.5g absmax %.4g", c, sum, absmax);
            line += buf;
        }
        std::fprintf(stderr, "%s\n", line.c_str());
    }

    // One executor round as a software pipeline over the stages: the decode lanes are
    // partitioned into groups (lane % groups); when the round carries prefill lanes they all
    // ride with the first non-empty group as a mixed round (the executor's prefill-result
    // semantics stay those of one mixed call), the other groups run decode-only rounds. At
    // step t stage s consumes the group it launched at t-1 — parking its export in the
    // group's host slot, or collecting the last stage's tokens — then launches group t-s
    // after copying the previous stage's parked export into its own import buffer.
    // Consuming stage s before launching stage s+1 orders the data; the other stages keep
    // running meanwhile.
    void run_grouped_round(std::span<const std::uint32_t> prefill_lanes, std::span<const std::uint32_t> lanes,
                           std::span<const RoundBudget> budgets) {
        // Groups for this round: as many as the stages, but never narrower than
        // min_lanes_per_group_ (the partition is per round; every lane's state lives on every
        // stage, so lanes may move between groups from round to round).
        const std::uint32_t width_groups = static_cast<std::uint32_t>(std::max<std::size_t>(1, lanes.size() / min_lanes_per_group_));
        // Under speculation this path runs one group: the round's decision is adopted from the
        // last stage's consume and its resolution reads that stage's frame, so a second
        // group's round on that stage before the first resolved would overwrite both. The
        // executor's flight loop (tick) orders that per group; this synchronous path does
        // not, and is not the executor's.
        const std::uint32_t groups       = width_ > 1 ? 1U : std::max<std::uint32_t>(1, std::min(groups_, width_groups));
        std::vector<std::vector<std::uint32_t>> group_lanes(groups);
        std::vector<std::vector<RoundBudget>> group_budgets(groups);
        std::vector<std::vector<std::size_t>> group_rows(groups);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t g = static_cast<std::uint32_t>(row) % groups;
            group_lanes[g].push_back(lanes[row]);
            group_budgets[g].push_back(budgets[row]);
            group_rows[g].push_back(row);
        }
        std::vector<std::uint32_t> active;
        for (std::uint32_t g = 0; g < groups; ++g) {
            if (!group_lanes[g].empty()) { active.push_back(g); }
        }
        const bool mixed = !prefill_lanes.empty();
        if (mixed && active.empty()) {
            throw std::logic_error("pipeline mixed round needs decode lanes");
        }
        const std::uint32_t mixed_group = mixed ? active.front() : groups;
        const std::size_t N = stages_.size(), G = active.size();
        std::vector<RoundHandle> in_flight(N);
        std::vector<bool> in_flight_mixed(N, false);
        assembled_mixed_ = MixedRoundResult{};
        const std::uint32_t width_hint = width_hint_; // one decision for every stage of the round
        const auto dflash_window = stages_.front()->program->select_dflash_draft_window(lanes);
        for (Stage* stage : stages_) {
            stage->program->set_round_width_hint(width_hint);
            stage->program->set_dflash_draft_window(dflash_window);
        }
        for (std::size_t t = 0; t + 1 < G + N + 1; ++t) {
            for (std::size_t s = 0; s < N; ++s) {
                if (t >= s + 1 && t - s - 1 < G) {
                    const std::uint32_t g = active[t - s - 1];
                    select(s);
                    trace("consume", s, g);
                    if (in_flight_mixed[s]) {
                        MixedRoundResult part = stages_[s]->program->consume_mixed_round(in_flight[s]);
                        if (s + 1 == N) {
                            collect_tokens(group_rows[g], part.round);
                            assembled_mixed_.prefill_count = part.prefill_count;
                            assembled_mixed_.prefills      = part.prefills;
                            // Same reason as store_prefill_token: the spans point into the
                            // stage's egress buffer, which the next round overwrites.
                            for (std::size_t i = 0; i < assembled_mixed_.prefill_count &&
                                                    i < assembled_prefill_tokens_.size();
                                 ++i) {
                                PrefillStepResult& step = assembled_mixed_.prefills[i];
                                if (step.complete) {
                                    store_prefill_token(step, assembled_prefill_tokens_[i]);
                                }
                            }
                        }
                    } else {
                        const BatchedGeneratedRound part = stages_[s]->program->consume_decode_round(in_flight[s]);
                        if (s + 1 == N) { collect_tokens(group_rows[g], part); }
                    }
                    in_flight[s] = RoundHandle{};
                    trace("consumed", s, g);
                    if (s + 1 < N) {
                        std::memcpy(slots_[s][g].data(), stages_[s]->program->stage_export_buffer(), boundary_bytes_);
                    }
                }
                if (t >= s && t - s < G) {
                    const std::uint32_t g = active[t - s];
                    select(s);
                    if (s > 0) {
                        std::memcpy(stages_[s]->program->stage_import_buffer(), slots_[s - 1][g].data(), boundary_bytes_);
                    }
                    if (g == mixed_group) {
                        trace("launch_mixed_round", s, prefill_lanes.size() * 1000 + group_lanes[g].size());
                        in_flight[s]       = stages_[s]->program->launch_mixed_round(prefill_lanes, group_lanes[g], group_budgets[g]);
                        in_flight_mixed[s] = true;
                    } else {
                        trace("launch_decode_round", s, group_lanes[g].size());
                        in_flight[s]       = stages_[s]->program->launch_decode_round(group_lanes[g], group_budgets[g]);
                        in_flight_mixed[s] = false;
                    }
                }
            }
        }
    }
    // Copy a group's tokens from the last stage's round into the assembled result (the
    // program's buffers are reused by its next consume).
    void collect_tokens(const std::vector<std::size_t>& rows, const BatchedGeneratedRound& part) {
        const std::size_t stride = part.row_stride > 0 ? static_cast<std::size_t>(part.row_stride) : 1;
        for (std::size_t i = 0; i < rows.size(); ++i) {
            const std::int32_t count = part.row_counts.empty() ? 1 : part.row_counts[i];
            if (count < 0 || static_cast<std::uint32_t>(count) > width_ ||
                (count > 0 && i * stride + static_cast<std::size_t>(count) > part.tokens.size())) {
                throw std::logic_error("pipeline round licensed more tokens than its width");
            }
            assembled_counts_[rows[i]] = count;
            for (std::size_t j = 0; j < width_; ++j) {
                assembled_tokens_[rows[i] * width_ + j] =
                    j < static_cast<std::size_t>(count) ? part.tokens[i * stride + j] : 0;
            }
        }
        if (width_ > 1) {
            const std::span<const std::byte> outcome = stages_.back()->program->speculative_outcome();
            assembled_outcome_.assign(outcome.begin(), outcome.end());
        }
    }
    // The stages without the head recorded placeholder tokens this round: overwrite them with
    // the tokens the last stage sampled (one per lane; stages run single rounds). Under
    // speculation they recorded nothing and adopt the last stage's decision instead.
    void propagate_prefill_features(std::uint32_t lane, const PrefillStepResult& result) {
        if (result.processed_prompt_tokens == 0) { return; }
        const auto* packet = stages_.back()->program->stage_export_buffer();
        if (!packet) { return; }
        for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
            select(s);
            stages_[s]->program->adopt_pipeline_prefill_features(lane,
                std::span<const std::byte>(static_cast<const std::byte*>(packet), boundary_bytes_)
                    .subspan(result.feature_offset * column_bytes_),
                result.processed_prompt_tokens, result.feature_base);
        }
    }
    void propagate_round_tokens(std::span<const std::uint32_t> lanes, const BatchedGeneratedRound& round,
                                const std::vector<std::byte>& outcome, std::uint32_t feature_columns = 0) {
        if (stages_.size() < 2 || lanes.empty()) { return; }
        if (const auto* packet = stages_.back()->program->stage_export_buffer()) {
            for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
                select(s);
                stages_[s]->program->adopt_pipeline_decode_features(lanes,
                    std::span<const std::byte>(static_cast<const std::byte*>(packet), boundary_bytes_)
                        .subspan(feature_columns * column_bytes_));
            }
        }
        if (width_ > 1) {
            const std::span<const std::byte> bytes(outcome.data(), outcome.size());
            for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
                stages_[s]->program->adopt_speculative_outcome(lanes, bytes);
            }
            return;
        }
        // Decode rounds carry per-row counts and a stride; mixed rounds carry one token per
        // row and nothing else. Stages run single rounds, so every row has at most one token.
        const std::size_t stride = round.row_stride > 0 ? static_cast<std::size_t>(round.row_stride) : 1;
        std::vector<std::uint32_t> replaced_lanes;
        std::vector<TokenId> tokens;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::int32_t count = round.row_counts.empty() ? 1 : round.row_counts[row];
            if (count == 0) { continue; }
            if (count != 1 || row * stride >= round.tokens.size()) {
                throw std::logic_error("pipeline stages expect one sampled token per lane per round");
            }
            replaced_lanes.push_back(lanes[row]);
            tokens.push_back(round.tokens[row * stride]);
        }
        if (replaced_lanes.empty()) { return; }
        for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
            stages_[s]->program->replace_pending_tokens(replaced_lanes, tokens);
        }
    }
    // Copies a finished prompt's sampled token into flight-owned storage and re-points the
    // result's span at it.
    static void store_prefill_token(PrefillStepResult& step, TokenId& storage) {
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("pipeline stages expect one sampled token from a finished prefill");
        }
        storage          = step.round.tokens[0];
        step.round.tokens = std::span<const TokenId>(&storage, 1);
    }

    void propagate_prefill_token(std::uint32_t lane, const PrefillStepResult& step) {
        if (stages_.size() < 2) { return; }
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("pipeline stages expect one sampled token from a finished prefill");
        }
        const std::uint32_t lanes[1] = {lane};
        const TokenId tokens[1]      = {step.round.tokens[0]};
        for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
            stages_[s]->program->replace_pending_tokens(lanes, tokens);
        }
        if (width_ > 1) {
            // Only the stage with the head proposed anything for the first round.
            const std::span<const std::byte> drafts = stages_.back()->program->lane_draft_state(lane);
            for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
                stages_[s]->program->adopt_lane_draft_state(lane, drafts);
            }
        }
    }

    std::array<TokenId, runtime::kMaximumMixedPrefills> assembled_prefill_tokens_{};
    std::vector<Stage*> stages_;
    std::vector<int> devices_;
    bool trace_                = false;
    void trace(const char* op, std::size_t stage, std::size_t columns) const {
        if (!trace_) { return; }
        static const auto origin = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - origin).count();
        std::fprintf(stderr, "pipeline-trace: %10.1f ms %s stage %zu columns %zu\n", ms, op, stage, columns);
    }
    std::uint32_t groups_      = 1;
    std::uint32_t min_lanes_per_group_ = 4;
    std::size_t boundary_bytes_ = 0;
    std::size_t column_bytes_   = 0;
    std::vector<std::vector<std::vector<std::byte>>> slots_; // [boundary][group]
    std::vector<TokenId> assembled_tokens_;
    std::vector<std::int32_t> assembled_counts_;
    std::vector<std::byte> assembled_outcome_;
    MixedRoundResult assembled_mixed_{};
    /// Tokens a decode round may license per lane: 1, or the draft window plus one.
    std::uint32_t width_ = 1;
    /// The executor's latest round width; pinned per flight at launch (see set_round_width_hint).
    std::uint32_t width_hint_ = 0;
    /// Per stage: the group whose verifying round last ran there and has not been resolved,
    /// or -1 (see Flight::verifying).
    std::vector<int> held_by_;
    static bool same_lanes(const std::vector<std::uint32_t>& a, std::span<const std::uint32_t> b) {
        if (a.size() != b.size()) { return false; }
        std::vector<std::uint32_t> x(a.begin(), a.end()), y(b.begin(), b.end());
        std::sort(x.begin(), x.end());
        std::sort(y.begin(), y.end());
        return x == y;
    }
};

/// The executor's instance: owns the stage instances and their device contexts.
template <class Stage>
class PipelineInstance {
public:
    using Loaded = typename std::remove_reference_t<decltype(*std::declval<Stage>().loaded)>;
    struct Package {
        using PreparedPrompt  = typename Stage::Package::PreparedPrompt;
        using OutputSession   = typename Stage::Package::OutputSession;
        using Program         = PipelineProgram<Stage>;
        using RequestBasePlan = typename PipelineProgram<Stage>::RequestBasePlan;
        using RequestPlan     = typename PipelineProgram<Stage>::RequestPlan;
    };

    PipelineInstance(std::vector<std::unique_ptr<DeviceContext>> device_contexts,
                     std::vector<std::unique_ptr<Stage>> stage_instances)
        : devices(std::move(device_contexts)), stages(std::move(stage_instances)),
          loaded(stages.front()->loaded.get()),
          kv_capacity_resolution(stages.front()->kv_capacity_resolution),
          request_memory(pointers()), capacity(stages.front()->capacity),
          program(std::make_unique<PipelineProgram<Stage>>(pointers(), device_ids())) {
        for (const auto& stage : stages) {
            kv_capacity_resolution.resolved_tokens =
                std::min(kv_capacity_resolution.resolved_tokens, stage->kv_capacity_resolution.resolved_tokens);
        }
    }

    std::vector<std::unique_ptr<DeviceContext>> devices;
    std::vector<std::unique_ptr<Stage>> stages;
    Loaded* loaded; // stage 0's (the frontend lives there)
    KvCapacityResolution kv_capacity_resolution;
    PipelineRequestMemory<Stage> request_memory;
    const std::uint32_t capacity;
    std::unique_ptr<PipelineProgram<Stage>> program;

private:
    std::vector<Stage*> pointers() const {
        std::vector<Stage*> out;
        for (const auto& stage : stages) { out.push_back(stage.get()); }
        return out;
    }
    std::vector<int> device_ids() const {
        std::vector<int> out;
        for (const auto& device : devices) { out.push_back(device->device); }
        return out;
    }
};

} // namespace sinfer::runtime
