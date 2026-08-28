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

#include "api/types.h"
#include "core/device.h"
#include "runtime/contract/round_lifecycle.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "runtime/engine/request_memory.h"

#include <algorithm>
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

namespace ninfer::runtime {

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
        assembled_tokens_.resize(kMaximumConcurrency);
        assembled_counts_.resize(kMaximumConcurrency);
        flights_.resize(groups_);
        stage_owner_.assign(stages_.size(), -1);
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
            PreparedPrompt copy = (s + 1 == stages_.size()) ? std::move(prompt) : prompt.clone();
            trace("start_prefill_lane", s, defer_first_chunk ? 0 : 1);
            result = stages_[s]->program->start_prefill_lane(lane, std::move(copy), std::move(plan.stages[s]),
                                                            stages_[s]->request_memory.region(),
                                                            defer_first_chunk);
        }
        return result;
    }
    [[nodiscard]] PrefillStepResult advance_prefill_lane(std::uint32_t lane) {
        PrefillStepResult result{};
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            if (s > 0) {
                std::memcpy(stages_[s]->program->stage_import_buffer(),
                            stages_[s - 1]->program->stage_export_buffer(), boundary_bytes_);
            }
            trace("advance_prefill_lane", s, 1);
            result = stages_[s]->program->advance_prefill_lane(lane);
        }
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
            .tokens     = std::span<const TokenId>(assembled_tokens_.data(), lanes.size()),
            .row_counts = std::span<const std::int32_t>(assembled_counts_.data(), lanes.size()),
            .row_stride = 1};
        propagate_round_tokens(lanes, result);
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
            .tokens     = std::span<const TokenId>(assembled_tokens_.data(), lanes.size()),
            .row_counts = std::span<const std::int32_t>(assembled_counts_.data(), lanes.size()),
            .row_stride = 1};
        propagate_round_tokens(lanes, result.round);
        for (std::size_t i = 0; i < result.prefill_count && i < prefill_lanes.size(); ++i) {
            const PrefillStepResult& step = result.prefill_at(i);
            if (step.complete) { propagate_prefill_token(prefill_lanes[i], step); }
        }
        return result;
    }
    [[nodiscard]] bool mixed_round_supported(std::uint32_t prefill_lane) const noexcept {
        for (Stage* stage : stages_) {
            if (!stage->program->mixed_round_supported(prefill_lane)) { return false; }
        }
        return true;
    }
    [[nodiscard]] std::string last_mixed_round_description(std::size_t row) const {
        return stages_.back()->program->last_mixed_round_description(row);
    }
    void set_round_burst_limit(std::uint32_t limit) noexcept {
        for (Stage* stage : stages_) { stage->program->set_round_burst_limit(limit); }
    }
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
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept {
        return stages_.back()->program->generation_timings_lane(lane);
    }
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept {
        return stages_.back()->program->speculative_stats_lane(lane);
    }
    [[nodiscard]] MemorySummary memory_summary() const noexcept {
        return stages_.front()->program->memory_summary();
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
        std::vector<std::uint32_t> finished;
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
        std::vector<std::byte> park;
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
        f.result       = GroupResult{};
        std::vector<std::uint32_t> none;
        advance_parked(none);
    }
    void store_round(Flight& f, const BatchedGeneratedRound& part) {
        const std::size_t stride = part.row_stride > 0 ? static_cast<std::size_t>(part.row_stride) : 1;
        f.tokens.resize(f.lanes.size());
        f.counts.resize(f.lanes.size());
        for (std::size_t i = 0; i < f.lanes.size(); ++i) {
            const std::int32_t count = part.row_counts.empty() ? 1 : part.row_counts[i];
            if (count > 1) { throw std::logic_error("pipeline stages expect single-token rounds"); }
            f.counts[i] = count;
            f.tokens[i] = count == 1 ? part.tokens[i * stride] : 0;
        }
        f.result.round = BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(f.tokens.data(), f.lanes.size()),
            .row_counts = std::span<const std::int32_t>(f.counts.data(), f.lanes.size()),
            .row_stride = 1};
    }
    // The group finished stage s (consumed, or a synchronous prefill step): park it for the
    // next stage or complete it.
    void finish_stage(std::uint32_t g, std::size_t s, std::vector<std::uint32_t>& finished) {
        Flight& f = flights_[g];
        trace("finished-stage", s, g);
        if (s + 1 < stages_.size()) {
            f.park.resize(boundary_bytes_);
            std::memcpy(f.park.data(), stages_[s]->program->stage_export_buffer(), boundary_bytes_);
            f.between = true;
            f.stage   = s + 1;
            return;
        }
        // Last stage: commit tokens on the head-less stages and hand the group back.
        if (f.kind == FlightKind::Prefill) {
            if (f.result.prefill.complete) { propagate_prefill_token(f.prefill_lane, f.result.prefill); }
        } else {
            propagate_round_tokens(f.lanes, f.result.round);
            if (f.kind == FlightKind::Mixed) {
                for (std::size_t i = 0; i < f.result.mixed.prefill_count && i < f.prefill_lanes.size(); ++i) {
                    const PrefillStepResult& step = f.result.mixed.prefill_at(i);
                    if (step.complete) { propagate_prefill_token(f.prefill_lanes[i], step); }
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
                if (pick < 0 || f.stage_sequence < flights_[static_cast<std::size_t>(pick)].stage_sequence) {
                    pick = static_cast<int>(g);
                }
            }
            if (pick < 0) { return; }
            Flight& f           = flights_[static_cast<std::size_t>(pick)];
            const std::size_t s = f.stage;
            select(s);
            if (s > 0) {
                std::memcpy(stages_[s]->program->stage_import_buffer(), f.park.data(), boundary_bytes_);
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
        const std::uint32_t groups       = std::max<std::uint32_t>(1, std::min(groups_, width_groups));
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
            if (count > 1) { throw std::logic_error("pipeline stages expect single-token rounds"); }
            assembled_counts_[rows[i]] = count;
            assembled_tokens_[rows[i]] = count == 1 ? part.tokens[i * stride] : 0;
        }
    }
    // The stages without the head recorded placeholder tokens this round: overwrite them with
    // the tokens the last stage sampled (one per lane; stages run single rounds).
    void propagate_round_tokens(std::span<const std::uint32_t> lanes, const BatchedGeneratedRound& round) {
        if (stages_.size() < 2 || lanes.empty()) { return; }
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
    }

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
    std::vector<std::vector<std::vector<std::byte>>> slots_; // [boundary][group]
    std::vector<TokenId> assembled_tokens_;
    std::vector<std::int32_t> assembled_counts_;
    MixedRoundResult assembled_mixed_{};
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

} // namespace ninfer::runtime
