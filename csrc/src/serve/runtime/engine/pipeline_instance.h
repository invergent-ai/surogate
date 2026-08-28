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

#include <cstdint>
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
        : stages_(std::move(stages)), devices_(std::move(devices)) {}

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
            result = stages_[s]->program->advance_prefill_lane(lane);
        }
        if (result.complete) { propagate_prefill_token(lane, result); }
        return result;
    }
    [[nodiscard]] BatchedGeneratedRound decode_batch(std::span<const std::uint32_t> lanes,
                                                     std::span<const RoundBudget> budgets) {
        BatchedGeneratedRound result{};
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            result = stages_[s]->program->decode_batch(lanes, budgets);
        }
        propagate_round_tokens(lanes, result);
        return result;
    }
    [[nodiscard]] MixedRoundResult advance_prefill_mixed(std::span<const std::uint32_t> prefill_lanes,
                                                         std::span<const std::uint32_t> lanes,
                                                         std::span<const RoundBudget> budgets) {
        MixedRoundResult result{};
        for (std::size_t s = 0; s < stages_.size(); ++s) {
            select(s);
            result = stages_[s]->program->advance_prefill_mixed(prefill_lanes, lanes, budgets);
        }
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

private:
    void select(std::size_t stage) const noexcept { (void)cudaSetDevice(devices_[stage]); }
    // The stages without the head recorded placeholder tokens this round: overwrite them with
    // the tokens the last stage sampled (one per lane; stages run single rounds).
    void propagate_round_tokens(std::span<const std::uint32_t> lanes, const BatchedGeneratedRound& round) {
        if (stages_.size() < 2 || lanes.empty()) { return; }
        std::vector<TokenId> tokens(lanes.size());
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (round.row_counts.size() <= row || round.row_counts[row] != 1) {
                throw std::logic_error("pipeline stages expect one sampled token per lane per round");
            }
            tokens[row] = round.tokens[row * static_cast<std::size_t>(round.row_stride)];
        }
        for (std::size_t s = 0; s + 1 < stages_.size(); ++s) {
            stages_[s]->program->replace_pending_tokens(lanes, tokens);
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
