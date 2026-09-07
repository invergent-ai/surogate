#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "core/engine_context.h"
#include "api/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "api/family/frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "core/limits.h"
#include <deque>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <utility>
#include <vector>

namespace sinfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    // Pipeline programs (runtime/engine/pipeline_instance.h) expose per-group rounds; the
    // worker loop then keeps several groups in flight across the stages (step C3).
    static constexpr bool kPipelined = requires(typename Package::Program& p) { p.tick(); p.group_count(); };
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          admission_capacity_(instance.program->admission_capacity()) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        ops_context_ = options.ops_context;
        // A thread's CUDA device is its own. Everything this engine owns was allocated on the
        // device the options name, so the thread that launches against it has to be on that
        // device too -- otherwise the launches go to whatever the process default is, which is
        // device 0, and only a run that asked for device 0 works by coincidence.
        device_ = options.devices.empty() ? options.device
                                          : options.devices.front();
        worker_ = std::thread([this] { worker_loop(); });
    }

    /// Refuse new submissions while asleep. The caller drains in-flight work
    /// first; holding the execution mutex during the transition guarantees the
    /// worker loop is parked, not mid-round.
    void set_asleep(bool asleep) {
        {
            std::lock_guard lock(queue_mutex_);
            asleep_ = asleep;
        }
        queue_cv_.notify_all();
    }
    [[nodiscard]] bool asleep() const {
        std::lock_guard lock(queue_mutex_);
        return asleep_;
    }
    [[nodiscard]] std::unique_lock<std::mutex> pause_execution() {
        return std::unique_lock<std::mutex>(execution_mutex_);
    }
    /// Give back what idling holds: every retained (prefix-cache) lane is evicted, so an
    /// elastic pool's granules can go back. Runs between rounds; the scheduler calls it on an
    /// idle model before it would sleep one.
    void shrink_kv() {
        auto paused = pause_execution();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr && instance_.program->has_retained_lane(lane)) {
                instance_.program->evict_retained_lane(lane);
                invalidate_lane_plans(lane);
            }
        }
        instance_.program->kv_settle();
    }
    [[nodiscard]] bool any_active_lane() const {
        std::lock_guard lock(queue_mutex_);
        if (!pending_.empty()) { return true; }
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { return true; }
        }
        return false;
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(family::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex_);
            if (asleep_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "the model is asleep; wake it with POST /wake_up");
            }
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        return published_stats_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

private:
    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
        }
        snapshot.prefilling_requests = static_cast<std::uint32_t>(prefill_lanes_.size());
        const auto kv                       = instance_.program->kv_occupancy();
        snapshot.kv_pages                   = kv.page_group_count;
        snapshot.kv_pages_entitled          = kv.entitled_pages;
        snapshot.kv_pages_in_use            = kv.pages_in_use;
        snapshot.kv_granule_pages           = kv.granule_pages;
        snapshot.kv_pages_resident_at_granule = kv.resident_pages_at_granule;
        snapshot.kv_pages_mapped            = kv.mapped_pages;
        snapshot.kv_page_bytes              = kv.page_bytes;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(std::uint64_t request_identity, family::PreparedPrompt input,
                family::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                Clock::time_point limit, Clock::time_point submit_time)
            : id(request_identity), prompt(std::move(input)), output(std::move(output_session)),
              prompt_summary(summary), prepare_seconds(frontend_seconds),
              options(std::move(request_options)), deadline(limit), submitted(submit_time) {}

        const std::uint64_t id;
        family::PreparedPrompt prompt;
        family::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        /// One per entry of `generated`, from the round that sampled it. A route
        /// that does not produce them contributes nothing, so this is either empty
        /// or exactly as long as `generated` -- never partly filled, which would
        /// silently misalign a client reading the two together.
        std::vector<float> generated_logprobs;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane  = 0;
        bool evict_retained = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       family::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.token_logprobs          = std::move(request->generated_logprobs);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
        }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token, float logprob,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        request->generated_logprobs.push_back(logprob);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0) {
            throw std::logic_error("request service projection consumed zero quanta");
        }
        // Mixed-token rounds (PATCHES.md #30) legitimately take more steps
        // than the admission-time projection (smaller effective chunks plus
        // the zero-suffix finalize), so the projection saturates instead of
        // failing the engine.
        const std::uint64_t ceiling =
            request->remaining_service_work > 1 ? request->remaining_service_work - 1 : 0;
        request->remaining_service_work -= std::min<std::uint64_t>(work, ceiling);
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        bool changed = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            if (prefill_lanes_.contains(lane)) {
                prefill_lanes_.erase(lane);
                if (prefill_lanes_.empty()) { instance_.request_memory.deactivate(); }
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
            changed = true;
        }
        if (changed) { publish_runtime_stats(); }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
            have_pending = !pending_.empty();
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        if (!cancelled.empty() || !expired.empty()) { publish_runtime_stats(); }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step, bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        consume_service_work(request, 1);
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lanes_.contains(lane)) {
                prefill_lanes_.erase(lane);
                if (prefill_lanes_.empty()) { instance_.request_memory.deactivate(); }
            }
            instance_.program->abort_lane(lane);
            complete_cancelled(request);
            remove_completed_slot(lane);
            return;
        }
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lanes_.contains(*request->lane)) {
            prefill_lanes_.erase(*request->lane);
            if (prefill_lanes_.empty()) { instance_.request_memory.deactivate(); }
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        const float first_logprob =
            step.round.logprobs.empty() ? std::numeric_limits<float>::quiet_NaN()
                                        : step.round.logprobs.front();
        if (resolve_round(request, step.round.tokens.front(), first_logprob, false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (prefill_lanes_.empty()) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = prefill_lanes_.front();
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        resolve_prefill_step(request, step, cancel_at_boundary);
        publish_runtime_stats();
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane(lane, plan) &&
                (!selected || reuse > selected_reuse)) {
                selected       = LaneChoice{.lane = lane};
                selected_reuse = reuse;
            }
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                (!selected || reuse > selected_reuse)) {
                selected = LaneChoice{
                    .lane           = lane,
                    .evict_retained = true,
                };
                selected_reuse = reuse;
            }
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            publish_runtime_stats();
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        if (choice.evict_retained) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *request->lane_plans[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    instance_.program->evict_retained_lane(retained_lane);
                    invalidate_lane_plans(retained_lane);
                }
            }
            if (!instance_.program->can_admit_lane(lane, *request->lane_plans[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        Plan selected_plan = std::move(*request->lane_plans[lane]);
        request->lane_plans[lane].reset();
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        release_planning_state(request);

        const RequestPlanSummary summary = selected_plan.summary();
        if (backfill_class == BackfillClass::Temporal) {
            if (!protection_ || protection_->epoch_id != backfill_epoch ||
                summary.service_work_quanta > protection_->temporal_credit) {
                throw std::logic_error("temporal backfill lost its protected credit");
            }
            protection_->temporal_credit -= summary.service_work_quanta;
        }
        clear_protection_if_head(request);

        const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started      = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lanes_.add(lane);
                transient     = instance_.request_memory.region();
            }
            publish_runtime_stats();
            target_started                = true;
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient,
                /*defer_first_chunk=*/true);
            if (!first.complete && !prefill_lanes_.contains(lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            if (first.processed_prompt_tokens != 0 || first.complete || cancel_at_boundary) {
                resolve_prefill_step(request, first, cancel_at_boundary);
            }
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lanes_.contains(lane)) {
                prefill_lanes_.erase(lane);
                if (prefill_lanes_.empty()) { instance_.request_memory.deactivate(); }
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            throw;
        }
        return AdmissionProgress::RanGpuUnit;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        for (;;) {
            const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
            if (queued.empty()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }
            const std::shared_ptr<Request>& head = queued.front();
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    publish_runtime_stats();
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }
            // Elastic overcommit: the device gate, not the incumbents, refused the head. The
            // protection policy reasons on lanes and pages and would find the head unblocked;
            // the head simply waits for memory (retained lanes go back at the round boundary,
            // the other engines' reserves have been asked for) and retries next round.
            if (instance_.program->kv_under_pressure()) {
                protection_.reset();
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        publish_runtime_stats();
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate->lane_plans[candidate_lane->lane]->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
    }

    // ---- Steady-state pipeline (step C3) ------------------------------------------------
    struct GroupMeta {
        RoundMembership membership;
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> staged{};
        std::size_t staged_count  = 0;
        std::uint32_t prefill_lane = 0;
        std::uint32_t deferred     = 0;
        bool prefill_flight        = false;
    };
    std::vector<GroupMeta> group_meta_;

    void pipelined_iteration(bool have_pending) {
        auto& program = *instance_.program;
        const std::uint32_t groups = program.group_count();
        if (group_meta_.size() != groups) { group_meta_.resize(groups); }
        // Admission is CPU-only (deferred first chunk) and stages the prompt into the mixed
        // prefill set; top_up admits while that set has room and a lane is free, which is
        // exactly the bound the set can hold.
        if (have_pending) {
            const auto t_admit = Clock::now();
            top_up_prefill_lanes();
            seg_timer_.admit += std::chrono::duration<double>(Clock::now() - t_admit).count();
        }
        // The round's width, for a draft head deciding whether a verify pays: every lane that
        // is decode-ready, whichever group it rides in.
        {
            std::uint32_t decode_lanes = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                const auto& request = slots_[lane];
                if (request != nullptr && request->decode_ready) { ++decode_lanes; }
            }
            program.set_round_width_hint(decode_lanes);
        }
        // Launch every idle group that has work.
        bool launched = false;
        for (std::uint32_t g = 0; g < groups; ++g) {
            // A group whose flight finished inside an earlier launch of this same loop is
            // handed back by the next tick; until then its lane is neither decode-ready nor
            // free of its prefill, and relaunching it would advance a step that already ran.
            if (program.group_in_flight(g) || program.group_finished_pending(g)) { continue; }
            GroupMeta& meta = group_meta_[g];
            meta.membership = RoundMembership{};
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (lane % groups != g) { continue; }
                const auto& request = slots_[lane];
                if (request == nullptr || !request->decode_ready) { continue; }
                if (!request->budget) { throw std::logic_error("decode-ready request has no generation budget"); }
                meta.membership.lanes[meta.membership.size]   = lane;
                meta.membership.budgets[meta.membership.size] = request->budget->round_budget();
                ++meta.membership.size;
            }
            // Staged prompts of this group: any of them can take a lone prefill step; only the
            // ones the program can advance inside a mixed round ride with the decode lanes.
            meta.staged_count       = 0;
            std::uint32_t lone_lane = max_concurrency_;
            for (const std::uint32_t candidate : prefill_lanes_.span()) {
                if (candidate % groups != g) { continue; }
                const auto& owner = slots_[candidate];
                if (owner == nullptr || owner->decode_ready) { continue; }
                if (lone_lane == max_concurrency_) { lone_lane = candidate; }
                if (!program.mixed_round_supported(candidate)) { continue; }
                meta.staged[meta.staged_count++] = candidate;
            }
            // Prefill batching (#88) in the pipelined loop: let a group's staged set grow for a
            // few of its rounds before spending a mixed round on it, bounded as in the
            // single-round loop.
            if (meta.staged_count > 0 && !meta.membership.empty() &&
                meta.staged_count < mixed_prefill_batch_target() && !prefill_lanes_.full() &&
                meta.deferred < mixed_prefill_batch_wait_rounds()) {
                ++meta.deferred;
                meta.staged_count = 0;
            } else {
                meta.deferred = 0;
            }
            meta.prefill_flight = false;
            if (meta.staged_count > 0 && !meta.membership.empty()) {
                last_round_ = LastRound{"mixed", static_cast<std::uint32_t>(meta.membership.size), meta.staged[0], last_round_.index + 1};
                program.launch_group_mixed(g, std::span<const std::uint32_t>(meta.staged.data(), meta.staged_count),
                                           meta.membership.lane_span(), meta.membership.budget_span());
                launched = true;
            } else if (lone_lane != max_concurrency_) {
                // No decode lanes in this group: a lone prefill step (a zero-lane mixed round
                // needs the family's mixed body to accept batch 0 first — SUROGATE_SERVE_PIPELINE_ZERO_LANE_MIXED).
                static const bool zero_lane_mixed = std::getenv("SUROGATE_SERVE_PIPELINE_LONE_PREFILL") == nullptr;
                if (zero_lane_mixed && meta.staged_count > 0) {
                    last_round_ = LastRound{"mixed", 0, meta.staged[0], last_round_.index + 1};
                    program.launch_group_mixed(g, std::span<const std::uint32_t>(meta.staged.data(), meta.staged_count),
                                               std::span<const std::uint32_t>{}, std::span<const RoundBudget>{});
                    launched = true;
                    continue;
                }
                meta.staged_count   = 0;
                meta.prefill_lane   = lone_lane;
                meta.prefill_flight = true;
                last_round_ = LastRound{"prefill", 0, meta.prefill_lane, last_round_.index + 1};
                program.launch_group_prefill(g, meta.prefill_lane);
                launched = true;
            } else if (!meta.membership.empty()) {
                last_round_ = LastRound{"decode", static_cast<std::uint32_t>(meta.membership.size), 0, last_round_.index + 1};
                program.launch_group_decode(g, meta.membership.lane_span(), meta.membership.budget_span());
                launched = true;
            }
        }
        if (!program.any_in_flight() && !program.has_finished_pending()) {
            if (!launched) { std::this_thread::sleep_for(std::chrono::microseconds(200)); }
            return;
        }
        const auto t_round = Clock::now();
        const std::vector<std::uint32_t> finished = program.tick();
        seg_timer_.decode += std::chrono::duration<double>(Clock::now() - t_round).count();
        for (const std::uint32_t g : finished) {
            const GroupMeta& meta = group_meta_[g];
            const auto& result    = program.group_result(g);
            switch (result.kind) {
            case std::remove_reference_t<decltype(program)>::FlightKind::Decode:
                process_decode_round(meta.membership, result.round);
                ++cumulative_stats_.decode_rounds;
                seg_timer_.decode_rounds += 1;
                break;
            case std::remove_reference_t<decltype(program)>::FlightKind::Mixed:
                process_decode_round(meta.membership, result.mixed.round);
                ++cumulative_stats_.decode_rounds;
                seg_timer_.mixed_rounds += 1;
                for (std::size_t i = 0; i < result.mixed.prefill_count && i < meta.staged_count; ++i) {
                    const auto owner = slots_[meta.staged[i]];
                    if (owner == nullptr) { continue; }
                    const bool cancelled_now = owner->cancelled.load(std::memory_order_acquire);
                    resolve_prefill_step(owner, result.mixed.prefill_at(i), cancelled_now);
                }
                publish_runtime_stats();
                break;
            case std::remove_reference_t<decltype(program)>::FlightKind::Prefill: {
                const auto owner = slots_[meta.prefill_lane];
                if (owner != nullptr) {
                    const bool cancelled_now = owner->cancelled.load(std::memory_order_acquire);
                    resolve_prefill_step(owner, result.prefill, cancelled_now);
                }
                publish_runtime_stats();
                break;
            }
            }
        }
    }

    void run_decode_round(const RoundMembership& membership) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        // Round chaining (PATCHES.md #32): burst pure-decode stretches, but
        // admission rides the round cadence (one attempt per GPU unit), so
        // the burst length must yield to admission pressure: single rounds
        // while a prefill is staged or a queued request could admit into a
        // free lane, short bursts while the queue waits on full lanes, full
        // bursts only when nothing is waiting.
        std::uint32_t burst_limit = 8;
        if (!prefill_lanes_.empty()) {
            burst_limit = 1;
        } else {
            bool queue_waiting = false;
            {
                std::lock_guard lock(queue_mutex_);
                queue_waiting = !pending_.empty();
            }
            if (queue_waiting) {
                std::uint32_t free_lanes = 0;
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    free_lanes += slots_[lane] == nullptr ? 1U : 0U;
                }
                // With continuous admission (PATCHES.md #41) a queue backed up
                // behind full lanes has nothing to admit into, so the short
                // burst that used to protect admission cadence only costs host
                // round-trips. Free lanes still take the single round, so the
                // next iteration can refill them.
                // A free lane used to mean "take a single round so the next
                // iteration can admit into it". With continuous admission
                // (PATCHES.md #41) admission runs after any decode unit, so a
                // free lane no longer needs a single-round cadence — and under
                // 100-user load a lane is nearly always free, which pinned the
                // burst at 1 and left every round paying full host serial.
                // Keep a floor so the serial cost is amortised either way.
                // MEASURED (4B, 100 users, 90 s): 1 -> 2,661 tok/s, 2 -> 2,174,
                // 4 -> 1,708, with TTFT 1.7 s / 3.3 s / 5.5 s. Delaying a free
                // lane's refill by even ONE round costs 18%, which is far more
                // than the host serial a burst amortises. Keep it at 1; the
                // knob exists so the measurement can be repeated.
                static const std::uint32_t kFreeLaneBurst = [] {
                    const char* raw = std::getenv("SUROGATE_SERVE_FREE_LANE_BURST");
                    return raw != nullptr ? static_cast<std::uint32_t>(std::atoi(raw)) : 1U;
                }();
                // SUROGATE_SERVE_BURST_CAP caps the saturated-lane burst too (bisection knob:
                // the burst chains rounds device-side, and a mid-burst stop token makes the
                // resolution unwind — cap 1 removes that whole path).
                static const std::uint32_t kSaturatedBurst = [] {
                    const char* raw = std::getenv("SUROGATE_SERVE_BURST_CAP");
                    return raw != nullptr ? static_cast<std::uint32_t>(std::atoi(raw)) : 8U;
                }();
                burst_limit = free_lanes > 0 ? kFreeLaneBurst : kSaturatedBurst;
            }
        }
        instance_.program->set_round_burst_limit(burst_limit);
        instance_.program->set_round_width_hint(static_cast<std::uint32_t>(lanes.size()));
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, membership.budget_span());
        process_decode_round(membership, round);
        ++cumulative_stats_.decode_rounds;
    }

    // Mixed-token round (PATCHES.md #30): one forward advances the staged
    // prefill by a chunk and produces one decode token per active lane.
    // Stage more prompts alongside the ones already waiting, so a single mixed round prefills
    // several at once (#80). Admission with a deferred first chunk is CPU-only, so this costs
    // nothing on the GPU; it stops as soon as an admission would run a unit or finds no work.
    void top_up_prefill_lanes() {
        while (!prefill_lanes_.full()) {
            {
                std::lock_guard lock(queue_mutex_);
                if (pending_.empty()) { return; }
            }
            bool lane_free = false;
            for (std::uint32_t lane = 0; lane < max_concurrency_ && !lane_free; ++lane) {
                lane_free = slots_[lane] == nullptr;
            }
            if (!lane_free) { return; }
            const std::size_t before = prefill_lanes_.size();
            if (try_admit_one() != AdmissionProgress::ControlProgress) { return; }
            if (prefill_lanes_.size() == before) { return; }
        }
    }

    void run_mixed_round(const RoundMembership& membership) {
        if (prefill_lanes_.empty()) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = prefill_lanes_.front();
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const std::span<const std::uint32_t> lanes = membership.lane_span();
        // Only the prompts the program can still advance ride this round.
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> staged{};
        std::size_t staged_count = 0;
        for (const std::uint32_t candidate : prefill_lanes_.span()) {
            if (!instance_.program->mixed_round_supported(candidate)) { continue; }
            const auto& owner = slots_[candidate];
            if (owner == nullptr || owner->decode_ready) { continue; }
            staged[staged_count++] = candidate;
        }
        if (staged_count == 0) { throw std::logic_error("mixed round has no advanceable prefill"); }
        const MixedRoundResult mixed = instance_.program->advance_prefill_mixed(
            std::span<const std::uint32_t>(staged.data(), staged_count), lanes,
            membership.budget_span());
        process_decode_round(membership, mixed.round);
        ++cumulative_stats_.decode_rounds;
        // Resolve each staged prompt against its own result. resolve_prefill_step can retire a
        // lane and edit prefill_lanes_, so the lanes were captured before the round ran.
        for (std::size_t i = 0; i < mixed.prefill_count && i < staged_count; ++i) {
            const auto owner = slots_[staged[i]];
            if (owner == nullptr) { continue; }
            const bool cancelled_now = owner->cancelled.load(std::memory_order_acquire);
            resolve_prefill_step(owner, mixed.prefill_at(i), cancelled_now);
        }
        publish_runtime_stats();
    }

    void process_decode_round(const RoundMembership& membership,
                              const BatchedGeneratedRound& round) {
        const std::span<const std::uint32_t> lanes = membership.lane_span();

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        const auto t_preview = Clock::now();
        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            OutputDecision decision{};
            try {
                decision = request->output.preview(row_tokens, request->budget->remaining(),
                                                   request->budget->limit_reason());
            } catch (const std::exception& error) {
                // Name the row before the worker dies: which lane, how deep, and
                // the token ids it just produced. Cheap, and it is what turned a
                // "mixed rounds corrupt sometimes" report into a root cause.
                std::string ids;
                for (const TokenId id : row_tokens) {
                    ids += std::to_string(id);
                    ids += ' ';
                }
                std::fprintf(stderr,
                             "decode round row %zu/%zu lane %u gen=%zu tokens=[ %s]: %s (%s)\n",
                             row, lanes.size(), lane, request->generated.size(), ids.c_str(),
                             error.what(),
                             instance_.program->last_mixed_round_description(row).c_str());
                std::fflush(stderr);
                // SUROGATE_SERVE_SURVIVE_CORRUPTION=1: a diagnostic mode that
                // fails only the corrupted request and keeps serving, so one
                // run collects many attributed events. Never a production
                // setting — a corrupted stream must be fatal by default.
                static const bool survive =
                    std::getenv("SUROGATE_SERVE_SURVIVE_CORRUPTION") != nullptr;
                if (!survive) { throw; }
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                cancelled[row]      = 1;
                continue;
            }
            if (decision.accepted_tokens == 0 || decision.accepted_tokens > count ||
                (!decision.finished() && decision.accepted_tokens != count)) {
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        seg_timer_.preview += std::chrono::duration<double>(Clock::now() - t_preview).count();
        const auto t_resolve = Clock::now();
        if (!lanes.empty()) { // a zero-lane mixed round (pipelined prefill) has nothing to resolve
            instance_.program->resolve_pending_batch(
                lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
                std::span<const std::uint8_t>(terminal.data(), lanes.size()),
                std::span<const std::uint8_t>(cancelled.data(), lanes.size()));
        }
        seg_timer_.resolve += std::chrono::duration<double>(Clock::now() - t_resolve).count();

        const auto t_append = Clock::now();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                // A route without probabilities contributes NaN rather than nothing,
                // so the two vectors stay the same length and a client reading them
                // together cannot silently pair a token with another token's number.
                if (round.logprobs.size() >= round.tokens.size()) {
                    const auto row_logprobs = round.logprobs.subspan(
                        row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                    request->generated_logprobs.insert(request->generated_logprobs.end(),
                                                       row_logprobs.begin(), row_logprobs.end());
                } else {
                    request->generated_logprobs.insert(request->generated_logprobs.end(),
                                                       accepted[row],
                                                       std::numeric_limits<float>::quiet_NaN());
                }
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published));
            if (terminal[row]) {
                complete_success(request, finish_reasons[row]);
                remove_completed_slot(lane);
            }
        }
        seg_timer_.append += std::chrono::duration<double>(Clock::now() - t_append).count();
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row]) { cumulative_stats_.committed_decode_tokens += accepted[row]; }
        }
        const auto t_stats = Clock::now();
        publish_runtime_stats();
        seg_timer_.stats += std::chrono::duration<double>(Clock::now() - t_stats).count();
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (!prefill_lanes_.empty()) {
            instance_.request_memory.deactivate();
            prefill_lanes_.clear();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
        publish_runtime_stats();
    }

    void worker_loop() noexcept {
        // This thread was born with the process default device, not the engine's.
        if (cudaSetDevice(device_) != cudaSuccess) { return; }
        // Every round this thread runs must resolve op-plane state (Marlin
        // scratch, LoRA banks) in this engine's context -- the same one target
        // construction bound, so captured-graph addresses and eager calls agree.
        if (ops_context_ != nullptr) { ops::bind_ops_context(ops_context_); }
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                // Preemptive sleep parks the loop here between rounds even with
                // active lanes; their state is in the offloaded arenas and the
                // requests resume exactly where they stopped after wake.
                queue_cv_.wait(lock, [&] { return !asleep_ || stopping_; });
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock,
                                       [&] { return stopping_ || (!pending_.empty() && !asleep_); });
                    }
                } else if (!stopping_ && !pending_.empty()) {
                    // Requests waiting on the elastic overcommit gate with nothing running:
                    // memory comes back on other engines' round boundaries, not ours, so pace
                    // the retries instead of spinning through admission.
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active && instance_.program->kv_under_pressure()) {
                        queue_cv_.wait_for(lock, std::chrono::milliseconds(5),
                                           [&] { return stopping_; });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                const auto seg_t0                = Clock::now();
                const bool have_pending          = expire_pending_requests();
                auto cancelled_at_boundary = snapshot_cancellations();
                if constexpr (kPipelined) {
                    // A lane whose group is mid-pipeline is cancelled at its group's boundary
                    // (the snapshot is retaken every iteration), never under an in-flight round.
                    for (std::uint32_t g = 0; g < group_meta_.size(); ++g) {
                        if (!instance_.program->group_in_flight(g)) { continue; }
                        const GroupMeta& meta = group_meta_[g];
                        for (std::size_t row = 0; row < meta.membership.size; ++row) {
                            cancelled_at_boundary[meta.membership.lanes[row]] = false;
                        }
                        for (std::size_t i = 0; i < meta.staged_count; ++i) { cancelled_at_boundary[meta.staged[i]] = false; }
                        if (meta.prefill_flight) { cancelled_at_boundary[meta.prefill_lane] = false; }
                    }
                }
                cancel_active_requests(cancelled_at_boundary);
                if constexpr (kPipelined) {
                    seg_timer_.boundary += std::chrono::duration<double>(Clock::now() - seg_t0).count();
                    seg_timer_.maybe_report();
                    pipelined_iteration(have_pending);
                    continue;
                }
                // Elastic overcommit: a device short of KV asks every engine on it for its
                // prefix cache first. Retained lanes are the one thing an idle lane holds.
                if (instance_.program->kv_service_pressure()) {
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        if (slots_[lane] == nullptr && instance_.program->has_retained_lane(lane)) {
                            instance_.program->evict_retained_lane(lane);
                            invalidate_lane_plans(lane);
                        }
                    }
                }
                const RoundMembership membership = build_round_membership();
                seg_timer_.boundary += std::chrono::duration<double>(Clock::now() - seg_t0).count();
                seg_timer_.maybe_report();

                if (!prefill_lanes_.empty()) {
                    top_up_prefill_lanes();
                    // Let the staged set grow before spending a round on it (#88). Bounded, so
                    // a prompt whose neighbours never arrive still goes out promptly.
                    if (prefill_lanes_.size() < mixed_prefill_batch_target() &&
                        !prefill_lanes_.full() && !membership.empty() &&
                        deferred_mixed_rounds_ < mixed_prefill_batch_wait_rounds()) {
                        ++deferred_mixed_rounds_;
                        const auto t_decode = Clock::now();
                        last_round_         = LastRound{"decode",
                                                static_cast<std::uint32_t>(membership.size), 0,
                                                last_round_.index + 1};
                        run_decode_round(membership);
                        seg_timer_.decode +=
                            std::chrono::duration<double>(Clock::now() - t_decode).count();
                        seg_timer_.decode_rounds += 1;
                        previous_unit_was_decode = true;
                        continue;
                    }
                    deferred_mixed_rounds_ = 0;
                    if (!membership.empty() &&
                        instance_.program->mixed_round_supported(prefill_lanes_.front())) {
                        const auto t_mixed = Clock::now();
                        last_round_ = LastRound{"mixed",
                                                static_cast<std::uint32_t>(membership.size),
                                                prefill_lanes_.front(), last_round_.index + 1};
                        run_mixed_round(membership);
                        seg_timer_.mixed +=
                            std::chrono::duration<double>(Clock::now() - t_mixed).count();
                        seg_timer_.mixed_rounds += 1;
                        // The mixed round carried the decode batch, so the
                        // admission branch may run next (PATCHES.md #32).
                        previous_unit_was_decode = true;
                        continue;
                    }
                    if (!membership.empty() && !previous_unit_was_decode) {
                        run_decode_round(membership);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (membership.empty() || previous_unit_was_decode)) {
                    const auto t_admit         = Clock::now();
                    AdmissionProgress progress = try_admit_one();
                    // Continuous admission (PATCHES.md #41): with the deferred
                    // first chunk (#30) an admission is CPU-only — it stages
                    // the prompt and leaves the prefill to a mixed round — so
                    // one admission per GPU unit needlessly rations lane
                    // refills and shows up as TTFT. Keep admitting while the
                    // queue holds work, lanes are free and no GPU unit ran.
                    if (progress == AdmissionProgress::ControlProgress) {
                        for (std::uint32_t extra = 1; extra < max_concurrency_; ++extra) {
                            bool lane_free = false;
                            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                                lane_free = lane_free || slots_[lane] == nullptr;
                            }
                            if (!lane_free) { break; }
                            {
                                std::lock_guard lock(queue_mutex_);
                                if (pending_.empty()) { break; }
                            }
                            const AdmissionProgress more = try_admit_one();
                            if (more == AdmissionProgress::None) { break; }
                            if (more == AdmissionProgress::RanGpuUnit) {
                                progress = more;
                                break;
                            }
                        }
                    }
                    seg_timer_.admit +=
                        std::chrono::duration<double>(Clock::now() - t_admit).count();
                    if (progress == AdmissionProgress::RanGpuUnit) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    const auto t_decode = Clock::now();
                    last_round_ = LastRound{"decode",
                                            static_cast<std::uint32_t>(membership.size), 0,
                                            last_round_.index + 1};
                    run_decode_round(membership);
                    seg_timer_.decode +=
                        std::chrono::duration<double>(Clock::now() - t_decode).count();
                    seg_timer_.decode_rounds += 1;
                    previous_unit_was_decode = true;
                    continue;
                }
            } catch (const std::exception& error) {
                std::fprintf(stderr,
                             "engine worker loop fatal: %s [last round #%llu kind=%s batch=%u "
                             "prefill_lane=%u lanes=%u]\n",
                             error.what(), static_cast<unsigned long long>(last_round_.index),
                             last_round_.kind, last_round_.batch, last_round_.prefill_lane,
                             max_concurrency_);
                fail_all(std::current_exception());
                return;
            } catch (...) {
                std::fprintf(stderr, "engine worker loop fatal: unknown exception\n");
                fail_all(std::current_exception());
                return;
            }
        }
    }

    // Failure-path diagnostic (PATCHES.md #49): the wide-band corruption
    // surfaces as a bad token in the frontend, far from the round that
    // produced it. Recording each round's shape costs two stores and names
    // the producing path in the fatal line.
    struct LastRound {
        const char* kind  = "none";
        std::uint32_t batch = 0;
        std::uint32_t prefill_lane = 0;
        std::uint64_t index = 0;
    };
    LastRound last_round_;

    // Batched prefill (#88). A mixed round streams every touched expert once no matter how
    // many columns it carries, so on an MoE the per-prompt cost of prefill falls steeply with
    // width: the 35B's routed GEMMs measure 1.26 us per token at 640 columns and 0.65 at
    // 2,176. Waiting a few rounds to stage several prompts and prefilling them together is
    // therefore real work saved - unlike a dense model, where the column cost is flat and the
    // same trade is algebraically a no-op (see the 27B round-cost model in BENCHMARKS.md).
    //
    // A staged prompt is not decode-ready, so holding it back costs no decode throughput; it
    // costs that prompt's own latency, and only while the deferral bound allows.
    static std::size_t mixed_prefill_batch_target() {
        static const std::size_t target = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_PREFILL_BATCH");
            if (raw == nullptr || *raw == '\0') { return std::size_t{1}; }
            const long parsed = std::strtol(raw, nullptr, 10);
            if (parsed <= 1) { return std::size_t{1}; }
            return std::min(static_cast<std::size_t>(parsed), runtime::kMaximumMixedPrefills);
        }();
        return target;
    }

    static std::uint32_t mixed_prefill_batch_wait_rounds() {
        static const std::uint32_t rounds = [] {
            const char* raw = std::getenv("SUROGATE_SERVE_PREFILL_BATCH_WAIT");
            if (raw == nullptr || *raw == '\0') { return std::uint32_t{8}; }
            const long parsed = std::strtol(raw, nullptr, 10);
            return parsed > 0 ? static_cast<std::uint32_t>(parsed) : std::uint32_t{8};
        }();
        return rounds;
    }

    std::uint32_t deferred_mixed_rounds_ = 0;

    struct SegmentTimer {
        bool enabled = std::getenv("SUROGATE_SERVE_ROUND_TIMING") != nullptr;
        double boundary = 0, admit = 0, mixed = 0, decode = 0;
        double preview = 0, resolve = 0, append = 0, stats = 0;
        std::uint64_t mixed_rounds = 0, decode_rounds = 0;
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
        void maybe_report() {
            if (!enabled) { return; }
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last).count() < 5.0) { return; }
            std::fprintf(stderr,
                         "round-timing: boundary %.0fms admit %.0fms mixed %.0fms/%llu decode "
                         "%.0fms/%llu | preview %.0fms resolve %.0fms append %.0fms stats %.0fms "
                         "(per 5s)\n",
                         boundary * 1e3, admit * 1e3, mixed * 1e3,
                         static_cast<unsigned long long>(mixed_rounds), decode * 1e3,
                         static_cast<unsigned long long>(decode_rounds), preview * 1e3,
                         resolve * 1e3, append * 1e3, stats * 1e3);
            boundary = admit = mixed = decode = 0;
            preview = resolve = append = stats = 0;
            mixed_rounds = decode_rounds = 0;
            last         = now;
        }
    };
    SegmentTimer seg_timer_;

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const AdmissionResources admission_capacity_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    bool asleep_ = false; ///< guarded by queue_mutex_; set by sleep(), cleared by wake()
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    // Multi-prompt prefill (#80): several prompts can be staged at once and share a round.
    // The first one owns the round's card; the rest ride along as extra segments.
    class PrefillLaneSet {
      public:
        [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
        [[nodiscard]] std::size_t size() const noexcept { return size_; }
        [[nodiscard]] bool full() const noexcept { return size_ >= lanes_.size(); }
        [[nodiscard]] std::uint32_t front() const { return lanes_.at(0); }
        [[nodiscard]] std::span<const std::uint32_t> span() const noexcept {
            return std::span<const std::uint32_t>(lanes_.data(), size_);
        }
        [[nodiscard]] bool contains(std::uint32_t lane) const noexcept {
            return std::find(lanes_.begin(), lanes_.begin() + static_cast<std::ptrdiff_t>(size_),
                             lane) != lanes_.begin() + static_cast<std::ptrdiff_t>(size_);
        }
        void add(std::uint32_t lane) {
            if (full() || contains(lane)) { throw std::logic_error("prefill lane set overflow"); }
            lanes_[size_++] = lane;
        }
        // Erasing keeps the order so the round's card owner stays stable.
        void erase(std::uint32_t lane) noexcept {
            for (std::size_t i = 0; i < size_; ++i) {
                if (lanes_[i] != lane) { continue; }
                for (std::size_t j = i + 1; j < size_; ++j) { lanes_[j - 1] = lanes_[j]; }
                --size_;
                return;
            }
        }
        void clear() noexcept { size_ = 0; }

      private:
        std::array<std::uint32_t, runtime::kMaximumMixedPrefills> lanes_{};
        std::size_t size_ = 0;
    };
    PrefillLaneSet prefill_lanes_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    bool stopping_ = false;
    bool failed_   = false;
    ops::EngineOpsContext* ops_context_ = nullptr;
    int device_ = 0;
    std::thread worker_;
};

/// The executor variant matching a variant of target instances.
///
/// An engine holds whichever executor its loaded target needs, so the two
/// variants have to stay in step: one alternative each, in the same order, plus
/// the empty state an engine has before it constructs one. Writing the second
/// list out by hand meant that adding a target and forgetting this file gave a
/// std::get on the wrong alternative rather than a compile error, so it derives
/// from the first list instead.
template <class TargetVariant> struct ExecutorVariant;

template <class... Instances>
struct ExecutorVariant<std::variant<std::unique_ptr<Instances>...>> {
    using type =
        std::variant<std::monostate, std::unique_ptr<ConcurrentExecutor<Instances>>...>;
};

template <class TargetVariant>
using ExecutorVariantFor = typename ExecutorVariant<TargetVariant>::type;

} // namespace sinfer::runtime
