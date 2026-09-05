#include "api/family/prepared_prompt.h"
#include "api/engine.h"

#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "core/limits.h"
#include "runtime/contract/types.h"
#include "ops/linear/w8a8/w4fp4_plane.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/linear/w8a8/w8fp8_plane.h"
#include "api/ops/lora_store.h"
#include "core/engine_context.h"
#include "core/sleep.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/registry.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>

namespace sinfer {


// The ops layer bounds batched work by kMaximumBatchColumns; the serving
// layer hands it kMaximumConcurrency lanes. They must agree, or a raise on
// one side silently outruns the other (PATCHES.md #35).
static_assert(static_cast<std::int32_t>(kMaximumConcurrency) == kMaximumBatchColumns,
              "serving concurrency ceiling and the ops batch bound must match");namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse      = options.execution.allow_prefix_reuse;
    resolved.execution.lora_slot               = options.execution.lora_slot;
    resolved.execution.min_tokens              = options.execution.min_tokens;
    resolved.execution.stop_barrier            = options.execution.stop_barrier;
    resolved.execution.stop_barrier_count      = options.execution.stop_barrier_count;
    resolved.stop                              = std::move(options.stop);
    resolved.output                            = options.output;
    return resolved;
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, PromptPreparationStats preparation, SamplingMode mode,
         family::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare(std::move(preparation)), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    PromptPreparationStats prepare;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    family::PreparedPrompt value;
};

std::vector<TokenId> PreparedPrompt::token_ids() const {
    if (!impl_) { return {}; }
    return family::PreparedPromptAccess::view(impl_->value).token_ids;
}

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
}

const PromptPreparationStats& PreparedPrompt::preparation_stats() const noexcept {
    static const PromptPreparationStats empty;
    return impl_ != nullptr ? impl_->prepare : empty;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    // One ConcurrentExecutor per target instance, derived from the registry's
    // own list so a new target needs no edit here.
    using Executor = runtime::ExecutorVariantFor<targets::ActiveTarget>;

    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        // The engine's op-layer state home. Bound here so everything target
        // construction creates -- Marlin scratch and adoption, LoRA banks,
        // sleepable arenas (which record this as their owner) -- lands in THIS
        // engine's context; the executor's worker thread binds the same object,
        // so addresses captured into graphs now and used by rounds later agree.
        options.ops_context = &ops_context;
        ops::bind_ops_context(&ops_context);
        // surogate vendor patch (PATCHES.md #20): the engine opts into the
        // derived FP8 prefill plane (op tests stay int8-exact by default;
        // SUROGATE_SERVE_FP8_PREFILL=0 vetoes).
        // Sleep mode: every owning DeviceArena constructed while this is set
        // becomes a VMM-backed sleepable region. Scoped to target construction
        // so arenas made elsewhere (tests, later tools) stay ordinary.
        set_sleepable_allocations(options.sleep_enable);
        ops::detail::w8fp8_plane_set_enabled(true);
        ops::detail::marlin_plane_set_enabled(true);
        ops::detail::marlin_set_fixed_m(static_cast<int>(options.max_concurrency));
        // surogate vendor patch (PATCHES.md #21): NVFP4 prefill profile is
        // an explicit opt-in (quality class change).
        if (const char* mode = std::getenv("SUROGATE_SERVE_PREFILL_QUANT");
            mode != nullptr && std::string_view(mode) == "fp4") {
            if (ops::detail::w8_device_compute_capability() >= 120) {
                ops::detail::w8_prefill_quant_set_mode(ops::detail::PrefillQuantMode::Fp4);
            } else {
                // NVFP4 needs the sm_120a block-scale tensor cores; explicit
                // requests degrade loudly, never silently.
                std::fprintf(stderr,
                             "surogate-serve: SUROGATE_SERVE_PREFILL_QUANT=fp4 requires an "
                             "sm_120-class GPU (found CC %d.%d); falling back to the default "
                             "quant profile.\n",
                             ops::detail::w8_device_compute_capability() / 10,
                             ops::detail::w8_device_compute_capability() % 10);
            }
        }
        auto constructed = options.devices.size() > 1 ? targets::construct_pipeline_target(options)
                                                      : targets::construct_target(options, device);
        active            = std::move(constructed.active);
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        // An automatic context (max_context == 0) is resolved by the target against the device's
        // free memory; every consumer below — the executors, admission, the prompt ceiling —
        // must see the resolved value, not the request.
        if (constructed.resolved_max_context != 0) {
            options.max_context = constructed.resolved_max_context;
        }
        // Every arm of what used to be a twelve-way if-constexpr ladder here
        // built ConcurrentExecutor<Instance> from the instance it matched, so
        // the ladder only restated its own subject.
        executor = std::visit(
            [&](auto& target_ptr) -> Executor {
                using Instance =
                    typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                return std::make_unique<runtime::ConcurrentExecutor<Instance>>(*target_ptr,
                                                                              options);
            },
            active);
        set_sleepable_allocations(false);
        ops::bind_ops_context(nullptr);
    }

    ~Impl() noexcept {
        executor.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    // Declared first so it is destroyed last: the executor's teardown and the
    // arenas' teardown may still touch context-homed state.
    ops::EngineOpsContext ops_context;
    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    Executor executor;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input), control);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, sampling_mode, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_text(std::string_view text, bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared =
                target_ptr->loaded->frontend.prepare_text(text, allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            // A raw prompt has no reasoning turn to resume, so the session starts outside one.
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::NonThinking, std::move(prepared)));
        },
        impl_->active);
}

bool Engine::supports_chat() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) -> bool {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.supports_chat();
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(
                    RequestErrorKind::ContextLengthExceeded,
                    context_capacity_error(info.prompt_tokens, target_ptr->capacity));
            }
            const PromptPreparationStats preparation = prepared.preparation_stats();
            return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
                info, preparation, SamplingMode::Thinking, std::move(prepared)));
        },
        impl_->active);
}

std::vector<std::string> Engine::token_texts(std::span<const TokenId> ids) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.token_texts(ids);
        },
        impl_->active);
}

std::vector<TokenId> Engine::default_stop_tokens() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) -> std::vector<TokenId> {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            const StopPolicy& policy = target_ptr->loaded->frontend.default_stop_policy();
            return policy.token_ids;
        },
        impl_->active);
}

std::uint32_t Engine::count_tokens(PromptInput input, const PreparationControl& control) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input), control);
        },
        impl_->active);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.prompt_capabilities();
        },
        impl_->active);
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                std::chrono::steady_clock::time_point pending_deadline) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare.seconds;
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;

            GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return std::visit(
        [&](auto& executor) -> GenerationHandle {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                auto submission = executor->submit(std::move(prompt.impl_->value), prompt_summary,
                                                   prepare_seconds, std::move(resolved_options),
                                                   pending_deadline);
                return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
                    impl_, std::move(submission), resolved_sampling));
            }
        },
        impl_->executor);
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    return submit(std::move(prompt), std::move(options)).wait(sink, cancellation);
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

void Engine::sleep_begin() {
    auto& impl = *impl_;
    if (!impl.options.sleep_enable) {
        throw std::logic_error("the engine was built without sleep_enable");
    }
    std::visit(
        [&](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                executor->set_asleep(true);
            }
        },
        impl.executor);
}

void Engine::sleep(bool allow_active) {
    auto& impl = *impl_;
    if (!impl.options.sleep_enable) {
        throw std::logic_error("the engine was built without sleep_enable");
    }
    std::visit(
        [&](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                if (device_asleep(impl.device.device, &impl.ops_context)) { return; } // idempotent
                executor->set_asleep(true);
                if (!allow_active && executor->any_active_lane()) {
                    executor->set_asleep(false);
                    throw std::logic_error(
                        "sleep requires a drained engine; requests are still in flight");
                }
                // With submissions refused and no active lanes, the worker loop
                // is parked on its queue wait; holding the execution mutex makes
                // that certain before touching memory.
                auto paused = executor->pause_execution();
                impl.device.synchronize();
                const std::size_t released = sleep_device(impl.device.device, &impl.ops_context);
                std::fprintf(stderr, "engine: asleep, released %.2f GiB of device memory\n",
                             static_cast<double>(released) / (1024.0 * 1024.0 * 1024.0));
            }
        },
        impl.executor);
}

void Engine::wake() {
    auto& impl = *impl_;
    std::visit(
        [&](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                if (device_asleep(impl.device.device, &impl.ops_context)) {
                    auto paused = executor->pause_execution();
                    const std::size_t mapped = wake_device(impl.device.device, &impl.ops_context);
                    impl.device.synchronize();
                    std::fprintf(stderr, "engine: awake, restored %.2f GiB of device memory\n",
                                 static_cast<double>(mapped) / (1024.0 * 1024.0 * 1024.0));
                }
                executor->set_asleep(false);
            }
        },
        impl.executor);
}

void Engine::prepare_sleep_backup() {
    if (impl_->options.sleep_enable) { sleep_prepare_backups(&impl_->ops_context); }
}

std::size_t Engine::sleepable_bytes() const { return sleep_owned_bytes(&impl_->ops_context); }

ops::LoraStoreSet& Engine::lora_stores() { return impl_->ops_context.slot<ops::LoraStoreSet>(); }

int Engine::device() const { return impl_->device.device; }

void Engine::shrink_kv() {
    std::visit(
        [](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (!std::is_same_v<Executor, std::monostate>) { executor->shrink_kv(); }
        },
        impl_->executor);
}

bool Engine::is_sleeping() const {
    return std::visit(
        [](auto& executor) -> bool {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                return false;
            } else {
                return executor->asleep();
            }
        },
        impl_->executor);
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> MemorySummary {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->memory_summary();
            }
        },
        impl_->executor);
}

MediaCacheSummary Engine::media_cache_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.media_cache_summary();
        },
        impl_->active);
}

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [](const auto& executor) -> RuntimeStats {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (std::is_same_v<Executor, std::monostate>) {
                throw std::logic_error("concurrent Engine executor is unavailable");
            } else {
                return executor->runtime_stats();
            }
        },
        impl_->executor);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    std::visit(
        [](auto& executor) {
            using Executor = std::remove_cvref_t<decltype(executor)>;
            if constexpr (!std::is_same_v<Executor, std::monostate>) {
                executor->reset_memory_peaks();
            }
        },
        impl_->executor);
}

} // namespace sinfer
