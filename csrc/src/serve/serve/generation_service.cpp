#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "api/ops/lora_store.h"
#include "serve/lora_registry.h"
#include "serve/output_parsers.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <thread>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace sinfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const sinfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case sinfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case sinfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case sinfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case sinfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case sinfer::RequestErrorKind::Cancelled:
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case sinfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const sinfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case sinfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case sinfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case sinfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case sinfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case sinfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception,
                                      const char* code = "invalid_media") {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

sinfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(sinfer::product::media_acquire::Error(
            sinfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    sinfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = sinfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const sinfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }

    remaining_bytes -= source_bytes.size();
    sinfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? sinfer::MediaKind::Image : sinfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case sinfer::product::media_acquire::SourceKind::Path:
    case sinfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case sinfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case sinfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes = std::move(source_bytes);
    return media;
}

[[noreturn]] void throw_request_error(const sinfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(sinfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public sinfer::OutputSink {
public:
    ServiceOutputSink(const StreamSink& sink, bool filter_tool_calls)
        : sink_(&sink), filter_tool_calls_(filter_tool_calls) {}

    void publish(sinfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == sinfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            std::string visible =
                filter_tool_calls_ ? tool_filter_.feed(delta.text) : std::move(delta.text);
            publish_content(visible);
        }
    }

    std::size_t finish(bool is_tool_call_response) {
        if (filter_tool_calls_) { publish_content(tool_filter_.finish(is_tool_call_response)); }
        return content_bytes_;
    }

private:
    void publish_content(const std::string& text) {
        if (text.empty() || !sink_->on_content) { return; }
        sink_->on_content(text);
        content_bytes_ += text.size();
    }

    const StreamSink* sink_ = nullptr;
    bool filter_tool_calls_ = false;
    ToolCallStreamFilter tool_filter_;
    std::size_t content_bytes_ = 0;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)) {
    sinfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.device                   = options_.device;
    engine_options.devices                  = options_.devices;
    engine_options.max_context              = options_.max_context;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.expert_slots             = options_.expert_slots;
    engine_options.host_expert_bank         = options_.host_expert_bank;
    engine_options.cpu_moe_share            = options_.cpu_moe_share;
    engine_options.cpu_moe_min_tokens       = options_.cpu_moe_min_tokens;
    engine_options.cpu_moe_prefill_share    = options_.cpu_moe_prefill_share;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.kv_cache_skip_layers     = options_.kv_cache_skip_layers;
    engine_options.rewrite_checkpoints      = options_.rewrite_checkpoints;
    engine_options.elastic_kv               = options_.elastic_kv;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.speculative              = options_.speculative;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.chat_template_override   = options_.chat_template;
    engine_options.sleep_enable             = options_.enable_sleep_mode;
    // The adapter's tensors, decoded on the host. The target binds them to its own
    // weights; a module the target cannot place is refused there rather than
    // dropped, because a partly applied adapter is worse than none.
    if (options_.enable_lora) {
        // The engine prepares the adapter machinery even with no adapters named:
        // banks preallocated and the delta kernels captured, so adapters loaded
        // later through the runtime endpoints work under the graphs recorded at
        // startup.
        engine_options.lora_enable   = true;
        engine_options.lora_slots    = options_.max_loras;
        engine_options.lora_max_rank = options_.max_lora_rank;
        std::int32_t slot            = 0;
        if (!options_.lora_modules.empty()) {
            LoraRegistry registry;
            std::vector<std::pair<std::string, std::string>> modules;
            for (const auto& module : options_.lora_modules) {
                modules.emplace_back(module.name, module.path);
            }
            registry.load(modules, options_.max_lora_rank);
            // Slots are assigned here, once, in the order the deployment named
            // them, and a request's adapter is turned into its slot index at
            // admission. The engine below never sees a name.
            for (const auto& [name, adapter] : registry.adapters()) {
                std::vector<std::string> skipped;
                auto payloads = LoraRegistry::read_payloads(adapter, skipped);
                if (!skipped.empty()) {
                    throw std::invalid_argument(
                        "--lora-modules '" + name + "': module '" + skipped.front() +
                        "' carries no layer index, so it cannot be bound to a projection");
                }
                for (auto& payload : payloads) {
                    payload.slot = slot;
                    engine_options.lora_payloads.push_back(std::move(payload));
                }
                lora_slot_of_[name] = slot;
                ++slot;
            }
        }
        for (std::int32_t free = slot; free < static_cast<std::int32_t>(options_.max_loras);
             ++free) {
            lora_free_slots_.push_back(free);
        }
    }
    engine_options.load_progress            = std::move(load_progress);
    const bool lora_requested = engine_options.lora_enable;
    engine_              = std::make_unique<sinfer::Engine>(std::move(engine_options));
    // Only a target that binds adapters to its own weights can apply them, and most
    // do not yet. Without this check a request naming an adapter would be answered
    // by the base model on those targets -- served confidently, and wrong. The
    // directory is registered during load, so an empty one here means nothing can
    // bind, at startup or from the runtime endpoints.
    if (lora_requested && !engine_->lora_store().has_bindings()) {
        throw std::invalid_argument(
            "--enable-lora: this target does not apply adapters, so one would be loaded and "
            "silently ignored. Merge it into the checkpoint before conversion (`surogate merge`) "
            "to serve it here.");
    }
    prompt_capabilities_ = engine_->prompt_capabilities();
    request_capacity_    = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime() const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(sinfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        return std::make_shared<RequestLifetime>(
            request_capacity_, started,
            started + std::chrono::milliseconds(options_.pending_timeout_ms));
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    PreparedRequest prepared;
    sinfer::RequestOptions request_options = to_request_options(request, options_);
    // The adapter the request named, as its bank slot. Resolved here rather than
    // deeper because this is the last place the name exists: the round stages an
    // integer per lane and nothing below knows adapters by name.
    request_options.execution.lora_slot = lora_slot(request.lora_adapter);
    prepared.include_usage                 = request.include_usage;
    // --enable-auto-tool-choice, vLLM's gate. Without it a request may still name a
    // function or demand one; what it may not do is leave the choice to the model,
    // because the deployment has not said the model's calls can be read back.
    if (!options_.enable_auto_tool_choice && !request.tools.empty() &&
        request.tool_choice.mode == ToolChoiceMode::Auto) {
        ApiError error;
        error.message = "tool_choice 'auto' needs the server started with "
                        "--enable-auto-tool-choice (and --tool-call-parser)";
        error.param   = "tool_choice";
        error.code    = "auto_tool_choice_disabled";
        throw ApiException(std::move(error));
    }
    prepared.tool_capable                  = request.uses_tools() || request.has_tool_history();
    prepared.tool_name_max_length          = request.tool_name_max_length;
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    prepared.enable_thinking                   = semantics.enable_thinking;
    prepared.preserve_thinking                 = semantics.preserve_thinking;
    prepared.preserve_thinking_semantic_change = request.preserve_thinking_semantic_change;
    const bool request_has_media               = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime();

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
        sinfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        sinfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const sinfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
        sinfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const sinfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) {
        output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared.tool_capable);
    }
    sinfer::OutputSink* public_sink = output_sink.get();
    sinfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = sinfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    sinfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const sinfer::RequestError& exception) { throw_request_error(exception); }
    GenerationOutcome outcome;
    outcome.text              = std::move(result.content);
    outcome.reasoning         = std::move(result.reasoning);
    outcome.prompt_tokens     = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens  = static_cast<int>(result.reasoning_tokens);
    outcome.finish_reason     = result.finish_reason;

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds  = result.timings.decode_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    bool is_tool_call_response = false;
    if (prepared.tool_capable) {
        ParsedToolCalls parsed = parse_tool_calls(options_.tool_call_format, outcome.text,
                                                  prepared.tool_name_max_length);
        outcome.text           = std::move(parsed.content);
        is_tool_call_response  = parsed.is_tool_call_response;
        if (is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
    }
    // --reasoning-parser reconciles what the frontend already split. `none` folds
    // the span back into the answer rather than dropping it.
    ReasoningSplit split = split_reasoning(options_.reasoning_format,
                                           std::move(outcome.reasoning), std::move(outcome.text));
    outcome.reasoning    = std::move(split.reasoning);
    outcome.text         = std::move(split.content);
    if (output_sink) {
        outcome.streamed_content_bytes = output_sink->finish(is_tool_call_response);
    }
    return outcome;
}

void GenerationService::load_lora_adapter(const std::string& name, const std::string& path) {
    if (!options_.enable_lora) {
        throw std::invalid_argument("the server was started without --enable-lora");
    }
    ops::LoraStore& store = engine_->lora_store();
    if (!store.has_bindings()) {
        throw std::invalid_argument("this target does not apply adapters");
    }
    // The banks live in the engine's sleepable estate now, so uploading into a
    // slept model would write into unmapped memory.
    if (engine_->is_sleeping()) {
        throw std::invalid_argument(
            "the model is asleep; wake it (POST /wake_up, or send it a request) before loading "
            "an adapter");
    }
    // Parse and validate outside the lock -- reading safetensors can take a
    // moment and requests resolving names must not wait on it.
    LoraRegistry registry;
    registry.load({{name, path}}, options_.max_lora_rank);
    const auto& adapters = registry.adapters();
    const auto found     = adapters.find(name);
    if (found == adapters.end()) {
        throw std::invalid_argument("adapter '" + name + "' did not load");
    }
    std::vector<std::string> skipped;
    auto payloads = LoraRegistry::read_payloads(found->second, skipped);
    if (!skipped.empty()) {
        throw std::invalid_argument("module '" + skipped.front() +
                                    "' carries no layer index, so it cannot be bound");
    }

    std::int32_t slot = -1;
    {
        const std::lock_guard<std::mutex> lock(lora_mutex_);
        if (lora_slot_of_.count(name) != 0) {
            throw std::invalid_argument("adapter '" + name + "' is already loaded");
        }
        if (lora_free_slots_.empty()) {
            throw std::invalid_argument(
                "all " + std::to_string(options_.max_loras) +
                " adapter slots are in use -- unload one, or restart with a larger --max-loras");
        }
        slot = lora_free_slots_.front();
        lora_free_slots_.erase(lora_free_slots_.begin());
    }

    // The uploads write only this slot's regions, which no request can select yet
    // -- the name becomes routable below, after every module landed. On failure
    // the slot is scrubbed and returned, so a half-written adapter is never
    // servable.
    try {
        for (const auto& payload : payloads) {
            store.set_module_slot(payload.layer, payload.module, slot, payload.a, payload.b,
                                  payload.rank, payload.in_dim, payload.out_dim, payload.scale);
        }
    } catch (...) {
        store.clear_slot(slot);
        const std::lock_guard<std::mutex> lock(lora_mutex_);
        lora_free_slots_.push_back(slot);
        throw;
    }
    store.set_active(true);
    const std::lock_guard<std::mutex> lock(lora_mutex_);
    lora_slot_of_[name] = slot;
}

void GenerationService::unload_lora_adapter(const std::string& name) {
    std::int32_t slot = -1;
    {
        const std::lock_guard<std::mutex> lock(lora_mutex_);
        const auto found = lora_slot_of_.find(name);
        if (found == lora_slot_of_.end()) {
            throw std::invalid_argument("adapter '" + name + "' is not loaded");
        }
        slot = found->second;
        lora_slot_of_.erase(found);
    }
    if (engine_->is_sleeping()) {
        // Undo the erase and refuse: clearing banks in a slept model would write
        // into unmapped memory.
        const std::lock_guard<std::mutex> relock(lora_mutex_);
        lora_slot_of_[name] = slot;
        throw std::invalid_argument(
            "the model is asleep; wake it before unloading an adapter");
    }
    // Zeroed, not freed: a request already in flight that selected this slot adds
    // nothing from here on -- it degrades to the base model instead of reading
    // another adapter's weights. The slot goes to the back of the free list so it
    // is the last one a later load reuses.
    engine_->lora_store().clear_slot(slot);
    const std::lock_guard<std::mutex> lock(lora_mutex_);
    lora_free_slots_.push_back(slot);
}

void GenerationService::sleep(bool preempt) {
    if (!options_.enable_sleep_mode) {
        throw std::invalid_argument("the server was started without --enable-sleep-mode");
    }
    // Refusing new submissions first turns the drain below into a bounded wait:
    // the engine rejects anything that arrives after this line.
    // (Engine::sleep is idempotent, so two racing sleeps are both fine.)
    engine_->sleep_begin();
    static const bool env_preempt = std::getenv("SUROGATE_SLEEP_PREEMPT") != nullptr;
    if (preempt || env_preempt) {
        // Preemptive: in-flight generations park mid-flight and resume,
        // byte-identical, after the next wake.
        engine_->sleep(/*allow_active=*/true);
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (true) {
        {
            const std::lock_guard<std::mutex> lock(request_capacity_->mutex);
            if (request_capacity_->active == 0) { break; }
        }
        if (std::chrono::steady_clock::now() > deadline) {
            engine_->wake();
            throw std::runtime_error(
                "sleep timed out waiting for in-flight requests to finish; the model stays awake");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    engine_->sleep();
}

void GenerationService::wake_up() { engine_->wake(); }

std::size_t GenerationService::active_requests() const {
    const std::lock_guard<std::mutex> lock(request_capacity_->mutex);
    return request_capacity_->active;
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = ChatRole::User;
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens       = 4;
        request.max_tokens_set   = true;
        PreparedRequest prepared = prepare(request);
        run(prepared, nullptr);
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Warning,
                          std::string("warmup failed (continuing): ") + exception.what());
    }
}

} // namespace sinfer::serve
