#include "core/device.h"
#include "serve/generation_service.h"
#include <set>

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "api/ops/lora_store.h"
#include "serve/lora_registry.h"
#include "serve/output_parsers.h"
#include "serve/tool_call_parser.h"
#include "serve/tool_constraints.h"
#include "serve/translate.h"
#include "serve/parallel_decoding.h"
#include "serve/decisions_schema.h"

#include <algorithm>
#include <array>
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
    std::size_t awaiting_wake = 0;
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
        if (awaiting_wake) { --capacity->awaiting_wake; }
    }

    void finish_wake() {
        std::lock_guard lock(capacity->mutex);
        --capacity->awaiting_wake;
        awaiting_wake = false;
    }

    bool awaiting_wake = true;
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

void validate_token_media(const GenerationRequest& request) {
    if (!request.prompt_token_ids.empty() && request.media_item_count() != 0) {
        throw ApiException(ApiError{
            .status = 400,
            .message = "tokens cannot be combined with images or video; send messages without tokens for vision input",
            .param = "tokens"});
    }
}

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
    error.message = "client disconnected during request preparation";
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

[[noreturn]] void refuse(std::string message, std::string param, const char* code) {
    throw ApiException(ApiError{.status = 400, .message = std::move(message),
                                .param = std::move(param), .code = code});
}

/// Answer with a classified decisions fault, keeping an engine-internal detail out of the
/// response body and in the server log instead.
[[noreturn]] void throw_decisions_fault(DecisionsFault fault) {
    if (!fault.internal_detail.empty()) {
        write_console_log(ConsoleLogLevel::Error,
                          "decisions request failed inside the engine: " + fault.internal_detail);
    }
    throw ApiException(std::move(fault.error));
}

/// Run one interaction with the engine with the fault boundary drawn around it.
///
/// Everything inside is the engine's own thread, planner and scheduler working on the engine's
/// state. `ApiException` is this endpoint's own refusal (a cancellation, a queue timeout) and
/// `RequestError` is the engine's client-facing vocabulary, already mapped by
/// `request_error_to_api_error`; both pass through untouched. Anything else that comes out --
/// an `std::invalid_argument` from a scheduler invariant handed over by `fail_all`, a prefix
/// image that is no longer in this engine, a `std::logic_error` from a round -- is the
/// engine's fault and is reported as such rather than as a 400 against the caller. An
/// `InvalidRequest` raised by the planner is still the caller's and keeps its 400;
/// `classify_decisions_fault` is what decides.
template <class Fn>
decltype(auto) in_engine(Fn&& fn) {
    try {
        return fn();
    } catch (const ApiException&) {
        throw;
    } catch (const sinfer::RequestError&) {
        throw;
    } catch (const std::exception& cause) {
        throw_decisions_fault(classify_decisions_fault(cause, DecisionsFaultStage::Engine));
    }
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(sinfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

void populate_score_texts(Engine& engine, GenerationOutcome& outcome) {
    std::vector<TokenId> ids;
    for (const auto* scores : {&outcome.prompt_scores, &outcome.completion_scores}) {
        for (const auto& score : *scores) {
            if (score.selected.token_id >= 0) ids.push_back(score.selected.token_id);
            for (const auto& candidate : score.top) ids.push_back(candidate.token_id);
        }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    if (!ids.empty()) {
        auto texts = engine.token_texts(ids);
        for (std::size_t i = 0; i < ids.size(); ++i) outcome.score_texts.emplace(ids[i], std::move(texts[i]));
    }
    for (const auto& score : outcome.completion_scores) {
        outcome.token_logprobs.push_back(score.selected.logprob);
        outcome.token_texts.push_back(outcome.score_texts.at(score.selected.token_id));
    }
}

class ServiceOutputSink final : public sinfer::OutputSink {
public:
    ServiceOutputSink(Engine& engine, const StreamSink& sink, bool filter_tool_calls, bool json_tools, bool muse_tools)
        : engine_(&engine), sink_(&sink), filter_tool_calls_(filter_tool_calls), muse_tools_(muse_tools), tool_filter_(json_tools,muse_tools) {}

    void publish(sinfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == sinfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            std::string visible = (filter_tool_calls_ && (!muse_tools_ || delta.channel == OutputChannel::Tool))
                ? tool_filter_.feed(delta.text) : std::move(delta.text);
            publish_content(visible);
        }
    }

    void prompt_ready(std::uint32_t reused_prompt_tokens) override {
        if (sink_->on_prompt_ready) { sink_->on_prompt_ready(reused_prompt_tokens); }
    }

    void publish_scores(TokenScoreDelta delta) override {
        if (!sink_->on_scores) return;
        GenerationOutcome outcome;
        outcome.prompt_scores = std::move(delta.prompt);
        outcome.completion_scores = std::move(delta.completion);
        populate_score_texts(*engine_, outcome);
        sink_->on_scores(outcome);
    }

    std::size_t finish(bool is_tool_call_response) {
        if (filter_tool_calls_) { publish_content(tool_filter_.finish(is_tool_call_response || muse_tools_)); }
        return content_bytes_;
    }

private:
    void publish_content(const std::string& text) {
        if (text.empty() || !sink_->on_content) { return; }
        sink_->on_content(text);
        content_bytes_ += text.size();
    }

    Engine* engine_ = nullptr;
    const StreamSink* sink_ = nullptr;
    bool filter_tool_calls_ = false;
    bool muse_tools_ = false;
    ToolCallStreamFilter tool_filter_;
    std::size_t content_bytes_ = 0;
};

} // namespace

void finalize_output_text(GenerationOutcome& outcome, ReasoningFormat format,
                          std::optional<std::size_t> streamed_content_bytes) {
    // The sink counts content bytes before reasoning is folded into the response.
    outcome.unstreamed_content.clear();
    if (streamed_content_bytes) {
        if (*streamed_content_bytes > outcome.text.size()) {
            throw std::logic_error("streamed content exceeds terminal content");
        }
        outcome.unstreamed_content = outcome.text.substr(*streamed_content_bytes);
    }
    auto split = split_reasoning(format, std::move(outcome.reasoning), std::move(outcome.text));
    outcome.reasoning = std::move(split.reasoning);
    outcome.text = std::move(split.content);
}

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)), lora_slots_(options_.max_loras) {
    // The CLI refuses a bad calibration temperature, but options can be built in code too; a
    // bad one must stop the service here, before any weights load, rather than turn every
    // decisions request into a 500.
    if (!valid_decision_temperature(options_.decision_temperature)) {
        throw std::invalid_argument("decision temperature must be finite and greater than zero");
    }
    sinfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.borrowed_weights         = options_.borrowed_weights;
    engine_options.device                   = options_.device;
    engine_options.devices                  = options_.devices;
    engine_options.max_context              = options_.max_context;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.host_moe_layers          = options_.host_moe_layers;
    engine_options.gpu_layers               = options_.gpu_layers;
    engine_options.offload_vision = options_.offload_vision;
    engine_options.offload_embeddings = options_.offload_embeddings;
    engine_options.offload_output_head = options_.offload_output_head;
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
    engine_options.elastic_kv               = options_.elastic_kv;
    engine_options.elastic_kv_overcommit    = options_.elastic_kv_overcommit;
    engine_options.gpu_memory_limit_bytes   = options_.gpu_memory_limit_bytes;
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
                auto update = lora_slots_.update(name, Clock::time_point::max());
                std::vector<std::string> skipped;
                auto payloads = LoraRegistry::read_payloads(adapter, skipped);
                if (!skipped.empty()) {
                    throw std::invalid_argument(
                        "--lora-modules '" + name + "': module '" + skipped.front() +
                        "' is unsupported for serving adapters. Merge the adapter into its base checkpoint "
                        "with `surogate merge`, then convert and serve the merged checkpoint.");
                }
                for (auto& payload : payloads) {
                    payload.slot = update.slot();
                    engine_options.lora_payloads.push_back(std::move(payload));
                }
                update.commit();
            }
        }
    }
    engine_options.load_progress            = std::move(load_progress);
    const bool lora_requested = engine_options.lora_enable;
    std::vector<std::pair<int, std::string>> startup_adapter_modules;
    for (const auto& payload : engine_options.lora_payloads) {
        startup_adapter_modules.emplace_back(payload.layer, payload.module);
    }
    engine_              = std::make_unique<sinfer::Engine>(std::move(engine_options));
    // Only a target that binds adapters to its own weights can apply them, and most
    // do not yet. Without this check a request naming an adapter would be answered
    // by the base model on those targets -- served confidently, and wrong. The
    // directory is registered during load, so an empty one here means nothing can
    // bind, at startup or from the runtime endpoints.
    if (lora_requested && !any_lora_bindings()) {
        throw std::invalid_argument(
            "--enable-lora: this target does not apply adapters, so one would be loaded and "
            "silently ignored. Merge it into the checkpoint before conversion (`surogate merge`) "
            "to serve it here.");
    }
    for (const auto& [layer, module] : startup_adapter_modules) {
        auto& stores = engine_->lora_stores();
        bool found = false;
        for (int device : stores.devices()) { found |= stores.peek(device)->covers_layer(layer); }
        if (!found) {
            throw std::invalid_argument("adapter module '" + module +
                "' is absent from the served model. Check that the adapter matches the base model and enable --vision "
                "for vision adapters. For unsupported serving modules, use `surogate merge`, then convert and serve the merged checkpoint.");
        }
    }
    prompt_capabilities_ = engine_->prompt_capabilities();
    request_capacity_    = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
}

std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime() const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (shared_training_) {
            throw_request_error(sinfer::RequestError(RequestErrorKind::Overloaded,
                                                     "rollouts are paused for a training step"));
        }
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(sinfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
        ++request_capacity_->awaiting_wake;
    }
    try {
        return std::make_shared<RequestLifetime>(
            request_capacity_, started,
            started + std::chrono::milliseconds(options_.pending_timeout_ms));
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        --request_capacity_->awaiting_wake;
        throw;
    }
}

std::shared_ptr<RequestLifetime> GenerationService::begin_request(
    const std::function<bool()>& is_cancelled, const PreparationGate& before_prepare) const {
    auto lifetime = acquire_request_lifetime();
    check_preparation_control(lifetime->deadline, is_cancelled);
    try {
        if (before_prepare) {
            before_prepare(PreparationControl{
                .deadline = lifetime->deadline, .cancellation = CancellationView(is_cancelled)});
        }
    } catch (const sinfer::RequestError& error) { throw_request_error(error); }
    check_preparation_control(lifetime->deadline, is_cancelled);
    lifetime->finish_wake();
    return lifetime;
}

PreparedRequest GenerationService::prepare(const GenerationRequest& original,
                                           std::function<bool()> is_cancelled,
                                           const PreparationGate& before_prepare) const {
    std::optional<GenerationRequest> classification_request;
    std::vector<ParallelField> parallel_fields;
    if (original.parallel_decoding) {
        classification_request = original;
        auto& request = *classification_request;
        validate_parallel_request(request);
        parallel_fields = parse_parallel_schema(request.json_schema);
        ChatTurn instruction;
        instruction.role = ChatRole::User;
        ContentPart text;
        text.text = "Classify the conversation above according to this JSON schema. "
                    "Evaluate each property independently using its description and allowed values. "
                    "Complete only the requested JSON property, without explanations. Schema: " + request.json_schema;
        instruction.content.push_back(std::move(text));
        request.messages.push_back(std::move(instruction));
    }
    const auto& request = classification_request ? *classification_request : original;
    validate_token_media(request);
    // A base model has no chat template, so there is no turn structure to render this into.
    // Refused here rather than in each handler: every chat-shaped endpoint arrives through
    // this one path, and /v1/completions is the shape that does fit.
    if (!request.raw_prompt.has_value() && !engine_->supports_chat()) {
        ApiError error;
        error.message = "this model publishes no chat template, which is what a base model "
                        "looks like; ask it through /v1/completions instead";
        error.code    = "chat_not_supported";
        throw ApiException(std::move(error));
    }
    PreparedRequest prepared;
    sinfer::RequestOptions request_options = to_request_options(request, options_);
    // A minimum length is honoured by barring the stop ids until it is reached, and
    // the round only knows numbers -- so resolve which ids those are here, where
    // the model's own stops and the request's are both in reach.
    request_options.execution.prompt_logprobs = request.prompt_logprobs;
    request_options.execution.top_logprobs = request.want_logprobs ? request.top_logprobs : -1;
    request_options.execution.min_tokens = static_cast<std::uint32_t>(request.min_tokens);
    if (request.min_tokens > 0) {
        std::vector<sinfer::TokenId> barrier = engine_->default_stop_tokens();
        for (const sinfer::TokenId id : request_options.stop.token_ids) { barrier.push_back(id); }
        std::sort(barrier.begin(), barrier.end());
        barrier.erase(std::unique(barrier.begin(), barrier.end()), barrier.end());
        if (barrier.size() > request_options.execution.stop_barrier.size()) {
            throw ApiException(ApiError{
                .status  = 400,
                .message = "min_tokens needs every stop token barred, and this model has more of "
                           "them than the sampler can bar at once",
                .param   = "min_tokens"});
        }
        request_options.execution.stop_barrier_count = static_cast<std::uint32_t>(barrier.size());
        for (std::size_t i = 0; i < barrier.size(); ++i) {
            request_options.execution.stop_barrier[i] = barrier[i];
        }
    }
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
    prepared.tool_capable                  = request.uses_tools();
    prepared.tools                         = request.tools;
    prepared.tool_name_max_length          = request.tool_name_max_length;
    ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    if (request.parallel_decoding) { semantics.enable_thinking = false; }
    request_options.execution.structural_tag = make_tool_constraint(request, options_.tool_call_format, semantics.enable_thinking);
    prepared.constrained_tools = !request_options.execution.structural_tag.empty();
    prepared.tool_choice = request.tool_choice;
    prepared.parallel_tool_calls = request.parallel_tool_calls;
    prepared.enable_thinking                   = semantics.enable_thinking;
    prepared.preserve_thinking                 = semantics.preserve_thinking;
    const bool request_has_media               = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = begin_request(is_cancelled, before_prepare);

    try {
        // Pin before preparation/submission, closing the race with replacement
        // and unload. The engine owns this lease through completion or cancellation.
        auto adapter = lora_slots_.acquire(request.lora_adapter, prepared.lifetime->deadline, is_cancelled);
        request_options.execution.lora_slot = adapter.slot;
        const auto acquisition_started = Clock::now();
        // Explicit tokens and raw text do not need structured message preparation.
        std::optional<sinfer::PromptInput> input;
        if (request.prompt_token_ids.empty() && !request.raw_prompt.has_value()) {
            std::size_t remaining_media_bytes =
                std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
            input = to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        }
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        // Ids given by the client win over anything the template would render: that
        // is the whole point of the token-in endpoint, and the messages are still
        // parsed above for the tools and the parser state they carry.
        sinfer::PreparedPrompt prompt;
        if (!request.prompt_token_ids.empty()) {
            try {
                prompt = engine_->prepare_tokens(request.prompt_token_ids);
            } catch (const std::out_of_range& error) {
                throw ApiException(ApiError{.status = 400, .message = error.what(), .param = "tokens"});
            }
        } else if (request.raw_prompt.has_value()) {
            prompt = engine_->prepare_text(*request.raw_prompt);
        } else {
            prompt = engine_->prepare(std::move(*input), control);
        }
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.prompt_logprobs = request.prompt_logprobs;
        prepared.top_logprobs = request.top_logprobs;
        prepared.want_logprobs    = request.want_logprobs;
        prepared.return_token_ids = request.return_token_ids;
        // Read before the submit takes the prompt: these are the ids the model
        // actually sees, which is the thing a trainer must score against and is not
        // reliably reproducible by re-rendering the messages.
        if (request.return_token_ids) { prepared.prompt_token_ids = prompt.token_ids(); }
        if (request.parallel_decoding) {
            auto state = std::make_shared<ParallelDecodingRequest>();
            state->plan = compile_parallel_plan(std::move(parallel_fields),
                [&](std::string_view text) { return engine_->encode_fragment(text); });
            state->prefix = prompt.token_ids();
            for (const auto& query : state->plan.queries) {
                if (state->prefix.size() + query.suffix.size() + 1 > engine_->options().max_context) {
                    throw ApiException({.message="parallel_decoding field exceeds the model context limit", .param="response_format"});
                }
            }
            state->max_tokens = request.max_tokens > 0 ? request.max_tokens : options_.default_max_tokens;
            // Reserve a conservative bound before doing GPU work: every possible
            // assembled result fits the client's requested completion budget.
            std::size_t bound = engine_->encode_fragment("{}").size();
            for (const auto& field : state->plan.fields) {
                std::size_t longest = 0;
                for (const auto& value : field.values)
                    longest = std::max(longest, engine_->encode_fragment(
                        nlohmann::json(field.name).dump() + ":" + value.dump() + ",").size());
                bound += longest;
            }
            if (bound > state->max_tokens) {
                throw ApiException({.message="max_tokens is too small for all parallel_decoding choices (need at least " +
                    std::to_string(bound) + ")", .param="max_tokens"});
            }
            state->adapter = adapter.lifetime;
            request_options.execution.json_schema.clear();
            request_options.execution.requested_output_tokens = 1;
            request_options.execution.cache_prompt = false;
            std::vector<bool> parents(state->plan.queries.size());
            for (const auto& query : state->plan.queries) if (query.parent >= 0) parents[query.parent] = true;
            const auto state_slots = 1U + static_cast<std::uint32_t>(std::count(parents.begin(), parents.end(), true));
            request_options.execution.save_gpu_prefix = std::make_shared<GpuPrefixKey>(GpuPrefixKey{state_slots});
            request_options.execution.allow_prefix_reuse = false;
            request_options.output.structured = true;
            auto& sampling = request_options.execution.sampling;
            sampling.top_k = 0;
            sampling.top_p = 1;
            sampling.min_p = 0;
            sampling.presence_penalty = 0;
            sampling.frequency_penalty = 0;
            sampling.repetition_penalty = 1;
            sampling.logit_bias.clear();
            // As a decision does (#14): the queue places the warm step and the readout waves will
            // need are taken before any GPU work, so the readouts are never refused after the
            // warm prefix has run.
            const auto places = static_cast<std::uint32_t>(std::max<std::size_t>(
                1, std::min(state->plan.queries.size(), candidate_wave_width())));
            request_options.execution.reservation = engine_->reserve_submissions(
                places, prepared.lifetime->deadline, CancellationView(is_cancelled));
            state->options = request_options;
            state->options.execution.gpu_prefix = request_options.execution.save_gpu_prefix;
            state->options.execution.save_gpu_prefix.reset();
            state->options.execution.allow_prefix_reuse = true;
            prepared.parallel_decoding = std::move(state);
        }
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              prepared.lifetime->deadline, std::move(adapter.lifetime));
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const sinfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled,
                                           const PreparationGate& before_prepare) const {
    validate_token_media(request);
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const auto lifetime = begin_request(is_cancelled, before_prepare);
    const auto deadline = lifetime->deadline;
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
    if (prepared.parallel_decoding) { return run_parallel(prepared, sink, is_cancelled); }
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) {
        output_sink = std::make_unique<ServiceOutputSink>(*engine_, *sink, prepared.tool_capable,
            options_.tool_call_format == ToolCallFormat::Llama3Json,
            options_.tool_call_format == ToolCallFormat::MuseAtem);
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
    outcome.stop_sequence     = std::move(result.stop_sequence);
    if (prepared.return_token_ids) {
        outcome.prompt_token_ids     = std::move(prepared.prompt_token_ids);
        outcome.completion_token_ids = result.generated_token_ids;
    }
    outcome.prompt_scores = std::move(result.prompt_logprobs);
    outcome.completion_scores = std::move(result.completion_logprobs);
    populate_score_texts(*engine_, outcome);

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
    outcome.metrics.speculative_rounds_per_draft_window =
        std::move(result.speculative.rounds_per_draft_window);
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    if (!prepared.tool_capable) { outcome.text += result.tool_content; }
    bool is_tool_call_response = false;
    if (prepared.tool_capable) {
        ParsedToolCalls parsed = parse_tool_calls(options_.tool_call_format,
            options_.tool_call_format == ToolCallFormat::MuseAtem ? result.tool_content : outcome.text,
                                                  prepared.tool_name_max_length, prepared.tools);
        if (options_.tool_call_format == ToolCallFormat::MuseAtem) { parsed.content = outcome.text; }
        if (!prepared.parallel_tool_calls && parsed.tool_calls.size() > 1) { parsed.tool_calls.resize(1); }
        if ((outcome.finish_reason != sinfer::FinishReason::StopToken &&
             outcome.finish_reason != sinfer::FinishReason::StopString) ||
            std::any_of(parsed.tool_calls.begin(), parsed.tool_calls.end(), [&](const ToolCall& call) {
                return std::none_of(prepared.tools.begin(), prepared.tools.end(),
                                    [&](const ToolDefinition& tool) { return tool.name == call.name; });
            })) {
            parsed = {};
            parsed.content = outcome.text;
        }
        if (prepared.constrained_tools && outcome.finish_reason == sinfer::FinishReason::StopToken) {
            for (const auto& call : parsed.tool_calls) {
                const auto tool = std::find_if(prepared.tools.begin(), prepared.tools.end(), [&](const auto& item) { return item.name == call.name; });
                if (tool == prepared.tools.end() || !tool_arguments_match_schema(*tool, call.arguments_json)) {
                    throw ApiException({.status=500, .type="server_error", .message="generated arguments failed validation for tool " + call.name, .code="tool_output_invalid"});
                }
            }
            const bool required = prepared.tool_choice.mode == ToolChoiceMode::Required || prepared.tool_choice.mode == ToolChoiceMode::Named;
            if ((required && parsed.tool_calls.empty()) ||
                (prepared.tool_choice.mode == ToolChoiceMode::Named &&
                 (parsed.tool_calls.size() != 1 || parsed.tool_calls.front().name != prepared.tool_choice.name))) {
                throw ApiException({.status=500, .type="server_error", .message="generated tool response violated the requested tool choice", .code="tool_output_invalid"});
            }
        }
        outcome.text           = std::move(parsed.content);
        is_tool_call_response  = parsed.is_tool_call_response;
        if (is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
    }
    // --reasoning-parser reconciles what the frontend already split. `none` folds
    // the span back into the answer rather than dropping it.
    finalize_output_text(outcome, options_.reasoning_format,
        output_sink ? std::optional(output_sink->finish(is_tool_call_response)) : std::nullopt);
    return outcome;
}

GenerationOutcome GenerationService::run_parallel(PreparedRequest& prepared, const StreamSink* sink,
    const std::function<bool()>& external_cancelled) {
    auto state = std::move(prepared.parallel_decoding);
    const auto cancelled = [&] {
        return (external_cancelled && external_cancelled()) ||
               (sink && sink->is_cancelled && sink->is_cancelled());
    };
    const auto deadline = prepared.lifetime->deadline;
    const CancellationView cancellation([&] { return cancelled() || Clock::now() >= deadline; });
    const auto check = [&] { check_preparation_control(deadline, cancelled); };
    GenerationOutcome outcome;
    try {
        check();
        const auto warm = prepared.generation.wait(nullptr, cancellation);
        check();
        if (warm.finish_reason == FinishReason::Cancelled)
            throw RequestError(RequestErrorKind::Cancelled, "parallel decoding was cancelled");
        outcome.prompt_tokens = prepared.prompt_tokens;
        outcome.metrics.prefix_cache_hit_tokens = warm.reused_prompt_tokens;
        outcome.metrics.prefix_reuse_path = warm.prefix_reuse_path;
        outcome.metrics.prefill_seconds = warm.timings.prefill_seconds;
        state->options.execution.sampling.temperature = prepared.sampling.temperature;
        CandidateReadout readout = read_candidates(state->plan.queries,
            [&](std::size_t i) {
                auto tokens = state->prefix;
                const auto& suffix = state->plan.queries[i].suffix;
                tokens.insert(tokens.end(), suffix.begin(), suffix.end());
                return engine_->prepare_tokens(std::move(tokens));
            },
            state->options, state->adapter, deadline, cancellation, check, "parallel decoding was cancelled");
        outcome.prompt_tokens += readout.suffix_tokens;
        outcome.metrics.prefill_seconds += readout.prefill_seconds;
        check();
        auto classified = resolve_parallel_plan(state->plan, readout.logits, prepared.sampling.temperature);
        outcome.text = classified.content.dump();
        outcome.completion_tokens = static_cast<int>(engine_->encode_fragment(outcome.text).size());
        if (std::size_t(outcome.completion_tokens) > state->max_tokens)
            throw ApiException({.message="assembled parallel_decoding result exceeds max_tokens", .param="max_tokens"});
        outcome.parallel_decoding_details = nlohmann::json{
            {"fields", classified.fields}, {"temperature", prepared.sampling.temperature}}.dump();
        outcome.finish_reason = FinishReason::StopToken;
        outcome.metrics.prepare_seconds = prepared.prepare_seconds;
        outcome.metrics.total_seconds = std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        outcome.metrics.ttft_seconds = outcome.metrics.total_seconds;
        if (sink && sink->on_content) sink->on_content(outcome.text);
        return outcome;
    } catch (const RequestError& exception) { throw_request_error(exception); }
}

std::size_t GenerationService::candidate_wave_width() const {
    return std::max<std::size_t>(1, std::min<std::size_t>(64, options_.max_concurrency));
}

GenerationService::CandidateReadout GenerationService::read_candidates(
    const std::vector<ParallelQuery>& queries,
    const std::function<sinfer::PreparedPrompt(std::size_t)>& make_prompt,
    const sinfer::RequestOptions& options, const std::shared_ptr<void>& adapter,
    Clock::time_point deadline, const sinfer::CancellationView& cancellation,
    const std::function<void()>& check, const char* cancelled_message) {
    CandidateReadout readout;
    readout.logits.resize(queries.size());
    // Keep work bounded and share the normal scheduler with other requests.
    // A one-lane engine runs the same finite-choice computation serially.
    const auto width = candidate_wave_width();
    std::vector<std::vector<std::size_t>> children(queries.size());
    std::vector<std::size_t> remaining(queries.size());
    std::vector<std::shared_ptr<const GpuPrefixKey>> keys(queries.size());
    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < queries.size(); ++i) {
        const int parent = queries[i].parent;
        if (parent < 0) ready.push_back(i);
        else { children[parent].push_back(i); ++remaining[parent]; }
    }
    for (std::size_t start = 0; start < ready.size();) {
        const auto end = std::min(start + width, ready.size());
        std::vector<PreparedPrompt> prompts;
        std::vector<RequestOptions> batch;
        for (std::size_t j = start; j < end; ++j) {
            check();
            const auto i = ready[j];
            const auto& query = queries[i];
            auto opt = options;
            opt.execution.next_token_candidates = query.candidates;
            if (query.parent >= 0) opt.execution.gpu_prefix = keys[query.parent];
            if (!children[i].empty()) {
                keys[i] = std::make_shared<GpuPrefixKey>();
                opt.execution.save_gpu_prefix = keys[i];
            }
            prompts.push_back(make_prompt(i));
            batch.push_back(std::move(opt));
            const auto parent_suffix = query.parent >= 0 ? queries[query.parent].suffix.size() : 0;
            readout.suffix_tokens += static_cast<int>(query.suffix.size() - parent_suffix);
        }
        const auto batch_started = Clock::now();
        auto pending = engine_->submit_batch(std::move(prompts), std::move(batch), deadline, adapter);
        for (std::size_t j = start; j < end; ++j) {
            const auto i = ready[j];
            const auto result = pending[j - start].wait(nullptr, cancellation);
            check();
            if (result.finish_reason == FinishReason::Cancelled)
                throw RequestError(RequestErrorKind::Cancelled, cancelled_message);
            readout.logits[i] = result.next_token_logits;
            ready.insert(ready.end(), children[i].begin(), children[i].end());
            const auto parent = queries[i].parent;
            if (parent >= 0 && --remaining[parent] == 0) keys[parent].reset();
        }
        // Concurrent rows report overlapping time. Count the wave once.
        readout.prefill_seconds += std::chrono::duration<double>(Clock::now() - batch_started).count();
        start = end;
    }
    return readout;
}

const std::vector<std::string>& GenerationService::decision_codes() {
    std::lock_guard lock(decision_codes_mutex_);
    if (!decision_codes_) {
        decision_codes_ = decision_codebook(
            [&](std::string_view text) { return engine_->encode_fragment(text); },
            [&](TokenId id) {
                const std::array<TokenId, 1> ids{id};
                return engine_->token_texts(std::span<const TokenId>(ids.data(), ids.size())).front();
            });
    }
    return *decision_codes_;
}

DecisionsOutcome GenerationService::decide(const DecisionsRequest& request,
                                           std::function<bool()> is_cancelled,
                                           const PreparationGate& before_prepare,
                                           const std::function<void(const std::string&)>& on_retry) {
    if (!engine_->supports_chat()) {
        ApiError error;
        error.message = "this model publishes no chat template, which is what a base model "
                        "looks like; decisions need one";
        error.code    = "chat_not_supported";
        throw ApiException(std::move(error));
    }
    if (!request.images.empty() && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    // Thinking off, spelled the way the chat endpoint spells it, so a template that cannot
    // be told is refused by the same rule. Every question is a fresh two-turn chat.
    GenerationRequest shape;
    shape.model           = request.model;
    shape.lora_adapter    = request.lora_adapter;
    shape.enable_thinking = false;
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(shape, options_, prompt_capabilities_);
    DecisionsOutcome outcome;
    const auto lifetime = begin_request(is_cancelled, before_prepare);
    const auto deadline = lifetime->deadline;
    const auto cancelled = [&] { return is_cancelled && is_cancelled(); };
    const CancellationView cancellation([&] { return cancelled() || Clock::now() >= deadline; });
    const auto check = [&] { check_preparation_control(deadline, is_cancelled); };
    try {
        auto adapter = lora_slots_.acquire(request.lora_adapter, deadline, is_cancelled);
        const PreparationControl control{
            .deadline = deadline, .cancellation = CancellationView(is_cancelled)};
        // Media are acquired once and handed to every question's prompt in order.
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
        std::vector<sinfer::OwnedMedia> media;
        for (const ContentPart& part : request.images) {
            media.push_back(acquire_media(part, deadline, is_cancelled, remaining_media_bytes));
        }
        check();
        const auto prepare_chat = [&](std::string_view system, std::string user_text) {
            GenerationRequest turns = shape;
            ChatTurn system_turn;
            system_turn.role = ChatRole::System;
            system_turn.content.push_back(ContentPart{ContentKind::Text, std::string(system), "text"});
            ChatTurn user_turn;
            user_turn.role = ChatRole::User;
            for (const ContentPart& part : request.images) { user_turn.content.push_back(part); }
            user_turn.content.push_back(ContentPart{ContentKind::Text, std::move(user_text), "text"});
            turns.messages = {std::move(system_turn), std::move(user_turn)};
            std::size_t next_media = 0;
            sinfer::PromptInput input = to_prompt_input(
                turns, semantics, [&](const ContentPart&) { return media.at(next_media++); });
            return engine_->prepare(std::move(input), control);
        };
        // The rendered text is read back from the ids (special tokens decode to their own
        // text, so this is the template's output); re-encoding it must give the ids back,
        // which is what makes "this text plus a label" the in-context tokenisation.
        const auto decode = [&](const std::vector<TokenId>& ids) {
            std::string text;
            for (const std::string& piece : engine_->token_texts(std::span<const TokenId>(ids))) {
                text += piece;
            }
            return text;
        };
        const auto encode = [&](std::string_view text) {
            return engine_->prepare_text(text, false).token_ids();
        };

        // One rendering of the state per system prompt, split at the question boundary
        // exactly as the reference does: everything before it is that variant's prefix.
        struct Variant {
            bool ready = false;
            std::string before;
            std::string after;
            std::vector<TokenId> prefix;
        };
        std::array<Variant, 2> variants;
        const auto variant_for = [&](bool extended) -> const Variant& {
            Variant& variant = variants[extended ? 1 : 0];
            if (variant.ready) { return variant; }
            const auto marked = prepare_chat(
                extended ? kDecisionExtendedSystemPrompt : kDecisionSystemPrompt,
                request.state_text + decision_boundary_marker());
            const std::string text    = decode(marked.token_ids());
            const std::string& marker = decision_boundary_marker();
            const auto at             = text.find(marker);
            if (at == std::string::npos || text.find(marker, at + marker.size()) != std::string::npos) {
                refuse("the chat template did not preserve the question boundary", "state",
                       "decisions_template_boundary");
            }
            variant.before = text.substr(0, at);
            variant.after  = text.substr(at + marker.size());
            variant.prefix = encode(variant.before);
            variant.ready  = true;
            return variant;
        };
        const bool any_extended = std::any_of(request.questions.begin(), request.questions.end(),
            [](const DecisionQuestion& question) { return question.extended(); });
        static const std::vector<std::string> no_codes;
        const std::vector<std::string>& codebook = any_extended ? decision_codes() : no_codes;

        struct Item {
            std::vector<TokenId> ids;
            std::vector<TokenId> candidates;
            const Variant* variant = nullptr;
            sinfer::PreparedPrompt prompt;
            // The rendering the prompt was prepared from: an attempt after the first prepares
            // it again, since the first attempt's engine consumed the prompt.
            std::string system;
            std::string user;
        };
        const bool with_images = !request.images.empty();
        const std::size_t max_context = engine_->options().max_context;
        std::vector<Item> items;
        items.reserve(request.questions.size());
        for (const DecisionQuestion& question : request.questions) {
            check();
            const RenderedDecisionQuestion rendered = render_decision_question(question, codebook);
            const Variant& variant = variant_for(rendered.extended);
            sinfer::PreparedPrompt prompt = prepare_chat(rendered.system, request.state_text + rendered.branch);
            Item item;
            item.ids     = prompt.token_ids();
            item.variant = &variant;
            item.system  = std::string(rendered.system);
            item.user    = request.state_text + rendered.branch;
            if (item.ids.size() + 1 > max_context) {
                throw ApiException(ApiError{.status = 400,
                    .message = "question '" + question.name + "' renders to " + std::to_string(item.ids.size()) +
                               " tokens; with its answer that exceeds the model context of " + std::to_string(max_context),
                    .param = "questions", .code = "context_length_exceeded"});
            }
            const std::string text = decode(item.ids);
            if (encode(text) != item.ids) {
                refuse("question '" + question.name + "' renders to text that does not re-tokenise to the same ids, "
                       "so its option labels cannot be verified in context", "questions",
                       "decisions_tokenizer_roundtrip");
            }
            if (text != variant.before + rendered.branch + variant.after) {
                refuse("the chat template did not render question '" + question.name +
                       "' as a continuation of the shared state", "questions", "decisions_template_boundary");
            }
            std::set<TokenId> seen;
            for (const std::string& label : rendered.labels) {
                const std::vector<TokenId> appended = encode(text + label);
                if (appended.size() != item.ids.size() + 1 ||
                    !std::equal(item.ids.begin(), item.ids.end(), appended.begin())) {
                    refuse("label '" + label + "' of question '" + question.name +
                           "' is not a single continuation token for this template", "questions",
                           "decisions_label_tokenization");
                }
                if (!seen.insert(appended.back()).second) {
                    refuse("option labels of question '" + question.name + "' share a token", "questions",
                           "decisions_label_tokenization");
                }
                item.candidates.push_back(appended.back());
            }
            item.prompt = std::move(prompt);
            items.push_back(std::move(item));
        }
        outcome.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - lifetime->started).count();

        // The queue places the decision will need are taken before any GPU work (#14): as many as
        // it ever has in the queue at once -- its shared prefix alone, then its questions in waves
        // (read_candidates) -- and each finished submission hands its place to the next. A busy
        // queue makes the decision wait for them here, until its deadline (then 503), rather than
        // refuse its questions after its shared prefix has run on the GPU. Only the client going
        // away cuts the wait short; the deadline is reserve's own.
        const auto places = static_cast<std::uint32_t>(std::max<std::size_t>(
            1, std::min(request.questions.size(), candidate_wave_width())));
        const auto reservation = in_engine([&] {
            return engine_->reserve_submissions(places, deadline, CancellationView(is_cancelled));
        });

        // The readout ignores sampling; keep it neutral anyway so nothing else is asked of
        // the round than the candidate logits.
        sinfer::RequestOptions base;
        base.execution.requested_output_tokens = 1;
        base.execution.cache_prompt            = false;
        base.execution.allow_prefix_reuse      = false;
        base.execution.lora_slot               = adapter.slot;
        base.execution.sampling.temperature    = 1.0F;
        base.execution.sampling.top_k          = 0;
        base.execution.sampling.top_p          = 1.0F;
        base.execution.sampling.min_p          = 0.0F;
        base.execution.sampling.presence_penalty   = 0.0F;
        base.execution.sampling.frequency_penalty  = 0.0F;
        base.execution.sampling.repetition_penalty = 1.0F;
        base.output.structured                 = true;
        base.execution.reservation             = reservation;

        std::vector<ParallelQuery> queries;
        queries.reserve(items.size());
        std::vector<TokenId> prefix;
        std::size_t shared = 0;
        // The runtime keeps saved GPU prefix states for text prompts only, so an image request
        // shares nothing on the GPU; its questions run as their own prompts below.
        if (!with_images) {
            std::vector<std::vector<TokenId>> prefixes;
            std::vector<std::vector<TokenId>> full;
            for (const Item& item : items) {
                prefixes.push_back(item.variant->prefix);
                full.push_back(item.ids);
            }
            shared = decision_shared_prefix(prefixes, full);
            if (shared == 0) {
                refuse("the chat template yielded an empty shared prefix", "questions", "decisions_shared_prefix");
            }
            // The prefill floor (kDecisionMinPrefillTokens): give tokens back until each suffix
            // is at least that long, and share nothing when the prefix itself would be shorter.
            std::vector<std::size_t> lengths;
            for (const Item& item : items) { lengths.push_back(item.ids.size()); }
            shared = decision_shared_prefix_floor(shared, lengths);
            // One question has nothing to share with; the warm step would only cost.
            if (items.size() == 1) { shared = 0; }
        }
        if (shared == 0) {
            // Each question is the full prompt a plain request would be, prefilled whole, with
            // the engine's prefix cache off so no earlier prompt can shorten the step.
            for (Item& item : items) {
                queries.push_back(ParallelQuery{.parent = -1, .suffix = item.ids, .candidates = item.candidates});
            }
        } else {
            prefix.assign(items.front().ids.begin(), items.front().ids.begin() + static_cast<std::ptrdiff_t>(shared));
            for (Item& item : items) {
                queries.push_back(ParallelQuery{.parent = -1,
                    .suffix = std::vector<TokenId>(item.ids.begin() + static_cast<std::ptrdiff_t>(shared), item.ids.end()),
                    .candidates = item.candidates});
            }
        }

        // One attempt runs the whole request on the GPU: the shared prefix (when there is one)
        // is prefilled and saved under a key of its own, and every question is read on it. A
        // model can return non-finite option logits for a row that is finite on another run
        // (2026-09-25: about one request per GPU-hour on RTX PRO 6000 servers, each fine when
        // sent again), so a failed attempt is run again, up to --decision-attempts in all.
        // Nothing of a failed attempt is reused: its prefix key is dropped with the attempt,
        // which releases the saved state and its KV pages before the next warm step is
        // planned, and the prompts are prepared again. The questions of every attempt are
        // prefilled with the prefix cache off, as the first attempt's are.
        const std::uint32_t attempts = std::max<std::uint32_t>(1, options_.decision_attempts);
        CandidateReadout readout;
        std::uint32_t attempt = 1;
        for (;; ++attempt) {
            sinfer::RequestOptions attempt_base = base;
            std::shared_ptr<GpuPrefixKey> key;
            std::function<sinfer::PreparedPrompt(std::size_t)> make_prompt;
            if (shared == 0) {
                make_prompt = [&](std::size_t i) {
                    if (attempt == 1) { return std::move(items[i].prompt); }
                    sinfer::PreparedPrompt prompt = prepare_chat(items[i].system, items[i].user);
                    if (prompt.token_ids() != items[i].ids) {
                        throw std::logic_error("a decisions question prepared again rendered to other tokens");
                    }
                    return prompt;
                };
            } else {
                // Prefill the shared prefix once and keep its GPU state; every question then
                // extends that state with its own suffix, exactly as parallel decoding does.
                key = std::make_shared<GpuPrefixKey>();
                sinfer::RequestOptions warm_options    = base;
                warm_options.execution.save_gpu_prefix = key;
                check();
                auto warm_prompt = engine_->prepare_tokens(prefix);
                auto warm = in_engine([&] {
                    return engine_->submit(std::move(warm_prompt), std::move(warm_options), deadline,
                                           adapter.lifetime);
                });
                const auto warm_result = in_engine([&] { return warm.wait(nullptr, cancellation); });
                check();
                if (warm_result.finish_reason == FinishReason::Cancelled) {
                    throw RequestError(RequestErrorKind::Cancelled, "decisions were cancelled");
                }
                outcome.prefill_seconds += warm_result.timings.prefill_seconds;
                attempt_base.execution.gpu_prefix         = key;
                attempt_base.execution.allow_prefix_reuse = true;
                make_prompt = [&](std::size_t i) {
                    auto tokens = prefix;
                    tokens.insert(tokens.end(), queries[i].suffix.begin(), queries[i].suffix.end());
                    return engine_->prepare_tokens(std::move(tokens));
                };
            }
            // `make_prompt` runs inside this: the tokens it hands the engine are this endpoint's
            // own rendering, not the caller's, so a refusal of them is a service fault too.
            readout = in_engine([&] {
                return read_candidates(queries, make_prompt, attempt_base, adapter.lifetime, deadline,
                                       cancellation, check, "decisions were cancelled");
            });
            check();
            outcome.prefill_seconds += readout.prefill_seconds;
            std::vector<std::size_t> nonfinite;
            for (std::size_t i = 0; i < readout.logits.size(); ++i) {
                if (std::any_of(readout.logits[i].begin(), readout.logits[i].end(),
                                [](float value) { return !std::isfinite(value); })) {
                    nonfinite.push_back(i);
                }
            }
            if (nonfinite.empty() || attempt >= attempts) { break; }
            if (on_retry) {
                std::string names;
                for (std::size_t k = 0; k < nonfinite.size() && k < 8; ++k) {
                    names += (k == 0 ? "" : ",") + request.questions[nonfinite[k]].name;
                }
                if (nonfinite.size() > 8) { names += ",..."; }
                on_retry("decisions attempt " + std::to_string(attempt) + " of " + std::to_string(attempts) +
                         " returned non-finite logits for " + std::to_string(nonfinite.size()) + " of " +
                         std::to_string(readout.logits.size()) + " questions (" + names +
                         "); running the request again from scratch");
            }
            // The attempt's key, options and prompts go out of scope here, before the next
            // attempt submits anything.
        }
        outcome.attempts             = attempt;
        outcome.shared_prefix_tokens = shared;
        outcome.input_tokens  = static_cast<int>(outcome.shared_prefix_tokens) + readout.suffix_tokens;
        outcome.output_tokens = static_cast<int>(request.questions.size());
        // Both routes above -- whole prompts, or suffixes on the shared GPU prefix -- and both
        // label schemes -- letters, or codebook codes past 26 options -- end in the same raw
        // candidate logits, one row per question. The server's calibration temperature is
        // applied here, once per row, and nowhere else.
        outcome.answers =
            resolve_decision_answers(request, readout.logits, options_.decision_temperature);
        outcome.total_seconds =
            std::chrono::duration<double>(Clock::now() - lifetime->started).count();
        return outcome;
    } catch (const ApiException&) { throw; } catch (const sinfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        // Preparation only: everything the engine answers with goes through `in_engine` above,
        // which has already classified it. Media acquisition raises its own ApiException. What
        // is left is the chat template or the tokenizer refusing what the caller sent, which is
        // theirs -- so this keeps the 400 it has always had.
        throw_decisions_fault(
            classify_decisions_fault(exception, DecisionsFaultStage::Preparation));
    }
}

std::vector<sinfer::TokenId> GenerationService::tokenize(const GenerationRequest& request,
    std::function<bool()> is_cancelled, const PreparationGate& before_prepare) {
    validate_token_media(request);
    const auto lifetime = begin_request(is_cancelled, before_prepare);
    try {
        std::vector<sinfer::TokenId> ids;
        if (!request.prompt_token_ids.empty()) {
            ids = request.prompt_token_ids;
        } else if (request.raw_prompt.has_value()) {
            ids = engine_->prepare_text(*request.raw_prompt).token_ids();
        } else {
            const auto semantics = resolve_prompt_semantics(request, options_, prompt_capabilities_);
            std::size_t remaining_media_bytes =
                std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
            auto input = to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, lifetime->deadline, is_cancelled, remaining_media_bytes);
            });
            ids = engine_->prepare(std::move(input), PreparationControl{
                .deadline = lifetime->deadline, .cancellation = CancellationView(is_cancelled)}).token_ids();
        }
        check_preparation_control(lifetime->deadline, is_cancelled);
        return ids;
    } catch (const ApiException&) { throw; } catch (const sinfer::RequestError& error) {
        throw_request_error(error);
    } catch (const std::invalid_argument& error) { throw_invalid_input(error); }
}

std::vector<std::string> GenerationService::token_texts(const std::vector<sinfer::TokenId>& ids) {
    if (ids.empty()) { return {}; }
    return engine_->token_texts(std::span<const sinfer::TokenId>(ids.data(), ids.size()));
}

std::uint32_t GenerationService::max_context() const {
    // The configured ceiling, not `memory_summary()`: that call takes the engine's
    // execution lock, and a tokenize request must not queue behind a round.
    return engine_->options().max_context;
}

bool GenerationService::any_lora_bindings() const {
    ops::LoraStoreSet& stores = engine_->lora_stores();
    for (int device : stores.devices()) {
        const ops::LoraStore* store = stores.peek(device);
        if (store != nullptr && store->has_bindings()) { return true; }
    }
    return false;
}

void GenerationService::clear_lora_slot_everywhere(std::int32_t slot) {
    ops::LoraStoreSet& stores = engine_->lora_stores();
    for (int device : stores.devices()) {
        if (ops::LoraStore* store = stores.peek(device); store != nullptr) {
            store->clear_slot(slot);
        }
    }
}

void GenerationService::load_lora_adapter(const std::string& name, const std::string& path) {
    if (!options_.borrowed_weights.empty()) {
        throw std::invalid_argument("shared GRPO adapters are published by the trainer");
    }
    if (!options_.enable_lora) {
        throw std::invalid_argument("the server was started without --enable-lora");
    }
    // Every upload below writes this engine's device memory from the caller's
    // thread, which is an HTTP handler and has bound no device of its own. Without
    // this the copies and the scrub resolve against whatever device that thread
    // was on, and an engine on any device but the first dies on the first load.
    const ScopedDevice on_engine_device(engine_->device());
    ops::LoraStoreSet& stores      = engine_->lora_stores();
    const std::vector<int> devices = stores.devices();
    if (!any_lora_bindings()) {
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
                                    "' is unsupported for serving adapters. Merge the adapter into its base checkpoint "
                        "with `surogate merge`, then convert and serve the merged checkpoint.");
    }

    // Validate every module before draining requests or changing a live adapter.
    // An incompatible replacement leaves the current policy untouched.
    for (const auto& payload : payloads) {
        bool applicable = false;
        for (int device : devices) {
            const ops::LoraStore* store = stores.peek(device);
            if (store == nullptr || !store->covers_layer(payload.layer)) { continue; }
            store->validate_module(payload.layer, payload.module, payload.a, payload.b,
                                   payload.rank, payload.in_dim, payload.out_dim, payload.scale);
            applicable = true;
        }
        if (!applicable) {
            throw std::invalid_argument("adapter '" + name + "': module '" + payload.module +
                "' names a layer absent from this model. Check that the adapter matches the base model and enable --vision "
                "for vision adapters. For unsupported serving modules, use `surogate merge`, then convert and serve the merged checkpoint.");
        }
    }
    {
        // DoRA normalization reads base weights; saved full matrices join the sleep estate.
        // Keep both mapped while staging, without holding this lock during request draining.
        std::lock_guard memory_lock(adapter_memory_mutex_);
        if (engine_->is_sleeping()) {
            throw std::invalid_argument("the model is asleep; wake it before loading an adapter");
        }
        for (int device : devices) {
            if (const auto* store = stores.peek(device)) { store->validate_payloads(payloads); }
        }
    }
    auto update = lora_slots_.update(name, Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms));
    const auto slot = update.slot();
    std::lock_guard memory_lock(adapter_memory_mutex_);
    if (engine_->is_sleeping()) {
        throw std::invalid_argument("the model is asleep; wake it before loading an adapter");
    }
    try {
        clear_lora_slot_everywhere(slot);
        for (const auto& payload : payloads) {
            for (int device : devices) {
                ops::LoraStore* store = stores.peek(device);
                if (store == nullptr || !store->covers_layer(payload.layer)) { continue; }
                store->set_payload(slot, payload);
            }
        }
        for (int device : devices) {
            if (ops::LoraStore* store = stores.peek(device)) { store->set_active(true); }
        }
        // A recycled slot can have retained prefixes even when its old adapter
        // name differs. Invalidate before reopening admission for the new bytes.
        engine_->shrink_kv();
    } catch (...) {
        // Device failures cannot expose partially uploaded weights. All users of
        // the old slot have finished, so removing it cannot change their policy.
        update.commit(true);
        throw;
    }
    update.commit();
}

void GenerationService::unload_lora_adapter(const std::string& name) {
    if (!options_.borrowed_weights.empty()) {
        throw std::invalid_argument("shared GRPO adapters are owned by the trainer");
    }
    auto update = lora_slots_.update(name, Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms), true);
    std::lock_guard memory_lock(adapter_memory_mutex_);
    if (engine_->is_sleeping()) {
        throw std::invalid_argument("the model is asleep; wake it before unloading an adapter");
    }
    try {
        clear_lora_slot_everywhere(update.slot());
        engine_->shrink_kv();
    } catch (...) {
        update.commit(true);
        throw;
    }
    update.commit(true);
}

void GenerationService::sleep(bool preempt) {
    if (!options_.enable_sleep_mode) {
        throw std::invalid_argument("the server was started without --enable-sleep-mode");
    }
    // Refusing new submissions first turns the drain below into a bounded wait:
    // the engine rejects anything that arrives after this line.
    // (Engine::sleep is idempotent, so two racing sleeps are both fine.)
    {
        std::lock_guard lock(adapter_memory_mutex_);
        engine_->sleep_begin();
    }
    static const bool env_preempt = std::getenv("SUROGATE_SLEEP_PREEMPT") != nullptr;
    if (preempt || env_preempt) {
        // Preemptive: in-flight generations park mid-flight and resume,
        // byte-identical, after the next wake.
        std::lock_guard lock(adapter_memory_mutex_);
        engine_->sleep(/*allow_active=*/true);
        return;
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (true) {
        {
            const std::lock_guard<std::mutex> lock(request_capacity_->mutex);
            // Wake waiters have not touched the engine yet and need the scheduler
            // worker to finish this eviction before they can proceed.
            if (request_capacity_->active == request_capacity_->awaiting_wake) { break; }
        }
        if (std::chrono::steady_clock::now() > deadline) {
            std::lock_guard lock(adapter_memory_mutex_);
            engine_->wake();
            throw std::runtime_error(
                "sleep timed out waiting for in-flight requests to finish; the model stays awake");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::lock_guard lock(adapter_memory_mutex_);
    engine_->sleep();
}

void GenerationService::wake_up() {
    // Shared training needs phase exclusion. Ordinary serving must leave admission
    // and lifetime release available while this potentially long copy is in progress.
    std::unique_lock phase_lock(request_capacity_->mutex, std::defer_lock);
    if (!options_.borrowed_weights.empty()) {
        phase_lock.lock();
        if (shared_training_) { throw std::logic_error("only the trainer can resume shared GRPO rollouts"); }
    }
    std::lock_guard memory_lock(adapter_memory_mutex_);
    engine_->wake();
}

void GenerationService::begin_shared_training() {
    if (options_.borrowed_weights.empty()) {
        throw std::logic_error("shared training requires borrowed base weights");
    }
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (shared_training_) { throw std::logic_error("already in the training phase"); }
        shared_training_ = true;
    }
    try {
        // Keep the worker running until admitted requests finish. sleep_begin()
        // parks active generations too, so it cannot precede this drain.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
        while (active_requests() != 0) {
            if (std::chrono::steady_clock::now() > deadline) {
                throw std::runtime_error("shared training timed out draining rollouts");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        engine_->sleep();
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        shared_training_ = false;
        throw;
    }
}

void GenerationService::publish_shared_adapter(const std::string& name,
                                                const std::vector<DeviceAdapterModule>& modules) {
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (!shared_training_ || request_capacity_->active != 0) {
            throw std::logic_error("publishing requires a drained training phase");
        }
    }
    if (name.empty() || modules.empty()) { throw std::invalid_argument("an adapter name and modules are required"); }
    auto& store = engine_->lora_stores().for_device(engine_->device());
    // Validate the complete update before waking or changing any adapter bytes.
    std::set<std::pair<int, std::string>> seen;
    for (const auto& module : modules) {
        store.validate_device_module(module);
        if (!seen.emplace(module.layer, module.module).second) {
            throw std::invalid_argument("duplicate adapter module");
        }
    }
    engine_->wake();
    store.clear_slot(0);
    for (const auto& module : modules) { store.set_device_module(0, module); }
    store.set_active(true);
    engine_->shrink_kv();
    lora_slots_.reset_to(name, 0);
    std::lock_guard lock(request_capacity_->mutex);
    shared_training_ = false;
}

std::size_t GenerationService::active_requests() const {
    const std::lock_guard<std::mutex> lock(request_capacity_->mutex);
    return request_capacity_->active;
}

std::size_t GenerationService::resumable_requests() const {
    const std::lock_guard lock(request_capacity_->mutex);
    return request_capacity_->active - request_capacity_->awaiting_wake;
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        request.max_tokens     = 4;
        request.max_tokens_set = true;
        // A base model would refuse the chat shape, so warm it the way it will be asked.
        if (!engine_->supports_chat()) {
            request.raw_prompt = "hi";
            PreparedRequest base = prepare(request);
            run(base, nullptr);
            return;
        }
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
