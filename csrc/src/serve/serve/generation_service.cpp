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
    ServiceOutputSink(Engine& engine, const StreamSink& sink, bool filter_tool_calls, bool json_tools)
        : engine_(&engine), sink_(&sink), filter_tool_calls_(filter_tool_calls), tool_filter_(json_tools) {}

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

    void publish_scores(TokenScoreDelta delta) override {
        if (!sink_->on_scores) return;
        GenerationOutcome outcome;
        outcome.prompt_scores = std::move(delta.prompt);
        outcome.completion_scores = std::move(delta.completion);
        populate_score_texts(*engine_, outcome);
        sink_->on_scores(outcome);
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

    Engine* engine_ = nullptr;
    const StreamSink* sink_ = nullptr;
    bool filter_tool_calls_ = false;
    ToolCallStreamFilter tool_filter_;
    std::size_t content_bytes_ = 0;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)), lora_slots_(options_.max_loras) {
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
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
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
    prepared.lifetime = acquire_request_lifetime();

    try {
        // Pin before preparation/submission, closing the race with replacement
        // and unload. The engine owns this lease through completion or cancellation.
        auto adapter = lora_slots_.acquire(request.lora_adapter, prepared.lifetime->deadline, is_cancelled);
        request_options.execution.lora_slot = adapter.slot;
        const auto acquisition_started = Clock::now();
        // A raw prompt carries no content parts, so there is nothing to acquire and no
        // template to render: the text is tokenized as written.
        std::optional<sinfer::PromptInput> input;
        if (!request.raw_prompt.has_value()) {
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
        sinfer::PreparedPrompt prompt =
            !request.prompt_token_ids.empty()
                ? engine_->prepare_tokens(request.prompt_token_ids)
                : (request.raw_prompt.has_value()
                       ? engine_->prepare_text(*request.raw_prompt)
                       : engine_->prepare(std::move(*input), control));
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
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              prepared.lifetime->deadline, std::move(adapter.lifetime));
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
        output_sink = std::make_unique<ServiceOutputSink>(*engine_, *sink, prepared.tool_capable,
            options_.tool_call_format == ToolCallFormat::Llama3Json);
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

    bool is_tool_call_response = false;
    if (prepared.tool_capable) {
        ParsedToolCalls parsed = parse_tool_calls(options_.tool_call_format, outcome.text,
                                                  prepared.tool_name_max_length, prepared.tools);
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
    ReasoningSplit split = split_reasoning(options_.reasoning_format,
                                           std::move(outcome.reasoning), std::move(outcome.text));
    outcome.reasoning    = std::move(split.reasoning);
    outcome.text         = std::move(split.content);
    if (output_sink) {
        outcome.streamed_content_bytes = output_sink->finish(is_tool_call_response);
    }
    return outcome;
}

std::vector<sinfer::TokenId> GenerationService::tokenize(const GenerationRequest& request) {
    if (!request.prompt_token_ids.empty()) { return request.prompt_token_ids; }
    if (request.raw_prompt.has_value()) {
        return engine_->prepare_text(*request.raw_prompt).token_ids();
    }
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    std::size_t remaining_media_bytes =
        std::min(options_.max_request_bytes, sinfer::kMaximumPromptMediaBytes);
    const auto never_cancelled = [] { return false; };
    sinfer::PromptInput input  = to_prompt_input(request, semantics, [&](const ContentPart& part) {
        return acquire_media(part, Clock::now() + std::chrono::seconds(30), never_cancelled,
                             remaining_media_bytes);
    });
    return engine_->prepare(std::move(input)).token_ids();
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
            if (request_capacity_->active == 0) { break; }
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
    std::lock_guard lock(request_capacity_->mutex);
    if (shared_training_) { throw std::logic_error("only the trainer can resume shared GRPO rollouts"); }
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
