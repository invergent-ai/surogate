#pragma once

// Product-side adapter between HTTP protocol requests and the public SInfer
// engine. It owns one Engine and keeps protocol concerns (aliases, usage,
// streaming callbacks, and tool-call parsing) outside the target package.

#include "api/engine.h"
#include "serve/lora_slots.h"
#include "serve/request.h"
#include "serve/serve_options.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <memory>
#include <string>
#include <vector>

namespace sinfer::serve {

struct RequestLifetime;
struct RequestCapacity;

struct GenerationMetrics {
    double prepare_seconds = 0.0;
    double ttft_seconds    = 0.0;
    double vision_seconds  = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds  = 0.0;
    double total_seconds   = 0.0;

    SpeculativeBackend speculative_backend    = SpeculativeBackend::None;
    std::uint32_t speculative_draft_window    = 0;
    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    std::vector<std::uint64_t> speculative_accepted_per_position;
    std::uint32_t prefix_cache_hit_tokens     = 0;
    sinfer::PrefixReusePath prefix_reuse_path = sinfer::PrefixReusePath::FullReset;
};

struct GenerationOutcome {
    std::string text;
    std::string reasoning;
    std::vector<ToolCall> tool_calls;
    int prompt_tokens                  = 0;
    int completion_tokens              = 0;
    int reasoning_tokens               = 0;
    std::size_t streamed_content_bytes = 0;
    /// Populated only when the request asked for them. `token_logprobs` is either
    /// empty or exactly as long as `completion_token_ids`.
    std::vector<sinfer::TokenId> prompt_token_ids;
    std::vector<sinfer::TokenId> completion_token_ids;
    std::vector<TokenScore> prompt_scores;
    std::vector<TokenScore> completion_scores;
    std::map<TokenId, std::string> score_texts;
    std::vector<float> token_logprobs;
    /// The text of each completion token, so a probability names the token it
    /// belongs to. Filled only alongside `token_logprobs`.
    std::vector<std::string> token_texts;
    sinfer::FinishReason finish_reason = sinfer::FinishReason::OutputLimit;
    GenerationMetrics metrics;
};

struct StreamSink {
    std::function<void(const GenerationOutcome&)> on_scores;
    std::function<void(const std::string& delta_text)> on_content;
    std::function<void(const std::string& delta_text)> on_reasoning;
    std::function<bool()> is_cancelled;
};

// Translate Engine request failures into the shared protocol-neutral HTTP error contract.
ApiError request_error_to_api_error(const sinfer::RequestError& exception);

// Preparation ends by synchronously submitting the owning prompt to the Engine FIFO. The returned
// request keeps its ingress/response lifetime reservation until the HTTP response is released and
// is consumed exactly once by run().
struct PreparedRequest {
    sinfer::GenerationHandle generation;
    sinfer::ResolvedSamplingParameters sampling;
    double prepare_seconds     = 0.0;
    double acquisition_seconds = 0.0;
    PromptPreparationStats preparation;
    int prompt_tokens                      = 0;
    bool include_usage                     = false;
    bool tool_capable                      = false;
    bool constrained_tools                 = false;
    ToolChoice tool_choice;
    bool parallel_tool_calls               = true;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length       = 64;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    /// What the client asked to be given back, and the prompt ids to give it.
    /// Snapshotted before the prompt is submitted, because submitting consumes it.
    int top_logprobs = 0;
    int prompt_logprobs = -1;
    bool want_logprobs    = false;
    bool return_token_ids = false;
    std::vector<sinfer::TokenId> prompt_token_ids;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, LoadProgress load_progress = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }
    /// The bank slot for an adapter name, or -1 for the base model.
    [[nodiscard]] std::int32_t lora_slot(const std::string& name) const {
        return lora_slots_.find(name);
    }
    /// The resident adapter names, for /v1/models and model routing.
    [[nodiscard]] std::vector<std::string> lora_adapter_names() const {
        return lora_slots_.names();
    }
    /// What a request's prompt tokenises to, without generating anything. Ids the
    /// client supplied are returned as given -- they are already the answer.
    [[nodiscard]] std::vector<sinfer::TokenId> tokenize(const GenerationRequest& request);
    /// The text of each token id, for a tokenize response that asks for it.
    [[nodiscard]] std::vector<std::string> token_texts(const std::vector<sinfer::TokenId>& ids);
    /// The context this engine was configured for, which a stitching client needs
    /// to know before it decides a prompt is too long.
    [[nodiscard]] std::uint32_t max_context() const;

    /// Whether any stage of this engine has adapter bindings.
    [[nodiscard]] bool any_lora_bindings() const;
    /// Scrub a drained adapter slot on every store.
    void clear_lora_slot_everywhere(std::int32_t slot);

    /// Load or replace a PEFT adapter. Replacement drains admitted requests;
    /// new requests for that adapter wait. Invalid payloads preserve the old
    /// adapter; RequestError reports a timeout before any bytes are changed.
    void load_lora_adapter(const std::string& name, const std::string& path);
    /// Drain admitted requests before removing the adapter and recycling its slot.
    void unload_lora_adapter(const std::string& name);

    [[nodiscard]] sinfer::LoadSummary load_summary() const { return engine_->load_summary(); }

    [[nodiscard]] sinfer::MemorySummary memory_summary() const { return engine_->memory_summary(); }

    [[nodiscard]] sinfer::RuntimeStats runtime_stats() const { return engine_->runtime_stats(); }

    [[nodiscard]] sinfer::MediaCacheSummary media_cache_summary() const {
        return engine_->media_cache_summary();
    }

    [[nodiscard]] sinfer::ModelSamplingDefaults sampling_defaults() const {
        return engine_->sampling_defaults();
    }

    [[nodiscard]] PreparedRequest prepare(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;
    [[nodiscard]] int count_prompt_tokens(const GenerationRequest& req,
                                          std::function<bool()> is_cancelled = {}) const;

    // Consumes prepared.generation. A PreparedRequest is single-use.
    GenerationOutcome run(PreparedRequest& prepared, const StreamSink* sink,
                          std::function<bool()> is_cancelled = {});

    void warmup();

    /// Sleep level 1 (vLLM parity): refuse new work, wait for in-flight
    /// requests to finish, then release the model's VRAM with its state parked
    /// in host RAM. `wake_up` restores it; requests then run against
    /// byte-identical state. Both are idempotent.
    void sleep(bool preempt = false);
    void wake_up();
    [[nodiscard]] bool is_sleeping() const { return engine_->is_sleeping(); }
    /// Return idle KV (prefix cache) so `resident_bytes` shrinks without sleeping.
    void shrink_kv() { engine_->shrink_kv(); }
    /// Requests currently inside this service (admission-counted).
    [[nodiscard]] std::size_t active_requests() const;
    void prepare_sleep_backup() {
        std::lock_guard lock(adapter_memory_mutex_);
        engine_->prepare_sleep_backup();
    }
    /// Drain and release serving allocations while the trainer retains the base.
    void begin_shared_training();
    /// Publish a complete adapter from GPU tensors, then reopen generation.
    void publish_shared_adapter(const std::string& name, const std::vector<DeviceAdapterModule>& modules);
    /// This model's VRAM footprint while awake (sleepable regions).
    [[nodiscard]] std::size_t resident_bytes(int device = -1) const {
        return engine_->sleepable_bytes(device);
    }
    [[nodiscard]] std::vector<int> devices() const { return engine_->devices(); }

private:
    [[nodiscard]] std::shared_ptr<RequestLifetime> acquire_request_lifetime() const;

    ServeOptions options_;
    LoraSlots lora_slots_;
    // Sleep/unmap and host uploads cannot access the adapter banks concurrently.
    mutable std::mutex adapter_memory_mutex_;
    std::unique_ptr<sinfer::Engine> engine_;
    sinfer::PromptCapabilities prompt_capabilities_;
    std::shared_ptr<RequestCapacity> request_capacity_;
    bool shared_training_ = false; // guarded by request_capacity_->mutex
};

} // namespace sinfer::serve
