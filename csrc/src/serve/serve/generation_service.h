#pragma once

// Product-side adapter between HTTP protocol requests and the public SInfer
// engine. It owns one Engine and keeps protocol concerns (aliases, usage,
// streaming callbacks, and tool-call parsing) outside the target package.

#include "api/engine.h"
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
    sinfer::FinishReason finish_reason = sinfer::FinishReason::OutputLimit;
    GenerationMetrics metrics;
};

struct StreamSink {
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
    std::size_t tool_name_max_length       = 64;
    bool enable_thinking                   = true;
    bool preserve_thinking                 = false;
    bool preserve_thinking_semantic_change = false;
    std::shared_ptr<RequestLifetime> lifetime;
};

class GenerationService {
public:
    explicit GenerationService(ServeOptions options, LoadProgress load_progress = {});

    [[nodiscard]] const ServeOptions& options() const noexcept { return options_; }
    /// The bank slot for an adapter name, or -1 for the base model.
    [[nodiscard]] std::int32_t lora_slot(const std::string& name) const {
        const std::lock_guard<std::mutex> lock(lora_mutex_);
        const auto found = lora_slot_of_.find(name);
        return found == lora_slot_of_.end() ? -1 : found->second;
    }
    /// The resident adapter names, for /v1/models and model routing.
    [[nodiscard]] std::vector<std::string> lora_adapter_names() const {
        const std::lock_guard<std::mutex> lock(lora_mutex_);
        std::vector<std::string> names;
        names.reserve(lora_slot_of_.size());
        for (const auto& [name, slot] : lora_slot_of_) { names.push_back(name); }
        return names;
    }
    /// Loads a PEFT adapter directory into a free slot, addressable by `name`
    /// from the next request on. Throws std::invalid_argument with the reason on
    /// refusal -- bad adapter, name taken, no free slot, module not applicable.
    void load_lora_adapter(const std::string& name, const std::string& path);
    /// Unloads by name. The slot is zeroed, so a request already in flight that
    /// selected it degrades to the base model rather than reading freed weights;
    /// the slot is reused last among the free ones.
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
    void sleep();
    void wake_up();
    [[nodiscard]] bool is_sleeping() const { return engine_->is_sleeping(); }
    /// Requests currently inside this service (admission-counted).
    [[nodiscard]] std::size_t active_requests() const;
    /// This model's VRAM footprint while awake (sleepable regions).
    [[nodiscard]] std::size_t resident_bytes() const { return engine_->sleepable_bytes(); }

private:
    [[nodiscard]] std::shared_ptr<RequestLifetime> acquire_request_lifetime() const;

    ServeOptions options_;
    /// Adapter name -> bank slot. A request carries the name; the round carries
    /// the slot, and nothing below this class knows the name. Guarded by
    /// lora_mutex_ because the runtime endpoints mutate it while request threads
    /// resolve names. Freed slots go to the back of the free list so a just-
    /// unloaded slot is the last to be reused -- an in-flight request still naming
    /// it reads zeros (base model), not another adapter's fresh weights.
    mutable std::mutex lora_mutex_;
    std::map<std::string, std::int32_t> lora_slot_of_;
    std::vector<std::int32_t> lora_free_slots_;
    std::unique_ptr<sinfer::Engine> engine_;
    sinfer::PromptCapabilities prompt_capabilities_;
    std::shared_ptr<RequestCapacity> request_capacity_;
};

} // namespace sinfer::serve
