#pragma once

#include "product/media_acquire/source.h"

#include <api/types.h>

// Internal, wire-format-independent representation of a generation request.
//
// OpenAI and Anthropic schemas both map into this wire-independent value.
// translate.cpp then produces the public PromptInput and RequestOptions consumed
// by Engine; media sources remain unresolved until the product service acquires
// owning bytes.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sinfer::serve {

// A structured API error mapped onto an error object + HTTP status. Wire-format
// independent: each protocol layer renders it into its own error body shape.
struct ApiError {
    int status       = 400;
    std::string type = "invalid_request_error";
    std::string message;
    std::string param; // optional
    std::string code;  // optional
    /// vLLM's error shape: `code` is the HTTP status as a number (and the keys come in vLLM's order). 0 = the
    /// string `code` above.
    int numeric_code = 0;
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

/// vLLM's CompletionRequest.max_tokens default (the OpenAI one): /v1/completions without a max_tokens asks for 16.
inline constexpr int kVllmCompletionDefaultMaxTokens = 16;

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    /// --default-max-tokens; 0 = none, a request without max_tokens gets the rest of the context (vLLM's
    /// get_max_tokens without a generation-config max_new_tokens), resolved where the prompt is measured.
    int default_max_tokens = 0;
    /// --overlong-max-tokens reject: a max_tokens below 1 is left for vLLM's check after the prompt is counted
    /// (serve/vllm_max_tokens.h), which is where vLLM refuses it, instead of being refused while parsing.
    bool vllm_max_tokens = false;
};

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
    /// The part of prompt_tokens served from the prefix cache instead of being prefilled.
    int cached_tokens     = 0;
};

/// Usage from an outcome's counts. The cached part is the prompt tokens the prefix cache supplied
/// (the request log's prefix_cache_hit_tokens), clamped to the prompt: the value the Responses
/// API reports as input_tokens_details.cached_tokens.
[[nodiscard]] inline CompletionUsage completion_usage(int prompt_tokens, int completion_tokens,
                                                      std::uint64_t prefix_cache_hit_tokens) {
    const auto prompt = static_cast<std::uint64_t>(prompt_tokens > 0 ? prompt_tokens : 0);
    return CompletionUsage{prompt_tokens, completion_tokens,
                           static_cast<int>(prefix_cache_hit_tokens < prompt ? prefix_cache_hit_tokens : prompt)};
}

enum class ContentKind {
    Text,
    Image,
    Video,
    InputAudio,
    Unsupported,
};

struct ContentPart {
    ContentKind kind = ContentKind::Text;
    std::string text;     // populated for Text
    std::string type_raw; // original OpenAI "type" string (diagnostics / future use)
    sinfer::product::media_acquire::Source source;
};

struct ToolDefinition {
    std::string name;
    std::string description;
    std::string parameters_json;
    std::string definition_json; // the tool as the chat template sees it (template_tool_json)
    bool strict = false;
    bool strict_set = false;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ToolChoiceMode {
    Auto,
    None,
    Required,
    Named,
};

struct ToolChoice {
    ToolChoiceMode mode = ToolChoiceMode::Auto;
    std::string name;
};

struct ChatTurn {
    ChatRole role = ChatRole::User;
    std::vector<ContentPart>
        content; // one or more parts; assistant content may be empty with tool_calls
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;      // populated for role=tool
    std::string reasoning_content; // assistant thinking carried across turns (round-tripped to the
                                   // template)
    /// The client sent a reasoning string, possibly empty. An empty one still reaches the
    /// template, which then renders an empty think block rather than looking for one in the
    /// content -- the difference between "no reasoning" and "reasoning not given".
    bool has_reasoning_content = false;
};

// OpenAI sampling fields carried by the protocol adapter.
struct SamplingParams {
    std::optional<double> temperature;
    std::optional<double> top_p;
    std::optional<int> top_k;
    /// vLLM's floor on a token's probability relative to the most likely one. The
    /// engine has always taken it; the wire layer simply never read it, so a
    /// request that asked for it got a distribution that ignored it.
    std::optional<double> min_p;
    /// vLLM's multiplicative repetition penalty. 1 asks for nothing.
    std::optional<double> repetition_penalty;
    std::optional<double> presence_penalty;
    std::optional<double> frequency_penalty;
    std::optional<std::uint64_t> seed;
    std::unordered_map<int, double> logit_bias;
    int n = 1;
};

// Protocol-level effort vocabulary. Each wire adapter accepts the values from
// its external contract; translation then resolves them against the capabilities
// advertised by the chat template embedded in the loaded artifact.
enum class RequestedReasoningEffort : std::uint8_t {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
};

[[nodiscard]] constexpr std::optional<RequestedReasoningEffort>
parse_requested_reasoning_effort(std::string_view value) noexcept {
    if (value == "none") { return RequestedReasoningEffort::None; }
    if (value == "minimal") { return RequestedReasoningEffort::Minimal; }
    if (value == "low") { return RequestedReasoningEffort::Low; }
    if (value == "medium") { return RequestedReasoningEffort::Medium; }
    if (value == "high") { return RequestedReasoningEffort::High; }
    if (value == "xhigh") { return RequestedReasoningEffort::XHigh; }
    if (value == "max") { return RequestedReasoningEffort::Max; }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view
requested_reasoning_effort_name(RequestedReasoningEffort effort) noexcept {
    switch (effort) {
    case RequestedReasoningEffort::None:
        return "none";
    case RequestedReasoningEffort::Minimal:
        return "minimal";
    case RequestedReasoningEffort::Low:
        return "low";
    case RequestedReasoningEffort::Medium:
        return "medium";
    case RequestedReasoningEffort::High:
        return "high";
    case RequestedReasoningEffort::XHigh:
        return "xhigh";
    case RequestedReasoningEffort::Max:
        return "max";
    }
    return {};
}

struct GenerationRequest {
    /// The caller's request id (X-Request-Id), already validated; empty when none was sent.
    /// Written to the request log only; it never changes what is generated.
    std::string client_request_id;
    std::string json_schema;
    bool parallel_decoding = false;
    std::string model;
    /// A `/v1/completions` prompt, served exactly as written with no chat template. Set for a
    /// completion request and empty for a chat one; the two are the same request otherwise, so
    /// everything below -- sampling, stops, streaming, adapters, limits -- is shared.
    std::optional<std::string> raw_prompt;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    bool parallel_tool_calls = true;
    std::vector<std::string> stop_strings;
    // Benchmark knob (vLLM-compatible): generation ignores the model's stop tokens and runs to
    // max_tokens, so throughput is measured on a fixed output length.
    bool ignore_eos = false;
    /// The fewest tokens to generate before a stop token may be drawn.
    int min_tokens = 0;
    /// 0 with max_tokens_set false: no server default, the rest of the context (RequestLimits::default_max_tokens).
    int max_tokens      = 0;
    bool max_tokens_set = false;
    /// The field max_tokens came from (max_tokens, max_completion_tokens, max_output_tokens), for vLLM's message.
    std::string max_tokens_param = "max_tokens";
    /// The client's max_tokens when it does not fit `max_tokens` (which then saturates); vLLM's rule words its
    /// refusal with the value as sent.
    std::optional<std::int64_t> max_tokens_wide;
    bool stream         = false;
    bool include_usage  = false;
    std::optional<bool> enable_thinking; // non-standard extension; falls back to server default
    /// Where this request spelled the thinking switch, so a refusal points at the
    /// field the client actually sent -- each wire has its own name for it.
    std::string enable_thinking_param = "enable_thinking";
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::string reasoning_effort_param = "reasoning_effort";
    std::optional<bool> preserve_thinking;
    /// Per-token log-probabilities in the response (`logprobs`), and the prompt and
    /// completion token ids beside them (`return_token_ids`, vLLM's extension). An
    /// RL trainer needs all three: it scores the exact ids it was given against the
    /// probabilities they were drawn with.
    int top_logprobs = 0;
    int prompt_logprobs = -1;
    bool want_logprobs    = false;
    bool return_token_ids = false;
    /// Rollout Routing Replay: the MoE expert ids every prompt and completion token was routed
    /// through, in choice.routed_experts (needs a server started with --capture-routing).
    bool return_routed_experts = false;
    /// The prompt as token ids, replacing whatever `messages` would have rendered
    /// to. Empty unless the client sent them.
    ///
    /// A multi-turn RL rollout holds the exact ids of the turn it just generated
    /// and needs the next prompt to extend them. Re-rendering the messages is not
    /// guaranteed to reproduce that prefix -- a template that strips earlier
    /// `<think>` blocks changes it -- and a prompt that is not an extension breaks
    /// the trajectory into separate training samples. `messages` is still parsed
    /// and still carries the tools and the parser state; only the ids change.
    std::vector<sinfer::TokenId> prompt_token_ids;
    /// Whether to append the template's generation prompt. Unset means yes, which
    /// is what generating needs.
    ///
    /// A turn-stitching client asks for it to be off when it tokenises a fragment
    /// it means to place *before* something else: it renders one message with the
    /// prompt and once without, and the difference is the separator the template
    /// inserts between turns. Ignoring the flag made both renders identical, the
    /// client's prefix check failed, and it silently fell back to re-rendering
    /// every turn -- which is the thing the token endpoint exists to avoid.
    std::optional<bool> add_generation_prompt;
    SamplingParams sampling;

    /// The adapter this request selected by naming it in `model`, empty for the
    /// base model. Resolved by the HTTP layer against the registry, so by the time
    /// the service sees it the name is known to exist.
    std::string lora_adapter;

    [[nodiscard]] bool uses_tools() const noexcept {
        return !tools.empty() && tool_choice.mode != ToolChoiceMode::None;
    }

    [[nodiscard]] std::size_t media_item_count() const noexcept {
        std::size_t count = 0;
        for (const ChatTurn& message : messages) {
            for (const ContentPart& part : message.content) {
                if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) { ++count; }
            }
        }
        return count;
    }

    [[nodiscard]] bool has_tool_history() const noexcept {
        for (const ChatTurn& message : messages) {
            if (!message.tool_calls.empty() || message.role == ChatRole::Tool) { return true; }
        }
        return false;
    }
};

} // namespace sinfer::serve
