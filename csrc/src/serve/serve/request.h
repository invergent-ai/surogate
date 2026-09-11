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
};

class ApiException : public std::runtime_error {
public:
    explicit ApiException(ApiError error)
        : std::runtime_error(error.message), error_(std::move(error)) {}

    [[nodiscard]] const ApiError& error() const noexcept { return error_; }

private:
    ApiError error_;
};

// Server-side context needed while parsing/validating a request.
struct RequestLimits {
    int default_max_tokens = 8192;
};

struct CompletionUsage {
    int prompt_tokens     = 0;
    int completion_tokens = 0;
};

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
    std::string definition_json; // normalized OpenAI function-tool object for Qwen prompt rendering
    bool strict = false;
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
    std::string json_schema;
    std::string model;
    /// A `/v1/completions` prompt, served exactly as written with no chat template. Set for a
    /// completion request and empty for a chat one; the two are the same request otherwise, so
    /// everything below -- sampling, stops, streaming, adapters, limits -- is shared.
    std::optional<std::string> raw_prompt;
    std::vector<ChatTurn> messages;
    std::vector<ToolDefinition> tools;
    std::size_t tool_name_max_length = 64;
    ToolChoice tool_choice;
    std::vector<std::string> stop_strings;
    // Benchmark knob (vLLM-compatible): generation ignores the model's stop tokens and runs to
    // max_tokens, so throughput is measured on a fixed output length.
    bool ignore_eos = false;
    /// The fewest tokens to generate before a stop token may be drawn.
    int min_tokens = 0;
    int max_tokens      = 0; // 0 => use server default
    bool max_tokens_set = false;
    bool stream         = false;
    bool include_usage  = false;
    std::optional<bool> enable_thinking; // non-standard extension; falls back to server default
    /// Where this request spelled the thinking switch, so a refusal points at the
    /// field the client actually sent -- each wire has its own name for it.
    std::string enable_thinking_param = "enable_thinking";
    std::optional<RequestedReasoningEffort> reasoning_effort;
    std::string reasoning_effort_param = "reasoning_effort";
    std::optional<bool> preserve_thinking;
    bool preserve_thinking_semantic_change = false;
    /// Per-token log-probabilities in the response (`logprobs`), and the prompt and
    /// completion token ids beside them (`return_token_ids`, vLLM's extension). An
    /// RL trainer needs all three: it scores the exact ids it was given against the
    /// probabilities they were drawn with.
    bool want_logprobs    = false;
    bool return_token_ids = false;
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
