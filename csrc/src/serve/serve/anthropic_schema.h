#pragma once

// Anthropic Messages API wire-format layer: parses /v1/messages request JSON into
// the internal GenerationRequest and serializes internal results back into
// Anthropic message bodies / SSE events. This is a sibling of openai_schema.h;
// both map to the same wire-agnostic GenerationRequest / GenerationOutcome, so the
// engine and generation service below know nothing about either protocol.

#include "serve/request.h"
#include "api/types.h"

#include <nlohmann/json.hpp>

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::serve {

// Parse an already-decoded Anthropic Messages body into a GenerationRequest.
// Top-level `system` becomes a leading system turn; system-role messages retain
// their array position; user tool_result blocks become ordered tool turns; and
// assistant tool_use blocks become tool calls. Throws ApiException on malformed /
// unsupported requests. The `model` field is accepted verbatim (any Claude model
// name) and echoed back, never validated against the loaded model.
GenerationRequest parse_messages_request(const nlohmann::json& body, const RequestLimits& limits);

// Map an internal finish reason (+ whether tool calls were produced) onto the
// Anthropic stop_reason wire value.
const char* messages_stop_reason(sinfer::FinishReason reason, bool has_tool_calls);

// Non-streaming Messages response body (JSON string). Content blocks are emitted
// in order: an optional `thinking` block (from reasoning), an optional `text`
// block (from content), then a `tool_use` block per tool call. When nothing was
// produced an empty text block is emitted so `content` is never empty.
std::string make_messages_response(const std::string& id, const std::string& model,
                                   const std::string& content, const std::string& reasoning,
                                   const std::vector<ToolCall>& tool_calls, const char* stop_reason,
                                   const CompletionUsage& usage, std::string_view stop_sequence = {});

// Streaming SSE event strings ("event: <type>\ndata: {...}\n\n"). The transport
// calls these pure builders. MessagesStreamBlocks keeps alternating text and
// thinking spans sequential; tool_use blocks follow those spans.
/// The stream's opening event. `usage` is the prompt's: input tokens with the cached part split
/// out (see messages_usage), and no output yet. The server sends it once the prompt is
/// prefilled, when the cache split is known, because gateways (agentgateway among them) take
/// the input tokens from this event and only the output tokens from message_delta.
std::string make_message_start(const std::string& id, const std::string& model, const CompletionUsage& usage);
std::string make_content_block_start_text(int index);
std::string make_content_block_start_thinking(int index);
std::string make_content_block_start_tool_use(int index, const ToolCall& call);
std::string make_content_block_delta_text(int index, const std::string& delta_text);
std::string make_content_block_delta_thinking(int index, const std::string& delta_text);
std::string make_content_block_delta_tool_json(int index, const std::string& partial_json);
std::string make_content_block_stop(int index);
/// The final usage of a streamed message, cumulative: input, cache reads and output tokens.
std::string make_message_delta(const char* stop_reason, const CompletionUsage& usage,
                               std::string_view stop_sequence = {});
std::string make_message_stop();
std::string make_messages_ping();

class MessagesStreamBlocks {
public:
    explicit MessagesStreamBlocks(std::function<void(const std::string&)> emit)
        : emit_(std::move(emit)) {}
    void append(OutputChannel channel, const std::string& text);
    void close();
    [[nodiscard]] int next_index() const noexcept { return next_index_; }

private:
    std::function<void(const std::string&)> emit_;
    std::optional<OutputChannel> open_;
    int next_index_ = 0;
};

// Error object body (Anthropic shape) and its SSE `event: error` form.
std::string make_messages_error_body(const ApiError& error);
std::string messages_sse_error_event(const ApiError& error);

// /v1/messages/count_tokens response body.
std::string make_count_tokens_response(int input_tokens);

// Message identifier ("msg_...").
std::string new_message_id();

} // namespace sinfer::serve
