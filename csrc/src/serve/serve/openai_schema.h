#pragma once

// OpenAI wire-format layer: parses request JSON into the internal GenerationRequest
// and serializes internal results back into OpenAI Chat Completions bodies/chunks.
// This layer knows nothing about the engine; it only speaks the OpenAI schema.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sinfer::serve {

// ApiError, ApiException, RequestLimits, and CompletionUsage are the wire-format
// independent request/error types; they live in request.h and are shared by the
// OpenAI and Anthropic schema layers.

// Parse an already-decoded JSON body into a GenerationRequest. Throws ApiException
// on malformed or unsupported requests (n>1, tools, non-text response_format, ...).
GenerationRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

std::optional<bool> parse_openai_preserve_thinking(const nlohmann::json& body);

// Parse a /v1/completions body. The prompt is carried verbatim in
// `GenerationRequest::raw_prompt`; everything else -- sampling, stops, streaming, the token
// budget -- is the same request a chat call makes, so the two share one type.
GenerationRequest parse_completion_request(const nlohmann::json& body,
                                           const RequestLimits& limits);

// Non-streaming /v1/completions body, and its streaming chunks. The shape differs from chat
// in more than a name: a choice carries `text` rather than a message, and there is no role.
std::string make_completion_response(const std::string& id, const std::string& model,
                                     std::int64_t created, const std::string& text,
                                     const char* finish_reason, const CompletionUsage& usage);
std::string make_completion_chunk_text(const std::string& id, const std::string& model,
                                       std::int64_t created, const std::string& delta_text,
                                       bool include_usage);
std::string make_completion_chunk_final(const std::string& id, const std::string& model,
                                        std::int64_t created, const char* finish_reason,
                                        bool include_usage);
std::string make_completion_chunk_usage(const std::string& id, const std::string& model,
                                        std::int64_t created, const CompletionUsage& usage);
std::string new_completion_id();

// Non-streaming chat completion response body (JSON string). When `reasoning` is
// non-empty it is attached as `message.reasoning_content` (the DeepSeek/vLLM-style
// convention consumed by Chatbox, Open WebUI, etc.), leaving `content` = answer.
/// The per-token detail a client asked to be given back: the ids it should score
/// and the probability each token was drawn with. Empty vectors render nothing, so
/// an ordinary chat response is byte-identical to what it was.
struct TokenDetail {
    std::vector<sinfer::TokenId> prompt_token_ids;
    std::vector<sinfer::TokenId> completion_token_ids;
    std::vector<float> logprobs;      ///< as long as `completion_token_ids`, or empty
    std::vector<std::string> texts;   ///< the text of each completion token, or empty
    bool include_token_ids = false;
    bool include_logprobs  = false;
};

std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage,
                                          const TokenDetail& detail = {});
std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const CompletionUsage& usage,
                                               const TokenDetail& detail = {});
std::string make_chat_chunk_token_detail(const std::string& id, const std::string& model,
                                         std::int64_t created, const TokenDetail& detail, bool include_usage);

// Streaming SSE event strings ("data: {...}\n\n"). The first chunk carries the
// assistant role; reasoning chunks carry `reasoning_content` deltas (the <think>
// block), content chunks carry `content` deltas; the final chunk carries the
// finish_reason with an empty delta. Per the OpenAI stream_options contract, when
// usage reporting is enabled every content-bearing chunk carries `usage: null`
// and a single dedicated usage chunk (empty choices) is emitted before [DONE];
// pass include_usage accordingly.
std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage);
std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage);
std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage);
std::string make_chat_chunk_tool_calls(const std::string& id, const std::string& model,
                                       std::int64_t created,
                                       const std::vector<ToolCall>& tool_calls, bool include_usage);
std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                  std::int64_t created, const char* finish_reason,
                                  bool include_usage);
// Dedicated usage chunk: `choices: []` with the request's token usage. Emitted
// only when stream_options.include_usage is true.
std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage);
std::string sse_done();

// /v1/models payloads.
std::string make_models_list(const std::string& model_id, std::int64_t created,
                             const std::vector<std::string>& adapters = {});
std::string make_model_object(const std::string& model_id, std::int64_t created);

// Error object body.
std::string make_error_body(const ApiError& error);

// Identifiers / timestamps.
std::string new_chat_completion_id();
std::int64_t unix_time_now();

} // namespace sinfer::serve
