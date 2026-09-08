#pragma once

// Named reasoning and tool-call parsers, selected the way vLLM selects them
// (`--reasoning-parser`, `--tool-call-parser`).
//
// Both formats were hard-wired before: reasoning split on `<think>`/`</think>`
// inside the target's frontend, and tool calls parsed Qwen's `<tool_call>` block
// in `tool_call_parser.cpp`. A served model whose template emits anything else
// had no way to say so. These registries name the formats instead, so an
// unsupported one is refused at startup with the list of what is supported
// rather than silently producing a reasoning block full of answer, or an answer
// full of unparsed JSON.
//
// The registry is closed on purpose. A parser here is a claim that the format
// round-trips, and every entry is one the engine has a decoder for.

#include "serve/request.h"

#include <string>
#include <string_view>
#include <vector>

namespace sinfer::serve {

/// How a model marks its reasoning span.
enum class ReasoningFormat {
    None,       ///< no reasoning span; whatever the model emits is content
    ThinkTags,  ///< `<think>` ... `</think>` (qwen3, deepseek_r1, glm4_moe)
};

/// How a model emits a tool call.
enum class ToolCallFormat {
    None,        ///< tool calls are not parsed out of the text
    QwenXml,     ///< `<tool_call>{"name":..,"arguments":{..}}</tool_call>` (qwen3_xml, hermes)
    Spark25,     ///< `<tool_call>name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`
    Llama3Json,  ///< bare `{"name":..,"parameters":{..}}`, optionally after `<|python_tag|>`
};

/// Parses `--reasoning-parser`. Throws std::invalid_argument naming the
/// supported values when `name` is not one of them.
[[nodiscard]] ReasoningFormat parse_reasoning_format(std::string_view name);

/// Parses `--tool-call-parser`; same contract.
[[nodiscard]] ToolCallFormat parse_tool_call_format(std::string_view name);

/// Comma-separated supported names, for usage text and error messages.
[[nodiscard]] std::string reasoning_parser_names();
[[nodiscard]] std::string tool_call_parser_names();

/// Splits `text` into (reasoning, content) for `format`.
///
/// The target frontends already split `ThinkTags` while decoding, so the server
/// applies this only to reconcile a *different* choice: `None` folds an already
/// split reasoning span back into the content, which is what a deployment that
/// does not want reasoning surfaced as a separate field asks for.
struct ReasoningSplit {
    std::string reasoning;
    std::string content;
};
[[nodiscard]] ReasoningSplit split_reasoning(ReasoningFormat format, std::string reasoning,
                                             std::string content);

/// Parses tool calls out of a completed generation for `format`.
struct ParsedToolCalls {
    bool is_tool_call_response = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
};
[[nodiscard]] ParsedToolCalls parse_tool_calls(ToolCallFormat format, const std::string& text,
                                               std::size_t max_tool_name_length);

} // namespace sinfer::serve
