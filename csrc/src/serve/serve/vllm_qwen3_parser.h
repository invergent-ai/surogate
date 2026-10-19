#pragma once

// A finished generation's reasoning, content and tool calls exactly as vLLM 0.31 returns them for a Qwen3.5 /
// Qwen3.6 model served with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`
// (qwen3_xml is the same parser there).
//
// vLLM 0.31 builds that parser as a DelegatingParser over two adapters of one declarative engine (vllm/parser/qwen3.py
// on vllm/parser/engine/{parser_engine,streaming_parser_engine,incremental_lexer}.py, vllm/parser/abstract_parser.py)
// and calls it for a finished request as chat_completion/serving.py does: `parser.parse(output.text, request)`. The
// token ids it is handed go unused, so this reads text only, in the same two passes:
//
//   1. Reasoning (ParserEngineReasoningAdapter.extract_reasoning; with thinking off the whole output is content): the
//      text up to `</think>`, or up to `<tool_call>`, which ends the reasoning implicitly and stays in the content.
//      A `<think>` inside the reasoning and a repeated `</think>` in the content are dropped, and so are the
//      tokenizer's special tokens other than the grammar's (transformers' `all_special_tokens`: `<|im_end|>`,
//      `<|endoftext|>`, the vision and audio markers). Whitespace is kept.
//   2. Tool calls (ParserEngineToolAdapter.extract_tool_calls, the engine started in content): the content is lexed
//      into `<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<function=`, `</function>`, `<parameter=`,
//      `</parameter>`, `>` and the special tokens' spellings, longest literal first, and run through the grammar: a call
//      opens at `<tool_call>` or at a bare `<function=`; its name runs to `>`; its body to `</function>`; text inside
//      `<tool_call>` before `<function=`, and between `</function>` and `</tool_call>`, is dropped; a call still open
//      at the end is closed there.
//
// The result: a call is kept when its stripped name is not empty -- any name, declared or not, a truncated one
// included -- with the parameters vLLM's `_qwen3_arg_converter` finds in its body (complete ones only) typed by the
// tool's schema (serve/vllm_tool_arguments.h). Content is the text before the first call; text after a call is
// dropped once a call has a name. With a call, content is stripped and whitespace-only content is dropped. With
// `tool_choice: "none"` and tools, the calls' text is dropped from the content and no call is returned. The finish
// reason plays no part: a call cut off by max_tokens is still a call, as in vLLM.

#include "serve/request.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sinfer::serve::vllm_qwen3 {

/// What the parser needs to know about the tokenizer: transformers' `all_special_tokens` spellings (any of the grammar's
/// own among them are ignored). They are dropped wherever they appear.
struct Vocabulary {
    std::vector<std::string> special_tokens;
};

struct Input {
    /// vLLM's `output.text`: the generated tokens decoded with their special tokens, the stop token that ended the
    /// generation excluded.
    std::string text;
    /// The prompt opened a reasoning turn (enable_thinking, default true).
    bool thinking = true;
    /// The request has tools and tool_choice "none".
    bool suppress_tool_calls = false;
};

struct Call {
    std::string name;
    std::string arguments;
};

struct Result {
    std::optional<std::string> reasoning;
    std::optional<std::string> content;
    /// In the order the model wrote them; empty when vLLM returns none.
    std::vector<Call> calls;
};

/// `tools`: the request's tools, for the arguments' types (by name, the first one wins).
[[nodiscard]] Result parse(const Input& input, const Vocabulary& vocabulary, const std::vector<ToolDefinition>& tools);

/// Python's `bytes.decode("utf-8", errors="replace")`: each maximal invalid subsequence becomes one U+FFFD.
[[nodiscard]] std::string utf8_replace(std::string_view bytes);

/// vLLM's `output.text` for generated token bytes (each token's bytes, the stop token's left out): their UTF-8
/// decoded as above, without a last character the output ended in the middle of -- vLLM's incremental
/// detokenizer holds such a character back until the next token completes it, and none does.
[[nodiscard]] std::string output_text(std::string_view bytes);

} // namespace sinfer::serve::vllm_qwen3
