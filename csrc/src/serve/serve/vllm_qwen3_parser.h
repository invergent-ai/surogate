#pragma once

// A finished generation's reasoning, content and tool calls exactly as vLLM 0.31 returns them for a Qwen3.5 /
// Qwen3.6 model served with `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`
// (qwen3_xml is the same parser there).
//
// vLLM 0.31 parses the whole output with one declarative engine (vllm/parser/qwen3.py on
// vllm/parser/engine/{parser_engine,streaming_parser_engine,incremental_lexer,token_id_scanner}.py), called as
// chat_completion/serving.py calls it for a non-streaming request: `parser.parse(output.text, request,
// model_output_token_ids=output.token_ids)`. This is that path, step for step:
//
//   * Special tokens are found by id. `<think>`, `</think>`, `<tool_call>` and `</tool_call>` are the grammar's
//     token-id terminals; the tokenizer's other special tokens (transformers' `all_special_tokens`: `<|im_end|>`,
//     `<|endoftext|>`, the vision and audio markers) are dropped. Those markers spelled by ordinary tokens are text.
//     The text between them is split as vLLM's TokenIDScanner splits it (each token decoded alone; when those
//     pieces do not spell the output, the markers are placed in the text from the right).
//   * The text is lexed into `<function=`, `</function>`, `<parameter=`, `</parameter>` and `>`, longest literal
//     first; the special tokens' spellings are literals too, read back as text.
//   * The state machine: reasoning until `</think>`, or until `<tool_call>` (an implicit end); a call opens at
//     `<tool_call>`, or at a bare `<function=` in content; its name runs to `>`; its body to `</function>`; text
//     inside `<tool_call>` before `<function=`, and between `</function>` and `</tool_call>`, is dropped; a call
//     still open when the output ends is closed there.
//   * The result: a call is kept when its stripped name is not empty -- any name, declared or not, a truncated one
//     included -- with the parameters vLLM's `_qwen3_arg_converter` finds in its body (complete ones only) typed by
//     the tool's schema (serve/vllm_tool_arguments.h). Content is the text before the first call; text after a
//     call is dropped once a call has a name. With a call, content is stripped and whitespace-only content is
//     dropped. Reasoning keeps its whitespace.
//   * `tool_choice: "none"` with tools: the calls' text is dropped from the content and no call is returned.
//
// The finish reason plays no part: a call cut off by max_tokens is still a call, as in vLLM.

#include "serve/request.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace sinfer::serve::vllm_qwen3 {

using TokenId = std::int32_t;

/// What the parser needs to know about the tokenizer.
struct Vocabulary {
    /// The grammar's token-id terminals; -1 when the vocabulary lacks one.
    TokenId think_start = -1;
    TokenId think_end   = -1;
    TokenId tool_start  = -1;
    TokenId tool_end    = -1;
    /// transformers' `all_special_tokens` minus the four above: id -> spelling. Dropped wherever they appear.
    std::unordered_map<TokenId, std::string> drop_tokens;
    /// `tokenizer.decode([id])`: one token's text, invalid UTF-8 replaced as Python replaces it.
    std::function<std::string(TokenId)> token_text;
};

struct Input {
    /// vLLM's `output.text`: the generated tokens decoded with their special tokens, the stop token that ended the
    /// generation excluded.
    std::string text;
    /// vLLM's `output.token_ids`: every generated token, the stop token included.
    std::vector<TokenId> token_ids;
    /// Each token's own text, as Vocabulary::token_text gives it; empty to ask the vocabulary.
    std::vector<std::string> token_texts;
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
