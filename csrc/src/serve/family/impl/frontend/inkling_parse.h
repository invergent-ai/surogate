#pragma once

// Inkling-2 frontend (ENGINE-INKLING lane f2): generated token ids -> (reasoning, content, tool calls, finish reason),
// parse-exact to vLLM's non-streaming chat response.
//
// The reference composes two passes of `vllm/parser/inkling.py::InklingParser` (vLLM's ParserEngine with the Inkling
// config) behind its DelegatingParser and runs them on the DECODED text (skip_special_tokens=False), not on token ids:
//   pass 1 (reasoning adapter, skip_tool_parsing): thinking blocks -> reasoning; everything else, tool blocks included
//          (their markers as text, the name header dropped), -> content;
//   pass 2 (tool adapter, from CONTENT): tool blocks -> calls (name = the first "name":"..." in the block's JSON, the
//          arguments = its verbatim "args" member, re-serialised only when the request's tool schema coerces a value),
//          the rest -> content;
// then the serving layer's tool_choice / include_reasoning / parallel_tool_calls rules and finish reason. Seven
// named special tokens are dropped from the text; marker spellings typed as plain text act as markers. This file
// ports exactly that for the Inkling config. Test: sinfer_inkling_frontend_test over vectors/parser.

#include "family/impl/frontend/inkling_json.h"
#include "family/impl/frontend/inkling_render.h"

#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::family::frontend_internal::inkling {

struct ParsedToolCall {
    std::string name;
    std::string arguments;  // JSON text
};

struct ParsedOutput {
    std::optional<std::string> reasoning;
    std::optional<std::string> content;
    std::vector<ParsedToolCall> tool_calls;
    std::string finish_reason;
};

struct ParserVocabulary {
    SpecialTokens tokens;
    /// transformers' all_special_tokens minus the parser's own markers: dropped from the text (`<|endoftext|>`,
    /// `<|message_user|>`, `<|message_system|>`, `<|message_tool|>`, `<|content_image|>`, `<|content_xml|>`,
    /// `<|begin_of_text|>` for Inkling-2-Mini).
    std::vector<std::string> drop_spellings;

    /// `named_special` = the tokenizer's all_special_tokens spellings.
    static ParserVocabulary from(const SpecialTokens& tokens, const std::vector<std::string>& named_special);
};

/// The text vLLM's detokenizer yields for these output ids (special tokens spelled out; a request that stopped on a
/// stop token does not detokenize that last token; bytes of an unfinished UTF-8 character are held back and dropped
/// at the end; an invalid sequence becomes U+FFFD). `token_bytes(id)` = the token's raw bytes.
[[nodiscard]] std::string detokenize_output(std::span<const int> output_ids, bool stop_terminated,
                                            const std::function<std::string(int)>& token_bytes);

/// The parsed response message for a complete generation.
[[nodiscard]] ParsedOutput parse_output(std::string_view output_text, const ChatRequest& request,
                                        std::string_view engine_finish_reason, const ParserVocabulary& vocabulary);

/// usage.completion_tokens_details.reasoning_tokens: tokens inside thinking blocks.
[[nodiscard]] int count_reasoning_tokens(std::span<const int> output_ids, const SpecialTokens& tokens);

} // namespace sinfer::family::frontend_internal::inkling
