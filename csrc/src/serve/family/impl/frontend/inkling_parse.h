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
#include <memory>
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
/// The same for the generated tokens' bytes given one string per token, in order.
[[nodiscard]] std::string detokenize_output(std::span<const std::string> per_token_bytes, bool stop_terminated);

/// The parsed response message for a complete generation.
[[nodiscard]] ParsedOutput parse_output(std::string_view output_text, const ChatRequest& request,
                                        std::string_view engine_finish_reason, const ParserVocabulary& vocabulary);

/// usage.completion_tokens_details.reasoning_tokens: tokens inside thinking blocks.
[[nodiscard]] int count_reasoning_tokens(std::span<const int> output_ids, const SpecialTokens& tokens);

// ------------------------------------------------------------------ streaming (stream=true)
//
// StreamParser reads the reply while it is generated and hands out what may be sent now, under one invariant: every
// piece it returns is a prefix-extension of the FINAL reading (parse_output of the whole generation), so the
// concatenated stream equals the non-streamed message byte for byte. It runs the same two passes incrementally (an
// incremental lexer in front of the same state machine; vLLM's DelegatingParser reads the same events per delta) and
// holds back only what the final reading can still change:
//   - text that may be the start of a marker (`<|...`) or of a stop string;
//   - a block's header text (`<|message_model|>NAME<|content_...|>`: dropped or kept by the next marker);
//   - trailing whitespace of the reasoning (the reading rstrips it; released when more reasoning follows);
//   - content that is whitespace only so far (dropped before tool calls, kept otherwise), and, after a tool block
//     that followed whitespace-only content, the content after it until a call has a name (vLLM's deferred
//     content rule decides between the two readings only then);
//   - tool calls: a call goes out (index, name) as soon as its "name" is written; its arguments stream as written
//     when the request's schema for that tool cannot change them (no tools / no properties), and otherwise arrive
//     whole when the call's block closes -- schema coercion (ParserEngine._fix_arg_types) re-serialises the whole
//     argument object when it changes any value, which only the complete object can tell (vLLM streams the verbatim
//     text there and its stream then disagrees with its own non-streamed message).
// finish() returns the rest of the final reading.

/// One streamed tool-call piece; `index` is the call's position in the response's tool_calls.
struct StreamToolDelta {
    int index = 0;
    bool opens = false;                   // the call's first piece: carries its name (and the caller's id / type)
    std::string name;                     // when opens
    std::optional<std::string> arguments; // argument text appended by this piece
};

struct StreamDelta {
    std::string reasoning;
    std::string content;
    std::vector<StreamToolDelta> tool_calls;
    [[nodiscard]] bool empty() const noexcept { return reasoning.empty() && content.empty() && tool_calls.empty(); }
};

class StreamParser {
public:
    /// `request` and `vocabulary` must outlive the parser. `stop_strings` = the request's stop strings (text after
    /// the first occurrence of any of them is never sent before finish()).
    StreamParser(const ChatRequest& request, const ParserVocabulary& vocabulary,
                 std::vector<std::string> stop_strings = {});
    ~StreamParser();
    StreamParser(StreamParser&&) noexcept;
    StreamParser& operator=(StreamParser&&) noexcept;

    /// The next generated tokens, one string of raw bytes per token (special tokens spelled out), in order. The stop
    /// token that ends a generation is not fed (vLLM does not detokenize it).
    [[nodiscard]] StreamDelta feed(std::span<const std::string> token_bytes);

    /// The rest of `final` (= parse_output of the whole generation) after everything fed() returned. `consistent` (if
    /// given) is false when something already returned is not a prefix of `final` -- never, by construction; the
    /// streaming test checks it on every vector.
    [[nodiscard]] StreamDelta finish(const ParsedOutput& final, bool* consistent = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::family::frontend_internal::inkling
