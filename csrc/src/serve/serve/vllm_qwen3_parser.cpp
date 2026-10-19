// vLLM 0.31's qwen3 parser (reasoning + qwen3_coder tool calls), non-streaming (serve/vllm_qwen3_parser.h).

#include "serve/vllm_qwen3_parser.h"

#include "serve/vllm_tool_arguments.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <string_view>
#include <utility>

namespace sinfer::serve::vllm_qwen3 {
namespace {

using Json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------------------------------------------
// Terminals (vllm/parser/qwen3.py qwen3_config) and the states of vllm/parser/engine/parser_engine_config.py.

enum class Terminal {
    Content, // the lexer's __CONTENT__
    ThinkStart,
    ThinkEnd,
    ToolStart,
    ToolEnd,
    FuncPrefix,
    FuncEnd,
    ParamStart,
    ParamEnd,
    CloseAngle,
    Drop, // a special token the grammar does not name (token_id_scanner.DROP_TERMINAL)
};

enum class State { Content, Reasoning, ToolPreamble, ToolName, ToolArgs, ToolBetween };

bool is_tool_state(State state) {
    return state == State::ToolPreamble || state == State::ToolName || state == State::ToolArgs ||
           state == State::ToolBetween;
}

/// The terminals vLLM matches by token id. Once a request carries token ids (always, when served), a lexed spelling
/// of one of them is text (StreamingParserEngine._process_lex_tokens, `strict`).
bool is_token_id_terminal(Terminal terminal) {
    return terminal == Terminal::ThinkStart || terminal == Terminal::ThinkEnd || terminal == Terminal::ToolStart ||
           terminal == Terminal::ToolEnd || terminal == Terminal::Drop;
}

enum class Event { Text, Reasoning, ReasoningEnd, ToolCallStart, ToolName, ArgValue, ToolCallEnd };

struct Transition {
    State next;
    std::array<Event, 2> events{};
    int event_count = 0;
};

/// qwen3_config's transition table; nullopt where it has no entry.
std::optional<Transition> transition(State state, Terminal terminal) {
    using T = Terminal;
    using S = State;
    using E = Event;
    switch (state) {
    case S::Reasoning:
        if (terminal == T::ThinkStart) { return Transition{S::Reasoning}; }
        if (terminal == T::ThinkEnd) { return Transition{S::Content, {E::ReasoningEnd}, 1}; }
        if (terminal == T::ToolStart) { return Transition{S::ToolPreamble, {E::ReasoningEnd, E::ToolCallStart}, 2}; }
        break;
    case S::Content:
        if (terminal == T::ThinkEnd) { return Transition{S::Content}; } // a repeated </think> is absorbed
        if (terminal == T::ToolStart) { return Transition{S::ToolPreamble, {E::ReasoningEnd, E::ToolCallStart}, 2}; }
        if (terminal == T::FuncPrefix) { return Transition{S::ToolName, {E::ToolCallStart}, 1}; }
        break;
    case S::ToolPreamble:
        if (terminal == T::ToolEnd) { return Transition{S::Content, {E::ToolCallEnd}, 1}; }
        if (terminal == T::FuncPrefix) { return Transition{S::ToolName}; }
        break;
    case S::ToolName:
        if (terminal == T::CloseAngle) { return Transition{S::ToolArgs}; }
        if (terminal == T::FuncEnd) { return Transition{S::ToolBetween, {E::ToolCallEnd}, 1}; }
        break;
    case S::ToolArgs:
        if (terminal == T::FuncEnd) { return Transition{S::ToolBetween, {E::ToolCallEnd}, 1}; }
        if (terminal == T::ParamStart || terminal == T::ParamEnd) { return Transition{S::ToolArgs, {E::ArgValue}, 1}; }
        break;
    case S::ToolBetween:
        if (terminal == T::ToolEnd) { return Transition{S::Content}; }
        if (terminal == T::ToolStart) { return Transition{S::ToolPreamble, {E::ToolCallStart}, 1}; }
        if (terminal == T::FuncPrefix) { return Transition{S::ToolName, {E::ToolCallStart}, 1}; }
        break;
    }
    return std::nullopt;
}

// ---------------------------------------------------------------------------------------------------------------
// vllm/parser/engine/incremental_lexer.py, over UTF-8 bytes (every literal is ASCII, so byte positions split the
// text where Python's code-point positions do).

struct Literal {
    std::string text;
    Terminal terminal;
};

struct LexToken {
    Terminal terminal;
    std::string value;
};

class Lexer {
public:
    explicit Lexer(std::vector<Literal> literals) : literals_(std::move(literals)) {
        for (const auto& literal : literals_) {
            first_chars_[static_cast<unsigned char>(literal.text.front())] = true;
            for (std::size_t i = 1; i < literal.text.size(); ++i) { prefixes_.push_back(literal.text.substr(0, i)); }
        }
        std::sort(prefixes_.begin(), prefixes_.end());
    }

    std::vector<LexToken> feed(std::string_view text) {
        buffer_.append(text);
        std::vector<LexToken> out;
        drain(out, false);
        return out;
    }

    std::vector<LexToken> flush() {
        std::vector<LexToken> out;
        if (!buffer_.empty()) { drain(out, true); }
        if (!buffer_.empty()) {
            out.push_back({Terminal::Content, std::move(buffer_)});
            buffer_.clear();
        }
        return out;
    }

private:
    bool is_first_char(char c) const { return first_chars_[static_cast<unsigned char>(c)]; }

    bool buffer_is_prefix() const { return std::binary_search(prefixes_.begin(), prefixes_.end(), buffer_); }

    void drain(std::vector<LexToken>& out, bool final) {
        while (!buffer_.empty()) {
            if (std::none_of(buffer_.begin(), buffer_.end(), [&](char c) { return is_first_char(c); })) {
                out.push_back({Terminal::Content, std::move(buffer_)});
                buffer_.clear();
                break;
            }
            const Literal* best = nullptr;
            for (const auto& literal : literals_) {
                if (literal.text.front() == buffer_.front() && buffer_.starts_with(literal.text) &&
                    (best == nullptr || literal.text.size() > best->text.size())) {
                    best = &literal;
                }
            }
            // A buffer that may still grow into a literal waits for more text, unless this is the last of it.
            if (!final && buffer_is_prefix()) {
                if (best == nullptr) { break; }
                const bool longer = std::any_of(literals_.begin(), literals_.end(), [&](const Literal& literal) {
                    return literal.text.size() > best->text.size() && literal.text.starts_with(buffer_);
                });
                if (longer) { break; }
            }
            if (best != nullptr) {
                out.push_back({best->terminal, best->text});
                buffer_.erase(0, best->text.size());
                continue;
            }
            const std::size_t end = content_boundary();
            out.push_back({Terminal::Content, buffer_.substr(0, end)});
            buffer_.erase(0, end);
        }
    }

    /// _find_content_boundary: the first position after the start where a literal (or the start of one, at the end
    /// of the buffer) begins.
    std::size_t content_boundary() const {
        const std::size_t n = buffer_.size();
        for (std::size_t i = 1; i < n; ++i) {
            if (!is_first_char(buffer_[i])) { continue; }
            for (const auto& literal : literals_) {
                const std::size_t k = std::min(n - i, literal.text.size());
                if (std::string_view(buffer_).substr(i, k) == std::string_view(literal.text).substr(0, k)) { return i; }
            }
        }
        return n;
    }

    std::vector<Literal> literals_;
    std::array<bool, 256> first_chars_{};
    std::vector<std::string> prefixes_;
    std::string buffer_;
};

// ---------------------------------------------------------------------------------------------------------------
// vllm/parser/engine/token_id_scanner.py, for one delta holding the whole output.

struct Item {
    bool terminal = false; // a PreLexedTerminal; else a TextChunk
    Terminal kind  = Terminal::Content;
    std::string text;
};

struct Scan {
    std::vector<Item> items;
    /// What the scanner holds back until finish(): a text that waited for a terminal's spelling, then those
    /// terminals (flush_pending).
    std::string deferred_text;
    std::vector<Item> deferred_terminals;
};

/// _rebuild_from_anchors: the terminals placed in the text from the right, each at the rightmost spelling of it
/// before the next one; the text between them taken from the output itself.
void rebuild_from_anchors(const std::string& text, const std::vector<Item>& results, Scan& scan) {
    std::vector<const Item*> anchors;
    for (const auto& item : results) {
        if (item.terminal) { anchors.push_back(&item); }
    }
    if (anchors.empty()) {
        scan.items.push_back({false, Terminal::Content, text});
        return;
    }
    std::vector<std::ptrdiff_t> positions(anchors.size(), -1);
    std::size_t search_end = text.size();
    for (std::size_t i = anchors.size(); i-- != 0;) {
        const std::string& spelling = anchors[i]->text;
        const std::size_t pos = spelling.size() <= search_end
                                    ? std::string_view(text).substr(0, search_end).rfind(spelling)
                                    : std::string_view::npos;
        if (pos != std::string_view::npos) {
            positions[i] = static_cast<std::ptrdiff_t>(pos);
            search_end   = pos;
        }
    }
    std::size_t consumed = 0;
    for (std::size_t i = 0; i < anchors.size(); ++i) {
        const std::ptrdiff_t pos = positions[i];
        if (pos >= static_cast<std::ptrdiff_t>(consumed)) {
            if (static_cast<std::size_t>(pos) > consumed) {
                scan.items.push_back({false, Terminal::Content, text.substr(consumed, pos - consumed)});
            }
            scan.items.push_back(*anchors[i]);
            consumed = static_cast<std::size_t>(pos) + anchors[i]->text.size();
        } else {
            const bool later = std::any_of(positions.begin() + static_cast<std::ptrdiff_t>(i) + 1, positions.end(),
                                           [](std::ptrdiff_t p) { return p >= 0; });
            if (!later && consumed < text.size() && anchors[i]->kind != Terminal::Drop) {
                scan.deferred_text += text.substr(consumed);
                consumed = text.size();
            }
            scan.deferred_terminals.push_back(*anchors[i]);
        }
    }
    if (consumed < text.size()) { scan.items.push_back({false, Terminal::Content, text.substr(consumed)}); }
}

Scan scan_output(const Input& input, const Vocabulary& vocabulary) {
    Scan scan;
    const auto terminal_of = [&](TokenId id) -> std::optional<Terminal> {
        if (id == vocabulary.think_start) { return Terminal::ThinkStart; }
        if (id == vocabulary.think_end) { return Terminal::ThinkEnd; }
        if (id == vocabulary.tool_start) { return Terminal::ToolStart; }
        if (id == vocabulary.tool_end) { return Terminal::ToolEnd; }
        if (vocabulary.drop_tokens.contains(id)) { return Terminal::Drop; }
        return std::nullopt;
    };
    const bool has_special = std::any_of(input.token_ids.begin(), input.token_ids.end(),
                                         [&](TokenId id) { return id >= 0 && terminal_of(id).has_value(); });
    if (!has_special) {
        if (!input.text.empty()) { scan.items.push_back({false, Terminal::Content, input.text}); }
        return scan;
    }
    std::vector<Item> results;
    std::string run;
    for (std::size_t i = 0; i < input.token_ids.size(); ++i) {
        const TokenId id = input.token_ids[i];
        const std::string piece =
            i < input.token_texts.size() ? input.token_texts[i] : vocabulary.token_text(id);
        if (const auto terminal = terminal_of(id)) {
            if (!run.empty()) { results.push_back({false, Terminal::Content, std::move(run)}); }
            run.clear();
            results.push_back({true, *terminal, piece});
        } else {
            run += piece;
        }
    }
    if (!run.empty()) { results.push_back({false, Terminal::Content, std::move(run)}); }
    if (input.text.empty()) {
        // No text to place the terminals in: they wait for finish().
        for (auto& item : results) {
            if (item.terminal) { scan.deferred_terminals.push_back(std::move(item)); }
        }
        return scan;
    }
    // _recover_holdback_text
    std::string reconstructed;
    for (const auto& item : results) { reconstructed += item.text; }
    if (reconstructed.empty()) {
        scan.items.push_back({false, Terminal::Content, input.text});
        scan.items.insert(scan.items.end(), results.begin(), results.end());
        return scan;
    }
    const std::size_t pos = input.text.find(reconstructed);
    if (pos == 0) {
        scan.items = std::move(results);
    } else if (pos != std::string::npos) {
        scan.items.push_back({false, Terminal::Content, input.text.substr(0, pos)});
        scan.items.insert(scan.items.end(), results.begin(), results.end());
    } else {
        rebuild_from_anchors(input.text, results, scan);
    }
    return scan;
}

// ---------------------------------------------------------------------------------------------------------------
// StreamingParserEngine (state machine) and ParserEngine (events -> result), one pass over the whole output.

struct Slot {
    std::string name;
    std::string args;
    bool name_sent = false;
};

/// ParserEngine._NAME_RE = r'"name"\s*:\s*"([^"]*)"', searched in a call's body.
std::optional<std::string> name_in_body(std::string_view body) {
    for (std::size_t at = body.find("\"name\""); at != std::string_view::npos; at = body.find("\"name\"", at + 1)) {
        std::size_t pos = at + 6;
        while (const std::size_t size = vllm_args::regex_space_size(body, pos)) { pos += size; }
        if (pos >= body.size() || body[pos] != ':') { continue; }
        ++pos;
        while (const std::size_t size = vllm_args::regex_space_size(body, pos)) { pos += size; }
        if (pos >= body.size() || body[pos] != '"') { continue; }
        const std::size_t close = body.find('"', pos + 1);
        if (close == std::string_view::npos) { continue; }
        std::string name(body.substr(pos + 1, close - pos - 1));
        if (name.empty()) { return std::nullopt; } // the first match decides
        return name;
    }
    return std::nullopt;
}

class Pass {
public:
    Pass(const Input& input, const std::vector<Literal>& literals)
        : state_(input.thinking ? State::Reasoning : State::Content), strict_(!input.token_ids.empty()),
          suppress_(input.suppress_tool_calls), lexer_(literals) {}

    void process(const std::vector<Item>& items) {
        for (const auto& item : items) {
            if (item.terminal) {
                lex(lexer_.flush());
                on_terminal(item.kind, item.text);
            } else if (!item.text.empty()) {
                lex(lexer_.feed(item.text));
            }
        }
    }

    void finish(Scan& scan) {
        std::vector<Item> pending;
        if (!scan.deferred_text.empty()) { pending.push_back({false, Terminal::Content, scan.deferred_text}); }
        pending.insert(pending.end(), scan.deferred_terminals.begin(), scan.deferred_terminals.end());
        process(pending);
        lex(lexer_.flush());
        if (is_tool_state(state_)) {
            if (tool_index_ >= 0) { on_event(Event::ToolCallEnd, {}); }
            state_ = State::Content;
        } else if (state_ == State::Reasoning) {
            on_event(Event::ReasoningEnd, {});
            state_ = State::Content;
        }
    }

    Result result(const std::vector<ToolDefinition>& tools) {
        // _events_to_delta(finished=True)
        if (!deferred_.empty() && (!seen_tool_event_ || !tool_delta_)) { content_ += deferred_; }
        if (!content_.empty() && vllm_args::python_strip(content_).empty() && !slots_.empty()) { content_.clear(); }
        Result out;
        if (!reasoning_.empty()) { out.reasoning = reasoning_; }
        // _build_extracted_result
        for (const auto& slot : slots_) {
            if (slot.name.empty() && slot.args.empty()) { continue; }
            std::optional<std::string> name = vllm_args::python_strip(slot.name);
            std::string arguments;
            const bool has_body = !vllm_args::python_strip(slot.args).empty();
            bool json_body = false;
            if (name->empty() && has_body) {
                auto [json_name, json_args] = vllm_args::extract_json_name_and_args(slot.args);
                name      = std::move(json_name);
                arguments = std::move(json_args);
                json_body = true;
            } else if (has_body) {
                arguments = vllm_args::qwen3_arguments_json(vllm_args::extract_qwen3_parameters(slot.args),
                                                            properties_of(*name, tools));
            } else {
                arguments = "{}";
            }
            if (!name || name->empty()) { continue; }
            if (json_body) { arguments = vllm_args::fix_arg_types(arguments, properties_of(*name, tools)); }
            out.calls.push_back({std::move(*name), std::move(arguments)});
        }
        // _single_pass_parse: content stripped when a call was made.
        if (!content_.empty()) {
            std::string content = out.calls.empty() ? content_ : vllm_args::python_strip(content_);
            if (!content.empty()) { out.content = std::move(content); }
        }
        return out;
    }

private:
    /// find_tool_properties: the first tool of that name, its parameters' `properties`, else none.
    static Json properties_of(const std::string& name, const std::vector<ToolDefinition>& tools) {
        for (const auto& tool : tools) {
            if (tool.name != name) { continue; }
            const Json parameters = Json::parse(tool.parameters_json, nullptr, false);
            if (parameters.is_object()) {
                if (const auto properties = parameters.find("properties"); properties != parameters.end()) {
                    return *properties;
                }
            }
            return Json::object();
        }
        return Json::object();
    }

    void lex(const std::vector<LexToken>& tokens) {
        for (const auto& token : tokens) {
            if (token.terminal == Terminal::Content || (strict_ && is_token_id_terminal(token.terminal))) {
                if (!token.value.empty()) { emit_for_state(token.value); }
            } else {
                on_terminal(token.terminal, token.value);
            }
        }
    }

    void on_terminal(Terminal terminal, const std::string& value) {
        const std::optional<Transition> next = transition(state_, terminal);
        if (!next) {
            if (terminal == Terminal::Drop) { return; }
            emit_for_state(value);
            return;
        }
        state_ = next->next;
        for (int i = 0; i < next->event_count; ++i) {
            if (next->events[i] == Event::ToolCallStart) { ++tool_index_; }
            on_event(next->events[i], value);
        }
    }

    void emit_for_state(const std::string& text) {
        switch (state_) {
        case State::ToolArgs: on_event(Event::ArgValue, text); return;
        case State::Content: on_event(Event::Text, text); return;
        case State::Reasoning: on_event(Event::Reasoning, text); return;
        case State::ToolName: on_event(Event::ToolName, text); return;
        case State::ToolPreamble:
        case State::ToolBetween: return; // no content event in these states: the text is dropped
        }
    }

    void on_event(Event event, const std::string& value) {
        switch (event) {
        case Event::Text:
            (seen_tool_event_ ? deferred_ : content_) += value;
            return;
        case Event::Reasoning: reasoning_ += value; return;
        case Event::ReasoningEnd: return;
        case Event::ToolCallStart:
            if (suppress_) { return; }
            seen_tool_event_ = true;
            while (slots_.size() <= static_cast<std::size_t>(tool_index_)) { slots_.emplace_back(); }
            return;
        case Event::ToolName:
            if (suppress_) { return; }
            seen_tool_event_ = true;
            slots_[tool_index_].name += value;
            return;
        case Event::ArgValue: {
            if (suppress_) { return; }
            seen_tool_event_ = true;
            Slot& slot       = slots_[tool_index_];
            slot.args += value;
            if (!slot.name_sent) {
                if (!slot.name.empty()) {
                    send_name(slot, slot.name);
                } else if (!value.empty()) {
                    if (auto name = name_in_body(slot.args)) { send_name(slot, *name); }
                }
            }
            return;
        }
        case Event::ToolCallEnd: {
            if (suppress_) { return; }
            seen_tool_event_ = true;
            if (tool_index_ < 0 || static_cast<std::size_t>(tool_index_) >= slots_.size()) { return; }
            Slot& slot = slots_[tool_index_];
            if (!slot.name_sent) {
                std::string name = slot.name;
                if (name.empty()) { name = name_in_body(slot.args).value_or(""); }
                if (!name.empty()) { send_name(slot, name); }
            }
            return;
        }
        }
    }

    /// _emit_name_delta: a name that is not empty is accepted (validate_tool_names is off for qwen3).
    void send_name(Slot& slot, std::string name) {
        slot.name      = std::move(name);
        slot.name_sent = true;
        tool_delta_    = true;
    }

    State state_;
    bool strict_;
    bool suppress_;
    Lexer lexer_;
    int tool_index_ = -1;
    std::vector<Slot> slots_;
    std::string content_;
    std::string deferred_;
    std::string reasoning_;
    bool seen_tool_event_ = false;
    bool tool_delta_      = false;
};

std::vector<Literal> literals_for(const Vocabulary& vocabulary) {
    std::vector<Literal> out{
        {"<think>", Terminal::ThinkStart},       {"</think>", Terminal::ThinkEnd},
        {"<tool_call>", Terminal::ToolStart},    {"</tool_call>", Terminal::ToolEnd},
        {"<function=", Terminal::FuncPrefix},    {"</function>", Terminal::FuncEnd},
        {"<parameter=", Terminal::ParamStart},   {"</parameter>", Terminal::ParamEnd},
        {">", Terminal::CloseAngle},
    };
    for (const auto& [id, spelling] : vocabulary.drop_tokens) {
        if (spelling.empty()) { continue; }
        const bool configured = std::any_of(out.begin(), out.begin() + 9, [&](const Literal& l) { return l.text == spelling; });
        const bool seen = std::any_of(out.begin() + 9, out.end(), [&](const Literal& l) { return l.text == spelling; });
        if (!configured && !seen) { out.push_back({spelling, Terminal::Drop}); }
    }
    return out;
}

} // namespace

Result parse(const Input& input, const Vocabulary& vocabulary, const std::vector<ToolDefinition>& tools) {
    Scan scan = scan_output(input, vocabulary);
    Pass pass(input, literals_for(vocabulary));
    pass.process(scan.items);
    pass.finish(scan);
    return pass.result(tools);
}

std::string output_text(std::string_view bytes) {
    // A trailing incomplete sequence: a lead byte and fewer continuation bytes than it announces.
    std::size_t start = bytes.size();
    for (std::size_t back = 1; back <= 3 && back <= bytes.size(); ++back) {
        const auto b = static_cast<unsigned char>(bytes[bytes.size() - back]);
        if ((b & 0xC0) == 0x80) { continue; }
        const std::size_t need = b >= 0xF0 && b <= 0xF4 ? 4 : b >= 0xE0 && b <= 0xEF ? 3 : b >= 0xC2 && b <= 0xDF ? 2 : 0;
        if (need > back) { start = bytes.size() - back; }
        break;
    }
    if (start < bytes.size() && utf8_replace(bytes.substr(start)) == "\xEF\xBF\xBD") {
        return utf8_replace(bytes.substr(0, start));
    }
    return utf8_replace(bytes);
}

std::string utf8_replace(std::string_view bytes) {
    std::string out;
    out.reserve(bytes.size());
    std::size_t i = 0;
    const auto replacement = [&] { out += "\xEF\xBF\xBD"; };
    while (i < bytes.size()) {
        const auto b0 = static_cast<unsigned char>(bytes[i]);
        if (b0 < 0x80) {
            out += static_cast<char>(b0);
            ++i;
            continue;
        }
        std::size_t need = 0;
        unsigned char lo = 0x80, hi = 0xBF; // the allowed range of the second byte
        if (b0 >= 0xC2 && b0 <= 0xDF) {
            need = 1;
        } else if (b0 >= 0xE0 && b0 <= 0xEF) {
            need = 2;
            if (b0 == 0xE0) { lo = 0xA0; }
            if (b0 == 0xED) { hi = 0x9F; }
        } else if (b0 >= 0xF0 && b0 <= 0xF4) {
            need = 3;
            if (b0 == 0xF0) { lo = 0x90; }
            if (b0 == 0xF4) { hi = 0x8F; }
        } else {
            replacement(); // not a lead byte
            ++i;
            continue;
        }
        std::size_t j = 1;
        for (; j <= need; ++j) {
            if (i + j >= bytes.size()) { break; }
            const auto b = static_cast<unsigned char>(bytes[i + j]);
            const unsigned char low  = j == 1 ? lo : 0x80;
            const unsigned char high = j == 1 ? hi : 0xBF;
            if (b < low || b > high) { break; }
        }
        if (j > need) {
            out.append(bytes.substr(i, need + 1));
            i += need + 1;
        } else {
            replacement(); // the valid start of a sequence, as one character
            i += j;
        }
    }
    return out;
}

} // namespace sinfer::serve::vllm_qwen3
