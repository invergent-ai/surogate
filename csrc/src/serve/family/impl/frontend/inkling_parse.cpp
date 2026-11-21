#include "family/impl/frontend/inkling_parse.h"

#include <algorithm>
#include <cmath>
#include <tuple>
#include <utility>

namespace sinfer::family::frontend_internal::inkling {
namespace {

// ------------------------------------------------------------------ the Inkling ParserEngineConfig, as a table

enum class St { Content, Reasoning, Header, ToolArgs };
enum class Term { MsgModel, TextStart, ThinkStart, ThinkEnd, EndSampling, ToolStart, ToolText, ToolError, Drop };
enum class Ev { Text, ReasoningStart, ReasoningChunk, ReasoningEnd, ToolCallStart, ArgChunk, ToolCallEnd };

struct Event {
    Ev type;
    std::string value;
    int tool_index = -1;
};

struct Transition {
    St next;
    std::vector<Ev> events;
};

// vllm/parser/inkling.py inkling_config().transitions
std::optional<Transition> transition(St s, Term t) {
    using E = Ev;
    switch (s) {
    case St::Content:
    case St::Header: {
        const bool header = s == St::Header;
        switch (t) {
        case Term::MsgModel: return Transition{St::Header, {}};
        case Term::TextStart: return Transition{St::Content, {E::ReasoningEnd}};
        case Term::ThinkStart: return Transition{St::Reasoning, {E::ReasoningStart}};
        case Term::ToolStart: return Transition{St::ToolArgs, {E::ReasoningEnd, E::ToolCallStart}};
        case Term::ToolText: return Transition{St::Content, {E::ReasoningEnd}};
        case Term::ToolError: return Transition{St::Content, {E::ReasoningEnd}};
        case Term::ThinkEnd:
        case Term::EndSampling:
            return header ? Transition{St::Content, {E::Text}} : Transition{St::Content, {}};
        default: return std::nullopt;
        }
    }
    case St::Reasoning:
        switch (t) {
        case Term::ThinkStart: return Transition{St::Reasoning, {}};
        case Term::ToolStart: return Transition{St::ToolArgs, {E::ReasoningEnd, E::ToolCallStart}};
        case Term::ThinkEnd: return Transition{St::Content, {}};
        case Term::EndSampling: return Transition{St::Content, {E::ReasoningEnd}};
        default: return std::nullopt;
        }
    case St::ToolArgs:
        switch (t) {
        case Term::ThinkEnd:
        case Term::EndSampling: return Transition{St::Content, {E::ToolCallEnd}};
        default: return std::nullopt;
        }
    }
    return std::nullopt;
}

// Terminals of transitions into or out of the tool state; the exits are the ones leaving it.
bool is_tool_terminal(Term t) { return t == Term::ToolStart || t == Term::ThinkEnd || t == Term::EndSampling; }
bool is_tool_exit(Term t) { return t == Term::ThinkEnd || t == Term::EndSampling; }

struct Literal {
    std::string text;
    Term term;
};

struct Lexeme {
    bool terminal = false;
    Term term = Term::Drop;
    std::string text;
};

// IncrementalLexer over a complete text: every literal is "<|...|>" and none is a prefix of another, so lexing is
// "match a literal at each position, else content" (content splits only where a literal starts).
std::vector<Lexeme> lex(std::string_view text, const std::vector<Literal>& literals) {
    std::vector<Lexeme> out;
    std::string content;
    std::size_t i = 0;
    while (i < text.size()) {
        if (text[i] == '<') {
            const Literal* hit = nullptr;
            for (const Literal& lit : literals) {
                if (text.substr(i, lit.text.size()) == lit.text && (hit == nullptr || lit.text.size() > hit->text.size())) {
                    hit = &lit;
                }
            }
            if (hit != nullptr) {
                if (!content.empty()) {
                    out.push_back({false, Term::Drop, std::move(content)});
                    content.clear();
                }
                out.push_back({true, hit->term, hit->text});
                i += hit->text.size();
                continue;
            }
        }
        content.push_back(text[i]);
        ++i;
    }
    if (!content.empty()) { out.push_back({false, Term::Drop, std::move(content)}); }
    return out;
}

// StreamingParserEngine for the Inkling config (no token ids, no JSON collection, tool_args_json; the argument
// stream's held-back braces only re-chunk the same characters, so whole runs are emitted).
class Engine {
public:
    Engine(St initial, bool skip_tool_parsing) : state_(initial), skip_tool_(skip_tool_parsing) {}

    std::vector<Event> run(std::string_view text, const std::vector<Literal>& literals) {
        for (Lexeme& lx : lex(text, literals)) {
            if (lx.terminal) {
                on_terminal(lx.term, lx.text);
            } else if (!lx.text.empty()) {
                emit_for_state(lx.text);
            }
        }
        finish();
        return std::move(events_);
    }

private:
    St state_;
    bool skip_tool_;
    int tool_index_ = -1;
    std::string header_;
    bool in_skipped_span_ = false;
    std::vector<Event> events_;

    void push(Ev type, std::string value) { events_.push_back({type, std::move(value), tool_index_}); }

    void emit_for_state(const std::string& text) {
        switch (state_) {
        case St::Header: header_ += text; return;
        case St::ToolArgs: push(Ev::ArgChunk, text); return;
        case St::Content: push(Ev::Text, text); return;
        case St::Reasoning: push(Ev::ReasoningChunk, text); return;
        }
    }

    void on_terminal(Term term, const std::string& value) {
        const std::optional<Transition> tr = transition(state_, term);
        if (!tr) {
            if (term == Term::Drop) { return; }
            if (skip_tool_ && is_tool_exit(term)) { in_skipped_span_ = false; }
            emit_for_state(value);
            return;
        }
        if (skip_tool_ && is_tool_terminal(term)) {
            const bool is_opener = tr->next == St::ToolArgs;
            const bool is_exit = is_tool_exit(term);
            const bool used_as_plain_closer = is_exit && !is_opener && !in_skipped_span_;
            if (!used_as_plain_closer) {
                if (is_opener) {
                    in_skipped_span_ = true;
                } else if (is_exit) {
                    in_skipped_span_ = false;
                }
                const bool leaving_header = state_ == St::Header;
                if (leaving_header) { header_.clear(); }
                if (std::find(tr->events.begin(), tr->events.end(), Ev::ReasoningEnd) != tr->events.end()) {
                    state_ = St::Content;
                    push(Ev::ReasoningEnd, value);
                    push(Ev::Text, value);
                    return;
                }
                if (leaving_header) {
                    state_ = St::Content;
                    push(Ev::Text, value);
                    return;
                }
                switch (state_) {
                case St::Content: push(Ev::Text, value); return;
                case St::Reasoning: push(Ev::ReasoningChunk, value); return;
                case St::ToolArgs: push(Ev::ArgChunk, value); return;
                case St::Header: return;
                }
                return;
            }
        }
        apply(*tr, value);
    }

    void apply(const Transition& tr, const std::string& value) {
        const St previous = state_;
        std::string header;
        if (previous == St::Header) {
            header = std::move(header_);
            header_.clear();
        }
        state_ = tr.next;
        for (const Ev e : tr.events) {
            if (e == Ev::ToolCallStart) { ++tool_index_; }
            std::string v = (previous == St::Header && e == Ev::Text) ? header : value;
            if (e == Ev::Text && v.empty()) { continue; }
            push(e, std::move(v));
        }
    }

    void finish() {
        if (state_ == St::ToolArgs) {
            if (tool_index_ >= 0) { push(Ev::ToolCallEnd, ""); }
        } else if (state_ == St::Reasoning) {
            push(Ev::ReasoningEnd, "");
        } else if (state_ == St::Header) {
            if (!header_.empty()) {
                push(Ev::Text, std::move(header_));
                header_.clear();
            }
        }
        state_ = St::Content;
    }
};

// ------------------------------------------------------------------ tool-call JSON helpers (vllm/parser/utils.py)

bool json_ws(char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; }

std::optional<std::size_t> scan_json_object_end(std::string_view raw, std::size_t start) {
    int depth = 0;
    bool in_string = false, escape = false;
    for (std::size_t i = start; i < raw.size(); ++i) {
        const char ch = raw[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (ch == '\\') { escape = true; }
            else if (ch == '"') { in_string = false; }
            continue;
        }
        if (ch == '"') { in_string = true; }
        else if (ch == '{') { ++depth; }
        else if (ch == '}') {
            if (--depth == 0) { return i + 1; }
        }
    }
    return std::nullopt;
}

// extract_json_member_object(raw, key): the verbatim span; nullopt when the value has not started; throws
// RequestError (the reference's ValueError) when the member is not an object.
std::optional<std::string> extract_json_member_object(std::string_view raw, std::string_view key) {
    int depth = 0;
    bool in_string = false, escape = false;
    std::size_t string_start = 0;
    std::optional<std::string_view> last_string;
    for (std::size_t i = 0; i < raw.size(); ++i) {
        const char ch = raw[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (in_string) {
            if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
                if (depth == 1) { last_string = raw.substr(string_start + 1, i - string_start - 1); }
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
            string_start = i;
        } else if (ch == ':' && depth == 1 && last_string && *last_string == key) {
            std::size_t value_start = i + 1;
            while (value_start < raw.size() && json_ws(raw[value_start])) { ++value_start; }
            if (value_start >= raw.size()) { return std::nullopt; }
            if (raw[value_start] != '{') { throw RequestError("member must be a JSON object"); }
            const auto end = scan_json_object_end(raw, value_start);
            if (!end) { return std::string(raw.substr(value_start)); }
            return std::string(raw.substr(value_start, *end - value_start));
        } else if (ch == '{' || ch == '[') {
            ++depth;
        } else if (ch == '}' || ch == ']') {
            --depth;
        }
    }
    return std::nullopt;
}

// The leftmost match of "name"\s*:\s*"([^"]*)" (ParserEngine._NAME_RE), group 1.
std::optional<std::string> search_name(std::string_view args) {
    static constexpr std::string_view kKey = "\"name\"";
    std::size_t from = 0;
    while (true) {
        const std::size_t at = args.find(kKey, from);
        if (at == std::string_view::npos) { return std::nullopt; }
        std::size_t i = at + kKey.size();
        const auto skip_space = [&] {  // \s*, one code point at a time
            while (i < args.size()) {
                const auto b = static_cast<unsigned char>(args[i]);
                std::size_t len = b < 0x80 ? 1 : (b >> 5) == 6 ? 2 : (b >> 4) == 14 ? 3 : 4;
                len = std::min(len, args.size() - i);
                if (!py_strip(args.substr(i, len)).empty()) { break; }
                i += len;
            }
        };
        skip_space();
        if (i < args.size() && args[i] == ':') {
            ++i;
            skip_space();
            if (i < args.size() && args[i] == '"') {
                const std::size_t close = args.find('"', i + 1);
                if (close != std::string_view::npos) { return std::string(args.substr(i + 1, close - i - 1)); }
            }
        }
        from = at + 1;
    }
}

// ------------------------------------------------------------------ schema coercion (ParserEngine._fix_arg_types)

void collect_types(const PyValue& schema, std::vector<std::string>& types) {
    if (!schema.is_object()) {
        types.emplace_back("string");
        return;
    }
    if (const PyValue* t = schema.find("type")) {
        if (t->is_string()) {
            types.push_back(t->text);
        } else if (t->is_array()) {
            for (const PyValue& x : t->items) {
                if (x.is_string()) { types.push_back(x.text); }
            }
        }
    }
    if (const PyValue* e = schema.find("enum"); e && e->is_array() && !e->items.empty()) {
        for (const PyValue& v : e->items) {
            switch (v.kind) {
            case PyValue::Kind::Null: types.emplace_back("null"); break;
            case PyValue::Kind::Bool: types.emplace_back("boolean"); break;
            case PyValue::Kind::Int: types.emplace_back("integer"); break;
            case PyValue::Kind::Float: types.emplace_back("number"); break;
            case PyValue::Kind::String: types.emplace_back("string"); break;
            case PyValue::Kind::Array: types.emplace_back("array"); break;
            case PyValue::Kind::Object: types.emplace_back("object"); break;
            }
        }
    }
    for (const char* field : {"anyOf", "oneOf", "allOf"}) {
        if (const PyValue* choices = schema.find(field); choices && choices->is_array()) {
            for (const PyValue& c : choices->items) {
                std::vector<std::string> sub;
                collect_types(c, sub);
                if (sub.empty()) { sub.emplace_back("string"); }
                types.insert(types.end(), sub.begin(), sub.end());
            }
        }
    }
}

std::vector<std::string> extract_types_from_schema(const PyValue& schema) {
    std::vector<std::string> types;
    collect_types(schema, types);
    if (types.empty()) { types.emplace_back("string"); }
    return types;
}

std::string normalize_type(const std::string& t) {
    static const std::pair<std::string_view, std::string_view> kAliases[] = {
        {"str", "string"}, {"text", "string"}, {"varchar", "string"}, {"char", "string"}, {"enum", "string"},
        {"int", "integer"}, {"int32", "integer"}, {"int64", "integer"}, {"uint", "integer"}, {"uint32", "integer"},
        {"uint64", "integer"}, {"long", "integer"}, {"short", "integer"}, {"unsigned", "integer"},
        {"float", "number"}, {"float32", "number"}, {"float64", "number"}, {"double", "number"},
        {"bool", "boolean"}, {"dict", "object"}, {"arr", "array"}, {"list", "array"}, {"sequence", "array"}};
    const std::string key = ascii_lower(py_strip(t));
    for (const auto& [from, to] : kAliases) {
        if (key == from) { return std::string(to); }
    }
    return key;
}

bool json_finite(const PyValue& v) {
    switch (v.kind) {
    case PyValue::Kind::Float: return std::isfinite(v.number);
    case PyValue::Kind::Array:
    case PyValue::Kind::Object:
        return std::all_of(v.items.begin(), v.items.end(), json_finite);
    default: return true;
    }
}

std::optional<PyValue> try_json_loads(std::string_view text) {
    try {
        return py_json_loads(text);
    } catch (const RequestError&) {
        return std::nullopt;
    }
}

struct Coerced {
    PyValue value;
    bool same_object = false;  // coerce_to_schema_type returned its input string itself
};

// coerce_to_schema_type(value, types)
Coerced coerce_to_schema_type(const std::string& value, const std::vector<std::string>& types) {
    std::vector<std::string> normalized;
    for (const std::string& t : types) { normalized.push_back(normalize_type(t)); }
    const auto has = [&](std::string_view t) { return std::find(normalized.begin(), normalized.end(), t) != normalized.end(); };
    for (const std::string_view candidate : {"null", "integer", "number", "boolean", "object", "array", "string"}) {
        if (!has(candidate)) { continue; }
        if (candidate == "null") {
            if (ascii_lower(value) == "null") { return {PyValue::null()}; }
            continue;
        }
        if (candidate == "string") { return {PyValue::from_string(value), true}; }
        if (candidate == "integer") {
            if (auto i = py_parse_int(value)) { return {PyValue::from_int_text(std::move(*i))}; }
            continue;
        }
        if (candidate == "number") {
            if (auto i = py_parse_int(value)) { return {PyValue::from_int_text(std::move(*i))}; }
            const auto f = py_parse_float(value);
            if (!f || !std::isfinite(*f)) { continue; }
            if (*f != std::trunc(*f)) { return {PyValue::from_float(*f)}; }
            return {PyValue::from_int_text(py_int_text_of_double(*f))};
        }
        if (candidate == "boolean") {
            const std::string lower = ascii_lower(py_strip(value));
            if (lower == "true" || lower == "1") { return {PyValue::from_bool(true)}; }
            if (lower == "false" || lower == "0") { return {PyValue::from_bool(false)}; }
            continue;
        }
        // object / array
        if (auto parsed = try_json_loads(value)) {
            if (json_finite(*parsed)) { return {std::move(*parsed)}; }
        }
    }
    auto parsed = try_json_loads(value);
    if (!parsed || !json_finite(*parsed)) { return {PyValue::from_string(value), true}; }
    return {std::move(*parsed)};
}

bool same_type_and_equal(const PyValue& a, const PyValue& b) {
    if (a.kind != b.kind) { return false; }
    switch (a.kind) {
    case PyValue::Kind::Null: return true;
    case PyValue::Kind::Bool: return a.boolean == b.boolean;
    case PyValue::Kind::Int: return a.text == b.text;
    case PyValue::Kind::Float: return a.number == b.number;
    case PyValue::Kind::String: return a.text == b.text;
    case PyValue::Kind::Array:
    case PyValue::Kind::Object:
        if (a.items.size() != b.items.size() || a.keys != b.keys) { return false; }
        for (std::size_t i = 0; i < a.items.size(); ++i) {
            if (!same_type_and_equal(a.items[i], b.items[i])) { return false; }
        }
        return true;
    }
    return false;
}

bool coerce_dict(PyValue& args, const PyValue& properties);

// ParserEngine._coerce_value; returns whether the value changed (and changes it in place).
bool coerce_value(PyValue& value, const PyValue& schema) {
    if (value.is_string()) {
        Coerced c = coerce_to_schema_type(value.text, extract_types_from_schema(schema));
        if (c.same_object) { return false; }
        value = std::move(c.value);
        return true;
    }
    if (value.is_object()) {
        const PyValue* nested = schema.find("properties");
        if (nested != nullptr && nested->is_object()) { return coerce_dict(value, *nested); }
        return false;
    }
    if (value.is_array()) {
        const PyValue* items = schema.find("items");
        if (items == nullptr || !items->is_object()) { return false; }
        bool changed = false;
        for (PyValue& item : value.items) { changed = coerce_value(item, *items) || changed; }
        return changed;
    }
    const std::string as_str = py_json_dumps_unicode(value);
    Coerced c = coerce_to_schema_type(as_str, extract_types_from_schema(schema));
    if (same_type_and_equal(c.value, value)) { return false; }
    value = std::move(c.value);
    return true;
}

bool coerce_dict(PyValue& args, const PyValue& properties) {
    bool changed = false;
    for (std::size_t i = 0; i < args.keys.size(); ++i) {
        const PyValue* prop = properties.find(args.keys[i]);
        if (prop == nullptr || !prop->is_object()) { continue; }
        changed = coerce_value(args.items[i], *prop) || changed;
    }
    return changed;
}

const PyValue* tool_properties(const ChatRequest& request, std::string_view name) {
    if (!request.tools) { return nullptr; }
    for (const PyValue& tool : request.tools->items) {
        const PyValue* fn = tool.find("function");
        const PyValue* n = fn ? fn->find("name") : nullptr;
        if (n == nullptr || !n->is_string() || n->text != name) { continue; }
        const PyValue* params = fn->find("parameters");
        if (params == nullptr || !params->is_object()) { return nullptr; }
        const PyValue* props = params->find("properties");
        return (props != nullptr && props->is_object()) ? props : nullptr;
    }
    return nullptr;
}

std::string fix_arg_types(const std::string& args_json, const std::string& name, const ChatRequest& request) {
    if (!request.tools || name.empty()) { return args_json; }
    auto args = try_json_loads(args_json);
    if (!args || !args->is_object()) { return args_json; }
    const PyValue* properties = tool_properties(request, name);
    if (properties == nullptr || properties->keys.empty()) { return args_json; }
    if (coerce_dict(*args, *properties)) { return py_json_dumps_unicode(*args); }
    return args_json;
}

// InklingParser._extract_args_value + ParserEngine._extract_name_and_args
std::pair<std::string, std::string> extract_name_and_args(std::string_view raw_body) {
    const std::string raw(py_strip(raw_body));
    auto parsed = try_json_loads(raw);
    if (!parsed || !parsed->is_object()) { return {"", raw}; }
    const PyValue* name_value = parsed->find("name");
    const std::string name = (name_value != nullptr && name_value->is_string()) ? name_value->text : std::string{};
    for (const char* key : {"args", "arguments", "parameters"}) {
        if (const PyValue* v = parsed->find(key)) {
            return {name, v->is_string() ? v->text : py_json_dumps_unicode(*v)};
        }
    }
    PyValue without = *parsed;
    without.erase("name");
    return {name, py_json_dumps_unicode(without)};
}

std::string extract_args_json(std::string_view raw) {
    if (py_strip(raw).empty()) { return "{}"; }
    return extract_name_and_args(raw).second;
}

std::optional<std::string> strip_content_whitespace(const std::string& content, bool tools_called) {
    if (tools_called && py_strip(content).empty()) { return std::nullopt; }  // drop_whitespace_only_content_before_tools
    if (content.empty()) { return std::nullopt; }
    return content;
}

struct ToolInfo {
    bool tools_called = false;
    std::vector<ParsedToolCall> calls;
    std::optional<std::string> content;
};

struct Slot {
    std::string args;
    std::string name;
    bool name_sent = false;
};

// ParserEngine.extract_tool_calls_from_content (InklingParser._single_pass_parse from CONTENT)
ToolInfo tool_pass(std::string_view content, const ChatRequest& request, const std::vector<Literal>& literals) {
    const bool suppress = request.tool_choice == ToolChoice::None && request.tools.has_value();
    const std::vector<Event> events = Engine(St::Content, /*skip_tool_parsing=*/false).run(content, literals);

    // _events_to_delta(events, finished=True)
    std::vector<Slot> slots;
    std::string deferred;
    std::string content_str;
    bool seen_tool_event = false;
    bool any_delta = false;
    const auto ensure = [&](int idx) {
        while (static_cast<int>(slots.size()) <= idx) { slots.emplace_back(); }
    };
    for (const Event& e : events) {
        switch (e.type) {
        case Ev::Text:
            if (seen_tool_event) {
                deferred += e.value;
            } else {
                content_str += e.value;
            }
            break;
        case Ev::ToolCallStart:
            if (!suppress) {
                seen_tool_event = true;
                ensure(e.tool_index);
            }
            break;
        case Ev::ArgChunk:
            if (!suppress) {
                seen_tool_event = true;
                Slot& slot = slots[static_cast<std::size_t>(e.tool_index)];
                if (!e.value.empty()) { slot.args += e.value; }
                if (!slot.name_sent && !e.value.empty()) {
                    const auto name = search_name(slot.args);
                    if (name && !name->empty()) {
                        slot.name = *name;
                        slot.name_sent = true;
                        any_delta = true;
                    }
                }
            }
            break;
        case Ev::ToolCallEnd:
            if (!suppress) {
                seen_tool_event = true;
                if (e.tool_index >= 0 && static_cast<std::size_t>(e.tool_index) < slots.size()) {
                    Slot& slot = slots[static_cast<std::size_t>(e.tool_index)];
                    if (!slot.name_sent) {
                        std::string name = slot.name;
                        if (name.empty()) { name = search_name(slot.args).value_or(""); }
                        if (!name.empty()) {
                            slot.name = name;
                            slot.name_sent = true;
                            any_delta = true;
                        }
                    }
                }
            }
            break;
        default: break;
        }
    }
    if (!deferred.empty() && (!seen_tool_event || !any_delta)) {
        content_str += deferred;
        deferred.clear();
    }
    if (!content_str.empty() && py_strip(content_str).empty() && !slots.empty()) { content_str.clear(); }
    std::optional<std::string> delta_content;
    if (!content_str.empty()) { delta_content = content_str; }

    // _build_extracted_result()
    ToolInfo info;
    for (const Slot& slot : slots) {
        if (slot.name.empty() && slot.args.empty()) { continue; }
        std::string name(py_strip(slot.name));
        std::string args_json;
        if (name.empty() && !py_strip(slot.args).empty()) {
            std::tie(name, args_json) = extract_name_and_args(slot.args);
        } else if (!py_strip(slot.args).empty()) {
            try {
                const auto span = extract_json_member_object(slot.args, "args");
                args_json = span ? *span : "{}";
            } catch (const RequestError&) {
                args_json = extract_args_json(slot.args);
            }
        } else {
            args_json = "{}";
        }
        if (!name.empty()) {
            info.calls.push_back({name, fix_arg_types(args_json, name, request)});
        }
    }
    info.tools_called = !info.calls.empty();

    std::optional<std::string> parsed_content;
    if (delta_content) { parsed_content = strip_content_whitespace(*delta_content, info.tools_called); }
    if (!deferred.empty()) {  // InklingParser._single_pass_parse: flush text after the tool blocks
        info.content = strip_content_whitespace(parsed_content.value_or("") + deferred, info.tools_called);
    }
    if (parsed_content && !info.content) { info.content = parsed_content; }
    return info;
}

std::optional<std::string> nonempty(const std::string& s) {
    if (s.empty()) { return std::nullopt; }
    return s;
}

} // namespace

ParserVocabulary ParserVocabulary::from(const SpecialTokens& tokens, const std::vector<std::string>& named_special) {
    static const char* const kMarkers[] = {"<|message_model|>", "<|content_text|>", "<|content_thinking|>",
                                           "<|end_message|>", "<|content_model_end_sampling|>",
                                           "<|content_invoke_tool_json|>", "<|content_invoke_tool_text|>",
                                           "<|content_tool_error|>"};
    ParserVocabulary v;
    v.tokens = tokens;
    for (const std::string& s : named_special) {
        if (s.empty() || std::find(std::begin(kMarkers), std::end(kMarkers), s) != std::end(kMarkers)) { continue; }
        if (std::find(v.drop_spellings.begin(), v.drop_spellings.end(), s) == v.drop_spellings.end()) {
            v.drop_spellings.push_back(s);
        }
    }
    return v;
}

std::string detokenize_output(std::span<const int> output_ids, bool stop_terminated,
                              const std::function<std::string(int)>& token_bytes) {
    std::size_t n = output_ids.size();
    if (stop_terminated && n > 0) { --n; }
    std::string out;
    std::string pending;
    for (std::size_t k = 0; k < n; ++k) {
        pending += token_bytes(output_ids[k]);
        // Lossy-decode the pending bytes (maximal invalid subparts -> U+FFFD); hold them while the result ends in U+FFFD.
        std::string decoded;
        bool ends_replacement = false;
        std::size_t i = 0;
        while (i < pending.size()) {
            const auto b0 = static_cast<unsigned char>(pending[i]);
            std::size_t need = 0;
            std::uint32_t lo = 0x80, hi = 0xBF;
            if (b0 < 0x80) {
                decoded.push_back(static_cast<char>(b0));
                ++i;
                ends_replacement = false;
                continue;
            }
            if (b0 >= 0xC2 && b0 <= 0xDF) { need = 1; }
            else if (b0 == 0xE0) { need = 2; lo = 0xA0; }
            else if (b0 >= 0xE1 && b0 <= 0xEC) { need = 2; }
            else if (b0 == 0xED) { need = 2; hi = 0x9F; }
            else if (b0 >= 0xEE && b0 <= 0xEF) { need = 2; }
            else if (b0 == 0xF0) { need = 3; lo = 0x90; }
            else if (b0 >= 0xF1 && b0 <= 0xF3) { need = 3; }
            else if (b0 == 0xF4) { need = 3; hi = 0x8F; }
            std::size_t j = i + 1;
            bool ok = need > 0;
            for (std::size_t c = 0; ok && c < need; ++c, ++j) {
                if (j >= pending.size()) {
                    ok = false;
                    break;
                }
                const auto b = static_cast<unsigned char>(pending[j]);
                const std::uint32_t l = c == 0 ? lo : 0x80, h = c == 0 ? hi : 0xBF;
                if (b < l || b > h) {
                    ok = false;
                    break;
                }
            }
            if (ok) {
                decoded.append(pending, i, need + 1);
                i += need + 1;
                ends_replacement = false;
            } else {
                decoded += "\xEF\xBF\xBD";
                i = std::max(j, i + 1);
                ends_replacement = true;
            }
        }
        if (!ends_replacement) {
            out += decoded;
            pending.clear();
        }
    }
    return out;
}

int count_reasoning_tokens(std::span<const int> output_ids, const SpecialTokens& t) {
    bool in_reasoning = false;
    int count = 0;
    for (const int id : output_ids) {
        if (id == t.content_thinking) {
            in_reasoning = true;
            continue;
        }
        if (id == t.end_message || id == t.content_model_end_sampling) {
            in_reasoning = false;
            continue;
        }
        if (in_reasoning) { ++count; }
    }
    return count;
}

ParsedOutput parse_output(std::string_view output_text, const ChatRequest& request,
                          std::string_view engine_finish_reason, const ParserVocabulary& vocabulary) {
    std::vector<Literal> literals = {
        {"<|message_model|>", Term::MsgModel},         {"<|content_text|>", Term::TextStart},
        {"<|content_thinking|>", Term::ThinkStart},    {"<|end_message|>", Term::ThinkEnd},
        {"<|content_model_end_sampling|>", Term::EndSampling},
        {"<|content_invoke_tool_json|>", Term::ToolStart}, {"<|content_invoke_tool_text|>", Term::ToolText},
        {"<|content_tool_error|>", Term::ToolError}};
    for (const std::string& s : vocabulary.drop_spellings) { literals.push_back({s, Term::Drop}); }

    // pass 1: the reasoning adapter's extract_reasoning (skip_tool_parsing, the config's MESSAGE_HEADER start)
    std::string reasoning_raw, content_raw;
    for (const Event& e : Engine(St::Header, /*skip_tool_parsing=*/true).run(output_text, literals)) {
        if (e.type == Ev::ReasoningChunk) { reasoning_raw += e.value; }
        else if (e.type == Ev::Text) { content_raw += e.value; }
    }
    std::optional<std::string> reasoning = nonempty(std::string(py_rstrip(reasoning_raw)));
    std::optional<std::string> content = nonempty(content_raw);

    // DelegatingParser._extract_tool_calls (enable_auto_tools; Inkling's tool parser supports no required / named)
    std::vector<ParsedToolCall> calls;
    if (request.tool_choice == ToolChoice::None) {
        const ToolInfo info = tool_pass(content.value_or(""), request, literals);
        content = info.content;
    } else {
        const ToolInfo info = tool_pass(content.value_or(""), request, literals);
        if (info.tools_called) {
            calls = info.calls;
            content = info.content;
            if (content && !content->empty() && py_strip(*content).empty()) { content.reset(); }
        } else {
            const bool forced = request.tool_choice == ToolChoice::Required || request.tool_choice == ToolChoice::Named;
            if (forced && (!content || py_strip(*content).empty())) {
                content.reset();
            } else {
                content = info.content ? nonempty(*info.content) : std::nullopt;
            }
        }
    }

    // assemble_chat_message_fields + resolve_finish_reason
    ParsedOutput out;
    out.reasoning = request.include_reasoning ? reasoning : std::nullopt;
    const bool forced = request.tool_choice == ToolChoice::Required || request.tool_choice == ToolChoice::Named;
    const bool keeps = forced || (request.tool_choice == ToolChoice::Auto && request.tools && !request.tools->items.empty());
    out.content = content;
    if (keeps) {
        if (forced && !out.content) { out.content = std::string(); }
        out.tool_calls = std::move(calls);
        if (!request.parallel_tool_calls && out.tool_calls.size() > 1) { out.tool_calls.resize(1); }
    }
    out.finish_reason = engine_finish_reason.empty() ? "stop" : std::string(engine_finish_reason);
    if (!out.tool_calls.empty() && out.finish_reason == "stop" && request.tool_choice != ToolChoice::Named) {
        out.finish_reason = "tool_calls";
    }
    return out;
}

} // namespace sinfer::family::frontend_internal::inkling
