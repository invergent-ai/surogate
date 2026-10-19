#include "serve/tool_call_parser.h"

#include "serve/tool_constraints.h"
#include "serve/vllm_tool_arguments.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string_view>

namespace sinfer::serve {
namespace {

// Ordered: a call's arguments are returned in the order the model wrote them, as vLLM returns
// them. A sorted map reordered them alphabetically, and the client sends that order back in the
// conversation history, where the template renders it to the model as its own earlier call.
using Json = nlohmann::ordered_json;

std::string trim_ascii(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(begin, end - begin));
}

std::string rtrim_ascii(std::string_view text) {
    std::size_t end = text.size();
    while (end != 0 && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return std::string(text.substr(0, end));
}

void skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])) != 0) { ++pos; }
}

bool starts_with_at(std::string_view text, std::size_t pos, std::string_view prefix) {
    return pos <= text.size() && text.substr(pos, prefix.size()) == prefix;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker) {
    const std::size_t maximum = std::min(text.size(), marker.size() - 1);
    for (std::size_t size = maximum; size != 0; --size) {
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

bool valid_function_name(std::string_view name, std::size_t max_name_length) {
    if (name.empty() || name.size() > max_name_length) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-' && c != '.') { return false; }
    }
    return true;
}

} // namespace

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

namespace {

Json function_schema(std::string_view name, const std::vector<ToolDefinition>& tools) {
    for (const auto& tool : tools) {
        if (tool.name == name) {
            auto schema = Json::parse(tool.parameters_json, nullptr, false);
            return schema.is_object() ? schema : Json::object();
        }
    }
    return Json::object();
}

unsigned schema_types(const Json& param, const Json& root, const std::string& raw,
                      const Json& parsed, unsigned depth = 0) {
    // Bits represent string, null, and other types. Resolve local definitions
    // and nullable unions without following external URLs or reference cycles.
    if (depth >= 16) { return 7; }
    if (!param.is_object()) { return param == false ? 0 : 7; }
    unsigned types = 7;
    if (param.contains("$ref")) {
        if (!param["$ref"].is_string()) { return 7; }
        const auto ref = param["$ref"].get<std::string>();
        if (ref.rfind("#/", 0) != 0) { return 7; }
        try { types &= schema_types(root.at(Json::json_pointer(ref.substr(1))), root, raw, parsed, depth + 1); }
        catch (const Json::exception&) { return 7; }
    }
    const auto kind = [](const Json& type) { return type == "string" ? 1U : type == "null" ? 2U : 4U; };
    if (param.contains("type")) {
        unsigned declared = 0;
        if (param["type"].is_array()) {
            for (const auto& type : param["type"]) { declared |= kind(type); }
        } else { declared = kind(param["type"]); }
        types &= declared;
    }
    // Enumerations can mix strings with JSON literals. Consider the two actual
    // interpretations, so enum ["123", 456] preserves 123 as text but parses 456.
    const auto literal_kind = [&](const Json& literal) {
        if (literal.is_string()) { return literal == raw ? 1U : 0U; }
        if (parsed.is_discarded() || literal != parsed) { return 0U; }
        return literal.is_null() ? 2U : 4U;
    };
    if (param.contains("const")) { types &= literal_kind(param["const"]); }
    if (param.contains("enum") && param["enum"].is_array()) {
        unsigned members = 0;
        for (const auto& literal : param["enum"]) { members |= literal_kind(literal); }
        types &= members;
    }
    for (const char* variant : {"anyOf", "oneOf", "allOf"}) {
        if (param.contains(variant) && param[variant].is_array()) {
            const bool all = std::string_view(variant) == "allOf";
            unsigned branches = all ? 7 : 0;
            for (const auto& option : param[variant]) {
                const auto branch = schema_types(option, root, raw, parsed, depth + 1);
                if (all) { branches &= branch; } else { branches |= branch; }
            }
            types &= branches;
        }
    }
    return types;
}

Json argument_value(const std::string& raw, const Json& schema, const std::string& key) {
    auto value = Json::parse(raw, nullptr, false);
    if (schema.contains("properties") && schema["properties"].is_object() && schema["properties"].contains(key)) {
        const auto types = schema_types(schema["properties"][key], schema, raw, value);
        if (types == 1 || (types == 3 && !value.is_null())) { return raw; }
    }
    return value.is_discarded() ? Json(raw) : value;
}

constexpr std::string_view kFunctionOpen  = "<function=";
constexpr std::string_view kFunctionClose = "</function>";

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length, ToolCall& out,
                        const std::vector<ToolDefinition>& tools, bool constrained) {
    // Qwen3/Hermes checkpoints use JSON; Qwen3.5 and the native renderer use
    // function/parameter XML. Both have the same outer tool_call delimiters.
    auto json = Json::parse(block, nullptr, false);
    if (json.is_object() && json.contains("name") && json["name"].is_string()) {
        const auto name = json["name"].get<std::string>();
        auto args = json.value("arguments", Json::object());
        if (args.is_string()) { args = Json::parse(args.get<std::string>(), nullptr, false); }
        if (!valid_function_name(name, max_name_length) || !args.is_object()) { return false; }
        out.id = new_tool_call_id(); out.name = name; out.arguments_json = vllm_args::python_json_dumps(args);
        return true;
    }
    std::size_t pos = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos) { return false; }
    // vLLM reads the name up to the first '>' and strips it.
    const std::string name = vllm_args::python_strip(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const vllm_args::Parameters parameters =
        vllm_args::extract_qwen3_parameters(block.substr(pos, function_end - pos));
    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    // The values as vLLM types them (serve/vllm_tool_arguments.h).
    const Json schema = function_schema(name, tools);
    out.arguments_json = vllm_args::qwen3_arguments_json(
        parameters, schema.contains("properties") ? schema.at("properties") : Json());
    if (constrained) {
        // The grammar wrote each value as the schema types it: a string as its bytes, anything else as JSON.
        // vLLM's reading loses that where the schema is one it does not read ($ref, const, an enum mixing
        // strings with numbers); keep the schema's reading then, so a strict call still matches its schema.
        const auto tool = std::find_if(tools.begin(), tools.end(), [&](const auto& item) { return item.name == name; });
        if (tool != tools.end() && !tool_arguments_match_schema(*tool, out.arguments_json)) {
            Json args = Json::object();
            for (const auto& [key, raw] : parameters) { args[key] = argument_value(raw, schema, key); }
            out.arguments_json = vllm_args::python_json_dumps(args);
        }
    }
    out.id   = new_tool_call_id();
    out.name = name;
    return true;
}

bool parse_spark_call(std::string_view block, std::size_t max_name_length, ToolCall& out,
                      const std::vector<ToolDefinition>& tools) {
    constexpr std::string_view key_open = "<arg_key>", key_close = "</arg_key>";
    constexpr std::string_view value_open = "<arg_value>", value_close = "</arg_value>";
    const auto name_end = block.find('<');
    const auto name = trim_ascii(block.substr(0, name_end));
    if (!valid_function_name(name, max_name_length)) { return false; }
    Json args = Json::object();
    const Json schema = function_schema(name, tools);
    std::size_t pos = name_end == std::string_view::npos ? block.size() : name_end;
    while (pos < block.size()) {
        skip_ws(block, pos);
        if (pos == block.size()) { break; }
        if (!starts_with_at(block, pos, key_open)) { return false; }
        pos += key_open.size();
        const auto end_key = block.find(key_close, pos);
        if (end_key == std::string_view::npos) { return false; }
        const auto key = trim_ascii(block.substr(pos, end_key - pos));
        if (key.empty() || args.contains(key)) { return false; }
        pos = end_key + key_close.size();
        skip_ws(block, pos);
        if (!starts_with_at(block, pos, value_open)) { return false; }
        pos += value_open.size();
        const auto end_value = block.find(value_close, pos);
        if (end_value == std::string_view::npos) { return false; }
        const auto raw = std::string(block.substr(pos, end_value - pos));
        args[key] = argument_value(raw, schema, key);
        pos = end_value + value_close.size();
    }
    out.id = new_tool_call_id(); out.name = name; out.arguments_json = args.dump();
    return true;
}

// Gemma 4 writes a call in its chat template's own notation rather than JSON (format_argument):
// a string between `<|"|>` marks, true or false, an object in braces with bare keys, a list in
// brackets, and anything else as Jinja prints it -- numbers, and Python's None for null.
constexpr std::string_view kGemmaQuote = "<|\"|>";
constexpr unsigned kGemmaMaxDepth      = 32;

bool parse_gemma_value(std::string_view text, std::size_t& pos, Json& out, std::string* bare,
                       unsigned depth);

// A key is bare (`city:`) as the template writes call arguments, or quoted (`<|"|>city<|"|>:`) as
// it writes them inside a tool's declaration; the model may copy either.
bool parse_gemma_key(std::string_view text, std::size_t& pos, std::string& key) {
    skip_ws(text, pos);
    if (starts_with_at(text, pos, kGemmaQuote)) {
        const std::size_t begin = pos + kGemmaQuote.size();
        const std::size_t end   = text.find(kGemmaQuote, begin);
        if (end == std::string_view::npos) { return false; }
        key = std::string(text.substr(begin, end - begin));
        pos = end + kGemmaQuote.size();
    } else {
        const std::size_t begin = pos;
        while (pos < text.size() && std::string_view(":,{}[]<").find(text[pos]) == std::string_view::npos) {
            ++pos;
        }
        key = trim_ascii(text.substr(begin, pos - begin));
    }
    skip_ws(text, pos);
    if (key.empty() || pos >= text.size() || text[pos] != ':') { return false; }
    ++pos;
    return true;
}

// `{key:value,...}` from its opening brace. At the top level (`schema` set) a bare value is typed
// by the tool's schema as the XML parsers type theirs: `code:123` stays text for a string field.
bool parse_gemma_object(std::string_view text, std::size_t& pos, Json& out, unsigned depth,
                        const Json* schema = nullptr) {
    if (depth > kGemmaMaxDepth || pos >= text.size() || text[pos] != '{') { return false; }
    ++pos;
    out = Json::object();
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == '}') {
        ++pos;
        return true;
    }
    for (;;) {
        std::string key;
        if (!parse_gemma_key(text, pos, key) || out.contains(key)) { return false; }
        Json value;
        std::string bare;
        if (!parse_gemma_value(text, pos, value, &bare, depth + 1)) { return false; }
        out[key] = schema != nullptr && !bare.empty() ? argument_value(bare, *schema, key)
                                                     : std::move(value);
        skip_ws(text, pos);
        if (pos >= text.size()) { return false; }
        if (text[pos] == ',') {
            ++pos;
            continue;
        }
        if (text[pos] != '}') { return false; }
        ++pos;
        return true;
    }
}

bool parse_gemma_value(std::string_view text, std::size_t& pos, Json& out, std::string* bare,
                       unsigned depth) {
    skip_ws(text, pos);
    if (pos >= text.size() || depth > kGemmaMaxDepth) { return false; }
    if (starts_with_at(text, pos, kGemmaQuote)) {
        const std::size_t begin = pos + kGemmaQuote.size();
        const std::size_t end   = text.find(kGemmaQuote, begin);
        if (end == std::string_view::npos) { return false; }
        out = std::string(text.substr(begin, end - begin));
        pos = end + kGemmaQuote.size();
        return true;
    }
    if (text[pos] == '{') { return parse_gemma_object(text, pos, out, depth); }
    if (text[pos] == '[') {
        ++pos;
        out = Json::array();
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return true;
        }
        for (;;) {
            Json item;
            if (!parse_gemma_value(text, pos, item, nullptr, depth + 1)) { return false; }
            out.push_back(std::move(item));
            skip_ws(text, pos);
            if (pos >= text.size()) { return false; }
            if (text[pos] == ',') {
                ++pos;
                continue;
            }
            if (text[pos] != ']') { return false; }
            ++pos;
            return true;
        }
    }
    if (text[pos] == '"') {
        // A JSON string the model wrote instead of the template's quote marks: read it whole, so a
        // comma or brace inside it does not end the value.
        std::size_t end = pos + 1;
        while (end < text.size() && text[end] != '"') { end += text[end] == '\\' ? 2 : 1; }
        if (end >= text.size()) { return false; }
        out = Json::parse(text.substr(pos, end + 1 - pos), nullptr, false);
        if (!out.is_string()) { return false; }
        pos = end + 1;
        return true;
    }
    // A bare scalar runs to the next delimiter.
    const std::size_t begin = pos;
    while (pos < text.size() && std::string_view(",}]<").find(text[pos]) == std::string_view::npos) {
        ++pos;
    }
    const std::string raw = trim_ascii(text.substr(begin, pos - begin));
    if (raw.empty()) { return false; }
    if (raw == "None" || raw == "True" || raw == "False") {
        out = raw == "None" ? Json(nullptr) : Json(raw == "True");
    } else {
        out = Json::parse(raw, nullptr, false);
        if (out.is_discarded() || out.is_structured()) { out = raw; }
    }
    if (bare != nullptr) { *bare = raw == "None" ? "null" : raw; }
    return true;
}

// The body between `<|tool_call>` and `<tool_call|>`: `call:NAME{arguments}`.
bool parse_gemma_call(std::string_view block, std::size_t max_name_length, ToolCall& out,
                      const std::vector<ToolDefinition>& tools) {
    constexpr std::string_view kCall = "call:";
    std::size_t pos = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kCall)) { return false; }
    pos += kCall.size();
    const std::size_t brace = block.find('{', pos);
    if (brace == std::string_view::npos) { return false; }
    const std::string name = trim_ascii(block.substr(pos, brace - pos));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos               = brace;
    const Json schema = function_schema(name, tools);
    Json args;
    if (!parse_gemma_object(block, pos, args, 0, &schema)) { return false; }
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }
    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

static ParsedToolCallOutput parse_tagged_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length, bool spark,
                                                 const std::vector<ToolDefinition>& tools,
                                                 bool constrained) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        const auto next = text.find(kToolOpen, pos);
        if (next == std::string::npos) {
            out.content += rtrim_ascii(std::string_view(text).substr(pos));
            break;
        }
        if (next != pos) {
            const auto between = std::string_view(text).substr(pos, next - pos);
            if (!trim_ascii(between).empty()) { out.content += between; }
        }
        const std::size_t inner_begin = next + kToolOpen.size();
        std::size_t close = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) { return fallback(text); }
        // A JSON string argument may itself contain the closing marker.
        std::size_t body_begin = inner_begin;
        skip_ws(text, body_begin);
        if (!spark && body_begin < text.size() && text[body_begin] == '{') {
            while (close != std::string::npos && Json::parse(text.substr(inner_begin, close - inner_begin), nullptr, false).is_discarded()) {
                close = text.find(kToolClose, close + kToolClose.size());
            }
            if (close == std::string::npos) { return fallback(text); }
        } else if (!spark && starts_with_at(text, body_begin, kFunctionOpen)) {
            // A parameter value may spell the closing marker itself (a file that holds a chat template). vLLM
            // reads such a value as text up to the call's </function>; close after it.
            const std::size_t name_end     = text.find('>', body_begin);
            const std::size_t function_end = name_end == std::string::npos ? std::string::npos
                                                                           : text.find(kFunctionClose, name_end);
            if (function_end != std::string::npos) {
                std::size_t after = function_end + kFunctionClose.size();
                skip_ws(text, after);
                if (starts_with_at(text, after, kToolClose)) { close = after; }
            }
        }
        ToolCall call;
        const std::string_view block = std::string_view(text).substr(inner_begin, close - inner_begin);
        if (spark ? !parse_spark_call(block, max_tool_name_length, call, tools)
                  : !parse_one_tool_call(block, max_tool_name_length, call, tools, constrained)) {
            return fallback(text);
        }
        out.tool_calls.push_back(std::move(call));
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

ParsedToolCallOutput parse_qwen_tool_call_output(const std::string& text, std::size_t max_name,
                                                const std::vector<ToolDefinition>& tools, bool constrained) {
    return parse_tagged_tool_call_output(text, max_name, false, tools, constrained);
}

ParsedToolCallOutput parse_spark_tool_call_output(const std::string& text, std::size_t max_name,
                                                 const std::vector<ToolDefinition>& tools) {
    return parse_tagged_tool_call_output(text, max_name, true, tools, false);
}

ParsedToolCallOutput parse_muse_tool_call_output(const std::string& text,
    std::size_t max_name, const std::vector<ToolDefinition>& tools) {
    constexpr std::string_view open = "<atem:function_calls>", close = "</atem:function_calls>";
    const auto first = text.find(open);
    if (first == std::string::npos) { return fallback(text); }
    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0,first));
    std::size_t pos=first;
    const auto attribute = [&](std::string_view tag, std::string& name) {
        if (!starts_with_at(text,pos,tag)) { return false; }
        pos += tag.size();
        const auto end = text.find("\">",pos);
        if (end == std::string::npos) { return false; }
        name = text.substr(pos,end-pos); pos=end+2;
        return !name.empty();
    };
    while (pos<text.size()) {
        skip_ws(text,pos);
        if (pos==text.size()) { break; }
        if (!starts_with_at(text,pos,open)) {
            out.content += text.substr(pos); break;
        }
        pos+=open.size();
        std::size_t count=0;
        for (;;) {
            skip_ws(text,pos);
            if (starts_with_at(text,pos,close)) { pos+=close.size(); break; }
            std::string name;
            if (!attribute("<atem:invoke name=\"",name)) { return fallback(text); }
            // Muse's template advertises bare tools as name.*, and the model may
            // emit name.name. Only collapse that exact registered doubled name.
            if (std::none_of(tools.begin(), tools.end(), [&](const auto& tool) { return tool.name == name; })) {
                for (const auto& tool : tools) {
                    if (name == tool.name + "." + tool.name && tool.name.find('.') == std::string::npos) {
                        name = tool.name; break;
                    }
                }
            }
            if (!valid_function_name(name,max_name)) { return fallback(text); }
            Json args=Json::object();
            const auto schema=function_schema(name,tools);
            for (;;) {
                skip_ws(text,pos);
                constexpr std::string_view end_invoke="</atem:invoke>", end_parameter="</atem:parameter>";
                if (starts_with_at(text,pos,end_invoke)) { pos+=end_invoke.size(); break; }
                std::string key;
                if (!attribute("<atem:parameter name=\"",key) || args.contains(key)) { return fallback(text); }
                const auto end=text.find(end_parameter,pos);
                if (end==std::string::npos) { return fallback(text); }
                // ATEM strings are literal parameter bodies, including boundary whitespace.
                args[key]=argument_value(text.substr(pos,end-pos),schema,key);
                pos=end+end_parameter.size();
            }
            out.tool_calls.push_back({.id=new_tool_call_id(),.name=name,.arguments_json=args.dump()});
            ++count;
        }
        if (!count) { return fallback(text); }
    }
    out.is_tool_call_response=!out.tool_calls.empty();
    return out;
}

ParsedToolCallOutput parse_gemma4_tool_call_output(const std::string& text, std::size_t max_name,
                                                   const std::vector<ToolDefinition>& tools) {
    constexpr std::string_view kToolOpen  = "<|tool_call>";
    constexpr std::string_view kToolClose = "<tool_call|>";
    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));
    std::size_t pos = first;
    while (pos < text.size()) {
        const auto next = text.find(kToolOpen, pos);
        if (next == std::string::npos) {
            out.content += rtrim_ascii(std::string_view(text).substr(pos));
            break;
        }
        if (next != pos) {
            const auto between = std::string_view(text).substr(pos, next - pos);
            if (!trim_ascii(between).empty()) { out.content += between; }
        }
        const std::size_t inner_begin = next + kToolOpen.size();
        // A string argument may itself spell the closing marker: take the first close after which
        // the call parses.
        ToolCall call;
        std::size_t close = text.find(kToolClose, inner_begin);
        while (close != std::string::npos &&
               !parse_gemma_call(std::string_view(text).substr(inner_begin, close - inner_begin),
                                 max_name, call, tools)) {
            close = text.find(kToolClose, close + kToolClose.size());
        }
        if (close == std::string::npos) { return fallback(text); }
        out.tool_calls.push_back(std::move(call));
        pos = close + kToolClose.size();
    }

    if (out.tool_calls.empty()) { return fallback(text); }
    out.is_tool_call_response = true;
    return out;
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    const std::string_view kToolOpen = muse_tools_    ? "<atem:function_calls>"
                                       : gemma_tools_ ? "<|tool_call>"
                                                      : "<tool_call>";
    pending_.append(text);
    std::size_t marker = pending_.find(kToolOpen);
    if (json_tools_) {
        for (const auto candidate : {"<|python_tag|>", "{", "["}) { marker = std::min(marker, pending_.find(candidate)); }
    }
    if (qwen3_coder_) { marker = std::min(marker, pending_.find("<function=")); }
    if (marker != std::string::npos) {
        std::size_t safe_end = marker;
        while (safe_end != 0 &&
               std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
            --safe_end;
        }
        std::string visible = pending_.substr(0, safe_end);
        tool_region_        = pending_.substr(safe_end);
        pending_.clear();
        saw_tool_marker_ = true;
        emitted_bytes_ += visible.size();
        return visible;
    }

    std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
    if (json_tools_) { prefix = std::max(prefix, longest_suffix_prefix(pending_, "<|python_tag|>")); }
    if (qwen3_coder_) { prefix = std::max(prefix, longest_suffix_prefix(pending_, "<function=")); }
    std::size_t safe_end     = pending_.size() - prefix;
    while (safe_end != 0 && std::isspace(static_cast<unsigned char>(pending_[safe_end - 1])) != 0) {
        --safe_end;
    }
    std::string visible = pending_.substr(0, safe_end);
    pending_.erase(0, safe_end);
    emitted_bytes_ += visible.size();
    return visible;
}

std::string ToolCallStreamFilter::finish(bool is_tool_call_response) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    finished_ = true;
    if (is_tool_call_response) {
        pending_.clear();
        tool_region_.clear();
        return {};
    }
    std::string tail = std::move(pending_);
    tail += tool_region_;
    tool_region_.clear();
    emitted_bytes_ += tail.size();
    return tail;
}

} // namespace sinfer::serve
