#include "serve/tool_call_parser.h"

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

using Json = nlohmann::json;

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

std::string new_tool_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "call_%016llx",
                  static_cast<unsigned long long>(dist(rng)));
    return std::string(buf.data());
}

Json function_schema(std::string_view name, const std::vector<ToolDefinition>& tools) {
    for (const auto& tool : tools) {
        if (tool.name == name) {
            auto schema = Json::parse(tool.parameters_json, nullptr, false);
            return schema.is_object() ? schema : Json::object();
        }
    }
    return Json::object();
}

unsigned schema_types(const Json& param, const Json& root, unsigned depth = 0) {
    // Bits represent string, null, and other types. Resolve local definitions
    // and nullable unions without following external URLs or reference cycles.
    if (!param.is_object() || depth >= 16) { return 4; }
    if (param.contains("$ref")) {
        if (!param["$ref"].is_string()) { return 4; }
        const auto ref = param["$ref"].get<std::string>();
        if (ref.rfind("#/", 0) != 0) { return 4; }
        try { return schema_types(root.at(Json::json_pointer(ref.substr(1))), root, depth + 1); }
        catch (const Json::exception&) { return 4; }
    }
    const auto kind = [](const Json& type) { return type == "string" ? 1U : type == "null" ? 2U : 4U; };
    unsigned types = 0;
    if (param.contains("type")) {
        if (param["type"].is_array()) {
            for (const auto& type : param["type"]) { types |= kind(type); }
        } else { types |= kind(param["type"]); }
    }
    for (const char* variant : {"anyOf", "oneOf", "allOf"}) {
        if (param.contains(variant) && param[variant].is_array()) {
            for (const auto& option : param[variant]) { types |= schema_types(option, root, depth + 1); }
        }
    }
    return types;
}

Json argument_value(const std::string& raw, const Json& schema, const std::string& key) {
    auto value = Json::parse(raw, nullptr, false);
    if (schema.contains("properties") && schema["properties"].is_object() && schema["properties"].contains(key)) {
        const auto types = schema_types(schema["properties"][key], schema);
        if (types == 1 || (types == 3 && !value.is_null())) { return raw; }
    }
    return value.is_discarded() ? Json(raw) : value;
}

bool parse_parameter(std::string_view inner, std::size_t& pos, Json& args, const Json& schema) {
    constexpr std::string_view kParamOpen  = "<parameter=";
    constexpr std::string_view kParamClose = "</parameter>";
    if (!starts_with_at(inner, pos, kParamOpen)) { return false; }
    const std::size_t name_begin = pos + kParamOpen.size();
    const std::size_t name_end   = inner.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string key       = std::string(inner.substr(name_begin, name_end - name_begin));
    if (args.contains(key)) { return false; }
    pos                         = name_end + 1;
    const std::size_t value_end = inner.find(kParamClose, pos);
    if (value_end == std::string_view::npos) { return false; }
    auto value = inner.substr(pos, value_end - pos);
    if (value.starts_with('\n')) { value.remove_prefix(1); }
    if (value.ends_with('\n')) { value.remove_suffix(1); }
    const std::string raw_value(value);
    args[key] = argument_value(raw_value, schema, key);
    pos                         = value_end + kParamClose.size();
    return true;
}

bool parse_one_tool_call(std::string_view block, std::size_t max_name_length, ToolCall& out,
                        const std::vector<ToolDefinition>& tools) {
    // Qwen3/Hermes checkpoints use JSON; Qwen3.5 and the native renderer use
    // function/parameter XML. Both have the same outer tool_call delimiters.
    auto json = Json::parse(block, nullptr, false);
    if (json.is_object() && json.contains("name") && json["name"].is_string()) {
        const auto name = json["name"].get<std::string>();
        auto args = json.value("arguments", Json::object());
        if (args.is_string()) { args = Json::parse(args.get<std::string>(), nullptr, false); }
        if (!valid_function_name(name, max_name_length) || !args.is_object()) { return false; }
        out.id = new_tool_call_id(); out.name = name; out.arguments_json = args.dump();
        return true;
    }
    constexpr std::string_view kFunctionOpen  = "<function=";
    constexpr std::string_view kFunctionClose = "</function>";
    std::size_t pos                           = 0;
    skip_ws(block, pos);
    if (!starts_with_at(block, pos, kFunctionOpen)) { return false; }
    const std::size_t name_begin = pos + kFunctionOpen.size();
    const std::size_t name_end   = block.find('>', name_begin);
    if (name_end == std::string_view::npos || name_end == name_begin) { return false; }
    const std::string name = std::string(block.substr(name_begin, name_end - name_begin));
    if (!valid_function_name(name, max_name_length)) { return false; }
    pos = name_end + 1;

    const std::size_t function_end = block.find(kFunctionClose, pos);
    if (function_end == std::string_view::npos) { return false; }
    const std::string_view params = block.substr(pos, function_end - pos);
    Json args                     = Json::object();
    const Json schema = function_schema(name, tools);
    std::size_t param_pos         = 0;
    for (;;) {
        skip_ws(params, param_pos);
        if (param_pos >= params.size()) { break; }
        if (!parse_parameter(params, param_pos, args, schema)) { return false; }
    }

    pos = function_end + kFunctionClose.size();
    skip_ws(block, pos);
    if (pos != block.size()) { return false; }

    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
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

ParsedToolCallOutput fallback(const std::string& text) {
    ParsedToolCallOutput out;
    out.content = text;
    return out;
}

} // namespace

static ParsedToolCallOutput parse_tagged_tool_call_output(const std::string& text,
                                                 std::size_t max_tool_name_length, bool spark,
                                                 const std::vector<ToolDefinition>& tools) {
    constexpr std::string_view kToolOpen  = "<tool_call>";
    constexpr std::string_view kToolClose = "</tool_call>";

    const std::size_t first = text.find(kToolOpen);
    if (first == std::string::npos) { return fallback(text); }

    ParsedToolCallOutput out;
    out.content = rtrim_ascii(std::string_view(text).substr(0, first));

    std::size_t pos = first;
    while (pos < text.size()) {
        skip_ws(text, pos);
        if (pos >= text.size()) { break; }
        if (!starts_with_at(text, pos, kToolOpen)) { return fallback(text); }
        const std::size_t inner_begin = pos + kToolOpen.size();
        const std::size_t close       = text.find(kToolClose, inner_begin);
        if (close == std::string::npos) { return fallback(text); }
        ToolCall call;
        const auto parser = spark ? parse_spark_call : parse_one_tool_call;
        if (!parser(std::string_view(text).substr(inner_begin, close - inner_begin),
                                 max_tool_name_length, call, tools)) {
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
                                                const std::vector<ToolDefinition>& tools) {
    return parse_tagged_tool_call_output(text, max_name, false, tools);
}

ParsedToolCallOutput parse_spark_tool_call_output(const std::string& text, std::size_t max_name,
                                                 const std::vector<ToolDefinition>& tools) {
    return parse_tagged_tool_call_output(text, max_name, true, tools);
}

std::string ToolCallStreamFilter::feed(std::string_view text) {
    if (finished_) { throw std::logic_error("tool-call stream filter is already finished"); }
    if (text.empty()) { return {}; }
    if (saw_tool_marker_) {
        tool_region_.append(text);
        return {};
    }

    constexpr std::string_view kToolOpen = "<tool_call>";
    pending_.append(text);
    const std::size_t marker = pending_.find(kToolOpen);
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

    const std::size_t prefix = longest_suffix_prefix(pending_, kToolOpen);
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
