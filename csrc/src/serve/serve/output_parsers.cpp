// Named reasoning and tool-call parsers (serve/output_parsers.h).

#include "serve/output_parsers.h"

#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>

namespace sinfer::serve {
namespace {

using Json = nlohmann::json;

/// vLLM's parser names, mapped onto the formats the engine can actually decode.
/// Several names share a format because the models share a convention: qwen3 and
/// deepseek_r1 both fence reasoning with `<think>`, hermes and qwen3_xml both emit
/// the same `<tool_call>` block. Aliases are kept because deployments name the
/// model's parser, not ours.
struct ReasoningName {
    std::string_view name;
    ReasoningFormat format;
};

constexpr std::array<ReasoningName, 6> kReasoningNames{{
    {"none", ReasoningFormat::None},
    {"off", ReasoningFormat::None},
    {"qwen3", ReasoningFormat::ThinkTags},
    {"deepseek_r1", ReasoningFormat::ThinkTags},
    {"glm4_moe", ReasoningFormat::ThinkTags},
    {"think", ReasoningFormat::ThinkTags},
}};

struct ToolCallName {
    std::string_view name;
    ToolCallFormat format;
};

constexpr std::array<ToolCallName, 7> kToolCallNames{{
    {"none", ToolCallFormat::None},
    {"off", ToolCallFormat::None},
    {"qwen3_xml", ToolCallFormat::QwenXml},
    {"hermes", ToolCallFormat::QwenXml},
    {"spark25", ToolCallFormat::Spark25},
    {"llama3_json", ToolCallFormat::Llama3Json},
    {"llama4_json", ToolCallFormat::Llama3Json},
}};

template <class Table>
std::string names_of(const Table& table) {
    std::string out;
    for (const auto& entry : table) {
        if (!out.empty()) { out += ", "; }
        out += std::string(entry.name);
    }
    return out;
}

std::string rtrim(std::string text) {
    while (!text.empty() && (std::isspace(static_cast<unsigned char>(text.back())) != 0)) {
        text.pop_back();
    }
    return text;
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

/// One Llama-3 style call: `{"name": "f", "parameters": {...}}`. The spelling
/// `arguments` is accepted too — the fine-tunes disagree with the model card and
/// with each other, and both are unambiguous here.
bool parse_llama3_object(const Json& node, std::size_t max_name_length, ToolCall& out) {
    if (!node.is_object() || !node.contains("name") || !node.at("name").is_string()) {
        return false;
    }
    const std::string name = node.at("name").get<std::string>();
    if (!valid_function_name(name, max_name_length)) { return false; }

    Json args = Json::object();
    for (const char* key : {"parameters", "arguments"}) {
        if (node.contains(key) && node.at(key).is_object()) {
            args = node.at(key);
            break;
        }
    }
    out.id             = new_tool_call_id();
    out.name           = name;
    out.arguments_json = args.dump();
    return true;
}

/// Llama-3 emits the call as bare JSON, sometimes behind `<|python_tag|>` and
/// sometimes as a `;`-separated run of objects. Anything that does not parse is
/// returned as content, which is the same fallback rule the XML parser uses: a
/// model that was answering in prose must not have its answer eaten.
ParsedToolCalls parse_llama3_json(const std::string& text, std::size_t max_tool_name_length) {
    constexpr std::string_view kPythonTag = "<|python_tag|>";
    ParsedToolCalls fallback;
    fallback.content = text;

    std::string_view body(text);
    std::string leading;
    if (const std::size_t tag = body.find(kPythonTag); tag != std::string_view::npos) {
        leading = rtrim(std::string(body.substr(0, tag)));
        body    = body.substr(tag + kPythonTag.size());
    }

    // Trim surrounding whitespace; the payload must be the whole remainder or the
    // text is prose that merely contains a brace.
    while (!body.empty() && (std::isspace(static_cast<unsigned char>(body.front())) != 0)) {
        body.remove_prefix(1);
    }
    while (!body.empty() && (std::isspace(static_cast<unsigned char>(body.back())) != 0)) {
        body.remove_suffix(1);
    }
    if (body.empty() || body.front() != '{' && body.front() != '[') { return fallback; }

    ParsedToolCalls out;
    out.content = leading;

    // A JSON array of calls, a single object, or objects separated by ';'.
    std::vector<std::string> chunks;
    if (body.front() == '[') {
        chunks.emplace_back(body);
    } else {
        std::size_t begin = 0;
        int depth         = 0;
        bool in_string    = false;
        bool escaped      = false;
        for (std::size_t i = 0; i < body.size(); ++i) {
            const char c = body[i];
            if (in_string) {
                if (escaped) {
                    escaped = false;
                } else if (c == '\\') {
                    escaped = true;
                } else if (c == '"') {
                    in_string = false;
                }
                continue;
            }
            if (c == '"') {
                in_string = true;
            } else if (c == '{') {
                ++depth;
            } else if (c == '}') {
                if (--depth == 0) {
                    chunks.emplace_back(body.substr(begin, i + 1 - begin));
                    std::size_t next = i + 1;
                    while (next < body.size() &&
                           ((std::isspace(static_cast<unsigned char>(body[next])) != 0) ||
                            body[next] == ';' || body[next] == ',')) {
                        ++next;
                    }
                    begin = next;
                    i     = next - 1;
                }
            }
        }
        if (depth != 0 || begin != body.size()) { return fallback; }
    }

    for (const std::string& chunk : chunks) {
        Json parsed = Json::parse(chunk, nullptr, false);
        if (parsed.is_discarded()) { return fallback; }
        if (parsed.is_array()) {
            for (const Json& element : parsed) {
                ToolCall call;
                if (!parse_llama3_object(element, max_tool_name_length, call)) { return fallback; }
                out.tool_calls.push_back(std::move(call));
            }
            continue;
        }
        ToolCall call;
        if (!parse_llama3_object(parsed, max_tool_name_length, call)) { return fallback; }
        out.tool_calls.push_back(std::move(call));
    }

    if (out.tool_calls.empty()) { return fallback; }
    out.is_tool_call_response = true;
    return out;
}

} // namespace

ReasoningFormat parse_reasoning_format(std::string_view name) {
    for (const auto& entry : kReasoningNames) {
        if (entry.name == name) { return entry.format; }
    }
    throw std::invalid_argument("unknown reasoning parser '" + std::string(name) +
                                "'; supported: " + reasoning_parser_names());
}

ToolCallFormat parse_tool_call_format(std::string_view name) {
    for (const auto& entry : kToolCallNames) {
        if (entry.name == name) { return entry.format; }
    }
    throw std::invalid_argument("unknown tool call parser '" + std::string(name) +
                                "'; supported: " + tool_call_parser_names());
}

std::string reasoning_parser_names() { return names_of(kReasoningNames); }
std::string tool_call_parser_names() { return names_of(kToolCallNames); }

ReasoningSplit split_reasoning(ReasoningFormat format, std::string reasoning,
                               std::string content) {
    ReasoningSplit out;
    switch (format) {
    case ReasoningFormat::None:
        // The frontend already lifted the span out; put it back. A deployment that
        // asked for no reasoning parser wants one stream of text, not a silently
        // discarded half of the answer.
        out.content = reasoning.empty() ? std::move(content) : reasoning + content;
        return out;
    case ReasoningFormat::ThinkTags:
        out.reasoning = std::move(reasoning);
        out.content   = std::move(content);
        return out;
    }
    out.content = std::move(content);
    return out;
}

ParsedToolCalls parse_tool_calls(ToolCallFormat format, const std::string& text,
                                 std::size_t max_tool_name_length, const std::vector<ToolDefinition>& tools) {
    ParsedToolCalls out;
    switch (format) {
    case ToolCallFormat::None:
        out.content = text;
        return out;
    case ToolCallFormat::QwenXml:
    case ToolCallFormat::Spark25: {
        ParsedToolCallOutput parsed = format == ToolCallFormat::Spark25
            ? parse_spark_tool_call_output(text, max_tool_name_length, tools)
            : parse_qwen_tool_call_output(text, max_tool_name_length, tools);
        out.is_tool_call_response   = parsed.is_tool_call_response;
        out.content                 = std::move(parsed.content);
        out.tool_calls              = std::move(parsed.tool_calls);
        return out;
    }
    case ToolCallFormat::Llama3Json:
        return parse_llama3_json(text, max_tool_name_length);
    }
    out.content = text;
    return out;
}

} // namespace sinfer::serve
