// The named reasoning and tool-call parsers (serve/output_parsers.h).
//
// The registries are contracts: a name that resolves is a promise the format
// round-trips, and a name that does not must say what does. The Llama-3 JSON
// decoder is new code and gets the most attention, including the cases where it
// must decline — a model answering in prose that happens to contain a brace has
// to keep its answer.

#include "serve/output_parsers.h"
#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using namespace sinfer::serve;
using Json = nlohmann::json;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void test_reasoning_names() {
    check(parse_reasoning_format("qwen3") == ReasoningFormat::ThinkTags, "qwen3 is think-tagged");
    check(parse_reasoning_format("deepseek_r1") == ReasoningFormat::ThinkTags,
          "deepseek_r1 is think-tagged");
    check(parse_reasoning_format("none") == ReasoningFormat::None, "none disables");

    bool threw = false;
    try {
        parse_reasoning_format("nope");
    } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).find("qwen3") != std::string::npos;
    }
    check(threw, "an unknown reasoning parser is refused and names the supported ones");
}

void test_tool_call_names() {
    check(parse_tool_call_format("hermes") == ToolCallFormat::QwenXml, "hermes shares the Qwen block");
    check(parse_tool_call_format("llama3_json") == ToolCallFormat::Llama3Json, "llama3_json");
    check(parse_tool_call_format("none") == ToolCallFormat::None, "none disables");

    bool threw = false;
    try {
        parse_tool_call_format("nope");
    } catch (const std::invalid_argument& error) {
        threw = std::string(error.what()).find("llama3_json") != std::string::npos;
    }
    check(threw, "an unknown tool call parser is refused and names the supported ones");
}

/// `none` must fold the span back rather than drop it: the frontend has already
/// split it off by the time the server sees the outcome.
void test_reasoning_split() {
    const ReasoningSplit kept = split_reasoning(ReasoningFormat::ThinkTags, "weighing", "answer");
    check(kept.reasoning == "weighing" && kept.content == "answer", "think-tags keeps the split");

    const ReasoningSplit folded = split_reasoning(ReasoningFormat::None, "weighing", "answer");
    check(folded.reasoning.empty(), "none reports no reasoning");
    check(folded.content == "weighinganswer", "none folds the span into the content");

    const ReasoningSplit empty = split_reasoning(ReasoningFormat::None, "", "answer");
    check(empty.content == "answer", "none leaves a plain answer alone");
}

void test_llama3_tool_calls() {
    constexpr std::size_t kMaxName = 64;

    const ParsedToolCalls one = parse_tool_calls(
        ToolCallFormat::Llama3Json, R"({"name": "get_weather", "parameters": {"city": "Paris"}})",
        kMaxName);
    check(one.is_tool_call_response, "a bare JSON object is a call");
    check(one.tool_calls.size() == 1, "one call");
    if (one.tool_calls.size() == 1) {
        check(one.tool_calls[0].name == "get_weather", "name decoded");
        check(Json::parse(one.tool_calls[0].arguments_json).at("city") == "Paris",
              "arguments decoded");
        check(one.tool_calls[0].id.rfind("call_", 0) == 0, "an id is minted");
    }

    // Llama emits the call behind a python tag, and the prose before it is content.
    const ParsedToolCalls tagged = parse_tool_calls(
        ToolCallFormat::Llama3Json,
        "Let me look.\n<|python_tag|>{\"name\": \"f\", \"arguments\": {\"x\": 1}}", kMaxName);
    check(tagged.is_tool_call_response, "python-tagged call is a call");
    check(tagged.content == "Let me look.", "text before the tag survives as content");
    check(tagged.tool_calls.size() == 1 && tagged.tool_calls[0].name == "f",
          "`arguments` is accepted as well as `parameters`");

    const ParsedToolCalls many = parse_tool_calls(
        ToolCallFormat::Llama3Json,
        R"({"name": "a", "parameters": {}}; {"name": "b", "parameters": {}})", kMaxName);
    check(many.tool_calls.size() == 2, "semicolon-separated calls");

    const ParsedToolCalls array =
        parse_tool_calls(ToolCallFormat::Llama3Json,
                         R"([{"name": "a", "parameters": {}}, {"name": "b", "parameters": {}}])",
                         kMaxName);
    check(array.tool_calls.size() == 2, "a JSON array of calls");

    // Everything below must decline, and declining means returning the text whole.
    const ParsedToolCalls prose = parse_tool_calls(
        ToolCallFormat::Llama3Json, "The set {1, 2} is closed under addition.", kMaxName);
    check(!prose.is_tool_call_response, "prose containing a brace is not a call");
    check(prose.content == "The set {1, 2} is closed under addition.", "prose survives verbatim");

    const ParsedToolCalls broken =
        parse_tool_calls(ToolCallFormat::Llama3Json, R"({"name": "f", "parameters": )", kMaxName);
    check(!broken.is_tool_call_response && !broken.content.empty(),
          "truncated JSON falls back to content");

    const ParsedToolCalls nameless =
        parse_tool_calls(ToolCallFormat::Llama3Json, R"({"parameters": {"x": 1}})", kMaxName);
    check(!nameless.is_tool_call_response, "an object without a name is not a call");

    const ParsedToolCalls bad_name = parse_tool_calls(
        ToolCallFormat::Llama3Json, R"({"name": "drop table", "parameters": {}})", kMaxName);
    check(!bad_name.is_tool_call_response, "a name outside [A-Za-z0-9_-] is refused");

    const ParsedToolCalls trailing = parse_tool_calls(
        ToolCallFormat::Llama3Json, R"({"name": "f", "parameters": {}} and then some prose)",
        kMaxName);
    check(!trailing.is_tool_call_response, "a call with prose after it is not a clean call");
}

void test_tool_call_dispatch() {
    const std::string qwen =
        "<tool_call><function=get_weather><parameter=city>Paris</parameter></function></tool_call>";
    const ParsedToolCalls parsed = parse_tool_calls(ToolCallFormat::QwenXml, qwen, 64);
    check(parsed.is_tool_call_response, "the Qwen block still parses through the registry");

    const ParsedToolCalls off = parse_tool_calls(ToolCallFormat::None, qwen, 64);
    check(!off.is_tool_call_response && off.content == qwen,
          "`none` leaves the block in the content untouched");
}

void test_spark_tool_calls() {
    check(parse_tool_call_format("spark25") == ToolCallFormat::Spark25, "Spark parser name");
    const std::string text = "Looking now.\n<tool_call>weather<arg_key>city</arg_key>"
        "<arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value>"
        "<arg_key>options</arg_key><arg_value>{\"units\":\"C\"}</arg_value></tool_call>"
        "<tool_call>refresh</tool_call>";
    const auto parsed = parse_tool_calls(ToolCallFormat::Spark25, text, 64);
    check(parsed.is_tool_call_response && parsed.tool_calls.size() == 2, "Spark multiple calls");
    check(parsed.content == "Looking now.", "Spark preserves preceding content");
    if (parsed.tool_calls.size() == 2) {
        const auto args = Json::parse(parsed.tool_calls[0].arguments_json);
        check(args["city"] == "Paris" && args["days"] == 3 && args["options"]["units"] == "C",
              "Spark string, number and nested JSON arguments");
        check(parsed.tool_calls[1].arguments_json == "{}", "Spark zero arguments");
    }
    for (std::size_t split=0; split<=text.size(); ++split) {
        ToolCallStreamFilter filter;
        auto content = filter.feed(std::string_view(text).substr(0,split));
        content += filter.feed(std::string_view(text).substr(split));
        content += filter.finish(true);
        check(content == parsed.content, "Spark split-stream content");
    }
    for (const auto malformed : {
        "<tool_call>bad.name</tool_call>", "<tool_call>weather<arg_key>city</arg_key></tool_call>",
        "<tool_call>weather<arg_key>x</arg_key><arg_value>1</arg_value><arg_key>x</arg_key><arg_value>2</arg_value></tool_call>",
        "<tool_call>weather", "<tool_call>weather</tool_call>trailing"}) {
        const auto result = parse_tool_calls(ToolCallFormat::Spark25, malformed, 64);
        check(!result.is_tool_call_response && result.content == malformed, "Spark malformed fallback");
    }
}

} // namespace

int main() {
    test_reasoning_names();
    test_tool_call_names();
    test_reasoning_split();
    test_llama3_tool_calls();
    test_tool_call_dispatch();
    test_spark_tool_calls();

    if (failures != 0) {
        std::cerr << "output_parsers: " << failures << " case(s) failed\n";
        return 1;
    }
    std::cout << "OK output_parsers\n";
    return 0;
}
