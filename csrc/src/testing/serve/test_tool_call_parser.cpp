#include "serve/tool_call_parser.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

int test_single_call() {
    const sinfer::serve::ParsedToolCallOutput parsed =
        sinfer::serve::parse_qwen_tool_call_output("Calling weather.\n"
                                                   "<tool_call>\n"
                                                   "<function=get_weather>\n"
                                                   "<parameter=city>\nParis\n</parameter>\n"
                                                   "<parameter=days>\n2\n</parameter>\n"
                                                   "</function>\n"
                                                   "</tool_call>",
                                                   64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "single call parsed as tool response");
    failures += check(parsed.content == "Calling weather.", "content prefix trimmed");
    failures += check(parsed.tool_calls.size() == 1, "one parsed call");
    failures += check(parsed.tool_calls[0].id.rfind("call_", 0) == 0, "generated call id prefix");
    failures += check(parsed.tool_calls[0].name == "get_weather", "function name parsed");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("city") == "Paris", "string parameter parsed");
    failures += check(args.at("days") == 2, "number parameter parsed");
    return failures;
}

int test_multiple_calls_and_json_values() {
    const sinfer::serve::ParsedToolCallOutput parsed = sinfer::serve::parse_qwen_tool_call_output(
        "<tool_call>\n"
        "<function=first>\n"
        "<parameter=payload>\n{\"ok\":true,\"items\":[1,2]}\n</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "<tool_call>\n"
        "<function=second>\n"
        "<parameter=value>\nplain text\n</parameter>\n"
        "</function>\n"
        "</tool_call>",
        64);

    int failures = 0;
    failures += check(parsed.is_tool_call_response, "multiple calls parsed as tool response");
    failures += check(parsed.tool_calls.size() == 2, "two parsed calls");
    failures += check(parsed.tool_calls[0].name == "first", "first call name");
    failures += check(parsed.tool_calls[1].name == "second", "second call name");
    const Json first = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(first.at("payload").at("ok") == true, "object parameter bool");
    failures += check(first.at("payload").at("items").at(1) == 2, "object parameter array");
    const Json second = Json::parse(parsed.tool_calls[1].arguments_json);
    failures += check(second.at("value") == "plain text", "plain text parameter string");
    return failures;
}

int test_malformed_falls_back_to_text() {
    const std::string text = "<tool_call>\n<function=get_weather>\n";
    const sinfer::serve::ParsedToolCallOutput parsed =
        sinfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "malformed xml is not tool response");
    failures += check(parsed.content == text, "malformed xml preserved as text");
    failures += check(parsed.tool_calls.empty(), "malformed xml has no calls");
    return failures;
}

int test_suffix_after_tool_falls_back_to_text() {
    const std::string text = "<tool_call>\n"
                             "<function=get_weather>\n"
                             "<parameter=city>\nParis\n</parameter>\n"
                             "</function>\n"
                             "</tool_call>\n"
                             "extra answer";
    const sinfer::serve::ParsedToolCallOutput parsed =
        sinfer::serve::parse_qwen_tool_call_output(text, 64);
    int failures = 0;
    failures += check(!parsed.is_tool_call_response, "non-whitespace suffix falls back to text");
    failures += check(parsed.content == text, "suffix fallback preserves text");
    return failures;
}

int test_configured_name_limit() {
    const std::string name(128, 'a');
    const std::string text = "<tool_call>\n<function=" + name + ">\n</function>\n</tool_call>";

    const sinfer::serve::ParsedToolCallOutput anthropic =
        sinfer::serve::parse_qwen_tool_call_output(text, 128);
    const sinfer::serve::ParsedToolCallOutput openai =
        sinfer::serve::parse_qwen_tool_call_output(text, 64);
    const std::string too_long_text =
        "<tool_call>\n<function=" + std::string(129, 'a') + ">\n</function>\n</tool_call>";
    const sinfer::serve::ParsedToolCallOutput too_long =
        sinfer::serve::parse_qwen_tool_call_output(too_long_text, 128);

    int failures = 0;
    failures += check(anthropic.is_tool_call_response && anthropic.tool_calls.size() == 1 &&
                          anthropic.tool_calls[0].name == name,
                      "128-character name accepted with Anthropic limit");
    failures +=
        check(!openai.is_tool_call_response, "128-character name rejected with OpenAI limit");
    failures +=
        check(!too_long.is_tool_call_response, "129-character name rejected with Anthropic limit");
    return failures;
}

int test_incremental_filter_valid_tool() {
    sinfer::serve::ToolCallStreamFilter filter;
    std::string visible;
    visible += filter.feed("Calling weather.  \n<tool_");
    visible += filter.feed("call>\n<function=get_weather>");
    visible += filter.feed("\n</function>\n</tool_call>");
    visible += filter.finish(true);
    int failures = 0;
    failures += check(visible == "Calling weather.",
                      "valid tool filter did not stream the trimmed content prefix");
    failures +=
        check(filter.emitted_bytes() == visible.size(), "valid tool filter byte count mismatch");
    return failures;
}

int test_incremental_filter_fallback() {
    const std::string original = "prefix  \n<tool_call>\n<function=broken>";
    sinfer::serve::ToolCallStreamFilter malformed;
    std::string restored;
    restored += malformed.feed(original.substr(0, 10));
    restored += malformed.feed(original.substr(10));
    restored += malformed.finish(false);

    sinfer::serve::ToolCallStreamFilter normal;
    std::string ordinary;
    ordinary += normal.feed("ordinary text  ");
    ordinary += normal.finish(false);

    int failures = 0;
    failures += check(restored == original, "malformed tool filter fallback lost raw bytes");
    failures +=
        check(ordinary == "ordinary text  ", "ordinary filtered output lost trailing whitespace");
    return failures;
}

int test_checkpoint_json_and_schema_strings() {
    using sinfer::serve::parse_qwen_tool_call_output;
    sinfer::serve::ToolDefinition tool;
    tool.name = "files.write";
    tool.parameters_json = R"({"type":"object","properties":{"text":{"$ref":"#/$defs/Text"},"count":{"type":"integer"}},"$defs":{"Text":{"type":"string"}}})";
    const auto parsed = parse_qwen_tool_call_output(
        "<tool_call><function=files.write><parameter=text>\n  007\n\n</parameter>"
        "<parameter=count>\n3\n</parameter></function></tool_call>", 64, {tool});
    int failures = check(parsed.tool_calls.size() == 1, "namespaced function parsed");
    if (parsed.tool_calls.empty()) { return failures; }
    auto args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args["text"] == "  007\n" && args["count"] == 3, "schema preserves string bytes and numeric types");
    const auto json = parse_qwen_tool_call_output(
        R"(<tool_call>{"name":"files.write","arguments":{"text":"007","count":3}}</tool_call>)", 64, {tool});
    failures += check(json.tool_calls.size() == 1, "Qwen3/Hermes JSON accepted");
    if (!json.tool_calls.empty()) {
        args = Json::parse(json.tool_calls[0].arguments_json);
        failures += check(args["text"] == "007" && args["count"] == 3, "JSON types preserved");
    }
    const std::string duplicate = "<tool_call><function=f><parameter=x>1</parameter>"
                                  "<parameter=x>2</parameter></function></tool_call>";
    failures += check(!parse_qwen_tool_call_output(duplicate, 64).is_tool_call_response,
                      "duplicate parameters do not execute");
    tool.parameters_json = R"({"properties":{"text":{"anyOf":[{"type":"string"},{"type":"null"}]}}})";
    const auto nullable = parse_qwen_tool_call_output(
        "<tool_call><function=files.write><parameter=text>123</parameter></function></tool_call>", 64, {tool});
    failures += check(nullable.tool_calls.size() == 1 &&
                      Json::parse(nullable.tool_calls[0].arguments_json)["text"] == "123",
                      "nullable string does not become a number");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_single_call();
    failures += test_multiple_calls_and_json_values();
    failures += test_malformed_falls_back_to_text();
    failures += test_suffix_after_tool_falls_back_to_text();
    failures += test_configured_name_limit();
    failures += test_incremental_filter_valid_tool();
    failures += test_incremental_filter_fallback();
    failures += test_checkpoint_json_and_schema_strings();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
