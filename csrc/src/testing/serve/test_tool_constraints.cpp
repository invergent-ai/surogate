#include "runtime/contract/constraint.h"
#include "serve/tool_constraints.h"
#include "serve/tool_call_parser.h"
#include <nlohmann/json.hpp>
#include <cassert>
#include <iostream>
using namespace sinfer;
using namespace sinfer::serve;
int main() {
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    ToolConstraintCompiler compiler(vocab, {256});
    GenerationRequest request;
    request.tools.push_back({.name="weather", .parameters_json=R"({"type":"object","properties":{"city":{"type":"string","enum":["Paris"]}},"required":["city"],"additionalProperties":false})"});
    assert(make_tool_constraint(request, ToolCallFormat::QwenXml, false).empty());
    request.parallel_tool_calls = false;
    assert(make_tool_constraint(request, ToolCallFormat::QwenXml, false).empty());
    request.tools[0].strict = true;
    const auto accepts = [&](ToolCallFormat format, const std::string& text, bool thinking=false) {
        auto grammar = compiler.compile(make_tool_constraint(request, format, thinking));
        auto state = grammar->create_state();
        std::vector<int32_t> mask(9);
        for (unsigned char token : text) {
            state->fill(mask);
            if (!((static_cast<uint32_t>(mask[token / 32]) >> (token % 32)) & 1U)) { return false; }
            if (!state->try_accept(token)) { return false; }
        }
        state->fill(mask);
        return bool((static_cast<uint32_t>(mask[8]) & 1U));
    };
    const std::string json = R"(<tool_call>{"name": "weather", "arguments": {"city":"Paris"}}</tool_call>)";
    const std::string xml = "<tool_call>\n<function=weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>";
    const std::string spark = "<tool_call>weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    const std::string llama = R"({"name": "weather", "parameters": {"city":"Paris"}})";
    assert(accepts(ToolCallFormat::QwenXml, "Hello!"));
    assert(accepts(ToolCallFormat::QwenXml, json));
    assert(accepts(ToolCallFormat::QwenXml, "Checking. " + json + " Done."));
    assert(accepts(ToolCallFormat::QwenXml, json + json)); // auto false is filtered after generation, like vLLM
    assert(!accepts(ToolCallFormat::QwenXml, R"(<tool_call>{"name": "unknown", "arguments": {}}</tool_call>)"));
    assert(!accepts(ToolCallFormat::QwenXml, R"(<tool_call>{"name": "weather", "arguments": {"city":"Rome"}}</tool_call>)"));
    auto parsed = parse_qwen_tool_call_output("Checking. " + json + " Done.", 64, request.tools);
    assert(parsed.tool_calls.size() == 1 && parsed.content == "Checking. Done.");
    assert(parse_qwen_tool_call_output(json + "\n\n" + json, 64, request.tools).content.empty());
    request.tools.push_back({.name="loose", .parameters_json=R"({"type":"object","properties":{"city":{"const":"Paris"}},"required":["city"]})", .strict=false, .strict_set=true});
    assert(accepts(ToolCallFormat::QwenXml, R"(<tool_call>{"name": "loose", "arguments": {"anything":42}}</tool_call>)"));
    request.tools.back().strict_set = false; // vLLM constrains omitted strict when another tool is strict
    assert(!accepts(ToolCallFormat::QwenXml, R"(<tool_call>{"name": "loose", "arguments": {"anything":42}}</tool_call>)"));
    request.tools.pop_back();
    request.tool_choice.mode = ToolChoiceMode::Required;
    assert(!accepts(ToolCallFormat::QwenXml, "Hello!"));
    assert(accepts(ToolCallFormat::QwenXml, xml));
    assert(accepts(ToolCallFormat::Spark25, spark));
    assert(accepts(ToolCallFormat::Llama3Json, llama));
    assert(accepts(ToolCallFormat::QwenXml, "Let me check.</think>\n\n" + xml, true));
    assert(!accepts(ToolCallFormat::QwenXml, xml + "\n" + xml));
    request.parallel_tool_calls = true;
    assert(accepts(ToolCallFormat::QwenXml, xml + "\n" + xml));
    request.tool_choice = {ToolChoiceMode::Named, "weather"};
    assert(!accepts(ToolCallFormat::QwenXml, xml + "\n" + xml));
    assert(tool_arguments_match_schema(request.tools[0], R"({"city":"Paris"})"));
    assert(!tool_arguments_match_schema(request.tools[0], R"({"city":"Rome"})"));
    assert(!tool_arguments_match_schema(request.tools[0], R"({"city":"Paris","extra":1})"));
    assert(!tool_arguments_match_schema(request.tools[0], R"({})"));
    ToolCallStreamFilter json_stream(true);
    assert(json_stream.feed("<|py").empty());
    assert(json_stream.feed("thon_tag|>" + llama).empty());
    assert(json_stream.finish(true).empty());
    ToolCallStreamFilter ordinary_stream(true);
    std::string visible = ordinary_stream.feed("An ordinary answer with {braces}.");
    visible += ordinary_stream.finish(false);
    assert(visible == "An ordinary answer with {braces}.");
    request.tool_choice.mode = ToolChoiceMode::None;
    assert(make_tool_constraint(request, ToolCallFormat::QwenXml, false).empty());
    std::cout << "native tool constraints, automatic choice, strict schemas and call counts passed\n";
}
