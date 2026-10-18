// Tool-call arguments as vLLM 0.31's qwen3_coder / qwen3_xml parser returns them (serve/vllm_tool_arguments.h).
//
// fixtures/serve/vllm_tool_arguments/parity.jsonl: raw model outputs (calls the model made in the 2026-10-18
// engine-vs-vLLM measurement on BFCL, IFEval and EnterpriseOps-Gym, rebuilt as the model writes them, plus synthetic
// values under every schema shape and markup edge cases), each with the calls vLLM 0.31.0's own parser returned for
// it (`vllm`). The engine must return the same names and byte-identical `arguments`. The few cases where it does not
// on purpose carry `expect` (what the engine returns) and `exception` (why).
// fixtures/serve/vllm_tool_arguments/python_scalars.jsonl: CPython 3.12's repr(float), int(float), int(str) and
// float(str) on random and edge inputs.

#include "serve/output_parsers.h"
#include "serve/tool_call_parser.h"
#include "serve/tool_constraints.h"
#include "serve/vllm_tool_arguments.h"

#include <nlohmann/json.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

namespace {

using OJson = nlohmann::ordered_json;
using namespace sinfer::serve;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

std::string fixture(const char* name) {
    return std::string(SINFER_SOURCE_DIR) + "/fixtures/serve/vllm_tool_arguments/" + name;
}

double from_bits(const std::string& hex) { return std::bit_cast<double>(std::stoull(hex, nullptr, 16)); }

std::string to_bits(double value) {
    char buffer[24];
    std::snprintf(buffer, sizeof buffer, "%016llx",
                  static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(value)));
    return buffer;
}

std::vector<ToolDefinition> tools_of(const OJson& tools) {
    std::vector<ToolDefinition> out;
    for (const auto& item : tools) {
        const OJson& fn = item.at("function");
        ToolDefinition tool;
        tool.name            = fn.at("name").get<std::string>();
        tool.parameters_json = fn.contains("parameters") && fn.at("parameters").is_object()
                                   ? fn.at("parameters").dump()
                                   : R"({"type":"object","properties":{}})"; // as serve/openai_schema.cpp fills it
        out.push_back(std::move(tool));
    }
    return out;
}

void test_python_scalars() {
    std::ifstream file(fixture("python_scalars.jsonl"));
    check(file.good(), "python_scalars.jsonl readable");
    std::string line;
    std::size_t rows = 0;
    while (std::getline(file, line)) {
        const OJson row = OJson::parse(line);
        const std::string fn = row.at("fn").get<std::string>();
        ++rows;
        if (fn == "repr") {
            const std::string got = vllm_args::python_float_repr(from_bits(row.at("bits").get<std::string>()));
            check(got == row.at("out").get<std::string>(), "repr " + row.at("out").get<std::string>() + " != " + got);
        } else if (fn == "int_of_float") {
            const std::string got = vllm_args::python_int_of_float(from_bits(row.at("bits").get<std::string>()));
            check(got == row.at("out").get<std::string>(), "int(float) " + row.at("out").get<std::string>() + " != " + got);
        } else if (fn == "int") {
            const auto got = vllm_args::python_int(row.at("in").get<std::string>());
            const bool expected = !row.at("out").is_null();
            check(got.has_value() == expected && (!expected || *got == row.at("out").get<std::string>()),
                  "int(" + OJson(row.at("in")).dump() + ") = " + (got ? *got : "error"));
        } else if (fn == "float") {
            const auto got      = vllm_args::python_float(row.at("in").get<std::string>());
            const bool expected = !row.at("bits").is_null();
            check(got.has_value() == expected && (!expected || to_bits(*got) == row.at("bits").get<std::string>()),
                  "float(" + OJson(row.at("in")).dump() + ") = " + (got ? to_bits(*got) : "error"));
        }
    }
    check(rows > 6000, "python_scalars.jsonl has its rows");
}

void test_parity_corpus() {
    std::ifstream file(fixture("parity.jsonl"));
    check(file.good(), "parity.jsonl readable");
    std::string line;
    std::size_t cases = 0, exceptions = 0;
    while (std::getline(file, line)) {
        const OJson c = OJson::parse(line);
        const OJson& expected = c.contains("expect") ? c.at("expect") : c.at("vllm");
        if (c.contains("expect")) { ++exceptions; }
        const ParsedToolCalls parsed =
            parse_tool_calls(ToolCallFormat::QwenXml, c.at("text").get<std::string>(), 64, tools_of(c.at("tools")));
        OJson got = OJson::array();
        for (const auto& call : parsed.tool_calls) {
            got.push_back(OJson{{"name", call.name}, {"arguments", call.arguments_json}});
        }
        check(got == expected && parsed.is_tool_call_response == !expected.empty(),
              c.at("id").get<std::string>() + ": expected " + expected.dump() + ", got " + got.dump());
        ++cases;
    }
    check(cases > 5000, "parity.jsonl has its cases");
    std::cout << cases << " parity cases (" << exceptions << " documented exceptions)\n";
}

std::string arguments(const std::string& body, const OJson& properties) {
    return vllm_args::qwen3_arguments_json(vllm_args::extract_qwen3_parameters(body), properties);
}

void test_examples() {
    const OJson properties = OJson::parse(R"({"flag":{"type":"boolean"},"n":{"type":"number"},"i":{"type":"integer"},
        "s":{"type":"string"},"o":{"type":"object"},"u":{"type":["integer","null"]},"e":{"enum":[1,2]}})");
    // The conversions the measurement turned on: a Python-style boolean, a whole number, an undeclared parameter.
    check(arguments("<parameter=flag>\nTrue\n</parameter>\n<parameter=n>\n5.0\n</parameter>\n<parameter=x>\n5\n</parameter>\n",
                    properties) == R"({"flag": true, "n": 5, "x": "5"})",
          "True, 5.0 and an undeclared 5");
    check(arguments("<parameter=i>\n 007 \n</parameter><parameter=u>\nNULL\n</parameter><parameter=e>\n2\n</parameter>",
                    properties) == R"({"i": 7, "u": null, "e": 2})",
          "int() of padded digits, null in any case, an integer enum");
    check(arguments("<parameter=s>\n  x\n\n</parameter><parameter=o>\n{\"a\": 1.50, \"b\": \"True\"}\n</parameter>",
                    properties) == R"({"s": "  x\n", "o": {"a": 1.5, "b": "True"}})",
          "string bytes kept, one wrapping newline trimmed, nested values not converted");
    check(arguments("<parameter=i>1</parameter><parameter=s>a</parameter><parameter=i>2</parameter>", properties) ==
              R"({"i": 2, "s": "a"})",
          "a repeated parameter keeps its first place and its last value");
    check(arguments("< parameter = i>3</ parameter >junk<parameter=s>a<parameter=n>1e3</parameter>"
                    "<parameter=flag>yes",
                    properties) == R"({"i": 3, "s": "a", "n": 1000})",
          "spaced tags, junk, a value ended by the next tag, an unterminated last value dropped");
    check(arguments("<parameter=i >3</parameter>", properties) == R"({"i ": "3"})",
          "space before '>' stays in the name, which the schema then does not declare");
    check(arguments("<parameter=n>\n1e16\n</parameter><parameter=o>\n[1e16, 0.1]\n</parameter>", properties) ==
              R"({"n": 10000000000000000, "o": [1e+16, 0.1]})",
          "a whole float is an exact integer; a float inside JSON keeps Python's repr");
    check(arguments("<parameter=flag>\nTrue\n</parameter>", OJson()) == R"({"flag": "True"})",
          "no schema: every value is text");
}

void test_call_structure() {
    ToolDefinition tool;
    tool.name            = "write";
    tool.parameters_json = R"({"type":"object","properties":{"text":{"type":"string"},"n":{"type":"integer"}}})";
    // A value that spells the closing marker: vLLM reads it as text up to the call's </function>.
    const auto parsed = parse_qwen_tool_call_output(
        "<tool_call>\n<function= write >\n<parameter=text>\nends with </tool_call> here\n</parameter>\n"
        "<parameter=n>\n2\n</parameter>\n</function>\n</tool_call>",
        64, {tool});
    check(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "write" &&
              parsed.tool_calls[0].arguments_json == R"({"text": "ends with </tool_call> here", "n": 2})",
          "closing marker inside a value, padded function name");
    // The JSON form: the arguments as written, serialised as Python's json.dumps.
    const auto json = parse_qwen_tool_call_output(
        R"(<tool_call>{"name": "write", "arguments": {"text":"a","n":2,"x":1.50}}</tool_call>)", 64, {tool});
    check(json.tool_calls.size() == 1 && json.tool_calls[0].arguments_json == R"({"text": "a", "n": 2, "x": 1.5})",
          "JSON call form serialised as Python does");
}

// Under the tool grammar a value was written as the schema types it. Where vLLM's reading would take it out of the
// schema -- `const` and `$ref` are not read, an enum mixing text and numbers reads "123" as 123 -- the call keeps
// the schema's reading, so a strict call still matches its schema; elsewhere it is vLLM's reading.
void test_constrained_calls() {
    ToolDefinition tool;
    tool.name            = "pick";
    tool.strict          = true;
    tool.strict_set      = true;
    tool.parameters_json = R"({"type":"object","properties":{"code":{"enum":["123",456]},"on":{"const":true},
        "ref":{"$ref":"#/$defs/N"},"n":{"type":"number"}},"required":["code","on","ref","n"],
        "$defs":{"N":{"type":"integer"}}})";
    const std::string text = "<tool_call>\n<function=pick>\n<parameter=code>\n123\n</parameter>\n"
                             "<parameter=on>\ntrue\n</parameter>\n<parameter=ref>\n7\n</parameter>\n"
                             "<parameter=n>\n5.0\n</parameter>\n</function>\n</tool_call>";
    const auto loose = parse_qwen_tool_call_output(text, 64, {tool});
    check(loose.tool_calls.size() == 1 &&
              loose.tool_calls[0].arguments_json == R"({"code": 123, "on": "true", "ref": "7", "n": 5})",
          "unconstrained: vLLM's reading");
    const auto strict = parse_qwen_tool_call_output(text, 64, {tool}, true);
    check(strict.tool_calls.size() == 1 &&
              strict.tool_calls[0].arguments_json == R"({"code": "123", "on": true, "ref": 7, "n": 5.0})" &&
              tool_arguments_match_schema(tool, strict.tool_calls[0].arguments_json),
          "constrained: the schema's reading where vLLM's breaks the schema");
    tool.parameters_json = R"({"type":"object","properties":{"n":{"type":"number"},"b":{"type":"boolean"}}})";
    const auto valid = parse_qwen_tool_call_output(
        "<tool_call>\n<function=pick>\n<parameter=n>\n5.0\n</parameter>\n<parameter=b>\nfalse\n</parameter>\n"
        "</function>\n</tool_call>",
        64, {tool}, true);
    check(valid.tool_calls.size() == 1 && valid.tool_calls[0].arguments_json == R"({"n": 5, "b": false})",
          "constrained: vLLM's reading where it matches the schema");
}

} // namespace

int main() {
    test_python_scalars();
    test_examples();
    test_call_structure();
    test_constrained_calls();
    test_parity_corpus();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
