// A finished generation's reasoning, content and tool calls exactly as vLLM 0.31's qwen3 parser returns them
// (serve/vllm_qwen3_parser.h), against vLLM's own results.
//
// fixtures/serve/vllm_qwen3_parser/corpus.jsonl: Surogate3.7 outputs (BFCL v4 answers and calls from our engine
// and from vLLM, the tool-argument cases of #324) and mutations of them (unknown names, missing closers, text after
// and between calls, calls inside reasoning, thinking off, tool_choice none, no tools, special tokens inside,
// markers spelled by ordinary tokens, bare <function=, JSON bodies, multibyte text, every token prefix of some),
// each with the token ids and text vLLM's parser was handed and what it returned. vllm_reference.py beside it
// re-derives the vLLM results. Line 1 holds the tokenizer facts the parser needs (marker ids, the special tokens
// vLLM drops, each used token's own text), so the test needs no model files; with
// SUROGATE_QWEN3_PARSER_TOKENIZER_DIR set, the engine's tokenizer is checked against them too.
//
// SUROGATE_QWEN3_PARSER_CORPUS names a larger corpus of the same shape to run instead.

#ifndef SUROGATE_PARSER_TEST_NO_TOKENIZER // a stand-alone build of this test (no engine tokenizer)
#include "family/impl/frontend/tokenizer.h"
#endif
#include "serve/vllm_qwen3_parser.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using Json = nlohmann::json;
using namespace sinfer::serve;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        if (failures <= 30) { std::cerr << "FAIL: " << message << '\n'; }
    }
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

Json optional_json(const std::optional<std::string>& value) { return value ? Json(*value) : Json(nullptr); }

std::vector<ToolDefinition> tools_of(const Json& tools) {
    std::vector<ToolDefinition> out;
    if (!tools.is_array()) { return out; }
    for (const auto& tool : tools) {
        const Json& function = tool.at("function");
        ToolDefinition definition;
        definition.name            = function.at("name").get<std::string>();
        definition.parameters_json = function.contains("parameters") ? function.at("parameters").dump() : "null";
        out.push_back(std::move(definition));
    }
    return out;
}

} // namespace

int main() {
    const char* override_path = std::getenv("SUROGATE_QWEN3_PARSER_CORPUS");
    const std::string path    = override_path != nullptr
                                    ? std::string(override_path)
                                    : std::string(SINFER_SOURCE_DIR) + "/fixtures/serve/vllm_qwen3_parser/corpus.jsonl";
    std::ifstream in(path);
    check(in.good(), "corpus missing: " + path);
    std::string line;
    std::getline(in, line);
    const Json header = Json::parse(line);

    std::unordered_map<std::int32_t, std::string> texts;
    for (const auto& [id, text] : header.at("token_texts").items()) { texts[std::stoi(id)] = text.get<std::string>(); }
    vllm_qwen3::Vocabulary vocabulary;
    for (const auto& [id, text] : header.at("drop_tokens").items()) { vocabulary.special_tokens.push_back(text); }

    // With the checkpoint's tokenizer at hand: the engine's tokenizer gives every token the text transformers gives it
    // (one token decoded alone), and the transformers' all_special_tokens it reads are the ones vLLM drops.
#ifndef SUROGATE_PARSER_TEST_NO_TOKENIZER
    std::unique_ptr<sinfer::family::frontend_internal::Tokenizer> tokenizer;
    std::string tokenizer_json, config_json, generation_json; // the resources the tokenizer reads at construction
    if (const char* dir = std::getenv("SUROGATE_QWEN3_PARSER_TOKENIZER_DIR")) {
        const std::string root = dir;
        tokenizer_json  = read_file(root + "/tokenizer.json");
        config_json     = read_file(root + "/tokenizer_config.json");
        generation_json = read_file(root + "/generation_config.json");
        tokenizer = std::make_unique<sinfer::family::frontend_internal::Tokenizer>(
            sinfer::family::frontend_internal::TokenizerResources{tokenizer_json, config_json, generation_json});
        int mismatched = 0;
        for (const auto& [id, text] : texts) {
            if (vllm_qwen3::utf8_replace(tokenizer->decode_token_bytes(id, false)) != text) { ++mismatched; }
        }
        check(mismatched == 0, std::to_string(mismatched) + " token texts differ from the engine tokenizer's");
        std::vector<std::string> named;
        for (const auto& special : tokenizer->named_special_tokens()) { named.push_back(special.second); }
        std::vector<std::string> expected = vocabulary.special_tokens;
        std::sort(named.begin(), named.end());
        std::sort(expected.begin(), expected.end());
        check(named == expected, "the engine tokenizer's all_special_tokens differ from transformers'");
        std::cout << "engine tokenizer: " << texts.size() << " token texts checked, " << mismatched << " differ; "
                  << named.size() << " special tokens\n";
    }
#endif

    int rows = 0, skipped = 0, with_calls = 0, calls = 0, nonempty_content = 0, decoded = 0;
    std::int32_t stop_id = header.value("stop_id", -1); // the generation's stop token, <|im_end|>
    for (const auto& [id, text] : texts) {
        if (stop_id < 0 && text == "<|im_end|>") { stop_id = id; }
    }
    while (std::getline(in, line)) {
        const Json row = Json::parse(line);
        const Json& expected = row.at("vllm");
        if (expected.contains("error")) { // vLLM refused the request itself (e.g. its tools)
            ++skipped;
            continue;
        }
        ++rows;
        vllm_qwen3::Input input;
        input.text     = row.at("text").get<std::string>();
        input.thinking = row.at("thinking").get<bool>();
        // The text the engine hands the parser for these token ids (GenerationService: the stop token's own text
        // left out): vLLM's output.text.
#ifndef SUROGATE_PARSER_TEST_NO_TOKENIZER
        if (tokenizer) {
            std::vector<std::int32_t> ids = row.at("ids").get<std::vector<std::int32_t>>();
            if (!ids.empty() && ids.back() == stop_id) { ids.pop_back(); }
            std::string raw;
            for (const auto id : ids) { raw += tokenizer->decode_token_bytes(id, false); }
            const bool cut = row.at("id").get<std::string>().find("/cut") != std::string::npos ||
                             std::isdigit(static_cast<unsigned char>(row.at("id").get<std::string>().back()));
            if (!cut) {
                ++decoded;
                check(vllm_qwen3::output_text(raw) == input.text, row.at("id").get<std::string>() + ": decoded text");
            }
        }
#endif
        const bool has_tools      = row.at("tools").is_array() && !row.at("tools").empty();
        input.suppress_tool_calls = has_tools && row.at("tool_choice") == "none";
        const vllm_qwen3::Result result = vllm_qwen3::parse(input, vocabulary, tools_of(row.at("tools")));
        // Serving (DelegatingParser._extract_tool_calls, GenerationService::run): calls are returned for tool_choice
        // "auto" with tools; a request without tools defaults to tool_choice "none".
        const bool returns_calls = has_tools && row.at("tool_choice") != "none";
        Json got_calls = Json::array();
        for (const auto& call : result.calls) {
            if (returns_calls) { got_calls.push_back({{"name", call.name}, {"arguments", call.arguments}}); }
        }
        const Json got = {{"reasoning", optional_json(result.reasoning)},
                          {"content", optional_json(result.content)},
                          {"calls", got_calls}};
        const Json want = {{"reasoning", expected.at("reasoning")},
                           {"content", expected.at("content")},
                           {"calls", expected.at("calls").is_null() ? Json::array() : expected.at("calls")}};
        check(got == want, row.at("id").get<std::string>() + "\n  engine " + got.dump().substr(0, 600) + "\n  vllm   " +
                               want.dump().substr(0, 600));
        with_calls += !want.at("calls").empty();
        calls += static_cast<int>(want.at("calls").size());
        nonempty_content += !want.at("content").is_null();
    }
    check(rows > 1000 && with_calls > 500, "corpus smaller than expected");

    // utf8_replace is Python's errors="replace".
    check(vllm_qwen3::utf8_replace("a\xE2\x82") == "a\xEF\xBF\xBD", "truncated sequence");
    check(vllm_qwen3::utf8_replace("\xE0\x80x") == "\xEF\xBF\xBD\xEF\xBF\xBDx", "overlong E0 80");
    check(vllm_qwen3::utf8_replace("\xF0\x9F\x98\x8A") == "\xF0\x9F\x98\x8A", "valid 4-byte");
    check(vllm_qwen3::utf8_replace("\xED\xA0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD", "surrogate");

    std::cout << "vllm_qwen3_parser: " << rows << " outputs (" << with_calls << " with " << calls << " calls, "
              << nonempty_content << " with content; " << decoded << " decoded by the engine), " << skipped << " that vLLM itself refused; failures="
              << failures << '\n';
    return failures == 0 ? 0 : 1;
}
