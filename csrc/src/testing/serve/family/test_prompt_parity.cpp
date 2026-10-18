// Golden prompt-render parity: a Chat Completions request goes through the serving layer exactly as
// /v1/chat/completions takes it (wire parse, translation, frontend preparation), and the prompt it
// produces must be byte-identical to the one vLLM's OpenAI server renders for the same request.
//
// The expected prompts are vLLM's own (fixtures/frontend/prompt_parity/make_golden.py runs vLLM's
// request validation, message parsing and chat-template call). Two templates: Surogate3.7's own
// Jinja, rendered by the tokenizer, and the stock Qwen3.5 template, which this family reproduces by
// hand. The tokenizer is a byte-level one built here with no merges, so token texts concatenate back
// to the exact prompt and the test needs no checkpoint.

#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>

#include "family/impl/frontend/test_access.h"
#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using OJson = nlohmann::ordered_json;
using sinfer::family::Frontend;
using sinfer::family::FrontendResources;
using sinfer::family::FrontendTestAccess;
namespace serve = sinfer::serve;

const std::string kFixtureDir = SINFER_SOURCE_DIR "/fixtures/frontend/prompt_parity/";

std::string read_file(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("failed to open test resource: " + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

/// GPT-2's byte-to-symbol map, the one byte-level BPE vocabularies are written in.
std::string byte_symbol(unsigned byte) {
    unsigned code = byte;
    if (!((byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255))) {
        unsigned shifted = 0;
        for (unsigned b = 0; b < byte; ++b) {
            if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))) { ++shifted; }
        }
        code = 256 + shifted;
    }
    std::string out;
    if (code < 0x80) {
        out += static_cast<char>(code);
    } else {
        out += static_cast<char>(0xC0 | (code >> 6));
        out += static_cast<char>(0x80 | (code & 0x3F));
    }
    return out;
}

/// Every byte is a token and nothing merges, so any prompt encodes and its token texts concatenate
/// back to it; the markers the templates write are added tokens, as in the real vocabulary.
FrontendResources byte_level_resources(const std::string& chat_template) {
    OJson vocab = OJson::object();
    for (unsigned byte = 0; byte < 256; ++byte) { vocab[byte_symbol(byte)] = byte; }
    const std::vector<std::pair<std::string, bool>> markers = {
        {"<|im_start|>", true}, {"<|im_end|>", true}, {"<|endoftext|>", true},
        {"<think>", false}, {"</think>", false}, {"<tool_call>", false}, {"</tool_call>", false},
        {"<tool_response>", false}, {"</tool_response>", false}, {"<|vision_start|>", true},
        {"<|vision_end|>", true}, {"<|image_pad|>", true}, {"<|video_pad|>", true}};
    OJson added   = OJson::array();
    OJson decoder = OJson::object();
    int id        = 300;
    for (const auto& [content, special] : markers) {
        OJson token = {{"id", id},          {"content", content},    {"single_word", false},
                       {"lstrip", false},   {"rstrip", false},       {"normalized", false},
                       {"special", special}};
        added.push_back(token);
        token.erase("id");
        decoder[std::to_string(id)] = std::move(token);
        ++id;
    }
    FrontendResources resources;
    resources.chat_template_jinja = chat_template;
    resources.tokenizer_json =
        OJson{{"added_tokens", added},
              {"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", OJson::array()}}}}.dump();
    resources.tokenizer_config_json = OJson{{"add_bos_token", false},
                                            {"add_prefix_space", false},
                                            {"eos_token", "<|im_end|>"},
                                            {"pad_token", "<|endoftext|>"},
                                            {"chat_template", chat_template},
                                            {"added_tokens_decoder", decoder}}
                                          .dump();
    resources.generation_config_json = R"({"eos_token_id":[301]})";
    return resources;
}

/// The prompt the serving layer prepares for a Chat Completions body.
std::string engine_prompt(const Frontend& frontend, const OJson& body) {
    const serve::GenerationRequest request = serve::parse_chat_completion_request(body, serve::RequestLimits{});
    const serve::ResolvedPromptSemantics semantics =
        serve::resolve_prompt_semantics(request, serve::ServeOptions{}, frontend.prompt_capabilities());
    sinfer::PromptInput input = serve::to_prompt_input(request, semantics, [](const serve::ContentPart&) -> sinfer::OwnedMedia {
        throw std::logic_error("the parity cases carry no media");
    });
    const auto prepared = frontend.prepare(std::move(input));
    std::string text;
    for (const std::string& piece : frontend.token_texts(FrontendTestAccess::inspect(prepared).token_ids)) {
        text += piece;
    }
    return text;
}

std::string excerpt(const std::string& text, std::size_t at) {
    const std::size_t begin = at > 60 ? at - 60 : 0;
    return OJson(text.substr(begin, 160)).dump(-1, ' ', false, OJson::error_handler_t::replace);
}

} // namespace

int main() {
    const OJson golden = OJson::parse(read_file(kFixtureDir + "cases.json"));
    int failures = 0;
    int checked  = 0;
    for (const auto& [name, file] : golden.at("templates").items()) {
        const Frontend frontend = FrontendTestAccess::create_component(
            byte_level_resources(read_file(kFixtureDir + file.get<std::string>())), /*vision_enabled=*/false);
        for (const OJson& item : golden.at("cases")) {
            const std::string label = item.at("name").get<std::string>() + " [" + name + "]";
            const OJson& expected   = item.at("expected").at(name);
            std::string actual;
            try {
                actual = engine_prompt(frontend, item.at("request"));
            } catch (const std::exception& error) {
                if (expected.is_object()) { ++checked; continue; } // both refuse it
                std::cerr << "FAIL: " << label << ": the engine refused a request vLLM renders: "
                          << error.what() << '\n';
                ++failures;
                continue;
            }
            ++checked;
            if (expected.is_object()) {
                std::cerr << "FAIL: " << label << ": the engine rendered a request vLLM refuses ("
                          << expected.dump() << ")\n";
                ++failures;
                continue;
            }
            const std::string& want = expected.get_ref<const std::string&>();
            if (actual == want) { continue; }
            std::size_t at = 0;
            while (at < actual.size() && at < want.size() && actual[at] == want[at]) { ++at; }
            std::cerr << "FAIL: " << label << ": prompt differs from vLLM's at byte " << at << " (engine "
                      << actual.size() << " bytes, vLLM " << want.size() << ")\n  vLLM:   "
                      << excerpt(want, at) << "\n  engine: " << excerpt(actual, at) << '\n';
            ++failures;
        }
    }
    if (checked == 0) {
        std::cerr << "FAIL: no parity cases were checked\n";
        return 1;
    }
    if (failures == 0) { std::cout << "ok (" << checked << " renders)\n"; }
    return failures == 0 ? 0 : 1;
}
