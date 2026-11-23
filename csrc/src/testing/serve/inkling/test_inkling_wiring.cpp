// The Inkling-2 path through the serving layer, against vLLM's own Python.
//
// What sinfer_inkling_frontend_test proves for the renderer and parser functions, this proves for the code that carries
// a request to them: the chat-completions parser in raw-chat mode (RequestLimits::raw_chat) accepts every body vLLM
// renders, translate hands the body over (PromptOptions::raw_chat_request), and the family Frontend built from the
// checkpoint's own files detects Inkling-2 (PromptCapabilities::inkling2_chat, no chat template), renders the vLLM
// prompt ids, counts them, refuses what vLLM refuses, and decodes generated ids through Frontend::token_texts (the
// bytes GenerationService reads) to vLLM's output text and reply. CPU only; no device, no weights.
// Skips (77) without INKLING_MODEL_DIR / INKLING_VECTORS (as sinfer_inkling_frontend_test).

#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>
#include <api/family/prepared_prompt.h>

#include "family/impl/frontend/inkling_json.h"
#include "family/impl/frontend/inkling_parse.h"
#include "family/impl/frontend/inkling_render.h"
#include "family/impl/frontend/test_access.h"
#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace ink = sinfer::family::frontend_internal::inkling;
using sinfer::family::FrontendResources;
using sinfer::family::FrontendTestAccess;

namespace {

int g_reported = 0;

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + p.string()); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

void report(const std::string& what) {
    if (g_reported++ < 30) { std::cout << "  MISMATCH " << what << "\n"; }
}

std::vector<int> ints(const ink::PyValue* v) {
    std::vector<int> out;
    if (v == nullptr || !v->is_array()) { return out; }
    for (const ink::PyValue& x : v->items) { out.push_back(std::stoi(x.text)); }
    return out;
}

// The renderer vectors to check against: renderer-099 (the engine's default since 2026-11-21: a request that sets no reasoning
// effort renders at 0.99; those vectors expect vLLM's ids for the same body with chat_template_kwargs.reasoning_effort
// = 0.99, every other vector is vLLM's own as in renderer/). renderer/ (vLLM's default 0.9) only where 099 is missing.
fs::path renderer_set(const fs::path& vectors) {
    const fs::path set = fs::is_directory(vectors / "renderer-099") ? vectors / "renderer-099" : vectors / "renderer";
    static bool said = false;
    if (!said) {
        said = true;
        std::cout << "renderer vectors: " << set.filename().string() << "\n";
    }
    return set;
}

std::vector<fs::path> vector_files(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".json" && e.path().filename() != "INDEX.json") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// The chat-completions body through the serving layer to the prompt the Frontend prepares.
sinfer::PromptInput serve_input(const std::string& body) {
    sinfer::serve::RequestLimits limits;
    limits.raw_chat = true;
    sinfer::serve::GenerationRequest request =
        sinfer::serve::parse_chat_completion_request(nlohmann::ordered_json::parse(body), limits);
    request.raw_chat_body = std::make_shared<const std::string>(body);
    const sinfer::serve::ResolvedPromptSemantics semantics{};
    return sinfer::serve::to_prompt_input(request, semantics, {});
}

} // namespace

int main() {
    const char* model_dir = std::getenv("INKLING_MODEL_DIR");
    const char* vectors_dir = std::getenv("INKLING_VECTORS");
    if (model_dir == nullptr || vectors_dir == nullptr) {
        std::cout << "SKIP: set INKLING_MODEL_DIR and INKLING_VECTORS\n";
        return 77;
    }
    const fs::path model(model_dir), vectors(vectors_dir);
    FrontendResources resources;
    resources.tokenizer_json = read_file(model / "tokenizer.json");
    resources.tokenizer_config_json = read_file(model / "tokenizer_config.json");
    resources.generation_config_json = read_file(model / "generation_config.json");
    const sinfer::family::Frontend frontend = FrontendTestAccess::create_component(resources, false);

    int failures = 0;
    const sinfer::PromptCapabilities caps = frontend.prompt_capabilities();
    const bool detected = caps.inkling2_chat && frontend.supports_chat() && frontend.reasoning_close() == "<|end_message|>";
    std::cout << "detected Inkling-2: " << (detected ? "yes" : "NO") << " (reasoning close "
              << frontend.reasoning_close() << ")\n";
    if (!detected) { ++failures; }

    // A request without the body (another endpoint) is refused, not rendered from the lossy messages.
    try {
        sinfer::PromptInput input;
        input.messages.push_back(sinfer::ChatMessage{});
        (void)frontend.prepare(std::move(input));
        report("a prompt without raw_chat_request was rendered");
        ++failures;
    } catch (const std::invalid_argument&) {}

    // 1. prompts: body -> serving layer -> Frontend::prepare / count_tokens
    {
        int n = 0, bad = 0, refused_by_schema = 0, errors = 0;
        for (const fs::path& f : vector_files(renderer_set(vectors))) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
            const ink::PyValue& expected = *vec.find("expected");
            const bool want_error = expected.find("error") != nullptr;
            ++n;
            errors += want_error;
            sinfer::PromptInput input;
            try {
                input = serve_input(body);
            } catch (const std::exception& e) {
                if (!want_error) {
                    ++bad;
                    ++refused_by_schema;
                    report("schema refused " + name + ": " + e.what());
                }
                continue;
            }
            std::vector<int> got;
            std::string got_error;
            std::uint32_t counted = 0;
            try {
                const sinfer::PromptInput copy = input;
                const sinfer::family::PreparedPrompt prompt = frontend.prepare(std::move(input));
                const auto& ids = FrontendTestAccess::inspect(prompt).token_ids;
                got.assign(ids.begin(), ids.end());
                counted = frontend.count_tokens(copy);
            } catch (const std::invalid_argument& e) {
                got_error = e.what();
            }
            if (want_error) {
                if (got_error.empty()) {
                    ++bad;
                    report("render " + name + ": rendered, vLLM refuses");
                }
                continue;
            }
            const std::vector<int> want = ints(expected.find("prompt_token_ids"));
            if (!got_error.empty() || got != want || counted != want.size()) {
                ++bad;
                report("render " + name + (got_error.empty() ? " ids/count differ" : ": error " + got_error));
            }
        }
        std::cout << "prompts through the serving layer: " << n << " vectors (" << errors << " request errors), " << bad
                  << " mismatches (" << refused_by_schema << " refused by the chat-completions parser)\n";
        failures += bad;
        if (n == 0) { ++failures; }
    }

    // 2. replies: generated ids -> Frontend::token_texts (GenerationService's bytes) -> vLLM's output text and reply
    {
        int n = 0, bad = 0;
        std::vector<std::string> named;
        for (const auto& [id, spelling] : frontend.named_special_tokens()) { named.push_back(spelling); }
        const ink::SpecialTokens special = ink::SpecialTokens::resolve([&](std::string_view s) {
            const auto id = frontend.token_id(s);
            return id ? std::optional<int>(static_cast<int>(*id)) : std::nullopt;
        });
        const ink::ParserVocabulary vocabulary = ink::ParserVocabulary::from(special, named);
        for (const fs::path& f : vector_files(vectors / "parser")) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
            const ink::PyValue& expected = *vec.find("expected");
            const std::vector<int> out_ids = ints(vec.find("output_token_ids"));
            const std::string finish = vec.find("engine_finish_reason")->text;
            ++n;
            std::vector<sinfer::TokenId> ids(out_ids.begin(), out_ids.end());
            const std::vector<std::string> bytes = frontend.token_texts(ids);
            const std::string text = ink::detokenize_output(std::span<const std::string>(bytes), finish == "stop");
            std::string problem;
            if (text != expected.find("output_text")->text) {
                problem = " output_text differs";
            } else {
                const ink::ParsedOutput got = ink::parse_output(text, ink::parse_chat_request(body), finish, vocabulary);
                const ink::PyValue* calls = expected.find("tool_calls");
                if (got.tool_calls.size() != calls->items.size()) { problem += " call count"; }
                for (std::size_t k = 0; problem.empty() && k < got.tool_calls.size(); ++k) {
                    if (got.tool_calls[k].arguments != calls->items[k].find("arguments")->text) { problem += " args"; }
                }
                if (got.finish_reason != expected.find("finish_reason")->text) { problem += " finish"; }
            }
            if (!problem.empty()) {
                ++bad;
                report("reply " + name + problem);
            }
        }
        std::cout << "replies through Frontend::token_texts: " << n << " vectors, " << bad << " mismatches\n";
        failures += bad;
        if (n == 0) { ++failures; }
    }

    // 3. OutputOptions::report_token_ids (Inkling-2 streaming): the engine's output session reports exactly the ids
    //    each round accepted, without the stop token, and its text deltas are the same with the option on or off (every
    //    other request runs with it off: nothing changes for them).
    {
        int n = 0, bad = 0;
        const sinfer::family::PreparedPrompt prompt = frontend.prepare_tokens({1, 2, 3});
        for (const fs::path& f : vector_files(vectors / "parser")) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::vector<int> out_ids = ints(vec.find("output_token_ids"));
            if (out_ids.empty()) { continue; }
            ++n;
            std::vector<sinfer::TokenId> ids(out_ids.begin(), out_ids.end());
            struct Seen {
                std::string text[3];
                std::vector<sinfer::TokenId> reported;
                sinfer::FinishReason finish = sinfer::FinishReason::None;
                std::size_t empty_text_deltas = 0;
            };
            const auto run = [&](bool report_ids, std::size_t round) {
                Seen seen;
                sinfer::OutputOptions options;
                options.report_token_ids = report_ids;
                sinfer::family::OutputSession session = frontend.make_output_session(prompt, sinfer::StopPolicy{}, options);
                const auto budget = static_cast<std::uint32_t>(ids.size());
                std::size_t at = 0;
                while (at < ids.size()) {
                    const std::size_t take = std::min(round, ids.size() - at);
                    const auto decision = session.preview(
                        std::span<const sinfer::TokenId>(ids).subspan(at, take), budget - static_cast<std::uint32_t>(at),
                        sinfer::FinishReason::OutputLimit);
                    for (const sinfer::OutputDelta& delta : session.commit_preview()) {
                        if (!delta.token_ids.empty()) {
                            if (!delta.text.empty()) { ++seen.empty_text_deltas; }  // ids ride in a delta of their own
                            seen.reported.insert(seen.reported.end(), delta.token_ids.begin(), delta.token_ids.end());
                            continue;
                        }
                        seen.text[delta.channel == sinfer::OutputChannel::Reasoning ? 0
                                  : delta.channel == sinfer::OutputChannel::Tool    ? 2 : 1] += delta.text;
                    }
                    at += decision.accepted_tokens;
                    if (decision.finish_reason != sinfer::FinishReason::None) {
                        seen.finish = decision.finish_reason;
                        break;
                    }
                }
                return std::make_pair(seen, at);
            };
            for (const std::size_t round : {std::size_t{1}, std::size_t{3}}) {
                const auto [off, off_used] = run(false, round);
                const auto [on, on_used] = run(true, round);
                std::vector<sinfer::TokenId> want(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(on_used));
                if (on.finish == sinfer::FinishReason::StopToken && !want.empty()) { want.pop_back(); }
                std::string problem;
                if (off.text[0] != on.text[0] || off.text[1] != on.text[1] || off.text[2] != on.text[2] ||
                    off.finish != on.finish || off_used != on_used) {
                    problem += " text deltas differ with the option on";
                }
                if (!off.reported.empty()) { problem += " ids reported with the option off"; }
                if (on.reported != want) { problem += " reported ids differ from the accepted ids"; }
                if (on.empty_text_deltas != 0) { problem += " an id delta carried text"; }
                if (!problem.empty()) {
                    ++bad;
                    report("report_token_ids " + name + " round " + std::to_string(round) + problem);
                    break;
                }
            }
        }
        std::cout << "report_token_ids through OutputSession: " << n << " vectors x rounds of 1 and 3, " << bad
                  << " mismatches\n";
        failures += bad;
        if (n == 0) { ++failures; }
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " sinfer_inkling_wiring_test\n";
    return failures == 0 ? 0 : 1;
}
