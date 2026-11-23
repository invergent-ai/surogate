// The native Inkling-2 frontend against vLLM's own Python, token for token.
//
// Vectors (not in this repository; made by tools/inkling_ref_driver.py inside the vLLM Inkling image):
//   renderer/*.json          request body -> prompt token ids (or the request error vLLM raises)
//   renderer/tokenizer/tokenize.jsonl   text -> ids (special tokens split, and parsed)
//   parser/*.json            output token ids + request -> detokenized text, reasoning, content, tool calls, finish
// Runs on the CPU (no device). Exits 77 (skip) without INKLING_MODEL_DIR (tokenizer.json, tokenizer_config.json,
// generation_config.json) and INKLING_VECTORS. INKLING_MAX_REPORT bounds the printed mismatches (default 20).

#include "family/impl/frontend/inkling_json.h"
#include "family/impl/frontend/inkling_parse.h"
#include "family/impl/frontend/inkling_render.h"
#include "family/impl/frontend/tokenizer.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace fi = sinfer::family::frontend_internal;
namespace ink = sinfer::family::frontend_internal::inkling;

namespace {

int g_reported = 0;
int g_max_report = 20;

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + p.string()); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::vector<int> ints(const ink::PyValue* v) {
    std::vector<int> out;
    if (v == nullptr || !v->is_array()) { return out; }
    for (const ink::PyValue& x : v->items) { out.push_back(std::stoi(x.text)); }
    return out;
}

std::string show_opt(const std::optional<std::string>& s) {
    if (!s) { return "None"; }
    std::string q;
    ink::py_json_quote(q, *s, false);
    return q.size() > 400 ? q.substr(0, 400) + "..." : q;
}

std::optional<std::string> opt_string(const ink::PyValue* v) {
    if (v == nullptr || v->is_null()) { return std::nullopt; }
    return v->text;
}

void report(const std::string& what) {
    if (g_reported++ < g_max_report) { std::cout << "  MISMATCH " << what << "\n"; }
}

std::string first_diff(const std::vector<int>& a, const std::vector<int>& b) {
    std::size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) { ++i; }
    std::ostringstream o;
    o << "at " << i << " (got " << a.size() << " ids, want " << b.size() << ")";
    if (i < a.size() || i < b.size()) {
        o << " got [";
        for (std::size_t k = i; k < std::min(a.size(), i + 8); ++k) { o << a[k] << " "; }
        o << "] want [";
        for (std::size_t k = i; k < std::min(b.size(), i + 8); ++k) { o << b[k] << " "; }
        o << "]";
    }
    return o.str();
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
    if (!fs::is_directory(dir)) { return files; }
    for (const auto& e : fs::directory_iterator(dir)) {
        if (e.is_regular_file() && e.path().extension() == ".json" && e.path().filename() != "INDEX.json") {
            files.push_back(e.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

} // namespace

int main() {
    const char* model_dir = std::getenv("INKLING_MODEL_DIR");
    const char* vectors_dir = std::getenv("INKLING_VECTORS");
    if (model_dir == nullptr || vectors_dir == nullptr) {
        std::cout << "SKIP: set INKLING_MODEL_DIR and INKLING_VECTORS\n";
        return 77;
    }
    if (const char* m = std::getenv("INKLING_MAX_REPORT")) { g_max_report = std::atoi(m); }
    const fs::path model(model_dir), vectors(vectors_dir);
    const std::string tokenizer_json = read_file(model / "tokenizer.json");
    const std::string tokenizer_config = read_file(model / "tokenizer_config.json");
    const std::string generation_config = read_file(model / "generation_config.json");
    const fi::Tokenizer tokenizer(fi::TokenizerResources{tokenizer_json, tokenizer_config, generation_config, {}, false});

    const ink::TextEncoder encode = [&](std::string_view text) {
        return tokenizer.encode(text, fi::EncodeOptions{.parse_added_tokens = false, .add_bos = false});
    };
    const ink::SpecialTokens special =
        ink::SpecialTokens::resolve([&](std::string_view s) { return tokenizer.token_id(s); });
    std::vector<std::string> named;
    for (const auto& [id, spelling] : tokenizer.named_special_tokens()) { named.push_back(spelling); }
    const ink::ParserVocabulary vocabulary = ink::ParserVocabulary::from(special, named);
    std::cout << "named special tokens " << named.size() << ", dropped by the parser " << vocabulary.drop_spellings.size() << "\n";

    int failures = 0;

    // 1. tokenizer: text -> ids, special tokens split (the renderer's encode) and parsed
    {
        int n = 0, bad_split = 0, bad_parse = 0;
        std::ifstream in(vectors / "renderer" / "tokenizer" / "tokenize.jsonl");
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) { continue; }
            const ink::PyValue item = ink::py_json_loads(line);
            const std::string& text = item.find("text")->text;
            ++n;
            const std::vector<int> got = encode(text);
            const std::vector<int> want = ints(item.find("ids_split_special"));
            if (got != want) {
                ++bad_split;
                report("tokenize(split) " + item.find("case")->text + " " + first_diff(got, want));
            }
            const std::vector<int> got_p = tokenizer.encode(text, fi::EncodeOptions{.parse_added_tokens = true, .add_bos = false});
            const std::vector<int> want_p = ints(item.find("ids_parse_special"));
            if (got_p != want_p) {
                ++bad_parse;
                report("tokenize(parse) " + item.find("case")->text + " " + first_diff(got_p, want_p));
            }
        }
        std::cout << "tokenizer: " << n << " texts, " << bad_split << " split-special mismatches, " << bad_parse
                  << " parse-special mismatches\n";
        failures += bad_split + bad_parse;
        if (n == 0) { ++failures; }
    }

    // 2. renderer: request body -> prompt ids
    {
        int n = 0, bad = 0, errors = 0, effort_default = 0, effort_default_bad = 0, effort_explicit = 0;
        for (const fs::path& f : vector_files(renderer_set(vectors))) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
            const ink::PyValue& expected = *vec.find("expected");
            const bool want_error = expected.find("error") != nullptr;
            const ink::PyValue* meta = vec.find("meta");
            const bool by_default = meta != nullptr && meta->is_object() && meta->find("effort_default") != nullptr;
            ++n;
            errors += want_error;
            effort_default += by_default;
            if (!want_error && !by_default) {
                // a request that names its effort (top-level or chat_template_kwargs, not null / "auto"): kept as sent
                const ink::PyValue* b = vec.find("body");
                const ink::PyValue* top = b->find("reasoning_effort");
                const ink::PyValue* ctk = b->find("chat_template_kwargs");
                const ink::PyValue* kw = ctk != nullptr && ctk->is_object() ? ctk->find("reasoning_effort") : nullptr;
                effort_explicit += (top != nullptr && !top->is_null()) ||
                                   (kw != nullptr && !kw->is_null() && !(kw->is_string() && kw->text == "auto"));
            }
            const int bad_before = bad;
            std::vector<int> got;
            std::string got_error;
            try {
                got = ink::render_chat(ink::parse_chat_request(body), encode, special);
            } catch (const std::exception& e) {
                got_error = e.what();
            }
            if (want_error) {
                if (got_error.empty()) {
                    ++bad;
                    report("render " + name + ": rendered " + std::to_string(got.size()) + " ids, vLLM refuses: " +
                           expected.find("error")->find("message")->text.substr(0, 160));
                }
                continue;
            }
            if (!got_error.empty()) {
                ++bad;
                report("render " + name + ": error '" + got_error + "', vLLM renders");
                effort_default_bad += by_default;
                continue;
            }
            const std::vector<int> want = ints(expected.find("prompt_token_ids"));
            if (got != want) {
                ++bad;
                report("render " + name + " " + first_diff(got, want));
            }
            if (by_default && bad != bad_before) { ++effort_default_bad; }
        }
        std::cout << "renderer: " << n << " vectors (" << errors << " request errors), " << bad << " mismatches\n";
        std::cout << "  effort: " << effort_default << " set none (rendered at the default 0.99), " << effort_default_bad
                  << " mismatches; " << effort_explicit << " set one (kept)\n";
        failures += bad;
        if (n == 0) { ++failures; }
    }

    // 3. parser: output ids -> text -> message fields
    {
        int n = 0, bad = 0;
        const auto token_bytes = [&](int id) { return tokenizer.decode_token_bytes(id, false); };
        for (const fs::path& f : vector_files(vectors / "parser")) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
            const ink::PyValue& expected = *vec.find("expected");
            const std::vector<int> out_ids = ints(vec.find("output_token_ids"));
            const std::string finish = vec.find("engine_finish_reason")->text;
            ++n;
            std::string problem;
            try {
                const ink::ChatRequest request = ink::parse_chat_request(body);
                const std::string text = ink::detokenize_output(out_ids, finish == "stop", token_bytes);
                if (text != expected.find("output_text")->text) {
                    problem = "output_text " + show_opt(text) + " want " + show_opt(expected.find("output_text")->text);
                } else {
                    const ink::ParsedOutput got = ink::parse_output(text, request, finish, vocabulary);
                    const auto want_reasoning = opt_string(expected.find("reasoning"));
                    const auto want_content = opt_string(expected.find("content"));
                    if (got.reasoning != want_reasoning) {
                        problem += " reasoning " + show_opt(got.reasoning) + " want " + show_opt(want_reasoning);
                    }
                    if (got.content != want_content) {
                        problem += " content " + show_opt(got.content) + " want " + show_opt(want_content);
                    }
                    const ink::PyValue* calls = expected.find("tool_calls");
                    if (calls->items.size() != got.tool_calls.size()) {
                        problem += " tool_calls " + std::to_string(got.tool_calls.size()) + " want " +
                                   std::to_string(calls->items.size());
                    } else {
                        for (std::size_t k = 0; k < got.tool_calls.size(); ++k) {
                            const std::string& wn = calls->items[k].find("name")->text;
                            const std::string& wa = calls->items[k].find("arguments")->text;
                            if (got.tool_calls[k].name != wn || got.tool_calls[k].arguments != wa) {
                                problem += " call" + std::to_string(k) + " " + show_opt(got.tool_calls[k].name) + " " +
                                           show_opt(got.tool_calls[k].arguments) + " want " + show_opt(wn) + " " + show_opt(wa);
                            }
                        }
                    }
                    if (got.finish_reason != expected.find("finish_reason")->text) {
                        problem += " finish " + got.finish_reason + " want " + expected.find("finish_reason")->text;
                    }
                    const int rt = ink::count_reasoning_tokens(out_ids, special);
                    if (std::to_string(rt) != expected.find("reasoning_tokens")->text) {
                        problem += " reasoning_tokens " + std::to_string(rt) + " want " + expected.find("reasoning_tokens")->text;
                    }
                }
            } catch (const std::exception& e) {
                problem = std::string("exception ") + e.what();
            }
            if (!problem.empty()) {
                ++bad;
                report("parse " + name + ":" + problem);
            }
        }
        std::cout << "parser: " << n << " vectors, " << bad << " mismatches\n";
        failures += bad;
        if (n == 0) { ++failures; }
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " sinfer_inkling_frontend_test\n";
    return failures == 0 ? 0 : 1;
}
