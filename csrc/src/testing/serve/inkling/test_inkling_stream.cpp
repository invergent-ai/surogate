// ENGINE-INKLING lane f2: Inkling-2 streaming (stream=true) -- the standing gate check for every Inkling build.
//
// Over the parser vectors (rl/engine-inkling-2026-11-21/vectors/parser: 554 replies, real ones and edge variants, each
// with vLLM's own non-streamed reading), the generated ids are fed to inkling::StreamParser the way GenerationService
// feeds it (one engine round at a time: rounds of 1 token, rounds of 1-4 tokens as with MTP, one round), then
// finish() with the non-streamed reading. Checks:
//   (a) EXACT: the concatenated deltas equal the non-streamed message byte for byte -- reasoning, content, every tool
//       call's name and arguments -- and vLLM's expected reading; no streamed piece is ever retracted (finish() reports
//       consistency); every call opens once, first, with contiguous indices.
//   (b) INCREMENTAL: with 1-token rounds, a token inside a thinking / text block that adds a non-whitespace character
//       produces a delta in its own round, except for marker lookahead (a '<' that may start "<|...|>"): the longest
//       run of such silent tokens is reported and bounded (INKLING_STREAM_MAX_SILENT, default 8); the share of the
//       reasoning + content bytes sent before the reply ends is reported (gate >= 99 %); tool calls: named before their
//       block closes, and argument bytes streamed before close where the schema cannot coerce them.
//   Stop strings: the same with a stop string taken from the reply (the stream never sends past it).
// CPU only (no device, no libsinfer): INKLING_MODEL_DIR (tokenizer.json, tokenizer_config.json,
// generation_config.json) and INKLING_VECTORS; skips (77) without them. Lines starting "GATE" are the pass lines.

#include "family/impl/frontend/inkling_json.h"
#include "family/impl/frontend/inkling_parse.h"
#include "family/impl/frontend/inkling_render.h"
#include "family/impl/frontend/tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
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

std::string shown(const std::string& s) {
    std::string q;
    ink::py_json_quote(q, s, false);
    return q.size() > 300 ? q.substr(0, 300) + "..." : q;
}

void report(const std::string& what) {
    if (g_reported++ < g_max_report) { std::cout << "  MISMATCH " << what << "\n"; }
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

// The parser vectors' variant tag ("...-literal", "-cut", "-ws", ...), or "real" for an unchanged real reply.
std::string variant_of(const std::string& name) {
    const std::string tail = name.substr(name.rfind('-') + 1);
    const bool tag = !tail.empty() && std::all_of(tail.begin(), tail.end(), [](char c) { return c >= 'a' && c <= 'z'; });
    return tag ? tail : "real";
}

struct Call {
    std::string name;
    std::string arguments;
};

// A client's assembly of the stream; `problem` collects protocol faults.
struct Assembled {
    std::string reasoning, content;
    std::vector<Call> calls;
    std::vector<bool> opened;
    std::string problem;

    void add(const ink::StreamDelta& d) {
        reasoning += d.reasoning;
        content += d.content;
        for (const ink::StreamToolDelta& t : d.tool_calls) {
            const auto at = static_cast<std::size_t>(t.index);
            if (t.opens) {
                if (at != calls.size()) { problem += " call " + std::to_string(t.index) + " opens out of order"; continue; }
                calls.push_back({t.name, t.arguments.value_or("")});
                opened.push_back(true);
            } else {
                if (at >= calls.size()) { problem += " call " + std::to_string(t.index) + " piece before it opens"; continue; }
                calls[at].arguments += t.arguments.value_or("");
            }
        }
    }
};

std::string compare(const Assembled& got, const ink::ParsedOutput& want) {
    std::string p = got.problem;
    if (got.reasoning != want.reasoning.value_or("")) {
        p += " reasoning " + shown(got.reasoning) + " want " + shown(want.reasoning.value_or(""));
    }
    if (got.content != want.content.value_or("")) {
        p += " content " + shown(got.content) + " want " + shown(want.content.value_or(""));
    }
    if (got.calls.size() != want.tool_calls.size()) {
        p += " calls " + std::to_string(got.calls.size()) + " want " + std::to_string(want.tool_calls.size());
    } else {
        for (std::size_t k = 0; k < got.calls.size(); ++k) {
            if (got.calls[k].name != want.tool_calls[k].name || got.calls[k].arguments != want.tool_calls[k].arguments) {
                p += " call" + std::to_string(k) + " " + shown(got.calls[k].name) + " " + shown(got.calls[k].arguments) +
                     " want " + shown(want.tool_calls[k].name) + " " + shown(want.tool_calls[k].arguments);
            }
        }
    }
    return p;
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
    int max_silent_allowed = 8;
    if (const char* m = std::getenv("INKLING_STREAM_MAX_SILENT")) { max_silent_allowed = std::atoi(m); }
    const fs::path model(model_dir), vectors(vectors_dir);
    const fi::Tokenizer tokenizer(fi::TokenizerResources{read_file(model / "tokenizer.json"),
                                                         read_file(model / "tokenizer_config.json"),
                                                         read_file(model / "generation_config.json"), {}, false});
    const ink::SpecialTokens special =
        ink::SpecialTokens::resolve([&](std::string_view s) { return tokenizer.token_id(s); });
    std::vector<std::string> named;
    for (const auto& [id, spelling] : tokenizer.named_special_tokens()) { named.push_back(spelling); }
    const ink::ParserVocabulary vocabulary = ink::ParserVocabulary::from(special, named);

    // INKLING_STREAM_REPLAY=<dir>: replies a live server generated (same JSON shape as the parser vectors, without
    // `expected`), replayed in 1-token rounds: which rounds the stream sends something in, and where and why it stays
    // silent (block kind of each silent token, the longest silent span). A live stream with as many payload chunks as
    // the replay has sending rounds was not merged by the server. Exactness is checked against our own final reading.
    if (const char* replay_dir = std::getenv("INKLING_STREAM_REPLAY")) {
        int items = 0, inconsistent = 0;
        for (const fs::path& f : vector_files(replay_dir)) {
            const ink::PyValue vec = ink::py_json_loads(read_file(f));
            const std::string name = vec.find("case")->text;
            const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
            const std::vector<int> ids = ints(vec.find("output_token_ids"));
            const std::string finish = vec.find("engine_finish_reason")->text;
            const ink::ChatRequest request = ink::parse_chat_request(body);
            std::vector<std::string> bytes;
            for (const int id : ids) { bytes.push_back(tokenizer.decode_token_bytes(id, false)); }
            const std::size_t fed = finish == "stop" && !bytes.empty() ? bytes.size() - 1 : bytes.size();
            ink::StreamParser parser(request, vocabulary);
            Assembled got;
            int emit_any = 0, emit_text = 0, emit_calls = 0;
            std::map<std::string, int> silent_by_block;
            std::string block = "start";
            std::size_t run_start = 0, run_len = 0, best_start = 0, best_len = 0;
            std::string best_block;
            for (std::size_t i = 0; i < fed; ++i) {
                const int id = ids[i];
                if (id == special.content_thinking) { block = "thinking"; }
                else if (id == special.content_text) { block = "text"; }
                else if (id == special.content_invoke_tool_json) { block = "tool"; }
                else if (id == special.content_invoke_tool_text || id == special.content_tool_error) { block = "tooltext"; }
                else if (id == special.message_model) { block = "header"; }
                else if (id == special.end_message || id == special.content_model_end_sampling) { block = "between"; }
                else if (id == special.message_user || id == special.message_system || id == special.message_tool) {
                    block = "other-role";
                }
                const ink::StreamDelta d = parser.feed(std::span<const std::string>(bytes).subspan(i, 1));
                got.add(d);
                const bool text = !d.reasoning.empty() || !d.content.empty();
                emit_text += text;
                emit_calls += !d.tool_calls.empty();
                if (!d.empty()) {
                    ++emit_any;
                    run_len = 0;
                } else {
                    ++silent_by_block[block];
                    if (run_len == 0) { run_start = i; }
                    if (++run_len > best_len) {
                        best_len = run_len;
                        best_start = run_start;
                        best_block = block;
                    }
                }
            }
            const std::string all = ink::detokenize_output(std::span<const std::string>(bytes), finish == "stop");
            const ink::ParsedOutput final = ink::parse_output(all, request, finish, vocabulary);
            bool consistent = true;
            got.add(parser.finish(final, &consistent));
            const std::string problem = compare(got, final);
            if (!consistent || !problem.empty()) { ++inconsistent; }
            std::string excerpt;
            for (std::size_t k = best_start; k < best_start + best_len && k < fed && excerpt.size() < 240; ++k) { excerpt += bytes[k]; }
            std::cout << "REPLAY " << name << " tokens=" << fed << " rounds_sending=" << emit_any
                      << " rounds_sending_text=" << emit_text << " rounds_sending_calls=" << emit_calls << " silent_by_block={";
            bool first = true;
            for (const auto& [b, c] : silent_by_block) {
                std::cout << (first ? "" : ",") << b << ":" << c;
                first = false;
            }
            std::cout << "} longest_silent=" << best_len << "@" << best_start << "(" << best_block << ") "
                      << shown(excerpt) << " final: reasoning=" << final.reasoning.value_or("").size()
                      << "B content=" << final.content.value_or("").size() << "B calls=" << final.tool_calls.size()
                      << (consistent && problem.empty() ? " exact" : " NOT-EXACT" + problem) << "\n";
            ++items;
        }
        std::cout << "replay: " << items << " replies, " << inconsistent << " not exact\n";
        return inconsistent == 0 && items > 0 ? 0 : 1;
    }

    int n = 0, bad_exact = 0, bad_consistent = 0, runs = 0;
    int stop_cases = 0, bad_stop = 0;
    // (b) incrementality over 1-token rounds
    long long text_tokens = 0, silent_tokens = 0;
    int max_silent = 0;
    std::map<std::string, int> silent_texts, silent_cases;
    std::string max_silent_case;
    long long text_bytes_total = 0, text_bytes_early = 0;
    int calls_total = 0, calls_opened_early = 0, calls_args_early = 0;

    for (const fs::path& f : vector_files(vectors / "parser")) {
        const ink::PyValue vec = ink::py_json_loads(read_file(f));
        const std::string name = vec.find("case")->text;
        const std::string body = ink::py_json_dumps(*vec.find("body"), ink::PyDumpStyle{.ensure_ascii = false});
        const ink::PyValue& expected = *vec.find("expected");
        const std::vector<int> out_ids = ints(vec.find("output_token_ids"));
        const std::string finish = vec.find("engine_finish_reason")->text;
        ++n;
        ink::ChatRequest request;
        try {
            request = ink::parse_chat_request(body);
        } catch (const std::exception& e) {
            ++bad_exact;
            report("request " + name + ": " + e.what());
            continue;
        }
        // The ids a stream is fed: the stop token that ended the generation is not (OutputSession reports it so).
        std::vector<std::string> bytes;
        for (const int id : out_ids) { bytes.push_back(tokenizer.decode_token_bytes(id, false)); }
        const bool stop_terminated = finish == "stop" && !bytes.empty();
        const std::size_t fed = stop_terminated ? bytes.size() - 1 : bytes.size();
        const std::string text = ink::detokenize_output(std::span<const std::string>(bytes), finish == "stop");
        const ink::ParsedOutput final = ink::parse_output(text, request, finish, vocabulary);

        // vLLM's reading itself (sinfer_inkling_frontend_test checks final == it; here the stream is checked against
        // both).
        ink::ParsedOutput vllm;
        if (const ink::PyValue* r = expected.find("reasoning"); r && !r->is_null()) { vllm.reasoning = r->text; }
        if (const ink::PyValue* c = expected.find("content"); c && !c->is_null()) { vllm.content = c->text; }
        for (const ink::PyValue& call : expected.find("tool_calls")->items) {
            vllm.tool_calls.push_back({call.find("name")->text, call.find("arguments")->text});
        }

        std::mt19937 rng(static_cast<unsigned>(n) * 2654435761u);
        for (int scheme = 0; scheme < 3; ++scheme) {
            ++runs;
            ink::StreamParser parser(request, vocabulary);
            Assembled got;
            std::size_t i = 0;
            // (b) bookkeeping, 1-token rounds only
            enum class Block { Between, Thinking, Text, Tool } block = Block::Between;
            int silent = 0;
            while (i < fed) {
                std::size_t take = 1;
                if (scheme == 1) { take = 1 + rng() % 4; }
                if (scheme == 2) { take = fed - i; }
                take = std::min(take, fed - i);
                const ink::StreamDelta d = parser.feed(std::span<const std::string>(bytes).subspan(i, take));
                got.add(d);
                if (scheme == 0) {
                    const int id = out_ids[i];
                    if (id == special.content_thinking) { block = Block::Thinking; }
                    else if (id == special.content_text) { block = Block::Text; }
                    else if (id == special.content_invoke_tool_json) { block = Block::Tool; }
                    else if (id == special.end_message || id == special.message_model ||
                             id == special.content_model_end_sampling) { block = Block::Between; }
                    else {
                        const bool counts = (block == Block::Text) ||
                                            (block == Block::Thinking && request.include_reasoning);
                        const std::string& b = bytes[i];
                        const bool nonws = std::any_of(b.begin(), b.end(), [](char c) {
                            return c != ' ' && c != '\t' && c != '\n' && c != '\r';
                        });
                        if (counts && nonws) {
                            ++text_tokens;
                            if (d.reasoning.empty() && d.content.empty()) {
                                ++silent_tokens;
                                ++silent_texts[b];
                                ++silent_cases[variant_of(name)];
                                ++silent;
                            } else {
                                silent = 0;
                            }
                            if (silent > max_silent) {
                                max_silent = silent;
                                max_silent_case = name + " @token " + std::to_string(i);
                            }
                        }
                    }
                }
                i += take;
            }
            if (scheme == 0) { text_bytes_early += static_cast<long long>(got.reasoning.size() + got.content.size()); }
            const std::size_t streamed_calls = got.calls.size();
            std::vector<std::size_t> streamed_args;
            for (const Call& c : got.calls) { streamed_args.push_back(c.arguments.size()); }
            bool consistent = true;
            got.add(parser.finish(final, &consistent));
            if (!consistent) {
                ++bad_consistent;
                report("consistency " + name + " scheme " + std::to_string(scheme));
            }
            std::string problem = compare(got, final);
            if (problem.empty()) { problem = compare(got, vllm); }
            if (!problem.empty()) {
                ++bad_exact;
                report("stream " + name + " scheme " + std::to_string(scheme) + ":" + problem);
            }
            if (scheme == 0) {
                text_bytes_total += static_cast<long long>(final.reasoning.value_or("").size() +
                                                           final.content.value_or("").size());
                calls_total += static_cast<int>(final.tool_calls.size());
                calls_opened_early += static_cast<int>(streamed_calls);
                for (std::size_t k = 0; k < streamed_args.size() && k < final.tool_calls.size(); ++k) {
                    // arguments that went out before the reply ended (whole at block close, or as written)
                    if (streamed_args[k] != 0) { ++calls_args_early; }
                }
            }
        }

        // Stop strings: a snippet of the reply's text as the request's stop string; the engine's reply is cut at its
        // first occurrence. The stream must not send past it and must add up to the cut reading.
        if (text.size() > 40) {
            for (const std::size_t at : {text.size() / 3, text.size() / 2}) {
                std::size_t start = at;
                while (start < text.size() && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) { ++start; }
                std::size_t end = start + 4;
                while (end < text.size() && (static_cast<unsigned char>(text[end]) & 0xC0) == 0x80) { ++end; }
                if (end > text.size()) { continue; }
                const std::string stop = text.substr(start, end - start);
                const std::size_t cut = text.find(stop);
                const ink::ParsedOutput cut_final =
                    ink::parse_output(std::string_view(text).substr(0, cut), request, "stop", vocabulary);
                ++stop_cases;
                ink::StreamParser parser(request, vocabulary, {stop});
                Assembled got;
                for (std::size_t k = 0; k < fed; ++k) { got.add(parser.feed(std::span<const std::string>(bytes).subspan(k, 1))); }
                bool consistent = true;
                got.add(parser.finish(cut_final, &consistent));
                const std::string problem = compare(got, cut_final);
                if (!consistent || !problem.empty()) {
                    ++bad_stop;
                    report("stop-string " + name + " stop " + shown(stop) + (consistent ? "" : " inconsistent") + problem);
                }
            }
        }
    }

    const double early = text_bytes_total == 0 ? 1.0 : static_cast<double>(text_bytes_early) / static_cast<double>(text_bytes_total);
    std::cout << "vectors " << n << ", stream runs " << runs << " (rounds of 1 token, 1-4 tokens, whole reply)\n";
    std::cout << "(a) exact: " << bad_exact << " mismatches, " << bad_consistent << " retractions\n";
    std::cout << "    stop strings: " << stop_cases << " cases, " << bad_stop << " mismatches\n";
    std::cout << "(b) text tokens (non-whitespace, inside thinking / text blocks) " << text_tokens << ", silent in their own round "
              << silent_tokens << ", longest silent run " << max_silent << (max_silent_case.empty() ? "" : " (" + max_silent_case + ")")
              << "\n";
    {
        std::vector<std::pair<int, std::string>> top;
        for (const auto& [t, c] : silent_texts) { top.emplace_back(c, t); }
        std::sort(top.rbegin(), top.rend());
        std::cout << "    silent tokens by text:";
        for (std::size_t k = 0; k < top.size() && k < 12; ++k) { std::cout << " " << shown(top[k].second) << "x" << top[k].first; }
        std::cout << "\n    silent tokens by vector kind:";
        for (const auto& [kind, c] : silent_cases) { std::cout << " " << kind << "=" << c; }
        std::cout << "\n";
    }
    std::cout << "    reasoning + content bytes sent before the reply ended: " << text_bytes_early << " / " << text_bytes_total
              << " = " << early * 100.0 << " %\n";
    std::cout << "    tool calls: " << calls_total << ", opened before the reply ended " << calls_opened_early
              << ", with arguments sent before the reply ended " << calls_args_early << "\n";
    const bool pass_a = bad_exact == 0 && bad_consistent == 0 && bad_stop == 0 && n > 0;
    const bool pass_b = max_silent <= max_silent_allowed && early >= 0.99;
    std::cout << "GATE inkling-stream (a) exact " << (pass_a ? "PASS" : "FAIL") << "\n";
    std::cout << "GATE inkling-stream (b) incremental " << (pass_b ? "PASS" : "FAIL") << "\n";
    const bool pass = pass_a && pass_b;
    std::cout << (pass ? "PASS" : "FAIL") << " sinfer_inkling_stream_test\n";
    return pass ? 0 : 1;
}
