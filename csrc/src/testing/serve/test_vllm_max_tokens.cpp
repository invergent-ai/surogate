// prompt + max_tokens against the context, decided as vLLM 0.31 decides it (serve/vllm_max_tokens.h).
//
// fixtures/serve/vllm_max_tokens/parity.jsonl: what vLLM 0.31.0 answered over HTTP (its GPU-less render server, which
// runs `vllm serve`'s request validation, on Surogate3.7-35B-A3B's tokenizer and chat template; vllm_reference.py
// beside it re-derives every row) for chat and completions requests at the edges of a 4096- and a 262144-token
// context, with and without a generation-config default: max_tokens / max_completion_tokens absent, null, 0, -1, 1,
// 16, 1000, 32768, the rest of the context and one either side of it, max_model_len - 1, max_model_len and one more,
// and the three around the max_tokens where the prompt text's character bound starts to refuse; min_tokens against an
// explicit and a defaulted budget; stream false and true. Each row has the prompt's token count and character count,
// and vLLM's answer: the max_tokens it would generate with, or the HTTP status and the exact response bytes.
//
// Chat and completions text rows go through the engine's own request parser (parse_chat_completion_request /
// parse_completion_request, as the HTTP handlers call it) and max_tokens_query + decide_vllm_max_tokens, the steps
// GenerationService::prepare takes; a refusal must be byte-identical to vLLM's, an accepted budget equal. Rows with a
// token-id prompt (which the engine's /v1/completions does not take; its chat token-in path counts the same way)
// check the decision alone.
//
// Documented exception: /v1/completions without a max_tokens (absent or null). vLLM reads it as 16 (the OpenAI
// default) and checks that like a given value; the engine gives such a request its --default-max-tokens capped to the
// rest of the context, as vLLM does for chat. Those rows check that, handed vLLM's 16, the decision is vLLM's answer,
// and that the engine's own answer is the chat rule's.

#include "family/impl/frontend/tokenizer.h"
#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/serve_options.h"
#include "serve/vllm_max_tokens.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

using OJson = nlohmann::ordered_json;
using namespace sinfer::serve;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        if (failures <= 40) { std::cerr << "FAIL: " << message << '\n'; }
    }
}

std::string fixture(const char* name) {
    return std::string(SINFER_SOURCE_DIR) + "/fixtures/serve/vllm_max_tokens/" + name;
}

// The engine's answer in the fixture's terms.
OJson answer(const MaxTokensDecision& decision) {
    if (decision.error) {
        return OJson{{"status", decision.error->status}, {"http_body", make_error_body(*decision.error)}};
    }
    return OJson{{"status", 200}, {"max_tokens", decision.budget}};
}

ServeOptions parse_options(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

std::string describe(const OJson& row) {
    return "row " + std::to_string(row.at("id").get<int>()) + " " + row.at("endpoint").get<std::string>() + "/" +
           row.at("prompt").get<std::string>() + " C=" + std::to_string(row.at("max_model_len").get<long>()) +
           " P=" + std::to_string(row.at("prompt_tokens").get<long>()) +
           " chars=" + std::to_string(row.at("prompt_characters").get<long>()) +
           " default=" + row.at("default_max_tokens").dump() + " " + row.at("body").dump();
}

} // namespace

int main() {
    std::ifstream in(fixture("parity.jsonl"));
    check(in.good(), "parity.jsonl missing");
    std::string line;
    int rows = 0, parsed = 0, exceptions = 0, refusals = 0, char_refusals = 0, at_least = 0, streams = 0;
    std::int64_t vocabulary_chars = 0;
    while (std::getline(in, line)) {
        const OJson row = OJson::parse(line);
        ++rows;
        const std::string id      = describe(row);
        const bool chat           = row.at("endpoint") == "chat";
        const bool text           = row.at("prompt") == "text";
        const std::int64_t C      = row.at("max_model_len");
        const std::int64_t P      = row.at("prompt_tokens");
        const std::int64_t chars  = row.at("prompt_characters");
        const std::int64_t L      = row.at("max_chars_per_token");
        const OJson& body         = row.at("body");
        const OJson& def          = row.at("default_max_tokens");
        const OJson& expected     = row.at("vllm");
        vocabulary_chars          = L;
        if (expected.at("status") != 200) {
            ++refusals;
            const std::string bytes = expected.at("http_body");
            if (bytes.find("\"param\":\"input_text\"") != std::string::npos) { ++char_refusals; }
            if (bytes.find("contains at least") != std::string::npos) { ++at_least; }
        }
        if (body.value("stream", false)) { ++streams; }

        RequestLimits limits;
        limits.default_max_tokens = def.is_null() ? INT_MAX : def.get<int>();
        limits.vllm_max_tokens    = true;
        const bool absent = (!body.contains("max_tokens") || body.at("max_tokens").is_null()) &&
                            (!body.contains("max_completion_tokens") || body.at("max_completion_tokens").is_null());

        if (!text) {
            // A token-id prompt: the decision alone, with vLLM's CompletionRequest default of 16.
            MaxTokensQuery query;
            query.max_model_len = C;
            query.prompt_tokens = P;
            query.requested     = absent ? 16 : body.at("max_tokens").get<std::int64_t>();
            query.min_tokens    = body.value("min_tokens", 0);
            query.text_prompt   = false;
            query.max_chars_per_token = L;
            const OJson got = answer(decide_vllm_max_tokens(query));
            check(got == expected, id + "\n  engine " + got.dump() + "\n  vllm   " + expected.dump());
            continue;
        }

        OJson request = body;
        request["model"] = "m";
        if (chat) {
            request["messages"] = OJson::array({OJson{{"role", "user"}, {"content", "x"}}});
        } else {
            request["prompt"] = "x";
        }
        GenerationRequest parsed_request;
        try {
            parsed_request = chat ? parse_chat_completion_request(request, limits) : parse_completion_request(request, limits);
        } catch (const ApiException& error) {
            check(false, id + ": the parser refused it: " + error.error().message);
            continue;
        }
        ++parsed;
        check(parsed_request.stream == body.at("stream").get<bool>(), id + ": stream not parsed");
        const MaxTokensQuery query = max_tokens_query(parsed_request, C, P, chars, L);
        check(query.text_prompt, id + ": a text prompt");
        const OJson got = answer(decide_vllm_max_tokens(query));
        if (!chat && absent) {
            ++exceptions;
            MaxTokensQuery as_vllm = query;
            as_vllm.requested      = 16;
            as_vllm.default_max_tokens.reset();
            const OJson with16 = answer(decide_vllm_max_tokens(as_vllm));
            check(with16 == expected, id + " with vLLM's 16\n  engine " + with16.dump() + "\n  vllm   " +
                                          expected.dump());
            // The engine's own rule: the chat one -- the default capped to the rest, refusals only for a prompt that
            // leaves nothing.
            const std::int64_t rest   = C - P;
            const std::int64_t budget = std::min<std::int64_t>(rest, def.is_null() ? INT_MAX : def.get<std::int64_t>());
            const std::int64_t min    = body.value("min_tokens", 0);
            if (rest >= 1 && min <= budget) {
                check(got == OJson{{"status", 200}, {"max_tokens", budget}}, id + " engine rule: " + got.dump());
            } else {
                check(got.at("status") == 400, id + " engine rule should refuse: " + got.dump());
            }
            continue;
        }
        check(got == expected, id + "\n  engine " + got.dump() + "\n  vllm   " + expected.dump());
    }

    // The tau3-Bench task_091 shape, spelled out: a ~256k-token chat prompt asking for 32768 on a 262144 context.
    // vLLM tokenizes the rendered prompt only up to 262144 - 32768 + 1 tokens, hence "at least".
    {
        GenerationRequest request;
        request.max_tokens     = 32768;
        request.max_tokens_set = true;
        const MaxTokensDecision d = decide_vllm_max_tokens(max_tokens_query(request, 262144, 255970, 1000000, 128));
        check(d.error.has_value() && d.error->status == 400, "task_091 shape not refused");
        if (d.error) {
            check(make_error_body(*d.error) ==
                      R"j({"error":{"message":"This model's maximum context length is 262144 tokens. However, you )j"
                      R"j(requested 32768 output tokens and your prompt contains at least 229377 input tokens, for a )j"
                      R"j(total of at least 262145 tokens. Please reduce the length of the input prompt or the number )j"
                      R"j(of requested output tokens. (parameter=input_tokens, value=229377)","type":"BadRequestError",)j"
                      R"j("param":"input_tokens","code":400}})j",
                  "task_091 body: " + make_error_body(*d.error));
        }
        request.max_tokens_set = false; // no max_tokens: the rest of the context, no error
        request.max_tokens     = 262144;
        const MaxTokensDecision rest = decide_vllm_max_tokens(max_tokens_query(request, 262144, 255970, 1000000, 128));
        check(!rest.error && rest.budget == 6174, "task_091 without max_tokens should get the remaining 6174");
    }
    // Responses names its field max_output_tokens.
    {
        GenerationRequest request;
        request.max_tokens       = 300000;
        request.max_tokens_set   = true;
        request.max_tokens_param = "max_output_tokens";
        const auto error = vllm_max_tokens_above_context(max_tokens_query(request, 262144, 0, 0, 0));
        check(error && error->param == "max_output_tokens" &&
                  error->message == "max_output_tokens=300000 cannot be greater than max_model_len=262144. Please "
                                    "request fewer output tokens. (parameter=max_output_tokens, value=300000)",
              "responses param name");
    }
    // A token-in request is counted as sent: no character bound, no truncated count.
    {
        GenerationRequest request;
        request.prompt_token_ids = {1, 2, 3};
        request.max_tokens       = 4090;
        request.max_tokens_set   = true;
        const MaxTokensQuery query = max_tokens_query(request, 4096, 100, 5000, 128);
        check(!query.text_prompt && query.prompt_characters == 0, "token-in prompt is not text");
        const MaxTokensDecision d = decide_vllm_max_tokens(query);
        check(d.error && d.error->message.find("contains 100 input tokens, for a total of 4190 tokens") !=
                             std::string::npos,
              "token-in count: " + (d.error ? d.error->message : std::string("accepted")));
    }
    // --overlong-max-tokens cap keeps the parser's earlier refusal of a max_tokens below 1, and the earlier shape.
    {
        RequestLimits limits; // vllm_max_tokens = false
        for (const char* field : {"max_tokens", "max_completion_tokens"}) {
            OJson request = {{"model", "m"}, {"messages", OJson::array({{{"role", "user"}, {"content", "x"}}})},
                             {field, 0}};
            bool refused = false;
            try { (void)parse_chat_completion_request(request, limits); } catch (const ApiException& error) {
                refused = error.error().message == "max_tokens must be positive" && error.error().numeric_code == 0;
            }
            check(refused, std::string("cap mode should refuse ") + field + "=0 while parsing");
        }
        ApiError old;
        old.message = "m";
        old.code    = "context_length_exceeded";
        check(make_error_body(old) ==
                  R"({"error":{"code":"context_length_exceeded","message":"m","param":null,"type":"invalid_request_error"}})",
              "the engine's own error shape changed: " + make_error_body(old));
    }
    // The flag.
    {
        check(parse_options({"s", "m.sinfer"}).overlong_max_tokens == OverlongMaxTokens::Reject, "default is reject");
        check(parse_options({"s", "m.sinfer", "--overlong-max-tokens", "cap"}).overlong_max_tokens ==
                  OverlongMaxTokens::Cap, "cap not parsed");
        check(parse_options({"s", "m.sinfer", "--overlong-max-tokens", "reject"}).overlong_max_tokens ==
                  OverlongMaxTokens::Reject, "reject not parsed");
        for (const char* bad : {"", "truncate", "Reject"}) {
            bool refused = false;
            try { (void)parse_options({"s", "m.sinfer", "--overlong-max-tokens", bad}); } catch (const std::invalid_argument&) {
                refused = true;
            }
            check(refused, std::string("--overlong-max-tokens accepted ") + bad);
        }
    }
    // The tokenizer's max_chars_per_token is Python's max(len(t) for t in get_vocab()): code points, added tokens
    // included.
    {
        using sinfer::family::frontend_internal::Tokenizer;
        const nlohmann::json vocab_fixture = {
            {"model", {{"type", "BPE"}, {"byte_fallback", false},
                       {"vocab", {{"a", 0}, {"ĠĠĠĠ", 1}, {"ééé", 2}}},
                       {"merges", nlohmann::json::array()}}},
            {"normalizer", nullptr},
            {"added_tokens", nlohmann::json::array({{{"id", 3}, {"content", "<|x|>"}, {"special", true},
                                                     {"single_word", false}, {"lstrip", false}, {"rstrip", false},
                                                     {"normalized", false}}})}};
        const Tokenizer small({vocab_fixture.dump(), "{}", R"({"eos_token_id":3})"});
        check(small.max_chars_per_token() == 5, "max_chars_per_token of the small vocabulary: " +
                                                    std::to_string(small.max_chars_per_token()));
        // The checkpoint the fixture was made with, when its files are at hand.
        if (const char* dir = std::getenv("SUROGATE_MAX_TOKENS_TOKENIZER_DIR")) {
            const std::string root = dir;
            const std::string tokenizer_json = read_file(root + "/tokenizer.json");
            const std::string config_json    = read_file(root + "/tokenizer_config.json");
            const std::string generation     = read_file(root + "/generation_config.json");
            const Tokenizer real({tokenizer_json, config_json, generation});
            check(static_cast<std::int64_t>(real.max_chars_per_token()) == vocabulary_chars,
                  "the checkpoint's max_chars_per_token " + std::to_string(real.max_chars_per_token()) +
                      ", vLLM's " + std::to_string(vocabulary_chars));
            std::cout << "checkpoint tokenizer: max_chars_per_token=" << real.max_chars_per_token() << '\n';
        }
    }

    check(rows > 3000 && parsed > 2000 && refusals > 1000 && char_refusals > 50 && at_least > 200 && streams > 1000,
          "fixture smaller than expected");
    std::cout << "vllm_max_tokens: " << rows << " vLLM answers (" << refusals << " refusals, of them " << char_refusals
              << " on the character bound and " << at_least << " worded 'at least'; " << streams << " streaming), "
              << parsed << " through the request parser, " << exceptions << " documented exceptions; failures="
              << failures << '\n';
    return failures == 0 ? 0 : 1;
}
