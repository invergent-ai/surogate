// Thinking levels on Gemma 4's own chat template and tokenizer: the thinking prompt the decisions
// endpoint renders for a question, beside the one-pass prompt v1 renders for it, built the way
// GenerationService::decide builds both (a system and a user turn through `to_prompt_input`, the
// template's thinking switch off for v1 and on for a level) and rendered by the serving frontend.
//
// It pins what the reference (jev scripts/think_when_unsure_pilot_v1.py) sent through
// /v1/chat/completions with `chat_template_kwargs: {"enable_thinking": true}`: the two prompts
// differ only in the system prompt, the template's `<|think|>` switch and the generation prompt
// (thinking off ends on the empty `<|channel>thought\n<channel|>` block, thinking on at
// `<|turn>model\n`, where the model opens the thought channel itself); the user message is the
// same bytes; `<channel|>` is one token, and each option letter after it is the one token v1 reads
// (and the bare letter the reference read).
//
// CPU only. Needs a Gemma 4 tokenizer directory (tokenizer.json, tokenizer_config.json,
// chat_template.jinja, generation_config.json): SUROGATE_GEMMA4_TOKENIZER_DIR, else the Hugging
// Face cache's google/gemma-4-26B-A4B-it snapshot. Without one it exits 77 (skipped).
#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>

#include "family/impl/frontend/test_access.h"
#include "serve/decisions_thinking.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

using namespace sinfer::serve;
using Frontend        = sinfer::family::Frontend;
using FrontendAccess  = sinfer::family::FrontendTestAccess;

namespace {

int failures = 0;
void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error("cannot read " + path.string()); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::filesystem::path tokenizer_dir() {
    std::vector<std::filesystem::path> candidates;
    if (const char* env = std::getenv("SUROGATE_GEMMA4_TOKENIZER_DIR"); env != nullptr && *env != '\0') {
        candidates.emplace_back(env);
    }
    if (const char* home = std::getenv("HOME"); home != nullptr) {
        const std::filesystem::path snapshots = std::filesystem::path(home) /
            ".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots";
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(snapshots, error)) {
            candidates.push_back(entry.path());
        }
    }
    for (const auto& dir : candidates) {
        bool complete = true;
        for (const char* name : {"tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"}) {
            complete = complete && std::filesystem::exists(dir / name);
        }
        if (complete) { return dir; }
    }
    return {};
}

/// GenerationService::decide's `prepare_chat_as`: one system and one user turn, the template's
/// switches resolved the way the endpoint resolves them.
sinfer::PromptInput decision_chat_input(std::string_view system, std::string user, bool enable_thinking,
                                        const sinfer::PromptCapabilities& capabilities) {
    GenerationRequest turns;
    turns.model           = "rune";
    turns.enable_thinking = enable_thinking;
    ChatTurn system_turn;
    system_turn.role = sinfer::ChatRole::System;
    system_turn.content.push_back(ContentPart{ContentKind::Text, std::string(system), "text"});
    ChatTurn user_turn;
    user_turn.role = sinfer::ChatRole::User;
    user_turn.content.push_back(ContentPart{ContentKind::Text, std::move(user), "text"});
    turns.messages = {std::move(system_turn), std::move(user_turn)};
    const ResolvedPromptSemantics semantics = resolve_prompt_semantics(turns, ServeOptions{}, capabilities);
    return to_prompt_input(turns, semantics, {});
}

std::string text_of(const Frontend& frontend, const std::vector<sinfer::TokenId>& ids) {
    std::string text;
    for (const std::string& piece : frontend.token_texts(std::span<const sinfer::TokenId>(ids))) { text += piece; }
    return text;
}

std::vector<sinfer::TokenId> encode(const Frontend& frontend, std::string_view text) {
    return FrontendAccess::inspect(frontend.prepare_text(text, false)).token_ids;
}

bool extends(const std::vector<sinfer::TokenId>& longer, const std::vector<sinfer::TokenId>& prefix, std::size_t by) {
    return longer.size() == prefix.size() + by && std::equal(prefix.begin(), prefix.end(), longer.begin());
}

} // namespace

int main() {
    const std::filesystem::path dir = tokenizer_dir();
    if (dir.empty()) {
        std::fprintf(stderr, "SKIP: no Gemma 4 tokenizer directory (set SUROGATE_GEMMA4_TOKENIZER_DIR)\n");
        return 77;
    }
    sinfer::family::FrontendResources resources;
    resources.tokenizer_json         = read_file(dir / "tokenizer.json");
    resources.tokenizer_config_json  = read_file(dir / "tokenizer_config.json");
    resources.chat_template_jinja    = read_file(dir / "chat_template.jinja");
    resources.generation_config_json = read_file(dir / "generation_config.json");
    const Frontend frontend          = FrontendAccess::create_component(resources, false);
    const sinfer::PromptCapabilities capabilities = frontend.prompt_capabilities();
    expect(capabilities.enable_thinking, "Gemma 4's template has a thinking switch the endpoint can drive");

    const DecisionsRequest request = parse_decisions_request(R"json({
      "model": "rune", "thinking": "medium",
      "state": {"ticket": "Comanda 8812 a ajuns târziu și cutia era strivită. Vreau banii înapoi.", "items": [1, 2.5]},
      "questions": {
        "tone": {"type": "choice", "instructions": "What is the tone?",
                 "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}},
        "refund": {"type": "noul", "instructions": "Is a refund requested?",
                   "criteria": {"true": "A refund is requested", "false": "No refund is requested"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["Not urgent", "Somewhat urgent", "Very urgent", "Critical"]}
      }})json");
    expect(request.thinking == DecisionThinkingLevel::Medium, "the level parses");

    const std::string empty_thought = "<|channel>thought\n<channel|>";
    const std::string close(kDecisionThinkingClose);
    for (const DecisionQuestion& question : request.questions) {
        const std::string& name = question.name;
        const RenderedDecisionQuestion rendered = render_decision_question(question, {});
        const std::string onepass_user          = request.state_text + rendered.branch;
        const DecisionChat chat                 = decision_thinking_chat(request, question);
        expect(chat.user == onepass_user, name + ": the thinking user message is the one-pass one");

        const auto onepass_ids = FrontendAccess::inspect(
            frontend.prepare(decision_chat_input(rendered.system, onepass_user, false, capabilities))).token_ids;
        const auto thinking_ids = FrontendAccess::inspect(
            frontend.prepare(decision_chat_input(chat.system, chat.user, chat.enable_thinking, capabilities))).token_ids;
        const std::string onepass  = text_of(frontend, onepass_ids);
        const std::string thinking = text_of(frontend, thinking_ids);

        // v1: thinking off ends on the empty thought block; a level: the template's thinking switch
        // on, the generation prompt ends at the model turn and the model opens the thought itself.
        expect(onepass.ends_with("<|turn>model\n" + empty_thought), name + ": one-pass prompt ends on the empty thought");
        expect(thinking.ends_with("<|turn>model\n") && !thinking.ends_with(empty_thought),
               name + ": thinking prompt ends at the model turn: " + thinking.substr(thinking.size() - 40));
        expect(thinking.find("<|think|>") != std::string::npos && onepass.find("<|think|>") == std::string::npos,
               name + ": only the thinking prompt carries the template's <|think|> switch");
        expect(thinking.find(std::string(kDecisionThinkingSystemPrompt)) != std::string::npos &&
                   thinking.find(std::string(kDecisionSystemPrompt)) == std::string::npos,
               name + ": the thinking system prompt");
        const std::string user_turn = "<|turn>user\n" + onepass_user + "<turn|>";
        expect(onepass.find(user_turn) != std::string::npos && thinking.find(user_turn) != std::string::npos,
               name + ": the same user turn, byte for byte, in both prompts");

        // Exactly those three differences: swap them back and the prompts are the same text.
        std::string expected = onepass;
        expected.replace(expected.find(std::string(kDecisionSystemPrompt)), kDecisionSystemPrompt.size(),
                         kDecisionThinkingSystemPrompt);
        expected.insert(expected.find("<|turn>system\n") + std::string("<|turn>system\n").size(), "<|think|>\n");
        expected.erase(expected.size() - empty_thought.size());
        expect(thinking == expected, name + ": the thinking prompt differs from the one-pass prompt only in the "
                                            "system prompt, the <|think|> switch and the generation prompt\n  got:      " +
                                         thinking + "\n  expected: " + expected);
        // The rendered text is the model's tokens: re-encoding it gives the same ids.
        expect(encode(frontend, thinking) == thinking_ids, name + ": the thinking prompt round-trips its tokens");

        // The close token, and the letters read after it: the one token each that v1 reads after
        // its own (empty) thought, and the bare letter the reference read.
        const auto closed = encode(frontend, thinking + close);
        expect(extends(closed, thinking_ids, 1) && text_of(frontend, {closed.back()}) == close,
               name + ": <channel|> is one token after a thinking prompt");
        for (const std::string& label : rendered.labels) {
            const auto after_close   = encode(frontend, thinking + close + label);
            const auto after_onepass = encode(frontend, onepass + label);
            const auto bare          = frontend.encode_fragment(label);
            expect(extends(after_close, closed, 1) && extends(after_onepass, onepass_ids, 1) && bare.size() == 1 &&
                       after_close.back() == after_onepass.back() && after_close.back() == bare.front(),
                   name + ": letter " + label + " is the same single token after a thought as in one pass");
        }
    }
    // The opener the model writes at the start of a thought is Gemma 4's own special token.
    const auto opener = encode(frontend, "<|channel>");
    expect(!opener.empty() && text_of(frontend, {opener.back()}) == "<|channel>", "<|channel> is one token");

    if (failures != 0) {
        std::cerr << failures << " Gemma 4 thinking-prompt check(s) failed (" << dir << ")\n";
        return 1;
    }
    std::cout << "decisions thinking prompts on Gemma 4 (" << dir.filename().string()
              << "): system prompt, thinking switch, user message and readout tokens as the reference\n";
    return 0;
}
