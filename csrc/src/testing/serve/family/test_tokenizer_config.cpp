// The tokenizer_config.json chat-template statement, in every form HF writes
// it, against the template the artifact serves.
//
// The check exists so that the template served is the checkpoint's own. It was
// once relaxed to compare only when the key held a string, which meant every
// other form -- HF's multi-template list, an explicit null, a stray object --
// skipped validation entirely and the artifact's template was served with no
// cross-check at all. That is a worse outcome than the refusal it replaced,
// because nothing surfaces: the model still answers, in the wrong format.
//
// Three JSON strings and no tokenizer, so this runs where the full frontend
// test skips for want of real tokenizer resources.
#include <api/family/frontend_resources.h>

#include "family/impl/frontend/test_access.h"

#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace {

// SentencePiece shape: byte_fallback true, so the byte-level pad and prefix
// rules do not apply and the chat-template contract is what is under test.
constexpr std::string_view kTokenizerJson =
    R"({"model":{"type":"BPE","byte_fallback":true},"added_tokens":[]})";
constexpr std::string_view kJinja = "{{ messages }}";

int failures = 0;

sinfer::family::FrontendResources resources(std::string_view tokenizer_config,
                                            std::string_view jinja = kJinja) {
    sinfer::family::FrontendResources out;
    out.tokenizer_json        = kTokenizerJson;
    out.tokenizer_config_json = tokenizer_config;
    out.chat_template_jinja   = jinja;
    return out;
}

std::string config_with(std::string_view chat_template_member) {
    std::string out = R"({"pad_token":"</s>")";
    if (!chat_template_member.empty()) {
        out += ",";
        out += chat_template_member;
    }
    out += "}";
    return out;
}

void accepts(const char* label, std::string_view chat_template_member,
             std::string_view jinja = kJinja) {
    const std::string config = config_with(chat_template_member);
    try {
        sinfer::family::FrontendTestAccess::check_tokenizer_config(resources(config, jinja));
    } catch (const std::exception& error) {
        std::cerr << label << ": refused what it should accept: " << error.what() << '\n';
        ++failures;
    }
}

void refuses(const char* label, std::string_view chat_template_member, std::string_view needle,
             std::string_view jinja = kJinja) {
    const std::string config = config_with(chat_template_member);
    try {
        sinfer::family::FrontendTestAccess::check_tokenizer_config(resources(config, jinja));
    } catch (const std::exception& error) {
        if (std::string_view(error.what()).find(needle) == std::string_view::npos) {
            std::cerr << label << ": refused without naming the reason (wanted \"" << needle
                      << "\"): " << error.what() << '\n';
            ++failures;
        }
        return;
    }
    std::cerr << label << ": accepted what it should refuse\n";
    ++failures;
}

} // namespace

int main() {
    // The two forms a checkpoint may state, and the one it may omit.
    accepts("string, equal", R"("chat_template":"{{ messages }}")");
    accepts("absent", "");
    accepts("list with a default entry",
            R"("chat_template":[{"name":"default","template":"{{ messages }}"}])");
    accepts("list, default among others",
            R"("chat_template":[{"name":"tool_use","template":"other"},)"
            R"({"name":"default","template":"{{ messages }}"}])");

    // Disagreement, in either form, is the case the check is for.
    refuses("string, different", R"("chat_template":"other")", "does not match");
    refuses("list, default differs",
            R"("chat_template":[{"name":"default","template":"other"}])", "does not match");

    // The forms that used to skip validation. Each must be refused by name:
    // accepting them is how a disagreeing template reached the engine unchecked.
    refuses("null", R"("chat_template":null)", "must be a string or a list");
    refuses("number", R"("chat_template":7)", "must be a string or a list");
    refuses("object", R"("chat_template":{"name":"default"})", "must be a string or a list");
    refuses("list of strings", R"("chat_template":["{{ messages }}"])",
            "not a {name, template} object");
    refuses("list entry missing template", R"("chat_template":[{"name":"default"}])",
            "not a {name, template} object");
    refuses("list with no default",
            R"("chat_template":[{"name":"tool_use","template":"a"},)"
            R"({"name":"rag","template":"b"}])",
            "no \"default\" entry");
    refuses("empty list", R"("chat_template":[])", "no \"default\" entry");

    // The artifact must carry a template at all, whatever the config says.
    refuses("no jinja", R"("chat_template":"{{ messages }}")", "no frontend/chat_template.jinja",
            "");

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " tokenizer_config chat template\n";
    return failures == 0 ? 0 : 1;
}
