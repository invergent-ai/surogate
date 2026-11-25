// The reply-phase sampler on Inkling-2's real vocabulary: Frontend::reply_phase_vocabulary built from the checkpoint's
// own tokenizer files names <|end_message|> as the close, <|content_thinking|> as the opening marker and
// <|message_model|> as the header, and the serving decision arms a chat request "opening". The ids are the ones the
// engine's DSpark code and the smoke read (200010 / 200008 / 200001). CPU only; skips (77) without INKLING_MODEL_DIR.
#include <api/family/frontend.h>
#include <api/family/frontend_resources.h>

#include "family/impl/frontend/test_access.h"
#include "runtime/contract/reply_phase.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {
std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot read " + p.string()); }
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
}
int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}
} // namespace

int main() {
    const char* model_dir = std::getenv("INKLING_MODEL_DIR");
    if (model_dir == nullptr) {
        std::cout << "SKIP: set INKLING_MODEL_DIR\n";
        return 77;
    }
    const fs::path model(model_dir);
    sinfer::family::FrontendResources resources;
    resources.tokenizer_json         = read_file(model / "tokenizer.json");
    resources.tokenizer_config_json  = read_file(model / "tokenizer_config.json");
    resources.generation_config_json = read_file(model / "generation_config.json");
    const sinfer::family::Frontend frontend = sinfer::family::FrontendTestAccess::create_component(resources, false);

    int failures = 0;
    const sinfer::ReplyPhaseVocabulary v = frontend.reply_phase_vocabulary();
    const auto id = [&](const char* spelling) { return frontend.token_id(spelling); };
    std::cout << "close " << (v.close ? *v.close : -1) << " open " << (v.open ? *v.open : -1) << " header "
              << v.headers[0] << " model_opens " << v.model_opens << "\n";
    failures += check(v.model_opens, "Inkling-2 does not open its own thought");
    failures += check(v.close && v.close == id("<|end_message|>") && *v.close == 200010, "close is not <|end_message|>");
    failures += check(v.open && v.open == id("<|content_thinking|>") && *v.open == 200008,
                      "open is not <|content_thinking|>");
    failures += check(v.headers[0] == 200001 && id("<|message_model|>") == 200001 && v.headers[1] == -1,
                      "header is not <|message_model|>");
    using Plan = sinfer::runtime::ReplyPhasePlan;
    failures += check(sinfer::runtime::plan_reply_phase(v, false, true, false) == Plan::ArmOpening,
                      "an Inkling chat request is not armed opening");
    failures += check(sinfer::runtime::plan_reply_phase(v, false, true, true) == Plan::Fold,
                      "an Inkling response_format request is not folded");
    // The walk on real ids: thought then answer; answer block first; tool call first.
    const auto walk = [&](std::vector<sinfer::TokenId> out) {
        sinfer::runtime::ReplyPhase p;
        p.armed   = true;
        p.state   = sinfer::runtime::ReplyPhase::State::Opening;
        p.close   = *v.close;
        p.open    = *v.open;
        p.headers = v.headers;
        (void)p.advance(out);
        return p.replying() ? static_cast<long>(p.frontier) : -1L;
    };
    const sinfer::TokenId text = *id("<|content_text|>"), tool = *id("<|content_invoke_tool_json|>");
    failures += check(walk({200001, 200008, 11, 12, 200010, 200001, text, 13}) == 5, "thought then answer: wrong switch");
    failures += check(walk({200001, text, 13, 14}) == 2, "answer block first: wrong switch");
    failures += check(walk({200001, tool, 13}) == 2, "tool call first: wrong switch");
    if (failures == 0) { std::cout << "reply_phase_inkling: ok\n"; }
    return failures == 0 ? 0 : 1;
}
