// Decisions v1 is stable: this test pins it.
//
// fixtures/serve/decisions_v1/golden.json holds a fixed set of requests and, for each, what v1
// must produce, computed by an independent Python implementation of the protocol
// (make_golden.py): the user turn's opening, every question's system prompt, option labels and
// question text, and the answer object for a fixed row of option logits at the default
// calibration temperature. The engine must reproduce all of it exactly -- texts byte for byte,
// numbers bit for bit (nlohmann writes a form that round-trips, so equal dumps mean equal
// doubles, and it tells 0 from 0.0 and -0.0). It also pins the shared-prefix rule and the
// prefill floor, which decide how each answer is computed. A failure here means a change alters
// the answers v1 clients rely on: it belongs in a new protocol version, not in v1, and the golden
// file must not be regenerated to pass.
//
// The expected numbers rely on Python's math.exp and C++'s std::exp both being the platform
// libm's exp (glibc here); a different libm could differ in the last bit.
//
// Model-free: the chat template and tokenizer belong to the model, not to the protocol; the
// served answers are compared by tests/serve/test_decisions_v1_golden.py on a GPU.
#include "serve/decisions_schema.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef SINFER_SOURCE_DIR
#error "SINFER_SOURCE_DIR must name csrc/src/testing/serve"
#endif

using namespace sinfer::serve;

namespace {

// Every one- and two-capital-letter code is a single token: the codebook is A..Z, AA, AB, ..
std::vector<sinfer::TokenId> encode(std::string_view s) {
    if (s.size() == 2) { return {256 + 26 * (s[0] - 'A') + (s[1] - 'A')}; }
    std::vector<sinfer::TokenId> ids;
    for (const unsigned char c : s) { ids.push_back(c); }
    return ids;
}
std::string decode(sinfer::TokenId id) {
    if (id < 256) { return std::string(1, static_cast<char>(id)); }
    const int index = id - 256;
    return std::string{static_cast<char>('A' + index / 26), static_cast<char>('A' + index % 26)};
}

int failures = 0;
void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "v1 changed: " << what << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    std::ifstream file(std::string(SINFER_SOURCE_DIR) + "/fixtures/serve/decisions_v1/golden.json");
    assert(file);
    std::stringstream text;
    text << file.rdbuf();
    const OrderedJson golden = OrderedJson::parse(text.str());
    assert(golden.at("version") == "v1" && golden.at("temperature") == 1.0);
    assert(kDecisionsProtocolVersion == "v1");

    const std::vector<std::string> codebook = decision_codebook(encode, decode);
    std::size_t questions = 0;
    for (const OrderedJson& item : golden.at("cases")) {
        const std::string name = item.at("name").get<std::string>();
        const DecisionsRequest request = parse_decisions_request(item.at("body").get<std::string>());
        expect(request.state_text == item.at("state_text").get<std::string>(), name + ": state text");
        const OrderedJson& expected = item.at("questions");
        expect(request.questions.size() == expected.size(), name + ": question count");
        if (request.questions.size() != expected.size()) { continue; }

        std::vector<std::vector<float>> rows;
        OrderedJson answers = OrderedJson::object();
        for (std::size_t i = 0; i < expected.size(); ++i) {
            const DecisionQuestion& question = request.questions[i];
            const OrderedJson& want          = expected[i];
            const std::string where          = name + "/" + want.at("name").get<std::string>();
            expect(question.name == want.at("name").get<std::string>(), where + ": question order");
            const RenderedDecisionQuestion rendered = render_decision_question(question, codebook);
            expect(std::string(rendered.system) == want.at("system").get<std::string>(), where + ": system prompt");
            expect(rendered.labels == want.at("labels").get<std::vector<std::string>>(), where + ": option labels");
            expect(rendered.branch == want.at("branch").get<std::string>(), where + ": question text");

            std::vector<float> logits;
            for (const auto& z : want.at("logits")) { logits.push_back(z.get<float>()); }
            const std::string answer = want.at("answer").dump();
            // The default temperature, and 1 spelled out, both read v1 exactly.
            expect(resolve_decision_answer(question, logits).dump() == answer, where + ": answer");
            expect(resolve_decision_answer(question, logits, 1.0).dump() == answer, where + ": answer at T=1");
            if (resolve_decision_answer(question, logits).dump() != answer) {
                std::cerr << "  got      " << resolve_decision_answer(question, logits).dump() << "\n  expected " << answer << '\n';
            }
            rows.push_back(std::move(logits));
            answers[question.name] = want.at("answer");
            ++questions;
        }
        // The whole response's `answers`, in request order, through the endpoint's own path.
        expect(resolve_decision_answers(request, rows).dump() == answers.dump(), name + ": answers object");
        expect(resolve_decision_answers(request, rows, kDecisionDefaultTemperature).dump() == answers.dump(),
               name + ": answers object at the default temperature");
    }
    expect(questions == 27, "the golden set has 27 questions");

    // The shared-prefix rule and the prefill floor: which tokens are prefilled once for all
    // questions, and that every question is either a whole prompt or at least 47 tokens beyond
    // the shared prefix (47 = the MoE wide-prefill width, see decisions_schema.h). They decide
    // which prefill route computes an answer, and the route moves near-even answers.
    {
        using Ids = std::vector<sinfer::TokenId>;
        expect(kDecisionMinPrefillTokens == 47, "the prefill floor is 47 tokens");
        expect(decision_shared_prefix({Ids{1, 2, 3}, Ids{1, 2, 3}}, {Ids{1, 2, 3, 7, 8}, Ids{1, 2, 3, 9}}) == 3,
               "the shared prefix is the common prefix of the question prompts");
        expect(decision_shared_prefix({Ids{1, 2, 3}, Ids{1, 2, 3}}, {Ids{1, 2, 3}, Ids{1, 2, 3, 4}}) == 2,
               "the shared prefix leaves every prompt at least one token");
        expect(decision_shared_prefix({Ids{1, 2, 3, 4}, Ids{1, 9, 3, 4}}, {Ids{1, 2, 3, 4, 5}, Ids{1, 9, 3, 4, 6}}) == 1,
               "the shared prefix stops at the first difference");
        expect(decision_shared_prefix({Ids{1, 2, 3}, Ids{1, 2, 3}}, {Ids{1, 2, 33, 4}, Ids{1, 2, 3, 4}}) == 2,
               "a token merged across the boundary shortens the shared prefix");
        expect(decision_shared_prefix_floor(154, {200}) == 153 && decision_shared_prefix_floor(153, {200}) == 153,
               "a suffix shorter than the floor gives tokens back");
        expect(decision_shared_prefix_floor(46, {400}) == 0 && decision_shared_prefix_floor(47, {400}) == 47,
               "a shared prefix shorter than the floor shares nothing");
        expect(decision_shared_prefix_floor(40, {48}) == 0 && decision_shared_prefix_floor(100, {300, 120}) == 73,
               "short prompts share nothing; the shortest suffix sets the prefix");
    }

    if (failures != 0) {
        std::cerr << failures << " v1 differences; see the header of this file before changing anything\n";
        return 1;
    }
    std::cout << "decisions v1 golden: " << golden.at("cases").size() << " requests, " << questions
              << " questions pinned (prompts byte for byte, answers bit for bit)\n";
    return 0;
}
