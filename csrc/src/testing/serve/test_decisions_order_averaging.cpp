// Model-free checks of the decisions endpoint's option-order averaging (decisions_schema.h,
// `"order_averaging": true`): the request field, which questions are read mirrored and how, and
// the arithmetic that combines a question's two readings -- the mean of the two T = 1
// distributions, handed to v1's readout as a row of logits so the calibration temperature applies
// once, to the mean.
//
// It also holds order averaging off to v1: a request without the field, with null or with false
// parses to the same request, and no golden request asks for it.
#include "serve/decisions_schema.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef SINFER_SOURCE_DIR
#error "SINFER_SOURCE_DIR must name csrc/src/testing/serve"
#endif

using namespace sinfer::serve;

namespace {

int failures = 0;
void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAILED: " << what << '\n';
        ++failures;
    }
}

/// The 400 a malformed `order_averaging` value gets: the endpoint's body-refusal code, pointed at
/// the field.
void refused(const std::string& body, const std::string& what) {
    try {
        (void)parse_decisions_request(body);
        expect(false, "not refused: " + what);
    } catch (const ApiException& e) {
        expect(e.error().status == 400 && e.error().code == "invalid_decisions_request" &&
                   e.error().param == "order_averaging" &&
                   e.error().message.starts_with("order_averaging must be a boolean"),
               "refusal of " + what + " (got " + std::to_string(e.error().status) + " " + e.error().code + " " +
                   e.error().param + ": " + e.error().message + ")");
    }
}

std::string body_with(const std::string& field) {
    return R"json({
  "model": "rune",
  "state": {"ticket": "Order 8812 arrived late and the box was crushed. I want a refund.", "lang": "en"},
  "questions": {
    "tone": {"type": "choice", "instructions": "What is the tone?",
             "criteria": {"calm": "Calm", "annoyed": "Annoyed", "furious": "Furious"}},
    "refund": {"type": "noul", "instructions": "Is a refund requested?",
               "criteria": {"true": "A refund is requested", "false": "No refund is requested"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["Not urgent", "Somewhat urgent", "Very urgent"]}
  })json" + field + "\n}";
}

/// The probabilities of a choice or score answer, in option order.
std::vector<double> probabilities(const OrderedJson& answer) {
    std::vector<double> out;
    for (const auto& item : answer.at("probabilities").items()) { out.push_back(item.value().get<double>()); }
    return out;
}

bool close(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

std::vector<float> reversed(std::vector<float> row) {
    std::reverse(row.begin(), row.end());
    return row;
}

/// The exact mean of the two readings, in double: softmax(as_sent) and softmax(mirrored) put back
/// in the question's order.
std::vector<double> reference_mean(const std::vector<float>& as_sent, const std::vector<float>& mirrored) {
    const auto softmax = [](const std::vector<float>& z) {
        const double top = *std::max_element(z.begin(), z.end());
        std::vector<double> p;
        double sum = 0.0;
        for (const float x : z) {
            p.push_back(std::exp(static_cast<double>(x) - top));
            sum += p.back();
        }
        for (double& v : p) { v /= sum; }
        return p;
    };
    const std::vector<double> p = softmax(as_sent);
    const std::vector<double> q = softmax(mirrored);
    std::vector<double> mean(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) { mean[i] = (p[i] + q[p.size() - 1 - i]) / 2.0; }
    return mean;
}

} // namespace

int main() {
    // ---- 1. The request field ----
    {
        const DecisionsRequest plain = parse_decisions_request(body_with(""));
        expect(!plain.order_averaging, "an absent field is off");
        expect(!parse_decisions_request(body_with(R"(, "order_averaging": null)")).order_averaging, "null is off");
        expect(!parse_decisions_request(body_with(R"(, "order_averaging": false)")).order_averaging, "false is off");
        expect(parse_decisions_request(body_with(R"(, "order_averaging": true)")).order_averaging, "true is on");
        const std::string first = R"json({"order_averaging": true, "model": "rune", "state": "s",
            "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}}})json";
        expect(parse_decisions_request(first).order_averaging, "the field's place is free");
        // Not with thinking yet: a thinking answer would replace the averaged one.
        try {
            (void)parse_decisions_request(body_with(R"(, "thinking": true, "order_averaging": true)"));
            expect(false, "thinking with order averaging was answered");
        } catch (const ApiException& e) {
            expect(e.error().status == 400 && e.error().code == "invalid_decisions_request" &&
                       e.error().param == "order_averaging" &&
                       e.error().message ==
                           "order_averaging cannot be combined with thinking yet: a question that thinks answers "
                           "from one thinking readout, which is not order-averaged",
                   "thinking with order averaging is refused (got " + e.error().param + ": " + e.error().message +
                       ")");
        }
        const DecisionsRequest thinking_only =
            parse_decisions_request(body_with(R"(, "thinking": true, "order_averaging": false)"));
        expect(thinking_only.thinking && !thinking_only.order_averaging, "thinking with order averaging false");
        const DecisionsRequest averaging_only =
            parse_decisions_request(body_with(R"(, "thinking": false, "order_averaging": true)"));
        expect(!averaging_only.thinking && averaging_only.order_averaging, "order averaging with thinking false");
        expect(parse_decisions_request(body_with(R"(, "thinking": true, "order_averaging": null)")).thinking,
               "thinking with order averaging null");
        expect(!parse_decisions_request(body_with(R"(, "thinking": true)")).order_averaging,
               "thinking alone leaves order averaging off");
        expect(!parse_decisions_request(body_with(R"(, "order_averaging": true)")).thinking,
               "order averaging alone leaves thinking off");

        // The flag never changes what v1 parses: the state text and every question are the same.
        for (const char* field : {R"(, "order_averaging": null)", R"(, "order_averaging": false)",
                                  R"(, "order_averaging": true)"}) {
            const DecisionsRequest with = parse_decisions_request(body_with(field));
            expect(with.state_text == plain.state_text && with.questions.size() == plain.questions.size() &&
                       with.model == plain.model && with.images.empty() && !with.thinking,
                   std::string("the rest of the request parses as v1 with ") + field);
            for (std::size_t i = 0; i < plain.questions.size(); ++i) {
                expect(with.questions[i].name == plain.questions[i].name &&
                           with.questions[i].kind == plain.questions[i].kind &&
                           with.questions[i].option_keys == plain.questions[i].option_keys &&
                           with.questions[i].option_texts == plain.questions[i].option_texts &&
                           with.questions[i].instructions == plain.questions[i].instructions,
                       std::string("question parses as v1 with ") + field);
            }
        }

        // Anything but a boolean is refused, never served silently with the flag off.
        refused(body_with(R"(, "order_averaging": "true")"), "the string true");
        refused(body_with(R"(, "order_averaging": "")"), "an empty string");
        refused(body_with(R"(, "order_averaging": 1)"), "the number 1");
        refused(body_with(R"(, "order_averaging": 0)"), "the number 0");
        refused(body_with(R"(, "order_averaging": {"enabled": true})"), "an object");
        refused(body_with(R"(, "order_averaging": [true])"), "an array");
        try {
            (void)parse_decisions_request(body_with(R"(, "order_averaging": "yes")"));
        } catch (const ApiException& e) {
            expect(e.error().message ==
                       "order_averaging must be a boolean: true to answer each choice and noul question from its "
                       "options as sent and in reverse order, false (the default) to read them as sent (got the "
                       "string 'yes')",
                   "the refusal says what is accepted and names the string: " + e.error().message);
        }
        // v1's refusals come first, then thinking's: the body is judged as v1 judges it first.
        try {
            (void)parse_decisions_request(R"json({"state": "s", "order_averaging": "yes",
                "questions": {"q": {"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}}})json");
            expect(false, "a body without a model was answered");
        } catch (const ApiException& e) {
            expect(e.error().param == "model", "a v1 refusal comes before the order averaging one");
        }
        try {
            (void)parse_decisions_request(body_with(R"(, "thinking": "low", "order_averaging": "yes")"));
            expect(false, "two bad extension fields were answered");
        } catch (const ApiException& e) {
            expect(e.error().param == "thinking", "the thinking refusal comes before the order averaging one");
        }
    }

    // ---- 2. The mirrored readings ----
    {
        const DecisionsRequest request = parse_decisions_request(body_with(R"(, "order_averaging": true)"));
        const DecisionQuestion& tone    = request.questions[0];
        const DecisionQuestion& refund  = request.questions[1];
        const DecisionQuestion& urgency = request.questions[2];
        expect(decision_mirrors(tone) && decision_mirrors(refund) && !decision_mirrors(urgency),
               "choice and noul questions are read mirrored, score questions are not");
        expect(decision_mirrored_questions(request) == std::vector<std::size_t>{0, 1},
               "the mirrored readings follow the questions in request order");

        const DecisionQuestion tone_mirrored = decision_mirrored_question(tone);
        expect(tone_mirrored.name == tone.name && tone_mirrored.kind == tone.kind &&
                   tone_mirrored.instructions == tone.instructions,
               "a mirrored question keeps its name, kind and instructions");
        expect(tone_mirrored.option_keys == std::vector<std::string>{"furious", "annoyed", "calm"} &&
                   tone_mirrored.option_texts == std::vector<std::string>{"Furious", "Annoyed", "Calm"} &&
                   tone_mirrored.option_values.size() == 3 && tone_mirrored.option_values[0] == "Furious",
               "a mirrored choice question has its options reversed");
        expect(render_decision_question(tone_mirrored, {}).branch ==
                   "QUESTION:\nWhat is the tone?\nOPTIONS:\nA: Furious\nB: Annoyed\nC: Calm\n"
                   "Answer with one option letter only.",
               "a mirrored choice question renders with v1's layout, the last option as A");
        const DecisionQuestion refund_mirrored = decision_mirrored_question(refund);
        expect(refund_mirrored.option_keys == std::vector<std::string>{"true", "false"},
               "a mirrored noul question is true then false");
        expect(render_decision_question(refund_mirrored, {}).branch ==
                   "QUESTION:\nIs a refund requested?\nOPTIONS:\nA: A refund is requested\nB: No refund is requested\n"
                   "Answer with one option letter only.",
               "a mirrored noul question renders true as A");
        const RenderedDecisionQuestion as_sent = render_decision_question(tone, {});
        const RenderedDecisionQuestion mirror  = render_decision_question(tone_mirrored, {});
        expect(as_sent.system == mirror.system && as_sent.labels == mirror.labels,
               "a mirrored question keeps its system prompt and labels");
        expect(decision_mirrored_question(tone_mirrored).option_keys == tone.option_keys,
               "mirroring twice is the question as sent");

        // A score-only request has nothing to mirror.
        const DecisionsRequest scores = parse_decisions_request(R"json({"model": "m", "state": "s",
            "order_averaging": true, "questions": {"s": {"type": "score", "instructions": "i", "criteria": ["a", "b"]}}})json");
        expect(decision_mirrored_questions(scores).empty(), "score questions are never mirrored");
    }

    // ---- 3. Combining two readings ----
    {
        const DecisionsRequest request = parse_decisions_request(body_with(R"(, "order_averaging": true)"));
        const DecisionQuestion& tone   = request.questions[0];
        const DecisionQuestion& refund = request.questions[1];

        // Readings that agree (the mirrored reading is the as-sent one reversed) leave v1's
        // answer as it is to float precision: the mean of two equal distributions is that
        // distribution, and the row handed to v1's readout is rounded to float. These logits'
        // differences are floats exactly, so here it is bit for bit.
        const std::vector<float> agree{0.5F, 2.25F, -1.75F};
        const std::vector<float> combined = decision_order_averaged_logits(agree, reversed(agree));
        expect(resolve_decision_answer(tone, combined).dump() == resolve_decision_answer(tone, agree).dump(),
               "agreeing readings give v1's answer: " + resolve_decision_answer(tone, combined).dump() + " vs " +
                   resolve_decision_answer(tone, agree).dump());
        expect(resolve_decision_answer(tone, combined, 2.0).dump() == resolve_decision_answer(tone, agree, 2.0).dump(),
               "agreeing readings give v1's tempered answer");
        expect(*std::max_element(combined.begin(), combined.end()) == 0.0F, "the combined row's maximum is 0");
        // Arbitrary agreeing logits: the same choice, probabilities to float precision.
        const std::vector<float> rough{0.3137F, 2.7183F, -1.4142F};
        const OrderedJson rough_v1  = resolve_decision_answer(tone, rough);
        const OrderedJson rough_avg = resolve_decision_answer(tone, decision_order_averaged_logits(rough, reversed(rough)));
        expect(rough_avg.at("choice") == rough_v1.at("choice"), "agreeing readings keep the choice");
        for (std::size_t i = 0; i < 3; ++i) {
            expect(close(probabilities(rough_avg)[i], probabilities(rough_v1)[i], 1e-6),
                   "agreeing readings keep the probabilities");
        }

        // A pure position preference cancels: both readings put 0.8 on option A, whatever it is.
        const std::vector<float> biased{1.3862944F, 0.0F}; // p(A) = 0.8
        const OrderedJson noul = resolve_decision_answer(refund, decision_order_averaged_logits(biased, biased));
        expect(noul.at("noul").get<double>() == 0.5, "a noul question read 0.8 false and then 0.8 true is 0.5: " +
                                                         noul.dump());
        const std::vector<float> three{2.0F, 1.0F, 0.0F};
        const OrderedJson choice = resolve_decision_answer(tone, decision_order_averaged_logits(three, three));
        const std::vector<double> mean = probabilities(choice);
        expect(mean[0] == mean[2], "the first and last options swap places and tie exactly");
        expect(choice.at("choice") == "calm", "the tie goes to the first option, as v1's does");
        const std::vector<double> expected = reference_mean(three, three);
        for (std::size_t i = 0; i < 3; ++i) {
            expect(close(mean[i], expected[i], 1e-6), "a position preference averages to the mean distribution");
        }

        // Readings that disagree: the answer is the mean distribution. As sent p = (0.9, 0.1); the
        // mirrored reading gives 0.6 to its A, the question's `true`.
        const std::vector<float> first{static_cast<float>(std::log(0.9)), static_cast<float>(std::log(0.1))};
        const std::vector<float> second{static_cast<float>(std::log(0.6)), static_cast<float>(std::log(0.4))};
        const OrderedJson mixed = resolve_decision_answer(refund, decision_order_averaged_logits(first, second));
        expect(close(mixed.at("noul").get<double>(), 0.35, 1e-6),
               "p(true) is the mean of 0.1 and 0.6: " + mixed.dump());
        // As sent the first option leads (0.60 calm, 0.37 furious); mirrored, furious is option A and
        // leads by far (0.93). The mean (0.33 calm, 0.65 furious) answers furious.
        const std::vector<float> wide{1.0F, -2.0F, 0.5F};
        const std::vector<float> wide_mirrored{2.5F, -1.0F, -0.5F};
        expect(resolve_decision_answer(tone, wide).at("choice") == "calm", "as sent alone the answer is calm");
        const OrderedJson averaged = resolve_decision_answer(tone, decision_order_averaged_logits(wide, wide_mirrored));
        const std::vector<double> want = reference_mean(wide, wide_mirrored);
        for (std::size_t i = 0; i < 3; ++i) {
            expect(close(probabilities(averaged)[i], want[i], 1e-6), "a choice answer is read from the mean");
        }
        expect(averaged.at("choice") == "furious", "the mean's most probable option: " + averaged.dump());
        expect(close(averaged.at("confidence").get<double>(), (want[2] - 1.0 / 3.0) / (1.0 - 1.0 / 3.0), 1e-6),
               "the confidence is v1's formula on the mean");

        // The calibration temperature applies once, to the mean: softmax(log mean / T).
        const OrderedJson tempered =
            resolve_decision_answer(tone, decision_order_averaged_logits(wide, wide_mirrored), 2.0);
        double sum = 0.0;
        for (const double m : want) { sum += std::sqrt(m); }
        for (std::size_t i = 0; i < 3; ++i) {
            expect(close(probabilities(tempered)[i], std::sqrt(want[i]) / sum, 1e-6),
                   "T = 2 tempers the mean distribution");
        }

        // An option both readings give a probability of exactly 0 in double (2000 logits down) is a
        // finite logit, not log(0): the row stays a readout v1 accepts.
        const std::vector<float> extreme{0.0F, -2000.0F};
        const std::vector<float> extreme_mirrored{-2000.0F, 0.0F};
        const std::vector<float> far = decision_order_averaged_logits(extreme, extreme_mirrored);
        expect(far[0] == 0.0F && far[1] == -2000.0F, "two readings that agree on an underflowing option keep its logit");
        expect(resolve_decision_answer(refund, far).at("noul").get<double>() == 0.0, "and it reads as probability 0");
        const std::vector<float> huge{3.0e38F, -3.0e38F};
        const std::vector<float> huge_mirrored{-3.0e38F, 3.0e38F};
        const std::vector<float> clamped = decision_order_averaged_logits(huge, huge_mirrored);
        expect(std::isfinite(clamped[0]) && std::isfinite(clamped[1]) && clamped[0] == 0.0F,
               "a row beyond float's range is clamped, not infinite");
        expect(resolve_decision_answer(refund, clamped).at("noul").get<double>() == 0.0, "and reads as probability 0");

        // What v1 refuses is refused: a non-finite reading on either side, and mis-sized rows.
        const auto throws = [](const std::vector<float>& a, const std::vector<float>& b, const std::string& message) {
            try {
                (void)decision_order_averaged_logits(a, b);
            } catch (const std::runtime_error& e) { return message == e.what(); }
            return false;
        };
        expect(throws({0.0F, NAN}, {0.0F, 1.0F}, "model returned non-finite logits"), "a NaN as sent is refused");
        expect(throws({0.0F, 1.0F}, {INFINITY, 1.0F}, "model returned non-finite logits"),
               "an infinite mirrored reading is refused");
        expect(throws({0.0F, 1.0F}, {0.0F, 1.0F, 2.0F}, "decision readout does not match its options"),
               "readings of different widths are refused");
        expect(throws({}, {}, "decision readout does not match its options"), "an empty readout is refused");
    }

    // ---- 4. A request's rows ----
    {
        const DecisionsRequest request = parse_decisions_request(body_with(R"(, "order_averaging": true)"));
        // Rows: tone, refund, urgency as sent, then tone and refund mirrored.
        const std::vector<float> tone{0.25F, 1.5F, -0.5F};
        const std::vector<float> refund{0.75F, -0.25F};
        const std::vector<float> urgency{0.1F, 0.7F, -0.3F};
        const std::vector<float> tone_mirrored{-1.0F, 0.5F, 0.0F};
        const std::vector<float> refund_mirrored{0.5F, 0.25F};
        const std::vector<std::vector<float>> rows{tone, refund, urgency, tone_mirrored, refund_mirrored};
        const std::vector<std::vector<float>> answered = decision_order_averaged_rows(request, rows);
        expect(answered.size() == 3, "one row per question");
        expect(answered[0] == decision_order_averaged_logits(tone, tone_mirrored), "tone is paired with its mirror");
        expect(answered[1] == decision_order_averaged_logits(refund, refund_mirrored), "refund is paired with its mirror");
        expect(answered[2] == urgency, "a score question keeps its row as read, bit for bit");
        const OrderedJson answers = resolve_decision_answers(request, answered);
        expect(answers.size() == 3 && answers.at("urgency").dump() == resolve_decision_answer(request.questions[2], urgency).dump(),
               "the score answer is v1's");
        expect(close(answers.at("refund").at("noul").get<double>(), reference_mean(refund, refund_mirrored)[1], 1e-6),
               "the noul answer is the mean's p(true)");

        bool threw = false;
        try {
            (void)decision_order_averaged_rows(request, {tone, refund, urgency});
        } catch (const std::runtime_error&) { threw = true; }
        expect(threw, "a readout without its mirrored rows is refused");
        threw = false;
        try {
            (void)decision_order_averaged_rows(request, {tone, refund, urgency, tone_mirrored, {0.0F, NAN}});
        } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "model returned non-finite logits"; }
        expect(threw, "a non-finite mirrored row is refused as v1 refuses a non-finite row");
    }

    // ---- 5. The v1 golden set: nothing asks for it, and every mirror renders with v1's layout ----
    {
        std::ifstream file(std::string(SINFER_SOURCE_DIR) + "/fixtures/serve/decisions_v1/golden.json");
        assert(file);
        std::stringstream text;
        text << file.rdbuf();
        const OrderedJson golden = OrderedJson::parse(text.str());
        // Codes as test_decisions_v1 has them: every one- and two-capital-letter code a single token.
        const std::vector<std::string> codebook = decision_codebook(
            [](std::string_view s) -> std::vector<sinfer::TokenId> {
                if (s.size() == 2) { return {256 + 26 * (s[0] - 'A') + (s[1] - 'A')}; }
                std::vector<sinfer::TokenId> ids;
                for (const unsigned char c : s) { ids.push_back(c); }
                return ids;
            },
            [](sinfer::TokenId id) {
                if (id < 256) { return std::string(1, static_cast<char>(id)); }
                const int index = id - 256;
                return std::string{static_cast<char>('A' + index / 26), static_cast<char>('A' + index % 26)};
            });
        std::size_t mirrors = 0;
        for (const OrderedJson& item : golden.at("cases")) {
            const DecisionsRequest parsed = parse_decisions_request(item.at("body").get<std::string>());
            expect(!parsed.order_averaging, "no golden request asks for order averaging");
            for (const DecisionQuestion& question : parsed.questions) {
                if (!decision_mirrors(question)) {
                    expect(question.kind == DecisionKind::Score, "only score questions are not mirrored");
                    continue;
                }
                const DecisionQuestion mirrored         = decision_mirrored_question(question);
                const RenderedDecisionQuestion as_sent  = render_decision_question(question, codebook);
                const RenderedDecisionQuestion rendered = render_decision_question(mirrored, codebook);
                std::string branch = "QUESTION:\n" + question.instructions + "\nOPTIONS:\n";
                for (std::size_t i = 0; i < question.option_count(); ++i) {
                    if (i > 0) { branch += "\n"; }
                    branch += as_sent.labels[i] + ": " + question.option_texts[question.option_count() - 1 - i];
                }
                branch += as_sent.extended ? "\nAnswer with one option code only." : "\nAnswer with one option letter only.";
                expect(rendered.branch == branch && rendered.labels == as_sent.labels && rendered.system == as_sent.system,
                       item.at("name").get<std::string>() + "/" + question.name + ": the mirror is v1's layout reversed");
                ++mirrors;
            }
        }
        expect(mirrors > 10, "the golden set's choice and noul questions were mirrored");
    }

    if (failures != 0) {
        std::cerr << failures << " order averaging check(s) failed\n";
        return 1;
    }
    std::cout << "decisions order averaging: request field, mirrored readings, combination and rows checks passed\n";
    return 0;
}
