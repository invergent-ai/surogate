#include "serve/decisions_thinking.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sinfer::serve {

namespace {

[[noreturn]] void invalid_thinking(std::string message) {
    throw ApiException(ApiError{.status  = 400,
                                .type    = "invalid_request_error",
                                .message = std::move(message),
                                .param   = "thinking",
                                .code    = "invalid_decisions_request"});
}

} // namespace

bool parse_decision_thinking(const OrderedJson& body) {
    if (!body.is_object()) { return false; }
    const auto found = body.find("thinking");
    if (found == body.end() || found->is_null()) { return false; }
    if (found->is_boolean()) { return found->get<bool>(); }
    // A string is named in the refusal when it is short and printable, like the rest of this
    // endpoint's refusals; "low" and "true" are the likely ones.
    std::string shown;
    if (found->is_string()) {
        const std::string& value = found->get_ref<const std::string&>();
        if (value.size() <= 32 && std::all_of(value.begin(), value.end(), [](char c) { return c >= 0x20 && c < 0x7F; })) {
            shown = " (got the string '" + value + "')";
        }
    }
    invalid_thinking("thinking must be a boolean: true to let unsure questions think, false (the default) "
                     "for one-pass answers" + shown);
}

double decision_onepass_confidence(const OrderedJson& answer) {
    if (answer.at("type").get_ref<const std::string&>() == "noul") {
        const double p = answer.at("noul").get<double>();
        return std::max(p, 1.0 - p);
    }
    double top = -std::numeric_limits<double>::infinity();
    for (const auto& item : answer.at("probabilities").items()) { top = std::max(top, item.value().get<double>()); }
    return top;
}

bool decision_should_think(const DecisionQuestion& question, double confidence) {
    if (question.option_count() > kDecisionLetterOptions) { return false; }
    return confidence < kDecisionThinkingGate;
}

DecisionChat decision_thinking_chat(const DecisionsRequest& request, const DecisionQuestion& question) {
    if (question.extended()) {
        throw std::logic_error("a question with more than 26 options never thinks");
    }
    // Letters only (26 options at most), so the codebook is never consulted: the branch is
    // v1's, byte for byte, which is what makes the user message the one-pass message.
    static const std::vector<std::string> no_codes;
    const RenderedDecisionQuestion rendered = render_decision_question(question, no_codes);
    return DecisionChat{.system          = kDecisionThinkingSystemPrompt,
                        .user            = request.state_text + rendered.branch,
                        .enable_thinking = true};
}

std::optional<std::size_t> decision_thinking_budget(std::size_t budget, std::size_t prompt_tokens,
                                                    std::size_t max_context) {
    // The readout prefills prompt + thought + close and reads one position beyond them.
    if (prompt_tokens + 2 > max_context) { return std::nullopt; }
    return std::min(budget, max_context - prompt_tokens - 2);
}

std::uint32_t decision_thinking_generation_limit(std::size_t budget) {
    if (budget >= std::numeric_limits<std::uint32_t>::max()) {
        throw std::logic_error("thinking budget does not fit a generation limit");
    }
    return static_cast<std::uint32_t>(budget + 1);
}

DecisionThought decision_thought(const std::vector<TokenId>& generated, TokenId close, std::size_t budget) {
    // The reference's reading of a greedy generation: the thought is everything before the first
    // close token (a model that writes its own channel opener keeps it in the thought), cut at
    // the budget; it closed on its own only if that close came within the budget. A generation
    // that ended without one (the budget, or a stop token of the model's own) is forced closed.
    const auto at        = std::find(generated.begin(), generated.end(), close);
    const auto natural   = static_cast<std::size_t>(at - generated.begin());
    const bool found     = at != generated.end();
    const std::size_t cut = std::min(natural, budget);
    DecisionThought thought;
    thought.tokens.assign(generated.begin(), generated.begin() + static_cast<std::ptrdiff_t>(cut));
    thought.closed = found && natural <= budget;
    return thought;
}

std::vector<TokenId> decision_thinking_readout(const std::vector<TokenId>& prompt, const DecisionThought& thought,
                                               TokenId close) {
    std::vector<TokenId> tokens;
    tokens.reserve(prompt.size() + thought.tokens.size() + 1);
    tokens.insert(tokens.end(), prompt.begin(), prompt.end());
    tokens.insert(tokens.end(), thought.tokens.begin(), thought.tokens.end());
    tokens.push_back(close);
    return tokens;
}

OrderedJson decision_thinking_answer(const DecisionQuestion& question, const std::vector<float>& logits,
                                     double temperature, const DecisionThought& thought, OrderedJson onepass) {
    // v1's own readout arithmetic, so a thinking answer means what a one-pass answer means.
    OrderedJson answer = resolve_decision_answer(question, logits, temperature);
    OrderedJson record = OrderedJson::object();
    record["tokens"]   = thought.tokens.size();
    record["closed"]   = thought.closed;
    record["onepass"]  = std::move(onepass);
    answer["thinking"] = std::move(record);
    return answer;
}

} // namespace sinfer::serve
