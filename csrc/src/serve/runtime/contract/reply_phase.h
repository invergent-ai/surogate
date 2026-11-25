#pragma once

#include "api/ops/sampling.h"
#include "api/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace sinfer::runtime {

using ::sinfer::ReplyPhaseVocabulary;

/// How a request with reply values is served (plan_reply_phase).
enum class ReplyPhasePlan : std::uint8_t {
    /// The output is answer from its first token: the reply values replace the request's own for every draw.
    Fold,
    /// The output starts inside a thought: the request's own values until the close token, the reply values after.
    ArmThinking,
    /// The model may open a thought itself: the request's own values until the output shows what it writes -- the
    /// opening marker starts a thought (then ArmThinking's rule), any other first content token starts the answer.
    ArmOpening,
    /// The output may think but the model has no single close token: the reply values are not used.
    Ignore,
};

/// The serving decision for a request that has reply values. `chat_rendered`: the prompt was rendered by the model's
/// own chat path (false for token-id and /v1/completions prompts, whose reasoning state is not known: those are
/// armed, never folded, so a thought a raw prompt opened is never drawn with the reply values).
/// `answer_from_first_token`: the prompt is the answer after a closed thought (the thinking guard's continuation), or
/// the output is structured from its first token (a JSON schema without json_after_reasoning).
/// `raw_tail_closed`: a raw prompt whose last reasoning marker is a close (prompt_tail): a model whose prompts open
/// its thoughts cannot think after it, so its output is answer (folded).
[[nodiscard]] constexpr ReplyPhasePlan plan_reply_phase(const ReplyPhaseVocabulary& vocabulary,
                                                        bool starts_in_reasoning, bool chat_rendered,
                                                        bool answer_from_first_token,
                                                        bool raw_tail_closed = false) noexcept {
    if (answer_from_first_token) { return ReplyPhasePlan::Fold; }
    if (!starts_in_reasoning && !chat_rendered && !vocabulary.model_opens && raw_tail_closed) {
        return ReplyPhasePlan::Fold;
    }
    if (starts_in_reasoning || (!chat_rendered && !vocabulary.model_opens)) {
        // A thought is open (or may be: a raw prompt of a model whose prompts open it).
        if (vocabulary.close) { return ReplyPhasePlan::ArmThinking; }
        return starts_in_reasoning ? ReplyPhasePlan::Ignore : ReplyPhasePlan::Fold;
    }
    if (vocabulary.model_opens) {
        return vocabulary.close && vocabulary.open ? ReplyPhasePlan::ArmOpening : ReplyPhasePlan::Ignore;
    }
    return ReplyPhasePlan::Fold;
}

/// Where a prompt the model's chat path did not render (token ids, /v1/completions) leaves the reasoning: the last of
/// its opening markers and close tokens. Unknown without both single tokens, or when the prompt holds neither.
enum class PromptTail : std::uint8_t { Unknown, InThought, AfterClose };

[[nodiscard]] inline PromptTail prompt_tail(std::span<const TokenId> prompt,
                                            const ReplyPhaseVocabulary& vocabulary) noexcept {
    if (!vocabulary.open || !vocabulary.close) { return PromptTail::Unknown; }
    for (std::size_t i = prompt.size(); i-- > 0;) {
        if (prompt[i] == *vocabulary.close) { return PromptTail::AfterClose; }
        if (prompt[i] == *vocabulary.open) { return PromptTail::InThought; }
    }
    return PromptTail::Unknown;
}

/// prompt_tail(...) == InThought.
[[nodiscard]] inline bool prompt_tail_in_thought(std::span<const TokenId> prompt,
                                                 const ReplyPhaseVocabulary& vocabulary) noexcept {
    return prompt_tail(prompt, vocabulary) == PromptTail::InThought;
}

/// The reply-phase sampler of one request (SamplingOverrides::reply_*): its thought is drawn with the request's own
/// config, every token after it with the reply values.
///
/// Host side, per round: advance() walks the tokens the request has generated since the last call; apply() turns
/// the request's own config into the one for the round's draws. Before the answer that is the request's config
/// unchanged except for the two phase_switch fields, which only the speculative verify reads
/// (ops::speculative_prepare_verify_inputs): a round accepts no draft at or after a drafted close (Thinking), or no
/// draft at all (Opening), so every token it commits is drawn exactly as without the option. In the answer, the same
/// config with the reply values. The random stream is keyed by position and purpose (ops/sampling.h), so it is the
/// same with the option on or off.
struct ReplyPhase {
    enum class State : std::uint8_t {
        /// The model has not yet shown whether it thinks (ArmOpening).
        Opening,
        /// Inside the thought: the answer starts after the close token.
        Thinking,
        /// The answer: every later draw uses `values`.
        Replying,
    };

    /// The request has reply values and its output starts in Opening or Thinking.
    bool armed  = false;
    State state = State::Thinking;
    TokenId close = -1;
    TokenId open  = -1;
    std::array<TokenId, 2> headers{-1, -1};
    /// Ledger index advance() has read up to; starts at the end of the request's own prompt.
    std::size_t frontier = 0;
    ReplySamplingParameters values;

    [[nodiscard]] bool replying() const noexcept { return state == State::Replying; }
    /// Before the answer (draws use the request's own values, the verify's draft cut is armed).
    [[nodiscard]] bool pending() const noexcept { return armed && state != State::Replying; }

    /// Walks ledger[frontier, size). Returns true when this call reached the answer.
    bool advance(std::span<const TokenId> ledger) noexcept {
        if (!pending()) { return false; }
        while (frontier < ledger.size()) {
            const TokenId token = ledger[frontier++];
            if (state == State::Opening) {
                if (token == open) {
                    state = State::Thinking;
                } else if (token != headers[0] && token != headers[1]) {
                    // The first content token is not a thought: the answer starts after it (it is the draw that
                    // decided, made with the request's own values).
                    state = State::Replying;
                    return true;
                }
            } else if (token == close) {
                state = State::Replying;
                return true;
            }
        }
        return false;
    }

    /// The config the request's next draws use; `config` is the request's own.
    [[nodiscard]] ops::SamplingConfig apply(ops::SamplingConfig config) const noexcept {
        if (!armed) { return config; }
        switch (state) {
        case State::Replying:
            config.temperature        = values.temperature;
            config.top_k              = values.top_k;
            config.top_p              = values.top_p;
            config.min_p              = values.min_p;
            config.phase_switch_armed = 0;
            break;
        case State::Thinking:
            config.phase_switch_token = close;
            config.phase_switch_armed = 1;
            break;
        case State::Opening:
            config.phase_switch_armed = 2;
            break;
        }
        return config;
    }
};

} // namespace sinfer::runtime
