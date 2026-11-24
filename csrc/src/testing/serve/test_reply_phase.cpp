// Reply-phase sampler (SamplingOverrides::reply_*), the host side: the request fields and server defaults reach the
// resolved parameters, invalid values are refused, and the per-round staging (runtime::ReplyPhase) leaves the thought's
// config bit for bit the request's own -- only the two former padding fields the verify reads change -- and switches
// to the reply values from the round after the close token. The device side (the verify's draft cut) is
// sinfer_speculative_round_test; the end-to-end check on a model is the GPU smoke described in the PR.
#include "runtime/contract/reply_phase.h"
#include "runtime/contract/sampling.h"
#include "serve/translate.h"

#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool throws_invalid(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

bool throws_any(const auto& operation) {
    try {
        operation();
    } catch (...) { return true; }
    return false;
}

// Every byte but the two reply-phase fields.
bool same_except_phase_fields(sinfer::ops::SamplingConfig a, sinfer::ops::SamplingConfig b) {
    a.phase_switch_token = b.phase_switch_token = 0;
    a.phase_switch_armed = b.phase_switch_armed = 0;
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

} // namespace

int main() {
    int failures = 0;
    const sinfer::ModelSamplingDefaults defaults{
        .thinking     = {.temperature = 1.0F, .top_k = 20, .top_p = 0.95F, .min_p = 0.0F},
        .non_thinking = {.temperature = 0.7F, .top_k = 20, .top_p = 0.8F, .min_p = 0.0F},
    };
    const auto mode = sinfer::SamplingMode::Thinking;

    // No reply field: no reply values, the resolved parameters as before.
    sinfer::SamplingOverrides overrides;
    overrides.temperature = 1.0F;
    auto resolved = sinfer::runtime::resolve_sampling(defaults, mode, overrides);
    failures += check(!resolved.reply.has_value(), "a request without reply fields got reply values");

    // reply_temperature alone: the other reply fields are the request's own resolved values.
    overrides.reply_temperature = 0.7F;
    resolved = sinfer::runtime::resolve_sampling(defaults, mode, overrides);
    failures += check(resolved.reply.has_value() && resolved.reply->temperature == 0.7F &&
                          resolved.reply->top_k == 20 && resolved.reply->top_p == 0.95F &&
                          resolved.reply->min_p == 0.0F,
                      "reply_temperature alone did not keep the request's top_k / top_p / min_p");
    failures += check(resolved.temperature == 1.0F && resolved.top_k == 20 && resolved.top_p == 0.95F,
                      "reply values changed the thought's values");
    overrides.reply_top_p = 0.9F;
    overrides.reply_top_k = 0;
    overrides.reply_min_p = 0.05F;
    resolved = sinfer::runtime::resolve_sampling(defaults, mode, overrides);
    failures += check(resolved.reply->top_p == 0.9F && resolved.reply->top_k == 0 && resolved.reply->min_p == 0.05F,
                      "explicit reply_top_p / reply_top_k / reply_min_p were not applied");
    // reply_temperature 0 is a real value (greedy answer), not "unset".
    overrides.reply_temperature = 0.0F;
    resolved = sinfer::runtime::resolve_sampling(defaults, mode, overrides);
    failures += check(resolved.reply->temperature == 0.0F, "reply_temperature 0 was lost");
    for (const auto& bad : std::vector<sinfer::SamplingOverrides>{
             {.reply_temperature = -0.1F}, {.reply_temperature = 2.5F}, {.reply_top_k = -1},
             {.reply_top_p = 1.5F}, {.reply_min_p = -0.5F}}) {
        failures += check(throws_invalid([&] { (void)sinfer::runtime::resolve_sampling(defaults, mode, bad); }),
                          "an invalid reply value was accepted");
    }

    // The wire fields and the server defaults (--reply-*): a request's field replaces the server's.
    sinfer::serve::ServeOptions server;
    sinfer::serve::GenerationRequest request;
    request.sampling.seed = 7;
    auto wire = [&] { return sinfer::serve::to_request_options(request, server).execution.sampling; };
    failures += check(!wire().has_reply(), "no reply field and no server default still produced reply values");
    server.sampling_overrides.reply_temperature = 0.7F;
    failures += check(wire().reply_temperature == 0.7F, "--reply-temperature did not reach the request");
    request.sampling.reply_temperature = 0.5;
    request.sampling.reply_top_k = -1;
    request.sampling.reply_top_p = 0.9;
    request.sampling.reply_min_p = 0.02;
    auto overridden = wire();
    failures += check(overridden.reply_temperature == 0.5F && overridden.reply_top_k == 0 &&
                          overridden.reply_top_p == 0.9F && overridden.reply_min_p == 0.02F,
                      "request reply fields did not replace the server default (or reply_top_k -1 was not 0)");
    request.sampling.reply_temperature = 3.0;
    failures += check(throws_any(wire), "reply_temperature 3 was accepted on the wire");
    request.sampling.reply_temperature = 0.5;
    server.greedy = true;
    failures += check(wire().reply_temperature == 0.0F, "--greedy left a sampled answer");
    server.greedy = false;

    // The staging: before the close the config is the request's own apart from the two phase fields.
    sinfer::ops::SamplingConfig own;
    own.temperature       = 1.0F;
    own.top_k             = 0;
    own.top_p             = 1.0F;
    own.min_p             = 0.0F;
    own.presence_penalty  = 0.25F;
    own.seed              = 1234567;
    own.speculative_coupled = 1;
    own.suppressed[0]     = 9;
    own.suppressed_count  = 1;
    constexpr sinfer::TokenId kClose = 200002;
    sinfer::runtime::ReplyPhase off;
    const sinfer::ops::SamplingConfig unchanged = off.apply(own);
    failures += check(std::memcmp(&own, &unchanged, sizeof(own)) == 0,
                      "a request without reply values got a different config");
    sinfer::runtime::ReplyPhase phase;
    phase.armed  = true;
    phase.state  = sinfer::runtime::ReplyPhase::State::Thinking;
    phase.close  = kClose;
    phase.values = {.temperature = 0.7F, .top_k = 50, .top_p = 0.9F, .min_p = 0.01F};
    // Prompt of 4 tokens that itself contains the close token (a past turn): not counted.
    std::vector<sinfer::TokenId> ledger = {1, kClose, 2, 3};
    phase.frontier = ledger.size();
    failures += check(!phase.advance(ledger), "a close token in the prompt switched the phase");
    auto thought = phase.apply(own);
    failures += check(same_except_phase_fields(thought, own) && thought.phase_switch_armed == 1 &&
                          thought.phase_switch_token == kClose,
                      "the thought's config is not the request's own with the draft cut armed");
    for (const sinfer::TokenId token : {10, 11, 12}) ledger.push_back(token);
    failures += check(!phase.advance(ledger) && !phase.replying(), "thought tokens switched the phase");
    // A round that commits the close as its last token: the next round answers.
    ledger.push_back(13);
    ledger.push_back(kClose);
    failures += check(phase.advance(ledger) && phase.replying(), "the close token did not switch the phase");
    auto answer = phase.apply(own);
    failures += check(answer.temperature == 0.7F && answer.top_k == 50 && answer.top_p == 0.9F &&
                          answer.min_p == 0.01F && answer.phase_switch_armed == 0,
                      "the answer's config does not carry the reply values");
    // Everything else -- seed (the random stream), penalties, barrier, coupling -- is the request's own.
    sinfer::ops::SamplingConfig answer_rest = answer;
    answer_rest.temperature = own.temperature;
    answer_rest.top_k       = own.top_k;
    answer_rest.top_p       = own.top_p;
    answer_rest.min_p       = own.min_p;
    failures += check(same_except_phase_fields(answer_rest, own),
                      "the answer's config changed more than temperature / top_k / top_p / min_p");
    ledger.push_back(kClose);
    failures += check(!phase.advance(ledger) && phase.replying(), "a later close token changed the phase again");

    // A request resumed after a KV preemption: its generated tokens are the end of its prefill's prompt; a close among
    // them answers from the prefill's own draw on (start_prefill_lane walks from the end of its own prompt).
    sinfer::runtime::ReplyPhase resumed = phase;
    resumed.state = sinfer::runtime::ReplyPhase::State::Thinking;
    const std::vector<sinfer::TokenId> prefill = {1, 2, 3, 20, kClose, 21};
    resumed.frontier = prefill.size() - 3; // own prompt: 3 tokens; 3 generated before the preemption
    failures += check(resumed.advance(prefill) && resumed.replying(), "a resumed request lost its answer phase");

    // Opening (a model that opens its thought itself; Inkling-2's shape): header, then a content marker decides.
    constexpr sinfer::TokenId kHeader = 200001, kOpen = 200008, kText = 200004, kToolJson = 200049;
    const auto opening = [&] {
        sinfer::runtime::ReplyPhase p = phase;
        p.state      = sinfer::runtime::ReplyPhase::State::Opening;
        p.open       = kOpen;
        p.headers    = {kHeader, -1};
        p.frontier   = 0;
        return p;
    };
    auto op = opening();
    const auto opening_config = op.apply(own);
    failures += check(same_except_phase_fields(opening_config, own) && opening_config.phase_switch_armed == 2,
                      "an opening row is not the request's own config with every draft cut");
    std::vector<sinfer::TokenId> out = {kHeader};
    failures += check(!op.advance(out) && op.pending(), "the header token ended the opening");
    out.push_back(kOpen);
    failures += check(!op.advance(out) && op.state == sinfer::runtime::ReplyPhase::State::Thinking,
                      "<|content_thinking|> did not start the thought");
    for (const sinfer::TokenId token : {5, 6, kHeader, kText}) out.push_back(token); // a thought may mention anything
    failures += check(!op.advance(out) && !op.replying(), "tokens inside the thought ended it");
    out.push_back(kClose);
    failures += check(op.advance(out) && op.replying(), "the thought's close did not start the answer");
    for (const sinfer::TokenId first : {kText, kToolJson, sinfer::TokenId{42}}) {
        auto direct = opening();
        const std::vector<sinfer::TokenId> no_thought = {kHeader, first, 7, 8};
        failures += check(direct.advance(no_thought) && direct.replying() && direct.frontier == 2,
                          "an output that opens with an answer block did not answer right after its marker");
    }

    // The serving decision (Engine::submit): fold, arm, or ignore.
    using Plan = sinfer::runtime::ReplyPhasePlan;
    sinfer::ReplyPhaseVocabulary qwen{.model_opens = false, .close = 151668, .open = 151667};
    sinfer::ReplyPhaseVocabulary inkling{.model_opens = true, .close = kClose, .open = kOpen, .headers = {kHeader, -1}};
    sinfer::ReplyPhaseVocabulary plain{}; // no reasoning tokens at all
    sinfer::ReplyPhaseVocabulary channel{.model_opens = true}; // reasoning as a message channel: no close token
    const auto plan = sinfer::runtime::plan_reply_phase;
    failures += check(plan(qwen, true, true, false) == Plan::ArmThinking, "a chat prompt that opens a thought was not armed");
    failures += check(plan(qwen, false, true, false) == Plan::Fold, "a chat prompt without a thought was not folded");
    // Review item 1: token-id and /v1/completions prompts are never folded for a model whose prompts open thoughts.
    failures += check(plan(qwen, false, false, false) == Plan::ArmThinking, "a raw prompt of a thinking model was folded");
    // Review item 2: a JSON schema from the first token (structured output) is answer throughout.
    failures += check(plan(qwen, true, true, true) == Plan::Fold, "structured output was not folded");
    failures += check(plan(inkling, false, true, true) == Plan::Fold, "Inkling structured output was not folded");
    failures += check(plan(inkling, false, true, false) == Plan::ArmOpening, "an Inkling chat prompt was not armed opening");
    failures += check(plan(inkling, false, false, false) == Plan::ArmOpening, "an Inkling raw prompt was not armed");
    failures += check(plan(inkling, true, false, false) == Plan::ArmThinking,
                      "an Inkling raw prompt ending inside a thought was not armed thinking");
    failures += check(plan(plain, false, false, false) == Plan::Fold && plan(plain, false, true, false) == Plan::Fold,
                      "a model without reasoning tokens was not folded");
    failures += check(plan(plain, true, true, false) == Plan::Ignore && plan(channel, false, true, false) == Plan::Ignore,
                      "a thought without a single close token was not left alone");
    // The raw prompt's tail: inside a thought iff its last opening marker follows its last close.
    using Span = std::span<const sinfer::TokenId>;
    const std::vector<sinfer::TokenId> open_tail = {1, kOpen, 2, kClose, 3, kHeader, kOpen, 9, 9};
    const std::vector<sinfer::TokenId> closed_tail = {1, kOpen, 2, kClose, 3, kHeader};
    failures += check(sinfer::runtime::prompt_tail_in_thought(Span(open_tail), inkling) &&
                          !sinfer::runtime::prompt_tail_in_thought(Span(closed_tail), inkling) &&
                          !sinfer::runtime::prompt_tail_in_thought(Span(open_tail), plain),
                      "prompt_tail_in_thought misread a prompt's tail");

    if (failures == 0) { std::cout << "reply_phase: ok\n"; }
    return failures == 0 ? 0 : 1;
}
