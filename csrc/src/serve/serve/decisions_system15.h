#pragma once

// The decisions protocol of a System 1.5 decision adapter (`model` naming one; api/targets/inkling2/decision.h): the
// same request as decisions v1, answered by the adapter's heads instead of a label readout. This file is the model's
// own prompt protocol (eltokh7/inkling-2-mini-decision `modeling.py`: `_render`, `_opt_text`, `_encode_state`,
// `_encode_question_suffix`), ported exactly, and the answer: v1's answer object computed from the adapter's
// calibrated distribution (`resolve_decision_answer` with the pass's exit temperature), plus a `system15` object.
//
// It is not decisions v1 and does not change v1: a request reaches it only through a decision adapter's name.

#include "serve/decisions_schema.h"

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::serve {

/// `_render(v)`: str() of a scalar (Python spelling: True / False, repr floats), "" for null, a list as "- " lines and
/// an object as "key: value" / "key:\n<nested>" lines, nested by two spaces.
[[nodiscard]] std::string system15_render(const OrderedJson& value, int indent = 0);

/// `<|name|>` -> `<¦name¦>` in text the tokenizer reads (the reference's _SPECIAL_RE escape).
[[nodiscard]] std::string system15_escape(std::string_view text);

struct System15Prompt {
    std::vector<TokenId> ids;
    std::int32_t decide_idx = 0;
    std::vector<std::int32_t> option_idx;
    std::vector<std::string> keys; ///< answer keys in option order
};

/// The token ids of the state prefix: [state marker] + encode(render(state)), truncated to `max_state` tokens.
[[nodiscard]] std::vector<TokenId> system15_state_tokens(const OrderedJson& state,
                                                         const std::function<std::vector<TokenId>(std::string_view)>& encode,
                                                         const std::vector<TokenId>& markers, std::size_t max_state);

/// One question's whole prompt after the state tokens. `markers` = the ids of the adapter's five special tokens
/// (state, question, option open, option close, decide).
[[nodiscard]] System15Prompt system15_question_prompt(const std::vector<TokenId>& state_tokens,
                                                      const DecisionQuestion& question,
                                                      const std::function<std::vector<TokenId>(std::string_view)>& encode,
                                                      const std::vector<TokenId>& markers);

} // namespace sinfer::serve
