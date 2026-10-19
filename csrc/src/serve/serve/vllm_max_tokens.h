#pragma once

// The output budget of a request, decided exactly as vLLM 0.31 decides it before the engine sees the request.
//
// vLLM refuses a request whose prompt and requested output do not fit the context: max_tokens (or
// max_completion_tokens) larger than max_model_len, or prompt + max_tokens > max_model_len, is HTTP 400 up front
// (vllm/renderers/params.py TokenizeParams). Without a max_tokens the budget is what is left of the context, or the
// generation-config default when that is smaller, and no error (vllm/entrypoints/serve/utils/api_utils.py
// get_max_tokens). The engine used to cap an overlong max_tokens to the rest of the context instead, so a ~256k-token
// prompt asking for 32768 tokens ran until the context was full and came back finish=length with reasoning only,
// which a client took for an empty assistant message (tau3-Bench task_091, 2026-10-19).
//
// The steps, in vLLM's order (M = the client's max_tokens, 0 when absent; C = max_model_len; P = prompt tokens;
// N = the prompt text's length in code points; L = the tokenizer's longest token in code points):
//   1. M > C                      -> "<param>=M cannot be greater than max_model_len=C. ..."  (params.py:229)
//   2. N > (C - M) * L            -> "... your prompt contains N characters (more than (C - M) * L characters, which
//                                     is the upper bound for C - M input tokens) ..."  (params.py:346 _text_len_check)
//   3. P' > C - M                 -> "This model's maximum context length is C tokens. However, you requested M
//                                     output tokens and your prompt contains P' input tokens, ..."  (params.py:485)
//      vLLM tokenizes prompt text -- the rendered chat template as much as a /v1/completions prompt -- with
//      truncation at C - M + 1 tokens (params.py:319 get_encode_kwargs), so P' = min(P, C - M + 1), and a count of
//      exactly C - M + 1 is worded "at least" (also when it is the true count). Token ids are counted as sent.
//   4. P > C                      -> "Input length (P) exceeds model's maximum context length (C)."
//      (reachable only with a negative M)                                                 (api_utils.py:184)
//      budget = min(C - P, M if given else the server default)
//   5. budget < 1                 -> "max_tokens must be at least 1, got <budget>."     (sampling_params.py:625)
//   6. min_tokens < 0             -> "min_tokens must be greater than or equal to 0, got <min_tokens>."
//      min_tokens > budget        -> "min_tokens must be less than or equal to max_tokens=<budget>, got ..."
// Every refusal is vLLM's body: {"error": {"message", "type": "BadRequestError", "param", "code": 400}}, the
// message carrying vLLM's " (parameter=..., value=...)" suffix where its VLLMValidationError has a parameter
// (vllm/exceptions.py), and HTTP 400 for a streaming request too: vLLM opens the stream only after these steps.
//
// One deliberate difference: vLLM's /v1/completions takes an absent max_tokens as 16 (the OpenAI default) and checks
// it like a given one; here every endpoint takes an absent max_tokens as "the server default, capped to the rest of
// the context" (--default-max-tokens), which is what vLLM does for chat, and for completions sent "max_tokens": null.
//
// --overlong-max-tokens cap (ServeOptions::overlong_max_tokens) skips all of this and keeps the engine's earlier
// behaviour, an overlong max_tokens silently capped to the rest of the context.

#include "serve/request.h"

#include <cstdint>
#include <optional>
#include <string>

namespace sinfer::serve {

struct MaxTokensQuery {
    std::int64_t max_model_len = 0;
    std::int64_t prompt_tokens = 0;
    /// The client's max_tokens / max_completion_tokens / max_output_tokens, any value it sent; nullopt when absent.
    std::optional<std::int64_t> requested;
    /// The field `requested` came from, as vLLM names it in step 1.
    std::string requested_param = "max_tokens";
    /// The budget when the client sent none (vLLM: generation_config max_new_tokens); nullopt = the whole rest.
    std::optional<std::int64_t> default_max_tokens;
    std::int64_t min_tokens = 0;
    /// The prompt was tokenized from text (a rendered chat template, or a completions prompt), so vLLM's character
    /// bound and its truncated count apply. False for token ids.
    bool text_prompt = true;
    /// N and L above. 0 skips step 2 (token ids, or a text the engine did not measure).
    std::int64_t prompt_characters   = 0;
    std::int64_t max_chars_per_token = 0;
};

struct MaxTokensDecision {
    /// Set when vLLM refuses the request; `budget` is then meaningless.
    std::optional<ApiError> error;
    std::int64_t budget = 0;
};

[[nodiscard]] MaxTokensDecision decide_vllm_max_tokens(const MaxTokensQuery& query);

/// Step 1 alone, which vLLM takes before it renders the prompt: the refusal of a max_tokens larger than the
/// context, or nullopt.
[[nodiscard]] std::optional<ApiError> vllm_max_tokens_above_context(const MaxTokensQuery& query);

/// The query for a parsed request: its max_tokens when the client sent one (max_tokens_set), else its server
/// default; a text prompt unless the client sent token ids.
[[nodiscard]] MaxTokensQuery max_tokens_query(const GenerationRequest& request, std::int64_t max_model_len,
                                              std::int64_t prompt_tokens, std::int64_t prompt_characters,
                                              std::int64_t max_chars_per_token);

} // namespace sinfer::serve
