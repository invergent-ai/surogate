#include "serve/vllm_max_tokens.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>

namespace sinfer::serve {

namespace {

// vllm/entrypoints/serve/exception_handling/error_response.py create_error_response: a VLLMValidationError (and a
// plain ValueError) is a 400 BadRequestError whose param is the error's parameter; str(VLLMValidationError) appends
// " (parameter=P, value=V)" for whichever of the two it has (vllm/exceptions.py).
ApiError bad_request(std::string message, const std::string& parameter, std::optional<std::int64_t> value) {
    if (!parameter.empty() || value) {
        std::string extras;
        if (!parameter.empty()) { extras = "parameter=" + parameter; }
        if (value) { extras += (extras.empty() ? "" : ", ") + std::string("value=") + std::to_string(*value); }
        message += " (" + extras + ")";
    }
    ApiError error;
    error.status       = 400;
    error.type         = "BadRequestError";
    error.message      = std::move(message);
    error.param        = parameter;
    error.numeric_code = 400;
    return error;
}

MaxTokensDecision refuse(ApiError error) {
    MaxTokensDecision decision;
    decision.error = std::move(error);
    return decision;
}

// build_tok_params: `max_output_tokens = max_tokens or 0`.
std::int64_t output_tokens(const MaxTokensQuery& query) { return query.requested.value_or(0); }

} // namespace

std::optional<ApiError> vllm_max_tokens_above_context(const MaxTokensQuery& query) {
    // TokenizeParams.__post_init__ (vllm/renderers/params.py:229-241).
    const std::int64_t M = output_tokens(query);
    if (M > query.max_model_len) {
        return bad_request(query.requested_param + "=" + std::to_string(M) + " cannot be greater than max_model_len=" +
                               std::to_string(query.max_model_len) + ". Please request fewer output tokens.",
                           query.requested_param, M);
    }
    return std::nullopt;
}

MaxTokensDecision decide_vllm_max_tokens(const MaxTokensQuery& query) {
    const std::int64_t C = query.max_model_len;
    const std::int64_t P = query.prompt_tokens;
    const std::int64_t M = output_tokens(query);

    // 1.
    if (std::optional<ApiError> error = vllm_max_tokens_above_context(query)) { return refuse(std::move(*error)); }

    // C - M, with M kept above -2^40 so the products below cannot overflow: no prompt reaches that bound, so
    // steps 2 and 3 never refuse (nor print these) for such an M, and step 5 prints M itself.
    const std::int64_t max_input = C - std::max<std::int64_t>(M, -(std::int64_t{1} << 40));
    if (query.text_prompt) {
        // 2. TokenizeParams._text_len_check (vllm/renderers/params.py:346), before the text is tokenized.
        const std::int64_t max_input_chars = max_input * query.max_chars_per_token;
        if (query.max_chars_per_token > 0 && query.prompt_characters > max_input_chars) {
            return refuse(bad_request(
                "This model's maximum context length is " + std::to_string(C) + " tokens. However, you requested " +
                    std::to_string(M) + " output tokens and your prompt contains " +
                    std::to_string(query.prompt_characters) + " characters (more than " +
                    std::to_string(max_input_chars) + " characters, which is the upper bound for " +
                    std::to_string(max_input) + " input tokens). Please reduce the length of the input prompt or the "
                    "number of requested output tokens.",
                "input_text", query.prompt_characters));
        }
    }

    // 3. TokenizeParams._token_len_check (vllm/renderers/params.py:485). Text was encoded with truncation=True,
    //    max_length=max_input_tokens + 1 (get_encode_kwargs), so its count stops there.
    const std::int64_t seen = query.text_prompt ? std::min(P, max_input + 1) : P;
    if (seen > max_input) {
        const std::string qualifier = seen == max_input + 1 ? "at least " : "";
        return refuse(bad_request(
            "This model's maximum context length is " + std::to_string(C) + " tokens. However, you requested " +
                std::to_string(M) + " output tokens and your prompt contains " + qualifier + std::to_string(seen) +
                " input tokens, for a total of " + qualifier + std::to_string(seen + M) +
                " tokens. Please reduce the length of the input prompt or the number of requested output tokens.",
            "input_tokens", seen));
    }

    // 4. get_max_tokens (vllm/entrypoints/serve/utils/api_utils.py:169); a plain ValueError, so no parameter.
    if (C < P) {
        return refuse(bad_request("Input length (" + std::to_string(P) + ") exceeds model's maximum context length (" +
                                      std::to_string(C) + ").",
                                  "", std::nullopt));
    }
    std::int64_t budget = C - P;
    if (query.requested) {
        budget = std::min(budget, *query.requested);
    } else if (query.default_max_tokens) {
        budget = std::min(budget, *query.default_max_tokens);
    }

    // 5./6. SamplingParams._verify_args (vllm/sampling_params.py:625-639), in its order.
    if (budget < 1) {
        return refuse(bad_request("max_tokens must be at least 1, got " + std::to_string(budget) + ".", "max_tokens",
                                  budget));
    }
    if (query.min_tokens < 0) {
        return refuse(bad_request("min_tokens must be greater than or equal to 0, got " +
                                      std::to_string(query.min_tokens) + ".",
                                  "", std::nullopt));
    }
    if (query.min_tokens > budget) {
        return refuse(bad_request("min_tokens must be less than or equal to max_tokens=" + std::to_string(budget) +
                                      ", got " + std::to_string(query.min_tokens) + ".",
                                  "", std::nullopt));
    }
    MaxTokensDecision decision;
    decision.budget = budget;
    return decision;
}

MaxTokensQuery max_tokens_query(const GenerationRequest& request, std::int64_t max_model_len,
                                std::int64_t prompt_tokens, std::int64_t prompt_characters,
                                std::int64_t max_chars_per_token) {
    MaxTokensQuery query;
    query.max_model_len = max_model_len;
    query.prompt_tokens = prompt_tokens;
    if (request.max_tokens_set) {
        query.requested       = request.max_tokens_wide.value_or(request.max_tokens);
        query.requested_param = request.max_tokens_param;
    } else if (request.max_tokens > 0) {
        query.default_max_tokens = request.max_tokens; // --default-max-tokens; none = the whole rest
    }
    query.min_tokens          = request.min_tokens;
    query.text_prompt         = request.prompt_token_ids.empty();
    query.prompt_characters   = query.text_prompt ? prompt_characters : 0;
    query.max_chars_per_token = max_chars_per_token;
    return query;
}

} // namespace sinfer::serve
