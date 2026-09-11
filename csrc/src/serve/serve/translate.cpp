#include "serve/translate.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer::serve {
namespace {

std::uint64_t random_seed() {
    static thread_local std::mt19937_64 rng(std::random_device{}());
    return rng();
}

[[noreturn]] void invalid_sampling(std::string message, std::string param) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    throw ApiException(std::move(error));
}

[[noreturn]] void invalid_prompt_option(std::string message, std::string param, std::string code) {
    ApiError error;
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

sinfer::SamplingOverrides resolve_sampling_overrides(const SamplingParams& request,
                                                     const ServeOptions& server) {
    sinfer::SamplingOverrides sampling = server.sampling_overrides;
    if (request.temperature) { sampling.temperature = static_cast<float>(*request.temperature); }
    if (request.top_p) { sampling.top_p = static_cast<float>(*request.top_p); }
    // vLLM spells "do not truncate" as -1 and every RL rollout sends it. The engine
    // contract takes 0 for the same thing and refuses a negative, so the wire value
    // is normalised here rather than loosened there.
    if (request.top_k) {
        sampling.top_k = *request.top_k < 0 ? 0 : static_cast<std::int32_t>(*request.top_k);
    }
    if (request.min_p) { sampling.min_p = static_cast<float>(*request.min_p); }
    if (request.repetition_penalty) {
        sampling.repetition_penalty = static_cast<float>(*request.repetition_penalty);
    }
    if (request.presence_penalty) {
        sampling.presence_penalty = static_cast<float>(*request.presence_penalty);
    }
    if (request.frequency_penalty) {
        sampling.frequency_penalty = static_cast<float>(*request.frequency_penalty);
    }
    for (const auto& [token, bias] : request.logit_bias) {
        if (token < 0 || !std::isfinite(bias) || bias < -100.0 || bias > 100.0) {
            invalid_sampling("logit_bias requires nonnegative token ids and finite values in [-100,100]", "logit_bias");
        }
        sampling.logit_bias[token] = static_cast<float>(bias);
    }
    if (request.seed) {
        sampling.seed = *request.seed;
    } else if (server.sampling_overrides.seed) {
        sampling.seed = *server.sampling_overrides.seed;
    } else {
        sampling.seed = random_seed();
    }

    const auto finite = [](const std::optional<float>& value) {
        return !value || std::isfinite(*value);
    };
    if (!finite(sampling.temperature) || !finite(sampling.top_p) || !finite(sampling.min_p) ||
        !finite(sampling.presence_penalty) || !finite(sampling.frequency_penalty)) {
        invalid_sampling("sampling parameters must be finite", "sampling");
    }
    if (sampling.temperature && (*sampling.temperature < 0.0F || *sampling.temperature > 2.0F)) {
        invalid_sampling("temperature must be in [0,2]", "temperature");
    }
    if (sampling.top_p && (*sampling.top_p < 0.0F || *sampling.top_p > 1.0F)) {
        invalid_sampling("top_p must be in [0,1]", "top_p");
    }
    // A negative top_k is how vLLM spells "do not truncate", and every rollout an
    // RL trainer sends carries -1. Refusing it rejected the request outright; it
    // means the same thing zero does here, which the sampler reads as "no limit of
    // the caller's own".
    if (sampling.top_k && *sampling.top_k < -1) {
        invalid_sampling("top_k must be -1 (no limit) or nonnegative", "top_k");
    }
    if (sampling.min_p && (*sampling.min_p < 0.0F || *sampling.min_p > 1.0F)) {
        invalid_sampling("min_p must be in [0,1]", "min_p");
    }
    if (sampling.presence_penalty &&
        (*sampling.presence_penalty < -2.0F || *sampling.presence_penalty > 2.0F)) {
        invalid_sampling("presence_penalty must be in [-2,2]", "presence_penalty");
    }
    if (sampling.frequency_penalty &&
        (*sampling.frequency_penalty < -2.0F || *sampling.frequency_penalty > 2.0F)) {
        invalid_sampling("frequency_penalty must be in [-2,2]", "frequency_penalty");
    }
    if (server.greedy) { sampling.temperature = 0.0F; }
    return sampling;
}

std::vector<std::string> effective_tool_jsons(const GenerationRequest& request) {
    std::vector<std::string> tools;
    if (!request.uses_tools()) { return tools; }
    if (request.tool_choice.mode == ToolChoiceMode::Named) {
        for (const ToolDefinition& tool : request.tools) {
            if (tool.name == request.tool_choice.name) {
                tools.push_back(tool.definition_json);
                break;
            }
        }
        return tools;
    }
    tools.reserve(request.tools.size());
    for (const ToolDefinition& tool : request.tools) { tools.push_back(tool.definition_json); }
    return tools;
}

/// The template's own vocabulary, for an error that says what to ask for instead of
/// only what not to. Empty when the loaded template chooses no effort at all.
std::string supported_efforts(const sinfer::ReasoningEffortCapabilities& capabilities) {
    std::string list;
    for (const sinfer::ReasoningEffort effort : sinfer::kReasoningEfforts) {
        if (!capabilities.supports(effort)) { continue; }
        if (!list.empty()) { list += ", "; }
        list += sinfer::reasoning_effort_name(effort);
    }
    return list;
}

/// The wire vocabulary and the template vocabulary are the same names. Which of them
/// a given template honours is its capabilities' business, checked below -- nothing
/// is remapped onto a neighbouring effort on the way, because a request for one
/// setting answered at another is the silent-wrong this path exists to prevent.
sinfer::ReasoningEffort to_template_effort(RequestedReasoningEffort requested) {
    switch (requested) {
    case RequestedReasoningEffort::Minimal:
        return sinfer::ReasoningEffort::Minimal;
    case RequestedReasoningEffort::Low:
        return sinfer::ReasoningEffort::Low;
    case RequestedReasoningEffort::Medium:
        return sinfer::ReasoningEffort::Medium;
    case RequestedReasoningEffort::High:
        return sinfer::ReasoningEffort::High;
    case RequestedReasoningEffort::XHigh:
        return sinfer::ReasoningEffort::XHigh;
    case RequestedReasoningEffort::Max:
        return sinfer::ReasoningEffort::Max;
    case RequestedReasoningEffort::None:
        break;
    }
    throw std::logic_error("reasoning effort 'none' names no template effort");
}

} // namespace

ResolvedPromptSemantics resolve_prompt_semantics(const GenerationRequest& request,
                                                 const ServeOptions& server,
                                                 const sinfer::PromptCapabilities& capabilities) {
    ResolvedPromptSemantics result{
        .enable_thinking   = request.enable_thinking.value_or(server.enable_thinking),
        .reasoning_effort  = std::nullopt,
        .preserve_thinking = request.preserve_thinking.value_or(server.preserve_thinking),
    };
    // Asking a template that always thinks, and carries no switch, to stop thinking.
    // It was accepted and dropped, and the answer came back with the reasoning in it;
    // saying so is the only honest answer. Asking such a template *to* think is not
    // refused -- it is already doing it.
    if (request.enable_thinking && !*request.enable_thinking && !capabilities.enable_thinking &&
        capabilities.reasoning_turn) {
        invalid_prompt_option("the loaded chat template always opens a reasoning turn and has no "
                              "switch to disable it",
                              request.enable_thinking_param, "thinking_toggle_not_supported");
    }
    if (!request.reasoning_effort) { return result; }

    const RequestedReasoningEffort requested = *request.reasoning_effort;
    const bool enables_thinking              = requested != RequestedReasoningEffort::None;
    if (request.enable_thinking && *request.enable_thinking != enables_thinking) {
        invalid_prompt_option("reasoning effort conflicts with enable_thinking",
                              request.reasoning_effort_param, "conflicting_template_option");
    }
    result.enable_thinking = enables_thinking;

    if (requested == RequestedReasoningEffort::None) {
        if (!capabilities.enable_thinking) {
            invalid_prompt_option("the loaded chat template cannot disable thinking",
                                  request.reasoning_effort_param, "reasoning_effort_not_supported");
        }
        return result;
    }

    result.reasoning_effort = to_template_effort(requested);
    if (!capabilities.reasoning_effort.supports(*result.reasoning_effort)) {
        const std::string accepted = supported_efforts(capabilities.reasoning_effort);
        invalid_prompt_option(
            "reasoning effort '" + std::string(requested_reasoning_effort_name(requested)) +
                "' is not supported by the loaded chat template" +
                (accepted.empty() ? std::string(", which chooses no reasoning effort")
                                  : ", which accepts " + accepted),
            request.reasoning_effort_param, "reasoning_effort_not_supported");
    }
    return result;
}

sinfer::PromptInput to_prompt_input(const GenerationRequest& request,
                                    const ResolvedPromptSemantics& semantics,
                                    const MediaAcquirer& acquire_media) {
    sinfer::PromptInput input;
    input.messages.reserve(request.messages.size());
    for (const ChatTurn& turn : request.messages) {
        sinfer::ChatMessage message;
        message.role              = turn.role;
        message.reasoning_content = turn.reasoning_content;
        message.tool_call_id      = turn.tool_call_id;
        message.tool_calls.reserve(turn.tool_calls.size());
        for (const ToolCall& call : turn.tool_calls) {
            message.tool_calls.push_back(sinfer::ToolCall{call.id, call.name, call.arguments_json});
        }

        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Text) {
                if (!message.parts.empty() && !part.text.empty() &&
                    message.parts.back().kind == sinfer::MessagePartKind::Text) {
                    sinfer::MessagePart newline;
                    newline.text = "\n";
                    message.parts.push_back(std::move(newline));
                }
                sinfer::MessagePart text;
                text.text = part.text;
                message.parts.push_back(std::move(text));
                continue;
            }
            if (part.kind == ContentKind::Image || part.kind == ContentKind::Video) {
                if (!acquire_media) {
                    throw std::logic_error("media acquisition callback is not configured");
                }
                sinfer::MessagePart media;
                media.kind  = sinfer::MessagePartKind::Media;
                media.media = acquire_media(part);
                message.parts.push_back(std::move(media));
                continue;
            }

            ApiError error;
            error.message = "content type '" + part.type_raw + "' is not supported";
            error.param   = "messages";
            error.code    = "modality_not_supported";
            throw ApiException(std::move(error));
        }
        input.messages.push_back(std::move(message));
    }

    input.options.add_generation_prompt = request.add_generation_prompt.value_or(true);
    input.options.enable_thinking       = semantics.enable_thinking;
    input.options.reasoning_effort      = semantics.reasoning_effort;
    input.options.preserve_thinking     = semantics.preserve_thinking;
    input.options.add_vision_id         = false;
    input.options.tool_jsons            = effective_tool_jsons(request);
    return input;
}

sinfer::RequestOptions to_request_options(const GenerationRequest& request,
                                          const ServeOptions& server) {
    sinfer::RequestOptions options;
    options.execution.requested_output_tokens = static_cast<std::uint32_t>(request.max_tokens);
    options.execution.allow_prefix_reuse      = server.allow_prefix_reuse;
    options.execution.json_schema = request.json_schema;
    if (!request.json_schema.empty() && (request.ignore_eos || request.min_tokens != 0 || request.uses_tools() || !request.stop_strings.empty())) {
        invalid_sampling("JSON response formats require ignore_eos=false, min_tokens=0, no custom stops, and no active tools", "response_format");
    }
    options.execution.sampling             = resolve_sampling_overrides(request.sampling, server);
    options.output.raw                     = false;
    options.output.preserve_special_tokens = request.uses_tools() || request.has_tool_history();
    options.stop.include_model_defaults = !request.ignore_eos;
    options.stop.strings.reserve(request.stop_strings.size());
    for (const std::string& stop : request.stop_strings) {
        if (!stop.empty()) {
            options.stop.strings.push_back(
                sinfer::StopString{.text              = stop,
                                   .channel           = sinfer::OutputChannel::Content,
                                   .include_in_output = false});
        }
    }
    return options;
}

const char* finish_reason_wire(sinfer::FinishReason reason) {
    switch (reason) {
    case sinfer::FinishReason::OutputLimit:
    case sinfer::FinishReason::ContextCapacity:
        return "length";
    case sinfer::FinishReason::None:
    case sinfer::FinishReason::StopToken:
    case sinfer::FinishReason::StopString:
    case sinfer::FinishReason::Cancelled:
        return "stop";
    }
    return "stop";
}

} // namespace sinfer::serve
