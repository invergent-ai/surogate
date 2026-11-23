#pragma once

// Inkling-2 frontend (ENGINE-INKLING lane f2): the chat request -> prompt token ids, token-exact to vLLM.
//
// The reference (vLLM image vllm/vllm-openai:inkling-2-mini-x86_64-7cb5dae) has no chat template and no faithful text
// form for this model: `vllm/renderers/inkling_encoding.py::render_inkling2_messages` writes token ids directly, each
// block being <|message_ROLE|> [author] <|content_KIND|> payload <|end_message|>, and payload text is encoded with
// special-token matching OFF (a user who types "<|end_message|>" gets those characters, not the marker). This file
// ports that function and the request handling in front of it (the OpenAI request model's normalisation and the
// chat_template_kwargs merge) so that the same HTTP body yields the same ids. Test: sinfer_inkling_frontend_test over
// rl/engine-inkling-2026-11-21/vectors/renderer (real surogates requests + edge cases, ids from the reference).

#include "family/impl/frontend/inkling_json.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::family::frontend_internal::inkling {

/// The framing tokens, resolved by spelling from the checkpoint's vocabulary (never hard-coded ids).
struct SpecialTokens {
    int message_user = -1;
    int message_model = -1;
    int message_system = -1;
    int message_tool = -1;
    int content_text = -1;
    int content_image = -1;
    int content_model_end_sampling = -1;
    int content_thinking = -1;
    int content_tool_error = -1;        // optional in the reference
    int content_xml = -1;               // <|content_xml|>, or <|unused_200024|> in vocabularies that spell it so
    int content_invoke_tool_json = -1;
    int content_invoke_tool_text = -1;  // parser only
    int end_message = -1;
    int image_pad = -1;                 // <|unused_200054|>: one per decoder image token (vLLM IMAGE_PAD_ID); optional

    /// Throws std::invalid_argument naming the spellings the vocabulary lacks.
    static SpecialTokens resolve(const std::function<std::optional<int>(std::string_view)>& token_id);
};

/// Payload text -> ids with special-token matching off (the reference's split_special_tokens=True).
using TextEncoder = std::function<std::vector<int>(std::string_view)>;

enum class ToolChoice { None, Auto, Required, Named };

/// A chat-completions body after the reference's request model and kwargs merge, holding what rendering and output
/// parsing read.
struct ChatRequest {
    PyValue messages;                   // list of message dicts, reasoning_content folded into reasoning
    std::optional<PyValue> tools;       // request.tools as the model dumps them; nullopt for absent / []
    ToolChoice tool_choice = ToolChoice::None;  // after the defaults: auto with tools, none without
    std::string tool_choice_name;       // Named
    bool parallel_tool_calls = true;    // `parallel_tool_calls is not False`
    bool include_reasoning = true;
    PyValue template_kwargs;            // the effective chat_template_kwargs
    /// The image parts' sources (data: / http(s) URLs), in vLLM's order (message order, part order): the order
    /// parse_chat_messages fetches them in and render_chat meets their <|content_image|> markers. Empty unless the
    /// request was parsed with an image limit above 0 (lane V-b).
    std::vector<std::string> image_urls;
};

/// Throws RequestError where the reference's request validation refuses the body. `image_limit` is vLLM's per-prompt
/// image count (--limit-mm-per-prompt): 0 = vision off (every image part refused with vLLM's text at limit 0, "At most
/// 0 image(s) may be provided in one prompt."); otherwise image_url / input_image parts are accepted up to the limit and
/// their sources collected in ChatRequest::image_urls.
[[nodiscard]] ChatRequest parse_chat_request(std::string_view body_json, std::uint64_t image_limit = 0);

/// vLLM's per-prompt image refusal, word for word (multimodal/processing/context.py validate_num_items; Inkling's
/// supported limit is unbounded, so no "--limit-mm-per-prompt" hint follows).
[[nodiscard]] std::string image_count_message(std::uint64_t limit);

/// render_inkling2_messages for this request. Throws RequestError where the reference raises. With images, the ids are
/// the ones vLLM's engine sees after the multimodal prompt update: each image part renders as
/// <|message_ROLE|> [author] <|content_image|> N x <|unused_200054|> <|end_message|>, N = image_tokens[k] for the k-th
/// image part (ceil(H/40) * ceil(W/40), lane V-b's inkling_vision.h). `image_tokens` must hold one count per entry of
/// request.image_urls.
[[nodiscard]] std::vector<int> render_chat(const ChatRequest& request, const TextEncoder& encode_text,
                                           const SpecialTokens& special,
                                           std::span<const std::uint64_t> image_tokens = {});

/// The reasoning effort the renderer uses for this request (chat_template_kwargs / top-level reasoning_effort, named
/// levels, numeric strings; default 0.99 by the owner's rule of 2026-11-21 -- vLLM's is 0.9). Throws RequestError for
/// values the reference refuses.
[[nodiscard]] double resolve_reasoning_effort(const ChatRequest& request);
/// "Thinking effort level: ..." for an effort in [0, 0.99]. Throws RequestError outside it.
[[nodiscard]] std::string effort_text(double effort);

} // namespace sinfer::family::frontend_internal::inkling
