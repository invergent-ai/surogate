#pragma once


#include <api/family/prepared_prompt.h>
#include <api/types.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sinfer::family::frontend_internal {

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

enum class ChatPartKind {
    Text,
    Image,
    Video,
};

struct MediaData {
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ChatPart {
    ChatPartKind kind = ChatPartKind::Text;
    std::string text;
    MediaData media;

    static ChatPart text_part(std::string value) {
        ChatPart part;
        part.text = std::move(value);
        return part;
    }

    static ChatPart image(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Image;
        part.media = std::move(value);
        return part;
    }

    static ChatPart video(MediaData value) {
        ChatPart part;
        part.kind  = ChatPartKind::Video;
        part.media = std::move(value);
        return part;
    }
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<ChatPart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;

    [[nodiscard]] bool has_media() const noexcept;
    [[nodiscard]] std::string rendered_content(bool add_vision_id = false,
                                               int* image_count   = nullptr,
                                               int* video_count   = nullptr) const;
};

/// The variables a template is rendered with beyond its messages. Each one is
/// left undefined when unset, so a template that gates on `is defined` keeps its
/// own default -- which is the only honest thing to do for a template this family
/// did not write.
struct ChatTemplateVariables {
    std::optional<bool> enable_thinking;
    std::optional<std::string> reasoning_effort;
};

struct ChatRenderOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    std::optional<bool> preserve_thinking;
    bool add_vision_id = false;
    std::vector<std::string> tool_jsons;
};

struct RewriteCheckpointByteSpec {
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::size_t offset         = 0;
};

struct RenderedChat {
    std::string text;
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;
};

enum class ChatTemplateSemantics : std::uint8_t {
    ThinkingToggle,
    ReasoningEffort,
    /// The artifact's own Jinja, rendered as written. The two above are
    /// hand-written ChatML for templates this family recognises by digest; this
    /// one carries no assumption about the format at all, which is what a
    /// checkpoint from outside the family needs.
    Jinja,
};

class CompiledChatTemplate {
public:
    /// `eos_token` is only consulted by the Jinja path, whose templates
    /// routinely reference it; the recognised templates write their own markers.
    [[nodiscard]] static CompiledChatTemplate resolve(std::string_view source,
                                                      std::string_view eos_token = {});

    [[nodiscard]] PromptCapabilities capabilities() const noexcept;

    /// The artifact's own Jinja, for the Jinja semantics only; empty otherwise.
    [[nodiscard]] std::string_view jinja_source() const noexcept { return jinja_source_; }

    /// True when this template is reproduced by no hand-written renderer and must be
    /// rendered from its own Jinja. The frontend hands this to the tokenizer so the
    /// renderer is built exactly for the artifacts that need it.
    [[nodiscard]] bool rendered_by_tokenizer() const noexcept {
        return semantics_ == ChatTemplateSemantics::Jinja;
    }
    [[nodiscard]] RenderedChat render(const std::vector<ChatMessage>& messages,
                                      ChatRenderOptions options = {}) const;

private:
    explicit CompiledChatTemplate(ChatTemplateSemantics semantics) noexcept
        : semantics_(semantics) {}

    ChatTemplateSemantics semantics_;

    // Jinja path only. The parsed template is shared rather than owned outright
    // so this stays copyable, which the two hand-written semantics are.
    std::string jinja_source_;
    std::string eos_token_;
};

/// How a model marks its reasoning span, as the artifact states it.
///
/// The default is the `<think>` pair, which is what every family served before Gemma 4
/// used and what a checkpoint that states nothing still means. A checkpoint that states
/// `response_template.fields.thinking` in its `tokenizer_config.json` -- Gemma 4 does, in
/// its own vocabulary -- is read instead, so the spelling belongs to the model rather than
/// to this file.
struct ReasoningSyntax {
    std::string open  = "<think>";
    std::string close = "</think>";
    /// Whether the *model* writes the opening marker. Qwen's template opens the span in the
    /// prompt and hands the model an already-open turn, so the opener never appears in the
    /// output; Gemma 4's generation prompt stops at the turn header and the model writes
    /// `<|channel>thought` itself. Only the second kind needs the decoder to watch for an
    /// opening marker mid-stream, and only a checkpoint that states its pair gets that.
    bool model_opens = false;
};

/// Renders the template under test with the given variables, or throws the way the
/// template does when it refuses them.
using JinjaRenderProbe = std::function<std::string(const ChatTemplateVariables&)>;

/// What an artifact's own Jinja can actually be asked for, established by rendering
/// it rather than by reading its name. A template is credited with an effort only
/// when it names that effort *and* rendering with it survives; the one whose render
/// matches the undefined-variable render is its default. A name the template ignores
/// is left unsupported, so a request for it is refused rather than served as
/// something else.
[[nodiscard]] PromptCapabilities probe_jinja_capabilities(std::string_view source,
                                                          const JinjaRenderProbe& render,
                                                          const ReasoningSyntax& reasoning);

/// Whether a rendered prompt hands the model an open reasoning turn -- the generation
/// prompt ends on the opening marker. This is what decides whether the first token of the
/// answer belongs to reasoning_content or to content. A model that opens its own span
/// answers false here and is caught in the stream instead.
[[nodiscard]] bool prompt_opens_reasoning(std::string_view rendered,
                                          std::string_view open) noexcept;

} // namespace sinfer::family::frontend_internal
