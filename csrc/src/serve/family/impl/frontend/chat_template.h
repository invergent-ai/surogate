#pragma once


#include <api/family/prepared_prompt.h>
#include <api/types.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
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

} // namespace sinfer::family::frontend_internal
