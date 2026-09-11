#pragma once

#include "api/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <memory>
#include <span>
#include <vector>

namespace sinfer::family {

inline constexpr std::size_t kTokenDomain = 248077;

struct FrontendOptions {
    bool vision_enabled                    = true;
    std::uint32_t max_context              = 2'048;
    std::size_t media_cache_bytes          = kDefaultMediaCacheBytes;
    std::size_t media_live_bytes           = kDefaultMediaLiveBytes;
    std::uint32_t media_preprocess_threads = 0;
    /// Replaces the artifact's chat template when non-empty (--chat-template).
    std::string chat_template_override;
    /// Whether the artifact's tokenizer is one of the family's registered
    /// Qwen3.5/3.6 checkpoints, whose exact 248,077-token domain and Vision
    /// token IDs are then asserted. A target outside that family (a dense Qwen3,
    /// Llama or Gemma stack) clears this: its tokenizer is a different, smaller
    /// domain and has no Vision tokens at all, and asserting the family's would
    /// reject a correct artifact.
    bool registered_tokenizer = true;
};

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] PromptPreparationStats preparation_stats() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;
    /// A deep copy (pipeline stages each start the same prompt on their own program).
    [[nodiscard]] PreparedPrompt clone() const;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
};

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }

    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }

    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                  std::uint32_t budget_remaining,
                                                  FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input,
                                         const PreparationControl& control = {}) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input,
                                             const PreparationControl& control = {}) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    /// A prompt served exactly as written, with no chat template applied.
    ///
    /// This is what `/v1/completions` sends and what a base model needs: a checkpoint that was
    /// never taught a turn structure has no template to render, and one that has a template
    /// still answers a raw continuation when asked for one. Tokenization is the tokenizer's
    /// own, so a SentencePiece checkpoint gets the leading BOS its encoder adds and a BPE one
    /// does not, matching what each produces on the chat path.
    [[nodiscard]] PreparedPrompt prepare_text(std::string_view text,
                                              bool allow_prefix_identity = true) const;
    /// Whether this artifact carries a chat template, and so whether the chat-shaped
    /// endpoints have anything to render with.
    [[nodiscard]] bool supports_chat() const noexcept;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] MediaCacheSummary media_cache_summary() const;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;
    /// The text of each token id, one string per id.
    ///
    /// Per-token rather than per-sequence: a client that asks for probabilities is
    /// given one number per token and needs the token each belongs to. Decoding the
    /// ids together instead would merge them into one string, which is the right
    /// answer to a different question.
    void validate_sampling_tokens(const ResolvedSamplingParameters& sampling) const;
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile_json_constraint(
        const std::string& schema) const;
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile_tool_constraint(
        const std::string& structural_tag) const;
    [[nodiscard]] std::vector<std::string> token_texts(std::span<const TokenId> ids) const;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, FrontendOptions options);

} // namespace sinfer::family
