#pragma once

#include "family/impl/frontend/chat_template.h"

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sinfer::family::frontend_internal {

struct EncodeOptions {
    bool parse_added_tokens = true;
};

struct DecodeOptions {
    bool skip_special_tokens = false;
    std::vector<int> stop_token_ids;
};

struct AddedToken {
    int id = -1;
    std::string content;
    bool single_word = false;
    bool lstrip      = false;
    bool rstrip      = false;
    bool normalized  = false;
    bool special     = false;
};

struct TokenizerResources {
    std::string_view tokenizer_json;
    std::string_view tokenizer_config_json;
    std::string_view generation_config_json;
    /// The chat template in force. The delegate renders it; a
    /// standalone template takes precedence over tokenizer_config.json's copy,
    /// which is the HF convention the project tokenizer implements.
    std::string_view chat_template_jinja;
    /// Set when the family reproduces no hand-written renderer for this template, so
    /// the artifact's own Jinja is the only thing that can render it. Rendering is
    /// independent of the encoding scheme: a BPE checkpoint carrying its own template
    /// -- what every GGUF is -- needs the renderer just as a SentencePiece one does.
    bool render_chat_template = false;
};

namespace project_delegate {
/// Opaque holder for the project tokenizer (csrc/src/tokenizer), so this header
/// does not drag its includes into every frontend translation unit. It is a BPE
/// tokenizer that also implements SentencePiece, and it owns the Jinja renderer.
struct Handle;
void destroy(Handle* handle);
struct Deleter {
    void operator()(Handle* handle) const { destroy(handle); }
};
} // namespace project_delegate

class Tokenizer {
public:
    explicit Tokenizer(TokenizerResources resources);
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    std::vector<int> encode(std::string_view text, EncodeOptions options = {}) const;
    std::string decode(std::span<const int> ids, DecodeOptions options = {}) const;
    std::string decode_token_bytes(int id, bool skip_special_tokens = false) const;

    [[nodiscard]] const std::vector<int>& default_stop_token_ids() const noexcept {
        return default_stop_token_ids_;
    }

    /// True when this checkpoint's chat template is rendered by the project
    /// tokenizer from the artifact's own Jinja, rather than reproduced by the
    /// family's hand-written ChatML.
    [[nodiscard]] bool renders_chat_template() const noexcept;
    /// Renders the artifact's own Jinja template. Messages are (role, content), and
    /// the variables are what the request asked the template for -- each one left
    /// undefined unless the template was found to honour it.
    [[nodiscard]] std::string
    render_chat_template(const std::vector<std::pair<std::string, std::string>>& messages,
                         bool add_generation_prompt,
                         const ChatTemplateVariables& variables = {}) const;

    [[nodiscard]] bool is_special_token(int id) const noexcept;
    [[nodiscard]] bool is_valid_token(int id) const noexcept;
    [[nodiscard]] bool has_exact_token_domain(std::size_t size) const noexcept;

private:
    std::vector<std::string> id_to_token_;
    std::vector<bool> valid_token_ids_;
    std::unordered_map<std::string, int> vocab_token_to_id_;
    std::unordered_map<std::string, int> bpe_merge_ranks_;
    bool has_bpe_merges_ = true;
    bool ignore_merges_ = false;
    bool normalize_nfc_ = false;
    /// Ordered Split stages preceding ByteLevel, when a single word rule is insufficient.
    std::vector<std::string> split_patterns_;
    /// How many digits a pre-token may hold. Every checkpoint this family served until GLM-5.3
    /// declares `\p{N}` and takes one; GLM-5.3 declares `\p{N}{1,3}` and takes up to three,
    /// which is what makes "3,344" tokenise as `3` `,` `34` `4` rather than as five digits. It
    /// is read from the pre-tokenizer the artifact declares, not compiled in: the rule was the
    /// Qwen family's, and it was the splitter's name that said so.
    std::size_t max_digit_run_ = 1;
    std::vector<AddedToken> added_tokens_;
    std::array<std::vector<std::size_t>, 256> added_token_candidates_;
    std::vector<int> default_stop_token_ids_;

    /// Set when this checkpoint uses the SentencePiece scheme, which the
    /// byte-level path above cannot represent. Encode and decode delegate to it;
    /// everything else -- the vocabulary, the added tokens, the stop ids -- is
    /// still read here, because the artifact contract is the same either way.
    std::unique_ptr<project_delegate::Handle, project_delegate::Deleter> delegate_;
    /// The delegate always renders when present; it encodes only for SentencePiece,
    /// so a BPE checkpoint keeps the encoder above and gains nothing but the renderer.
    bool delegate_encodes_ = false;
};

} // namespace sinfer::family::frontend_internal
