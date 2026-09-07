#pragma once

// The Gemma tokenizer, over upstream SentencePiece.
//
// Not a second implementation of the engine's BPE. Gemma's tokenizer is BPE by
// type but SentencePiece by everything else: spaces become U+2581 in the
// normalizer, so the pre-tokenizer's split finds nothing and the entire string
// is one BPE unit, with byte fallback for whatever the 262,144-entry vocabulary
// cannot spell. The engine's own tokenizer is ByteLevel BPE with a Qwen
// pretokenizer baked in and reads none of the pipeline stages from the file.
//
// The checkpoint ships `tokenizer.model` beside `tokenizer.json`, and upstream
// SentencePiece reproduces the HF ids from it exactly -- verified over 414 cases
// including empty strings, whitespace runs, emoji, CJK, Cyrillic and 400 random
// mixed-alphabet strings. It is also 4.7 MB against the JSON's 33 MB. Writing a
// 514,906-merge BPE by hand to match that would be strictly worse.
//
// `<bos>` and `<eos>` are added here rather than by SentencePiece: they come
// from the tokenizer.json post-processor's template, which the .model file does
// not carry.

#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace sinfer::encoder {

class GemmaTokenizer {
public:
    /// From a serialized SentencePiece model -- the artifact's
    /// `frontend/tokenizer.model` resource, as bytes.
    static GemmaTokenizer from_serialized_proto(std::span<const std::byte> proto);

    ~GemmaTokenizer();
    GemmaTokenizer(GemmaTokenizer&&) noexcept;
    GemmaTokenizer& operator=(GemmaTokenizer&&) noexcept;
    GemmaTokenizer(const GemmaTokenizer&)            = delete;
    GemmaTokenizer& operator=(const GemmaTokenizer&) = delete;

    /// Text to token ids, wrapped in `<bos>` ... `<eos>`, which is what the
    /// reference produces and therefore what the pooled mean is computed over.
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;

    [[nodiscard]] std::int32_t bos_id() const noexcept;
    [[nodiscard]] std::int32_t eos_id() const noexcept;
    [[nodiscard]] std::int32_t vocabulary_size() const noexcept;

private:
    GemmaTokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer::encoder
