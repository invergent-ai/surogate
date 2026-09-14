#pragma once

#include "artifact/reader.h"
#include "encoder/gemma_tokenizer.h"
#include "family/impl/frontend/tokenizer.h"

#include <memory>
#include <string_view>
#include <vector>

namespace sinfer::encoder {

class EmbeddingTokenizer {
public:
    static EmbeddingTokenizer from_artifact(const artifact::Reader& reader);
    [[nodiscard]] std::vector<std::int32_t> encode(std::string_view text) const;
    [[nodiscard]] std::int32_t vocabulary_size() const noexcept;

private:
    std::unique_ptr<GemmaTokenizer> sentencepiece_;
    std::unique_ptr<family::frontend_internal::Tokenizer> bpe_;
    std::int32_t eos_id_ = -1;
};

} // namespace sinfer::encoder
