#include "encoder/embedding_tokenizer.h"

#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace sinfer::encoder {

EmbeddingTokenizer EmbeddingTokenizer::from_artifact(const artifact::Reader& reader) {
    EmbeddingTokenizer tokenizer;
    if (reader.find("frontend/tokenizer.model")) {
        tokenizer.sentencepiece_ = std::make_unique<GemmaTokenizer>(
            GemmaTokenizer::from_serialized_proto(reader.payload("frontend/tokenizer.model").data));
    } else {
        const auto resource = [&](const char* name) {
            const auto bytes = reader.payload(name).data;
            return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        };
        const auto json = resource("frontend/tokenizer.json");
        const auto config = resource("frontend/tokenizer_config.json");
        const auto generation = resource("frontend/generation_config.json");
        tokenizer.bpe_ = std::make_unique<family::frontend_internal::Tokenizer>(
            family::frontend_internal::TokenizerResources{json, config, generation});
        const auto parsed = nlohmann::json::parse(config);
        if (parsed.value("add_eos_token", false)) {
            tokenizer.eos_id_ = parsed.at("eos_token_id").get<std::int32_t>();
            if (!tokenizer.bpe_->is_valid_token(tokenizer.eos_id_)) {
                throw std::invalid_argument("embedding tokenizer EOS is outside its vocabulary");
            }
        }
    }
    return tokenizer;
}

std::vector<std::int32_t> EmbeddingTokenizer::encode(std::string_view text) const {
    if (sentencepiece_) { return sentencepiece_->encode(text); }
    auto ids = bpe_->encode(text);
    if (eos_id_ >= 0) { ids.push_back(eos_id_); }
    return ids;
}

std::int32_t EmbeddingTokenizer::vocabulary_size() const noexcept {
    return sentencepiece_ ? sentencepiece_->vocabulary_size()
                         : static_cast<std::int32_t>(bpe_->vocabulary_size());
}

} // namespace sinfer::encoder
