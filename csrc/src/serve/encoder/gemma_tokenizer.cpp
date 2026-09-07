#include "encoder/gemma_tokenizer.h"

#include <sentencepiece_processor.h>

#include <stdexcept>
#include <string>

namespace sinfer::encoder {

struct GemmaTokenizer::Impl {
    sentencepiece::SentencePieceProcessor processor;
};

GemmaTokenizer::GemmaTokenizer() : impl_(std::make_unique<Impl>()) {}
GemmaTokenizer::~GemmaTokenizer()                                    = default;
GemmaTokenizer::GemmaTokenizer(GemmaTokenizer&&) noexcept            = default;
GemmaTokenizer& GemmaTokenizer::operator=(GemmaTokenizer&&) noexcept = default;

GemmaTokenizer GemmaTokenizer::from_serialized_proto(std::span<const std::byte> proto) {
    GemmaTokenizer tokenizer;
    const std::string_view bytes(reinterpret_cast<const char*>(proto.data()), proto.size());
    const auto status = tokenizer.impl_->processor.LoadFromSerializedProto(bytes);
    if (!status.ok()) {
        throw std::runtime_error("GemmaTokenizer: " + status.ToString());
    }
    if (tokenizer.impl_->processor.bos_id() < 0 || tokenizer.impl_->processor.eos_id() < 0) {
        throw std::runtime_error("GemmaTokenizer: model declares no bos/eos");
    }
    return tokenizer;
}

std::vector<std::int32_t> GemmaTokenizer::encode(std::string_view text) const {
    std::vector<int> pieces;
    const auto status = impl_->processor.Encode(text, &pieces);
    if (!status.ok()) { throw std::runtime_error("GemmaTokenizer::encode: " + status.ToString()); }

    // The template the .model file does not carry: <bos> ... <eos>.
    std::vector<std::int32_t> ids;
    ids.reserve(pieces.size() + 2);
    ids.push_back(impl_->processor.bos_id());
    ids.insert(ids.end(), pieces.begin(), pieces.end());
    ids.push_back(impl_->processor.eos_id());
    return ids;
}

std::int32_t GemmaTokenizer::bos_id() const noexcept { return impl_->processor.bos_id(); }
std::int32_t GemmaTokenizer::eos_id() const noexcept { return impl_->processor.eos_id(); }
std::int32_t GemmaTokenizer::vocabulary_size() const noexcept {
    return impl_->processor.GetPieceSize();
}

} // namespace sinfer::encoder
