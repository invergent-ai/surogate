#include "family/impl/frontend/tokenizer.h"
// The project tokenizer, shared with training. Included by path because this
// file is itself a tokenizer.h and would otherwise find itself.
#include "tokenizer/tokenizer.h"

#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace sinfer::family::frontend_internal {
namespace {

using Json    = nlohmann::json;
namespace uni = sinfer::text::unicode_internal;

constexpr std::int64_t kMaxTokenId = 1'000'000;

struct VocabMetadata {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::unordered_set<int> occupied_ids;
};

Json read_json_asset(std::string_view contents, std::string_view label) {
    try {
        return Json::parse(contents);
    } catch (const nlohmann::json::exception& ex) {
        throw std::invalid_argument("malformed " + std::string(label) + ": " + ex.what());
    }
}

const Json& require_object_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_object()) {
        throw std::invalid_argument("field " + std::string(field) + " must be object in " +
                                    std::string(label));
    }
    return value;
}

const Json& require_array_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field)) {
        throw std::invalid_argument("missing field " + std::string(field) + " in " +
                                    std::string(label));
    }
    const Json& value = object.at(field);
    if (!value.is_array()) {
        throw std::invalid_argument("field " + std::string(field) + " must be array in " +
                                    std::string(label));
    }
    return value;
}

int parse_token_id(const Json& value, const char* field, std::string_view label) {
    if (!value.is_number_integer()) {
        throw std::invalid_argument("field " + std::string(field) + " must be integer in " +
                                    std::string(label));
    }
    if (value.is_number_unsigned()) {
        const std::uint64_t id = value.get<std::uint64_t>();
        if (id > static_cast<std::uint64_t>(kMaxTokenId)) {
            throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                        std::string(label));
        }
        return static_cast<int>(id);
    }

    const std::int64_t id = value.get<std::int64_t>();
    if (id < 0) {
        throw std::invalid_argument("field " + std::string(field) + " has negative id in " +
                                    std::string(label));
    }
    if (id > kMaxTokenId) {
        throw std::invalid_argument("field " + std::string(field) + " id is out of range in " +
                                    std::string(label));
    }
    return static_cast<int>(id);
}

std::string require_string_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_string()) {
        throw std::invalid_argument("field " + std::string(field) + " must be string in " +
                                    std::string(label));
    }
    return object.at(field).get<std::string>();
}

bool require_bool_field(const Json& object, const char* field, std::string_view label) {
    if (!object.is_object() || !object.contains(field) || !object.at(field).is_boolean()) {
        throw std::invalid_argument("field " + std::string(field) + " must be boolean in " +
                                    std::string(label));
    }
    return object.at(field).get<bool>();
}

VocabMetadata load_vocab(const Json& model, std::string_view label) {
    if (!model.contains("type") || !model.at("type").is_string() ||
        model.at("type").get<std::string>() != "BPE") {
        throw std::invalid_argument("field model.type must be BPE in " + std::string(label));
    }
    const Json& vocab = require_object_field(model, "vocab", label);
    if (vocab.empty()) {
        throw std::invalid_argument("field model.vocab must not be empty in " + std::string(label));
    }

    int max_id = -1;
    VocabMetadata metadata;
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        if (!metadata.occupied_ids.insert(id).second) {
            throw std::invalid_argument("field model.vocab has duplicate id in " +
                                        std::string(label));
        }
        max_id = std::max(max_id, id);
    }

    metadata.id_to_token.resize(static_cast<std::size_t>(max_id + 1));
    for (const auto& item : vocab.items()) {
        const int id = parse_token_id(item.value(), "model.vocab", label);
        metadata.id_to_token.at(static_cast<std::size_t>(id)) = item.key();
        metadata.token_to_id.emplace(item.key(), id);
    }
    return metadata;
}

AddedToken parse_added_token(const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    if (!item.contains("id")) {
        throw std::invalid_argument("missing field added_tokens.id in " + std::string(label));
    }
    token.id          = parse_token_id(item.at("id"), "added_tokens.id", label);
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

AddedToken parse_added_token_decoder_entry(int id, const Json& item, std::string_view label) {
    if (!item.is_object()) {
        throw std::invalid_argument("field added_tokens_decoder item must be object in " +
                                    std::string(label));
    }
    AddedToken token;
    token.id          = id;
    token.content     = require_string_field(item, "content", label);
    token.single_word = require_bool_field(item, "single_word", label);
    token.lstrip      = require_bool_field(item, "lstrip", label);
    token.rstrip      = require_bool_field(item, "rstrip", label);
    token.normalized  = require_bool_field(item, "normalized", label);
    token.special     = require_bool_field(item, "special", label);
    return token;
}

void validate_supported_added_token(const AddedToken& token, bool normalizer_is_identity,
                                    std::string_view label) {
    if (token.content.empty()) {
        throw std::invalid_argument("added token content must not be empty in " +
                                    std::string(label));
    }
    // `normalized` says the token is normalised before it is matched. Where the tokenizer
    // declares no normalizer at all, normalising is the identity and the flag distinguishes
    // nothing -- so it is accepted there and refused everywhere else. LFM2 marks two ordinary
    // words this way (`Mathias`, `python`, both already in the base vocabulary at the same ids)
    // and declares a null normalizer, so refusing it turned a flag that changes no byte into a
    // checkpoint this engine would not load.
    if (token.single_word || token.lstrip || token.rstrip ||
        (token.normalized && !normalizer_is_identity)) {
        throw std::invalid_argument("Tokenizer only supports added tokens with single_word=false, "
                                    "lstrip=false, rstrip=false, and normalized=false (or a "
                                    "tokenizer that declares no normalizer) in " +
                                    std::string(label));
    }
}

bool same_added_token(const AddedToken& lhs, const AddedToken& rhs) noexcept {
    return lhs.id == rhs.id && lhs.content == rhs.content && lhs.single_word == rhs.single_word &&
           lhs.lstrip == rhs.lstrip && lhs.rstrip == rhs.rstrip &&
           lhs.normalized == rhs.normalized && lhs.special == rhs.special;
}

int parse_added_token_decoder_id(std::string_view key, std::string_view label) {
    std::int64_t parsed     = -1;
    const auto [end, error] = std::from_chars(key.data(), key.data() + key.size(), parsed);
    if (error != std::errc{} || end != key.data() + key.size() || parsed < 0 ||
        parsed > kMaxTokenId || std::to_string(parsed) != key) {
        throw std::invalid_argument("added_tokens_decoder key must be a nonnegative token id in " +
                                    std::string(label));
    }
    return static_cast<int>(parsed);
}

std::vector<AddedToken>
load_added_tokens(const Json& root, bool normalizer_is_identity, std::string_view label,
                  std::vector<std::string>& id_to_token,
                  const std::unordered_set<int>& occupied_vocab_ids,
                  const std::unordered_map<std::string, int>& occupied_vocab_tokens) {
    const Json& added = require_array_field(root, "added_tokens", label);
    std::vector<AddedToken> tokens;
    tokens.reserve(added.size());
    std::unordered_set<int> seen_added_ids;
    std::unordered_map<std::string, int> seen_added_contents;
    for (const Json& item : added) {
        AddedToken token = parse_added_token(item, label);
        validate_supported_added_token(token, normalizer_is_identity, label);
        const auto index = static_cast<std::size_t>(token.id);
        // A SentencePiece conversion states its specials twice: once in
        // model.vocab, where they hold real ids, and again in added_tokens so the
        // splitter keeps them whole. That is a restatement, not a collision --
        // the id and the content agree on both sides. A byte-level conversion
        // keeps its added tokens out of the vocabulary entirely, never reaches
        // this branch, and keeps the strictness it had.
        const auto vocab_entry    = occupied_vocab_tokens.find(token.content);
        const bool restates_vocab = vocab_entry != occupied_vocab_tokens.end() &&
                                    vocab_entry->second == token.id;
        if (occupied_vocab_ids.contains(token.id) && !restates_vocab) {
            throw std::invalid_argument("field added_tokens overlaps existing id in " +
                                        std::string(label));
        }
        if (!seen_added_ids.insert(token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate id in " +
                                        std::string(label));
        }
        if ((occupied_vocab_tokens.contains(token.content) && !restates_vocab) ||
            !seen_added_contents.emplace(token.content, token.id).second) {
            throw std::invalid_argument("field added_tokens has duplicate content mapping in " +
                                        std::string(label));
        }
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        id_to_token.at(static_cast<std::size_t>(token.id)) = token.content;
        tokens.push_back(std::move(token));
    }
    return tokens;
}

void merge_added_tokens_decoder(const Json& root, bool normalizer_is_identity,
                                std::string_view label,
                                std::vector<std::string>& id_to_token,
                                const std::unordered_set<int>& occupied_vocab_ids,
                                const std::unordered_map<std::string, int>& occupied_vocab_tokens,
                                std::vector<AddedToken>& tokens) {
    // tokenizer.json's added_tokens is the base and, in the transformers-5 config shape
    // (`TokenizersBackend`), the only place the special tokens are stated: that config has
    // no added_tokens_decoder at all. Nothing to merge, nothing to cross-check.
    if (!root.contains("added_tokens_decoder")) {
        // load_added_tokens returns file order; sorted-by-id is this function's postcondition
        // and the scan order added_token_candidates_ is built in.
        std::sort(tokens.begin(), tokens.end(),
                  [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
        return;
    }
    const Json& decoder = require_object_field(root, "added_tokens_decoder", label);
    std::unordered_map<int, std::size_t> token_by_id;
    std::unordered_map<std::string, int> token_by_content;
    std::unordered_set<int> decoder_ids;
    token_by_id.reserve(tokens.size() + decoder.size());
    token_by_content.reserve(tokens.size() + decoder.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        token_by_id.emplace(tokens[index].id, index);
        token_by_content.emplace(tokens[index].content, tokens[index].id);
    }

    for (const auto& item : decoder.items()) {
        const int id = parse_added_token_decoder_id(item.key(), label);
        if (!decoder_ids.insert(id).second) {
            throw std::invalid_argument("added_tokens_decoder has duplicate id mapping in " +
                                        std::string(label));
        }
        AddedToken token = parse_added_token_decoder_entry(id, item.value(), label);
        validate_supported_added_token(token, normalizer_is_identity, label);

        const auto existing_id = token_by_id.find(id);
        if (existing_id != token_by_id.end()) {
            if (!same_added_token(tokens.at(existing_id->second), token)) {
                throw std::invalid_argument("conflicting added-token definition for id " +
                                            std::to_string(id) +
                                            " between tokenizer.json and tokenizer_config.json");
            }
            continue;
        }
        if (occupied_vocab_ids.contains(id)) {
            throw std::invalid_argument("added_tokens_decoder overlaps vocabulary id " +
                                        std::to_string(id));
        }
        if (token_by_content.contains(token.content) ||
            occupied_vocab_tokens.contains(token.content)) {
            throw std::invalid_argument("conflicting added-token content mapping for " +
                                        token.content);
        }

        const auto index = static_cast<std::size_t>(id);
        if (index >= id_to_token.size()) { id_to_token.resize(index + 1); }
        if (!id_to_token[index].empty()) {
            throw std::invalid_argument("duplicate tokenizer mapping for id " + std::to_string(id));
        }
        id_to_token[index] = token.content;
        token_by_id.emplace(id, tokens.size());
        token_by_content.emplace(token.content, id);
        tokens.push_back(std::move(token));
    }
    std::sort(tokens.begin(), tokens.end(),
              [](const AddedToken& lhs, const AddedToken& rhs) { return lhs.id < rhs.id; });
}

std::vector<int> load_default_stop_token_ids(std::string_view contents) {
    constexpr std::string_view label = "generation_config.json";
    const Json root                  = read_json_asset(contents, label);
    if (!root.is_object() || !root.contains("eos_token_id")) {
        throw std::invalid_argument("missing field eos_token_id in generation_config.json");
    }

    const Json& eos = root.at("eos_token_id");
    if (eos.is_number_integer()) { return {parse_token_id(eos, "eos_token_id", label)}; }
    if (eos.is_array()) {
        if (eos.empty()) {
            throw std::invalid_argument(
                "field eos_token_id must not be empty in generation_config.json");
        }
        std::vector<int> ids;
        ids.reserve(eos.size());
        for (const Json& item : eos) { ids.push_back(parse_token_id(item, "eos_token_id", label)); }
        return ids;
    }
    throw std::invalid_argument(
        "field eos_token_id must be integer or array in generation_config.json");
}

std::string merge_pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + 1 + right.size());
    key.append(left);
    key.push_back('\0');
    key.append(right);
    return key;
}

std::unordered_map<std::string, int> load_bpe_merge_ranks(const Json& model,
                                                          std::string_view label) {
    const Json& merges = require_array_field(model, "merges", label);
    std::unordered_map<std::string, int> ranks;
    ranks.reserve(merges.size());
    int rank = 0;
    for (const Json& item : merges) {
        std::string left;
        std::string right;
        if (item.is_array() && item.size() == 2 && item[0].is_string() && item[1].is_string()) {
            left  = item[0].get<std::string>();
            right = item[1].get<std::string>();
        } else if (item.is_string()) {
            const std::string pair  = item.get<std::string>();
            const std::size_t space = pair.find(' ');
            if (space == std::string::npos || space == 0 || space + 1 >= pair.size() ||
                pair.find(' ', space + 1) != std::string::npos) {
                throw std::invalid_argument("malformed model.merges entry in " +
                                            std::string(label));
            }
            left  = pair.substr(0, space);
            right = pair.substr(space + 1);
        } else {
            throw std::invalid_argument("field model.merges must contain symbol pairs in " +
                                        std::string(label));
        }
        const auto [_, inserted] = ranks.emplace(merge_pair_key(left, right), rank++);
        if (!inserted) { throw std::invalid_argument("duplicate merge pair in model.merges"); }
    }
    return ranks;
}

std::unordered_map<std::uint32_t, char> build_byte_level_decoder() {
    std::unordered_map<std::uint32_t, char> decoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        decoder.emplace(codepoint, static_cast<char>(static_cast<unsigned char>(byte)));
    }
    return decoder;
}

std::unordered_map<unsigned char, std::string> build_byte_level_encoder() {
    std::unordered_map<unsigned char, std::string> encoder;
    std::uint32_t next = 256;
    for (int byte = 0; byte <= std::numeric_limits<unsigned char>::max(); ++byte) {
        const bool visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) ||
                             (byte >= 174 && byte <= 255);
        const std::uint32_t codepoint = visible ? static_cast<std::uint32_t>(byte) : next++;
        encoder.emplace(static_cast<unsigned char>(byte),
                        uni::codepoint_to_utf8(static_cast<std::int32_t>(codepoint)));
    }
    return encoder;
}

bool is_newline(std::int32_t codepoint) noexcept { return codepoint == '\r' || codepoint == '\n'; }

bool is_letter_or_mark(std::int32_t codepoint) noexcept {
    return uni::is_letter(codepoint) || uni::is_mark(codepoint);
}

bool is_non_newline_non_letter_non_number(std::int32_t codepoint) noexcept {
    return !is_newline(codepoint) && !uni::is_letter(codepoint) && !uni::is_number(codepoint);
}

bool is_non_space_non_letter_mark_number(std::int32_t codepoint) noexcept {
    return !uni::is_whitespace(codepoint) && !uni::is_letter(codepoint) &&
           !uni::is_mark(codepoint) && !uni::is_number(codepoint);
}

bool ascii_ci_matches(std::string_view text, std::size_t offset, std::string_view suffix) {
    if (offset + suffix.size() > text.size()) { return false; }
    for (std::size_t i = 0; i < suffix.size(); ++i) {
        const unsigned char lhs = static_cast<unsigned char>(text[offset + i]);
        const unsigned char rhs = static_cast<unsigned char>(suffix[i]);
        if (std::tolower(lhs) != std::tolower(rhs)) { return false; }
    }
    return true;
}

std::size_t span_end_offset(std::string_view text, const std::vector<uni::CodepointSpan>& spans,
                            std::size_t end) {
    if (end == spans.size()) { return text.size(); }
    return spans.at(end).offset;
}

/// The pre-tokenizer's word split. `max_digit_run` is the only thing that differs between the
/// checkpoints this family serves: one digit per word for the Qwen-family rule (`\p{N}`), up to
/// three for GLM-5.3's (`\p{N}{1,3}`).
std::vector<std::string_view> split_words(std::string_view text, std::size_t max_digit_run) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "Tokenizer::encode input");
    std::vector<std::string_view> words;
    for (std::size_t i = 0; i < spans.size();) {
        const std::size_t begin_offset = spans[i].offset;
        const std::int32_t cp          = spans[i].value;

        if (cp == '\'') {
            constexpr std::string_view suffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (std::string_view suffix : suffixes) {
                if (ascii_ci_matches(text, begin_offset + 1, suffix)) {
                    std::size_t end = i + 1;
                    while (end < spans.size() &&
                           spans[end].offset < begin_offset + 1 + suffix.size()) {
                        ++end;
                    }
                    words.emplace_back(text.substr(begin_offset, span_end_offset(text, spans, end) -
                                                                     begin_offset));
                    i = end;
                    goto next_word;
                }
            }
        }

        if (is_letter_or_mark(cp) ||
            (is_non_newline_non_letter_non_number(cp) && i + 1 < spans.size() &&
             is_letter_or_mark(spans[i + 1].value))) {
            std::size_t end = i;
            if (!is_letter_or_mark(spans[end].value)) { ++end; }
            while (end < spans.size() && is_letter_or_mark(spans[end].value)) { ++end; }
            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            goto next_word;
        }

        if (uni::is_number(cp)) {
            std::size_t end = i + 1;
            while (end < spans.size() && end - i < max_digit_run &&
                   uni::is_number(spans[end].value)) {
                ++end;
            }
            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            goto next_word;
        }

        if ((cp == ' ' && i + 1 < spans.size() &&
             is_non_space_non_letter_mark_number(spans[i + 1].value)) ||
            is_non_space_non_letter_mark_number(cp)) {
            std::size_t end = i;
            if (spans[end].value == ' ') { ++end; }
            while (end < spans.size() && is_non_space_non_letter_mark_number(spans[end].value)) {
                ++end;
            }
            while (end < spans.size() && is_newline(spans[end].value)) { ++end; }
            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
            i = end;
            goto next_word;
        }

        if (uni::is_whitespace(cp)) {
            std::size_t run_end      = i;
            std::size_t last_newline = std::string_view::npos;
            while (run_end < spans.size() && uni::is_whitespace(spans[run_end].value)) {
                if (is_newline(spans[run_end].value)) { last_newline = run_end; }
                ++run_end;
            }
            if (last_newline != std::string_view::npos) {
                const std::size_t end = last_newline + 1;
                words.emplace_back(
                    text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                goto next_word;
            }

            if (run_end == spans.size()) {
                words.emplace_back(text.substr(begin_offset));
                i = run_end;
                goto next_word;
            }

            if (run_end - i >= 2) {
                const std::size_t end = run_end - 1;
                words.emplace_back(
                    text.substr(begin_offset, span_end_offset(text, spans, end) - begin_offset));
                i = end;
                goto next_word;
            }

            words.emplace_back(
                text.substr(begin_offset, span_end_offset(text, spans, run_end) - begin_offset));
            i = run_end;
            goto next_word;
        }

        words.emplace_back(text.substr(begin_offset, spans[i].length));
        ++i;

    next_word:;
    }
    return words;
}

std::string byte_level_encode(std::string_view text) {
    static const std::unordered_map<unsigned char, std::string> byte_encoder =
        build_byte_level_encoder();
    std::string encoded;
    encoded.reserve(text.size());
    for (const unsigned char byte : text) { encoded += byte_encoder.at(byte); }
    return encoded;
}

std::vector<std::string> byte_level_symbols(std::string_view text) {
    const std::vector<uni::CodepointSpan> spans =
        uni::utf8_codepoints(text, "Tokenizer::encode byte-level text");
    std::vector<std::string> symbols;
    symbols.reserve(spans.size());
    for (const uni::CodepointSpan& span : spans) {
        symbols.emplace_back(text.substr(span.offset, span.length));
    }
    return symbols;
}

bool is_added_token_id(const std::vector<AddedToken>& added_tokens, int id) {
    return std::any_of(added_tokens.begin(), added_tokens.end(),
                       [id](const AddedToken& token) { return token.id == id; });
}

bool is_stop_token_id(std::span<const int> stop_token_ids, int id) {
    return std::find(stop_token_ids.begin(), stop_token_ids.end(), id) != stop_token_ids.end();
}

void append_symbol_id(std::vector<int>& ids,
                      const std::unordered_map<std::string, int>& token_to_id,
                      std::string_view symbol) {
    const auto direct = token_to_id.find(std::string(symbol));
    if (direct != token_to_id.end()) {
        ids.push_back(direct->second);
        return;
    }

    const std::vector<std::string> bytes = byte_level_symbols(symbol);
    if (bytes.size() <= 1) {
        throw std::invalid_argument("Tokenizer::encode produced token outside vocabulary: " +
                                    std::string(symbol));
    }
    for (const std::string& byte_symbol : bytes) {
        const auto byte_id = token_to_id.find(byte_symbol);
        if (byte_id == token_to_id.end()) {
            throw std::invalid_argument(
                "Tokenizer::encode produced byte symbol outside vocabulary: " + byte_symbol);
        }
        ids.push_back(byte_id->second);
    }
}

void append_bpe_ids(std::vector<int>& ids, std::string_view text, bool has_bpe_merges,
                    std::size_t max_digit_run,
                    const std::unordered_map<std::string, int>& merge_ranks,
                    const std::unordered_map<std::string, int>& token_to_id) {
    if (text.empty()) { return; }
    if (!has_bpe_merges) {
        throw std::invalid_argument(
            "Tokenizer::encode ordinary BPE text requires embedded merges.txt");
    }

    const std::string normalized = uni::normalize_nfc(text);
    for (const std::string_view word : split_words(normalized, max_digit_run)) {
        std::vector<std::string> symbols = byte_level_symbols(byte_level_encode(word));
        while (symbols.size() > 1) {
            int best_rank              = std::numeric_limits<int>::max();
            std::size_t best_pair_left = symbols.size();
            for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
                const auto rank = merge_ranks.find(merge_pair_key(symbols[i], symbols[i + 1]));
                if (rank != merge_ranks.end() && rank->second < best_rank) {
                    best_rank      = rank->second;
                    best_pair_left = i;
                }
            }
            if (best_pair_left == symbols.size()) { break; }
            symbols[best_pair_left] += symbols[best_pair_left + 1];
            symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_pair_left + 1));
        }
        for (const std::string& symbol : symbols) { append_symbol_id(ids, token_to_id, symbol); }
    }
}

} // namespace

namespace project_delegate {
struct Handle {
    ::tokenizer::Tokenizer inner;
};
void destroy(Handle* handle) { delete handle; }
} // namespace project_delegate

namespace {

/// True when the tokenizer.json describes the SentencePiece scheme rather than
/// the byte-level one: a normalizer that substitutes a word mark for a space, or
/// a model that declares byte fallback. Read off the file, never off the target
/// -- model_type "llama" is byte-level for Llama 3 and SentencePiece for Llama 2.
bool describes_sentencepiece(const Json& root) {
    const auto normalizer_marks_words = [](const Json& node, auto&& self) -> bool {
        if (!node.is_object() || !node.contains("type") || !node.at("type").is_string()) {
            return false;
        }
        const std::string type = node.at("type").get<std::string>();
        if (type == "Prepend" || type == "Replace") { return true; }
        if (type == "Sequence" && node.contains("normalizers") &&
            node.at("normalizers").is_array()) {
            for (const Json& item : node.at("normalizers")) {
                if (self(item, self)) { return true; }
            }
        }
        return false;
    };
    if (root.contains("normalizer") && !root.at("normalizer").is_null() &&
        normalizer_marks_words(root.at("normalizer"), normalizer_marks_words)) {
        return true;
    }
    return root.contains("model") && root.at("model").is_object() &&
           root.at("model").contains("byte_fallback") &&
           root.at("model").at("byte_fallback").is_boolean() &&
           root.at("model").at("byte_fallback").get<bool>();
}

} // namespace

Tokenizer::~Tokenizer()                            = default;
namespace {

/// How many digits the checkpoint's pre-tokenizer keeps in one word.
///
/// Read from the `Split` pattern `tokenizer.json` declares rather than compiled in. Every
/// checkpoint this family served until GLM-5.3 writes `\p{N}` and gets one digit a word; GLM-5.3
/// writes `\p{N}{1,3}` and gets up to three, which is the difference between "3,344" tokenising
/// as `3` `,` `34` `4` and as five separate digits -- and, for a model asked how many sheep are
/// left of seventeen, between reading "17" and reading "1" "7".
///
/// A pattern naming neither is left at one, which is what every such checkpoint has always had.
std::size_t declared_digit_run(const Json& root, std::string_view label) {
    (void)label;
    std::size_t run = 1;
    const auto scan = [&run](const Json& node) {
        if (!node.is_object() || !node.contains("pattern")) { return; }
        const Json& pattern = node.at("pattern");
        if (!pattern.is_object() || !pattern.contains("Regex")) { return; }
        const Json& regex = pattern.at("Regex");
        if (!regex.is_string()) { return; }
        const std::string text = regex.get<std::string>();
        if (text.find("\\p{N}{1,3}") != std::string::npos) { run = 3; }
    };
    if (!root.contains("pre_tokenizer") || root.at("pre_tokenizer").is_null()) { return run; }
    const Json& pre = root.at("pre_tokenizer");
    if (pre.is_object() && pre.contains("pretokenizers") && pre.at("pretokenizers").is_array()) {
        for (const Json& sub : pre.at("pretokenizers")) { scan(sub); }
    } else {
        scan(pre);
    }
    return run;
}

} // namespace

Tokenizer::Tokenizer(Tokenizer&&) noexcept         = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer::Tokenizer(TokenizerResources resources) {
    if (resources.tokenizer_json.empty() || resources.tokenizer_config_json.empty() ||
        resources.generation_config_json.empty()) {
        throw std::invalid_argument("embedded tokenizer resources are empty");
    }
    constexpr std::string_view tokenizer_label        = "tokenizer.json";
    constexpr std::string_view tokenizer_config_label = "tokenizer_config.json";
    const Json root = read_json_asset(resources.tokenizer_json, tokenizer_label);
    const Json tokenizer_config =
        read_json_asset(resources.tokenizer_config_json, tokenizer_config_label);
    max_digit_run_    = declared_digit_run(root, tokenizer_label);
    const Json& model = require_object_field(root, "model", tokenizer_label);

    VocabMetadata vocab_metadata = load_vocab(model, tokenizer_label);
    id_to_token_                 = std::move(vocab_metadata.id_to_token);
    vocab_token_to_id_           = std::move(vocab_metadata.token_to_id);
    valid_token_ids_.resize(id_to_token_.size());
    for (const int id : vocab_metadata.occupied_ids) {
        valid_token_ids_.at(static_cast<std::size_t>(id)) = true;
    }
    // Whether normalising is the identity, which is what decides if an added token's
    // `normalized` flag can change anything. Both files' added tokens are the same tokens, so
    // both are judged against the tokenizer.json that declares the normalizer.
    const bool normalizer_is_identity =
        !root.contains("normalizer") || root.at("normalizer").is_null();
    added_tokens_ = load_added_tokens(root, normalizer_is_identity, tokenizer_label, id_to_token_,
                                      vocab_metadata.occupied_ids, vocab_token_to_id_);
    merge_added_tokens_decoder(tokenizer_config, normalizer_is_identity, tokenizer_config_label,
                               id_to_token_, vocab_metadata.occupied_ids, vocab_token_to_id_,
                               added_tokens_);
    for (std::size_t index = 0; index < added_tokens_.size(); ++index) {
        const std::string& content = added_tokens_[index].content;
        if (!content.empty()) {
            added_token_candidates_[static_cast<unsigned char>(content.front())].push_back(index);
        }
    }
    if (valid_token_ids_.size() < id_to_token_.size()) {
        valid_token_ids_.resize(id_to_token_.size());
    }
    for (const AddedToken& token : added_tokens_) {
        valid_token_ids_.at(static_cast<std::size_t>(token.id)) = true;
    }
    bpe_merge_ranks_        = load_bpe_merge_ranks(model, tokenizer_label);
    has_bpe_merges_         = true;
    default_stop_token_ids_ = load_default_stop_token_ids(resources.generation_config_json);

    // The project tokenizer is built for two reasons that are independent of one
    // another. A SentencePiece checkpoint is *encoded* by it, because it implements
    // that scheme; any checkpoint whose template this family does not reproduce by
    // hand is *rendered* by it, because it owns the Jinja renderer. Everything above
    // still applies either way: the vocabulary, the added tokens and the stop ids are
    // the artifact's, and only the text-to-ids mapping differs.
    delegate_encodes_ = describes_sentencepiece(root);
    if (delegate_encodes_ || resources.render_chat_template) {
        ::tokenizer::Tokenizer::Sources sources;
        sources.tokenizer_json        = std::string(resources.tokenizer_json);
        sources.tokenizer_config_json = std::string(resources.tokenizer_config_json);
        sources.chat_template_jinja   = std::string(resources.chat_template_jinja);
        delegate_.reset(new project_delegate::Handle{::tokenizer::Tokenizer::from_sources(sources)});
    }
}

bool Tokenizer::renders_chat_template() const noexcept { return delegate_ != nullptr; }

std::string
Tokenizer::render_chat_template(const std::vector<std::pair<std::string, std::string>>& messages,
                                bool add_generation_prompt,
                                const ChatTemplateVariables& variables) const {
    if (!delegate_) {
        throw std::logic_error("Tokenizer::render_chat_template: no template renderer for this "
                               "checkpoint");
    }
    std::vector<::tokenizer::ChatMessage> converted;
    converted.reserve(messages.size());
    for (const auto& [role, content] : messages) { converted.push_back({role, content}); }
    return delegate_->inner.apply_chat_template(
        converted, add_generation_prompt,
        ::tokenizer::ChatTemplateVariables{.enable_thinking  = variables.enable_thinking,
                                           .reasoning_effort = variables.reasoning_effort});
}

std::vector<int> Tokenizer::encode(std::string_view text, EncodeOptions options) const {
    if (text.empty()) { return {}; }
    if (delegate_encodes_) {
        // The project tokenizer handles added tokens itself, so the option maps
        // onto which of its two entry points to call.
        const std::string owned(text);
        std::vector<int> ids = options.parse_added_tokens
                                   ? delegate_->inner.encode_with_special_tokens(owned)
                                   : delegate_->inner.encode_ordinary(owned);
        // Llama opens every sequence with its BOS. It comes from the
        // post-processor rather than the chat template, so nothing upstream has
        // added it, and a prompt without one is a prompt the model never saw.
        if (options.parse_added_tokens && delegate_->inner.adds_bos()) {
            ids.insert(ids.begin(), delegate_->inner.bos_token_id());
        }
        return ids;
    }
    if (!options.parse_added_tokens) {
        std::vector<int> ids;
        append_bpe_ids(ids, text, has_bpe_merges_, max_digit_run_, bpe_merge_ranks_,
                       vocab_token_to_id_);
        return ids;
    }

    std::vector<int> ids;
    std::size_t ordinary_begin = 0;
    std::size_t pos            = 0;
    while (pos < text.size()) {
        const AddedToken* match_token = nullptr;
        const auto& candidates = added_token_candidates_[static_cast<unsigned char>(text[pos])];
        for (const std::size_t index : candidates) {
            const AddedToken& token = added_tokens_[index];
            if (token.content.size() <= text.size() - pos &&
                text.compare(pos, token.content.size(), token.content) == 0) {
                match_token = &token;
                break;
            }
        }

        if (match_token == nullptr) {
            ++pos;
            continue;
        }

        if (pos > ordinary_begin) {
            append_bpe_ids(ids, text.substr(ordinary_begin, pos - ordinary_begin), has_bpe_merges_,
                           max_digit_run_,
                           bpe_merge_ranks_, vocab_token_to_id_);
        }

        ids.push_back(match_token->id);
        pos += match_token->content.size();
        ordinary_begin = pos;
    }
    if (ordinary_begin < text.size()) {
        append_bpe_ids(ids, text.substr(ordinary_begin), has_bpe_merges_, max_digit_run_,
                       bpe_merge_ranks_,
                       vocab_token_to_id_);
    }
    return ids;
}

std::string Tokenizer::decode(std::span<const int> ids, DecodeOptions options) const {
    std::string text;
    const std::size_t terminal_stop_index =
        (!ids.empty() && is_stop_token_id(options.stop_token_ids, ids.back())) ? ids.size() - 1
                                                                               : ids.size();

    if (delegate_encodes_) {
        // Decoding is not per-token here: the word mark becomes a space only in
        // the finished string, so the ids are handed over whole.
        std::vector<std::int32_t> kept;
        kept.reserve(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) {
            if (i == terminal_stop_index) { continue; }
            if (options.skip_special_tokens && is_special_token(ids[i])) { continue; }
            kept.push_back(ids[i]);
        }
        text = delegate_->inner.decode(kept);
        (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
        return text;
    }

    for (std::size_t i = 0; i < ids.size(); ++i) {
        const int id = ids[i];
        if (i == terminal_stop_index) { continue; }
        text += decode_token_bytes(id, options.skip_special_tokens);
    }
    (void)uni::utf8_codepoints(text, "Tokenizer::decode reconstructed output");
    return text;
}

std::string Tokenizer::decode_token_bytes(int id, bool skip_special_tokens) const {
    if (delegate_encodes_) {
        if (skip_special_tokens && is_special_token(id)) { return {}; }
        // One token, not a sequence: the leading space is this token's own.
        return delegate_->inner.decode(std::vector<std::int32_t>{id}, false);
    }
    static const std::unordered_map<std::uint32_t, char> byte_decoder = build_byte_level_decoder();

    if (skip_special_tokens && is_special_token(id)) { return {}; }
    if (id < 0) {
        throw std::invalid_argument("Tokenizer::decode received negative token id " +
                                    std::to_string(id));
    }
    const auto index = static_cast<std::size_t>(id);
    if (index >= id_to_token_.size() || index >= valid_token_ids_.size() ||
        !valid_token_ids_.at(index)) {
        throw std::out_of_range("Tokenizer::decode token id " + std::to_string(id) +
                                " is outside loaded vocabulary");
    }

    const std::string& token = id_to_token_.at(index);
    if (is_added_token_id(added_tokens_, id)) { return token; }

    std::string bytes;
    const std::vector<uni::CodepointSpan> codepoints =
        uni::utf8_codepoints(token, "Tokenizer::decode token id " + std::to_string(id));
    for (const uni::CodepointSpan& codepoint : codepoints) {
        const auto byte = byte_decoder.find(static_cast<std::uint32_t>(codepoint.value));
        if (byte == byte_decoder.end()) {
            throw std::invalid_argument("Tokenizer::decode token id " + std::to_string(id) +
                                        " contains a character outside the byte-level alphabet");
        }
        bytes.push_back(byte->second);
    }
    return bytes;
}

bool Tokenizer::is_special_token(int id) const noexcept {
    return std::any_of(added_tokens_.begin(), added_tokens_.end(),
                       [id](const AddedToken& token) { return token.id == id && token.special; });
}

bool Tokenizer::is_valid_token(int id) const noexcept {
    return id >= 0 && static_cast<std::size_t>(id) < valid_token_ids_.size() &&
           valid_token_ids_[static_cast<std::size_t>(id)];
}

bool Tokenizer::has_exact_token_domain(std::size_t size) const noexcept {
    return valid_token_ids_.size() == size &&
           std::find(valid_token_ids_.begin(), valid_token_ids_.end(), false) ==
               valid_token_ids_.end();
}

} // namespace sinfer::family::frontend_internal
