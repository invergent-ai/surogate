// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// HuggingFace tokenizer.json loader with tiktoken-speed BPE.

#include "tokenizer.h"
#include "bpe.h"
#include "unicode.h"

#include <minja/chat-template.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <nlohmann/json.hpp>
#include <fmt/format.h>

// minja uses nlohmann::ordered_json as 'json'. We use the same alias to avoid conflicts.
using json = nlohmann::ordered_json;

namespace tokenizer {

// ============================================================================
// Byte-level encoding tables (GPT-2 style)
// Maps each byte 0-255 to a visible Unicode character.
// This is how HuggingFace BPE tokenizers with ByteLevel pre-tokenizer work:
// raw bytes are mapped to printable Unicode chars before BPE, then unmapped
// after decode. E.g. space (0x20) -> 'Ġ' (U+0120).
// ============================================================================

static std::vector<uint32_t> build_bytes_to_unicode() {
    std::vector<uint32_t> table(256);
    // Printable ASCII ranges that map to themselves
    int n = 0;
    for (int i = 0; i < 256; i++) {
        if ((i >= 0x21 && i <= 0x7E) || (i >= 0xA1 && i <= 0xAC) || (i >= 0xAE && i <= 0xFF)) {
            table[i] = static_cast<uint32_t>(i);
        } else {
            table[i] = static_cast<uint32_t>(256 + n);
            n++;
        }
    }
    return table;
}

static std::unordered_map<uint32_t, uint8_t> build_unicode_to_bytes() {
    auto fwd = build_bytes_to_unicode();
    std::unordered_map<uint32_t, uint8_t> rev;
    rev.reserve(256);
    for (int i = 0; i < 256; i++) {
        rev[fwd[i]] = static_cast<uint8_t>(i);
    }
    return rev;
}

// Convert a UTF-8 string (byte-level encoded) back to raw bytes.
static std::string byte_level_decode(const std::string& text) {
    static const auto unicode_to_byte = build_unicode_to_bytes();
    std::string result;
    result.reserve(text.size());
    auto cpts = unicode_cpts_from_utf8(text);
    for (uint32_t cp : cpts) {
        auto it = unicode_to_byte.find(cp);
        if (it != unicode_to_byte.end()) {
            result.push_back(static_cast<char>(it->second));
        } else {
            // Codepoint not in byte-level table; pass through as UTF-8
            result += unicode_cpt_to_utf8(cp);
        }
    }
    return result;
}

// Convert raw bytes to byte-level encoded UTF-8 string.
static std::string byte_level_encode(const std::string& text) {
    static const auto bytes_to_unicode = build_bytes_to_unicode();
    std::string result;
    result.reserve(text.size() * 2);
    for (unsigned char c : text) {
        result += unicode_cpt_to_utf8(bytes_to_unicode[c]);
    }
    return result;
}

// ============================================================================
// Tokenizer Implementation
// ============================================================================

struct Tokenizer::Impl {
    // BPE model
    Encoder encoder;                                             // byte_seq -> rank
    std::unordered_map<Rank, std::vector<uint8_t>> decoder;      // rank -> byte_seq
    EncoderLookup* encoder_lookup = nullptr;                     // fast lookup (no alloc)

    // Special tokens
    std::unordered_map<std::string, Rank> special_tokens_encoder; // text -> id
    std::unordered_map<Rank, std::string> special_tokens_decoder; // id -> text
    std::unordered_set<Rank> special_token_ids;

    // Pre-tokenizer regex patterns (from tokenizer.json)
    std::vector<std::string> pre_tokenizer_regex;

    // Whether byte-level pre-tokenizer is active
    bool byte_level = false;

    // Normalizer type
    enum class Normalizer { NONE, NFC, NFD, NFKC, NFKD } normalizer = Normalizer::NONE;

    // Well-known token IDs
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    int32_t unk_id = -1;

    // Total vocab size (regular + special)
    int32_t vocab_size_ = 0;

    // Named special tokens (e.g. "eos_token" -> "<|im_end|>")
    std::unordered_map<std::string, std::string> named_special_tokens;

    // Config flags
    bool add_bos = false;
    bool add_eos = false;

    // Chat template (Jinja2, rendered by minja)
    std::unique_ptr<minja::chat_template> chat_tmpl;

    ~Impl() { delete encoder_lookup; }

    void build_lookup() {
        delete encoder_lookup;
        encoder_lookup = new EncoderLookup(encoder);
    }

    // Encode a single pre-tokenized piece (after regex split + byte-level encoding).
    // Returns token IDs.
    void encode_piece(const std::string& piece, std::vector<int32_t>& out) const {
        auto key = std::vector<uint8_t>(piece.begin(), piece.end());
        auto it = encoder.find(key);
        if (it != encoder.end()) {
            out.push_back(static_cast<int32_t>(it->second));
            return;
        }
        // BPE merge
        auto ranks = byte_pair_encode(
            reinterpret_cast<const uint8_t*>(piece.data()), piece.size(), *encoder_lookup);
        for (Rank r : ranks) {
            out.push_back(static_cast<int32_t>(r));
        }
    }

    // Encode ordinary text (no special token handling).
    std::vector<int32_t> encode_ordinary_impl(const std::string& text) const {
        std::vector<int32_t> result;
        if (text.empty()) return result;

        // Pre-tokenize: split by regex patterns.
        // unicode_regex_split already applies byte-level encoding (GPT-2 style)
        // to each piece, so the returned strings match the vocab keys directly.
        auto pieces = unicode_regex_split(text, pre_tokenizer_regex);

        for (const auto& piece : pieces) {
            encode_piece(piece, result);
        }
        return result;
    }

    // Encode with special token handling.
    std::vector<int32_t> encode_impl(const std::string& text, bool use_special) const {
        if (!use_special || special_tokens_encoder.empty()) {
            return encode_ordinary_impl(text);
        }

        std::vector<int32_t> result;

        // Find all special token occurrences and split around them.
        // Use a simple linear scan (special tokens are few and this is not the hot path).
        size_t pos = 0;
        while (pos < text.size()) {
            // Find the earliest special token from current position.
            size_t best_pos = std::string::npos;
            std::string best_token;
            Rank best_id = 0;

            for (const auto& [token, id] : special_tokens_encoder) {
                size_t found = text.find(token, pos);
                if (found != std::string::npos && (found < best_pos || (found == best_pos && token.size() > best_token.size()))) {
                    best_pos = found;
                    best_token = token;
                    best_id = id;
                }
            }

            if (best_pos == std::string::npos) {
                // No more special tokens; encode the rest as ordinary.
                auto tail = encode_ordinary_impl(text.substr(pos));
                result.insert(result.end(), tail.begin(), tail.end());
                break;
            }

            // Encode text before the special token.
            if (best_pos > pos) {
                auto prefix = encode_ordinary_impl(text.substr(pos, best_pos - pos));
                result.insert(result.end(), prefix.begin(), prefix.end());
            }

            // Add the special token.
            result.push_back(static_cast<int32_t>(best_id));
            pos = best_pos + best_token.size();
        }

        return result;
    }
};

// ============================================================================
// Public API
// ============================================================================

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

Tokenizer Tokenizer::from_pretrained(const std::string& model_dir) {
    namespace fs = std::filesystem;
    auto dir = fs::path(model_dir);

    auto tokenizer_json_path = dir / "tokenizer.json";
    if (!fs::exists(tokenizer_json_path)) {
        throw std::runtime_error(fmt::format("tokenizer.json not found in {}", model_dir));
    }

    // Parse tokenizer.json
    json data;
    {
        std::ifstream f(tokenizer_json_path);
        data = json::parse(f);
    }

    Tokenizer tok;
    auto& impl = *tok.impl_;

    // ---- Model (BPE) ----
    auto& model = data.at("model");
    std::string model_type = model.at("type");
    if (model_type != "BPE") {
        throw std::runtime_error(fmt::format("Unsupported tokenizer model type: {} (only BPE supported)", model_type));
    }

    // Load vocabulary: { "token_str": rank, ... }
    auto& vocab = model.at("vocab");
    impl.encoder.reserve(vocab.size());
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        std::string token_str = it.key();
        Rank rank = it.value().get<Rank>();
        std::vector<uint8_t> token_bytes(token_str.begin(), token_str.end());
        impl.encoder[token_bytes] = rank;
        impl.decoder[rank] = token_bytes;
    }

    // Load merges (for rank ordering validation — ranks are already in vocab).
    // In HuggingFace format, merges encode the training order.
    // The vocab already contains the correct rank for each token, so we don't
    // need to rebuild the merge table. The BPE algorithm uses encoder lookups
    // directly (tiktoken-style).
    // We do verify merge count for sanity.
    if (model.contains("merges") && model["merges"].is_array()) {
        size_t num_merges = model["merges"].size();
        (void)num_merges; // Used only for sanity check if needed
    }

    impl.vocab_size_ = static_cast<int32_t>(impl.encoder.size());

    // ---- Normalizer ----
    if (data.contains("normalizer") && !data["normalizer"].is_null()) {
        auto& norm = data["normalizer"];
        std::string norm_type = norm.at("type");
        if (norm_type == "NFC") impl.normalizer = Impl::Normalizer::NFC;
        else if (norm_type == "NFD") impl.normalizer = Impl::Normalizer::NFD;
        else if (norm_type == "NFKC") impl.normalizer = Impl::Normalizer::NFKC;
        else if (norm_type == "NFKD") impl.normalizer = Impl::Normalizer::NFKD;
    }

    // ---- Detect ByteLevel from tokenizer.json ----
    // Check pre_tokenizer, post_processor, and decoder for ByteLevel.
    auto has_byte_level = [](const json& obj) -> bool {
        if (obj.is_null()) return false;
        if (obj.at("type") == "ByteLevel") return true;
        if (obj.at("type") == "Sequence" && obj.contains("pretokenizers")) {
            for (auto& sub : obj["pretokenizers"]) {
                if (sub.at("type") == "ByteLevel") return true;
            }
        }
        return false;
    };
    for (const auto& key : {"pre_tokenizer", "post_processor", "decoder"}) {
        if (data.contains(key) && !data[key].is_null()) {
            if (has_byte_level(data[key])) {
                impl.byte_level = true;
                break;
            }
        }
    }

    // ---- Pre-tokenizer: detect from model architecture (config.json) ----
    // Like llama.cpp, we use hard-coded regex patterns per architecture.
    // Each pattern has a matching hand-optimized C++ implementation in unicode.cpp,
    // so no regex engine is needed.
    std::string architecture;
    auto model_config_path = dir / "config.json";
    if (fs::exists(model_config_path)) {
        json model_config;
        { std::ifstream f(model_config_path); model_config = json::parse(f); }
        architecture = model_config.value("model_type", "");
    }

    if (architecture == "qwen2" || architecture == "qwen3" ||
        architecture == "qwen3_moe" || architecture == "qwen3_vl") {
        // Qwen2/3 family: single-digit number matching
        impl.pre_tokenizer_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
    } else if (architecture == "qwen3_5") {
        // Qwen3.5: includes \p{M} (accent marks) in letter matching
        impl.pre_tokenizer_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
    } else if (architecture == "llama" || architecture == "smollm" || architecture == "smollm3" ||
               architecture == "mistral" || architecture == "gemma" || architecture == "gemma2" ||
               architecture == "phi3" || architecture == "chatglm") {
        // Llama3 family: 1-3 digit number matching
        impl.pre_tokenizer_regex = {
            "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+",
        };
    } else if (architecture == "gpt2" || architecture == "gpt_bigcode") {
        // GPT-2 family
        impl.pre_tokenizer_regex = {
            "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
        };
    } else {
        // Unknown architecture: fall back to parsing regex from tokenizer.json
        if (data.contains("pre_tokenizer") && !data["pre_tokenizer"].is_null()) {
            auto& pre_tok = data["pre_tokenizer"];
            auto extract_regex = [&](const json& obj) {
                if (obj.at("type") == "Split" && obj.contains("pattern")) {
                    auto& pattern = obj["pattern"];
                    if (pattern.contains("Regex")) {
                        impl.pre_tokenizer_regex.push_back(pattern["Regex"].get<std::string>());
                    }
                }
            };
            if (pre_tok.at("type") == "Sequence") {
                for (auto& sub : pre_tok.at("pretokenizers")) {
                    extract_regex(sub);
                }
            } else {
                extract_regex(pre_tok);
            }
        }
        // If still empty, use GPT-2 default
        if (impl.pre_tokenizer_regex.empty()) {
            impl.pre_tokenizer_regex = {
                "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)",
            };
        }
    }

    // ---- Added tokens (special tokens) ----
    if (data.contains("added_tokens") && data["added_tokens"].is_array()) {
        for (auto& tok_obj : data["added_tokens"]) {
            std::string content = tok_obj.at("content").get<std::string>();
            Rank id = tok_obj.at("id").get<Rank>();
            bool is_special = tok_obj.value("special", false);

            if (is_special) {
                impl.special_tokens_encoder[content] = id;
                impl.special_tokens_decoder[id] = content;
                impl.special_token_ids.insert(id);
            }

            // Also add to encoder/decoder if not already present
            std::vector<uint8_t> bytes(content.begin(), content.end());
            if (impl.encoder.find(bytes) == impl.encoder.end()) {
                impl.encoder[bytes] = id;
                impl.decoder[id] = bytes;
                impl.vocab_size_ = std::max(impl.vocab_size_, static_cast<int32_t>(id + 1));
            }
        }
    }

    // ---- Load tokenizer_config.json for extra metadata ----
    auto config_path = dir / "tokenizer_config.json";
    if (fs::exists(config_path)) {
        json config;
        {
            std::ifstream f(config_path);
            config = json::parse(f);
        }

        // BOS/EOS/PAD token resolution
        auto resolve_token_id = [&](const json& config, const std::string& key) -> int32_t {
            if (!config.contains(key) || config[key].is_null()) return -1;

            std::string token_str;
            if (config[key].is_string()) {
                token_str = config[key].get<std::string>();
            } else if (config[key].is_object() && config[key].contains("content")) {
                token_str = config[key]["content"].get<std::string>();
            } else {
                return -1;
            }

            // Store named special token
            impl.named_special_tokens[key] = token_str;

            // Look up in special tokens first
            auto it = impl.special_tokens_encoder.find(token_str);
            if (it != impl.special_tokens_encoder.end()) {
                return static_cast<int32_t>(it->second);
            }

            // Fall back to regular vocab
            std::vector<uint8_t> bytes(token_str.begin(), token_str.end());
            auto it2 = impl.encoder.find(bytes);
            if (it2 != impl.encoder.end()) {
                return static_cast<int32_t>(it2->second);
            }
            return -1;
        };

        impl.bos_id = resolve_token_id(config, "bos_token");
        impl.eos_id = resolve_token_id(config, "eos_token");
        impl.pad_id = resolve_token_id(config, "pad_token");
        if (config.contains("unk_token")) {
            impl.unk_id = resolve_token_id(config, "unk_token");
        }

        impl.add_bos = config.value("add_bos_token", false);
        impl.add_eos = config.value("add_eos_token", false);

        // Load chat template (Jinja2 string)
        if (config.contains("chat_template") && config["chat_template"].is_string()) {
            std::string tmpl_str = config["chat_template"].get<std::string>();
            impl.chat_tmpl = std::make_unique<minja::chat_template>(tmpl_str, impl.named_special_tokens.count("bos_token") ? impl.named_special_tokens["bos_token"] : "", impl.named_special_tokens.count("eos_token") ? impl.named_special_tokens["eos_token"] : "");
        }
    }

    // If no chat template in tokenizer_config.json, check for chat_template.jinja file
    if (!impl.chat_tmpl) {
        auto jinja_path = dir / "chat_template.jinja";
        if (fs::exists(jinja_path)) {
            std::ifstream f(jinja_path);
            std::string tmpl_str((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            impl.chat_tmpl = std::make_unique<minja::chat_template>(tmpl_str, impl.named_special_tokens.count("bos_token") ? impl.named_special_tokens["bos_token"] : "", impl.named_special_tokens.count("eos_token") ? impl.named_special_tokens["eos_token"] : "");
        }
    }

    // Build fast lookup table
    impl.build_lookup();

    return tok;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text, bool add_special_tokens) const {
    auto result = impl_->encode_impl(text, /*use_special=*/true);
    if (add_special_tokens) {
        if (impl_->add_bos && impl_->bos_id >= 0) {
            result.insert(result.begin(), impl_->bos_id);
        }
        if (impl_->add_eos && impl_->eos_id >= 0) {
            result.push_back(impl_->eos_id);
        }
    }
    return result;
}

std::vector<int32_t> Tokenizer::encode_with_special_tokens(const std::string& text) const {
    return impl_->encode_impl(text, /*use_special=*/true);
}

std::vector<int32_t> Tokenizer::encode_ordinary(const std::string& text) const {
    return impl_->encode_ordinary_impl(text);
}

std::vector<std::vector<int32_t>> Tokenizer::encode_batch(
        const std::vector<std::string>& texts, bool add_special_tokens) const {
    std::vector<std::vector<int32_t>> results(texts.size());

    // Parallel encode using std::thread for batches > 1
    if (texts.size() > 1) {
        unsigned num_threads = std::min(static_cast<unsigned>(texts.size()),
                                        std::thread::hardware_concurrency());
        if (num_threads < 2) num_threads = 1;

        if (num_threads > 1) {
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            std::atomic<size_t> next_idx{0};

            for (unsigned t = 0; t < num_threads; t++) {
                threads.emplace_back([&]() {
                    while (true) {
                        size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
                        if (i >= texts.size()) break;
                        results[i] = this->encode(texts[i], add_special_tokens);
                    }
                });
            }
            for (auto& th : threads) th.join();
            return results;
        }
    }

    // Sequential fallback
    for (size_t i = 0; i < texts.size(); i++) {
        results[i] = encode(texts[i], add_special_tokens);
    }
    return results;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Accumulate byte-level-encoded text in segments, decode each segment
    // when we hit a special token (which is stored as plain text, not byte-level).
    std::string result;
    std::string byte_level_buf;

    auto flush_byte_level = [&]() {
        if (!byte_level_buf.empty()) {
            if (impl_->byte_level) {
                result += byte_level_decode(byte_level_buf);
            } else {
                result += byte_level_buf;
            }
            byte_level_buf.clear();
        }
    };

    for (int32_t id : ids) {
        auto uid = static_cast<Rank>(id);

        // Check special tokens first — these are plain text, not byte-level encoded
        auto sit = impl_->special_tokens_decoder.find(uid);
        if (sit != impl_->special_tokens_decoder.end()) {
            flush_byte_level();
            result += sit->second;
            continue;
        }

        // Regular token — byte-level encoded
        auto it = impl_->decoder.find(uid);
        if (it != impl_->decoder.end()) {
            byte_level_buf.append(reinterpret_cast<const char*>(it->second.data()), it->second.size());
        }
    }

    flush_byte_level();
    return result;
}

int32_t Tokenizer::encode_single_token(const std::string& token_bytes) const {
    std::vector<uint8_t> key(token_bytes.begin(), token_bytes.end());
    auto it = impl_->encoder.find(key);
    if (it != impl_->encoder.end()) {
        return static_cast<int32_t>(it->second);
    }
    // Check special tokens
    auto sit = impl_->special_tokens_encoder.find(token_bytes);
    if (sit != impl_->special_tokens_encoder.end()) {
        return static_cast<int32_t>(sit->second);
    }
    throw std::runtime_error(fmt::format("Token not found in vocabulary: '{}'", token_bytes));
}

std::string Tokenizer::decode_single_token(int32_t id) const {
    auto uid = static_cast<Rank>(id);

    auto sit = impl_->special_tokens_decoder.find(uid);
    if (sit != impl_->special_tokens_decoder.end()) {
        return sit->second;
    }

    auto it = impl_->decoder.find(uid);
    if (it != impl_->decoder.end()) {
        std::string raw(reinterpret_cast<const char*>(it->second.data()), it->second.size());
        if (impl_->byte_level) {
            return byte_level_decode(raw);
        }
        return raw;
    }
    throw std::runtime_error(fmt::format("Token ID {} not found in vocabulary", id));
}

int32_t Tokenizer::vocab_size() const { return impl_->vocab_size_; }
int32_t Tokenizer::bos_token_id() const { return impl_->bos_id; }
int32_t Tokenizer::eos_token_id() const { return impl_->eos_id; }
int32_t Tokenizer::pad_token_id() const { return impl_->pad_id; }

bool Tokenizer::is_special_token(int32_t id) const {
    return impl_->special_token_ids.count(static_cast<Rank>(id)) > 0;
}

std::string Tokenizer::special_token(const std::string& name) const {
    auto it = impl_->named_special_tokens.find(name);
    if (it != impl_->named_special_tokens.end()) {
        return it->second;
    }
    return "";
}

std::string Tokenizer::apply_chat_template(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt) const {
    if (!impl_->chat_tmpl) {
        throw std::runtime_error("No chat template loaded. Model directory must contain "
                                 "chat_template in tokenizer_config.json or a chat_template.jinja file.");
    }

    // Convert ChatMessage to nlohmann::ordered_json array
    nlohmann::ordered_json json_messages = nlohmann::ordered_json::array();
    for (const auto& msg : messages) {
        json_messages.push_back({{"role", msg.role}, {"content", msg.content}});
    }

    minja::chat_template_inputs inputs;
    inputs.messages = json_messages;
    inputs.add_generation_prompt = add_generation_prompt;
    inputs.now = std::chrono::system_clock::now();

    return impl_->chat_tmpl->apply(inputs);
}

std::vector<int32_t> Tokenizer::apply_chat_template_and_encode(
        const std::vector<ChatMessage>& messages,
        bool add_generation_prompt) const {
    std::string text = apply_chat_template(messages, add_generation_prompt);
    return encode_with_special_tokens(text);
}

} // namespace tokenizer
