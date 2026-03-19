// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// High-performance BPE tokenizer that loads from HuggingFace tokenizer.json.
// Core BPE algorithm adapted from tiktoken (MIT License).
// Unicode support adapted from llama.cpp (MIT License).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tokenizer {

class Tokenizer {
public:
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // Load from a HuggingFace model directory (reads tokenizer.json + tokenizer_config.json)
    static Tokenizer from_pretrained(const std::string& model_dir);

    // Encode text to token IDs. Special tokens in the text are NOT encoded unless
    // they appear in allowed_special.
    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = false) const;

    // Encode text, treating all special tokens found in the text as special.
    std::vector<int32_t> encode_with_special_tokens(const std::string& text) const;

    // Encode without any special token handling (pure BPE on the full text).
    std::vector<int32_t> encode_ordinary(const std::string& text) const;

    // Batch encode for throughput.
    std::vector<std::vector<int32_t>> encode_batch(const std::vector<std::string>& texts, bool add_special_tokens = false) const;

    // Decode token IDs back to text.
    std::string decode(const std::vector<int32_t>& ids) const;

    // Single token encode/decode.
    int32_t encode_single_token(const std::string& token_bytes) const;
    std::string decode_single_token(int32_t id) const;

    // Vocabulary info.
    int32_t vocab_size() const;
    int32_t bos_token_id() const;
    int32_t eos_token_id() const;
    int32_t pad_token_id() const;

    // Check if a token ID is a special token.
    bool is_special_token(int32_t id) const;

    // Get the string content of a special token by name (e.g. "eos_token").
    std::string special_token(const std::string& name) const;

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tokenizer
