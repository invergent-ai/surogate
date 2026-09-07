// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// High-performance BPE tokenizer that loads from HuggingFace tokenizer.json.
// Core BPE algorithm adapted from tiktoken (MIT License).
// Unicode support adapted from llama.cpp (MIT License).

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tokenizer {

// Loss masking strategy for encode_for_training().
enum class LossStrategy {
    DEFAULT,        // Train on all assistant responses + suffix tokens
    LAST_ROUND,     // Train only on the last assistant response + suffix
    THINKING_ONLY,  // Train reasoning through </think>; mask final answers
    FINAL_ONLY,     // Train final answers after </think>; mask reasoning/direct turns
    ALL             // Train on all tokens (including system/user)
};

// Result of encode_for_training().
struct TrainingEncoded {
    std::vector<int32_t> input_ids;
    std::vector<int32_t> labels;  // -100 for masked (no loss), token_id for trainable
};

// A single message in a conversation.
struct ChatMessage {
    std::string role;  // "system", "user", "assistant", "tool"
    std::string content;
};

// Template variables beyond the messages and the generation prompt. Each is left
// undefined in the render when unset, so a template that gates on `is defined`
// keeps its own default. Both names are the ones the model cards use:
// enable_thinking for the Qwen family's switch, reasoning_effort for the templates
// that choose an effort instead (GLM names low/high/max, Qwen low/medium/xhigh).
struct ChatTemplateVariables {
    std::optional<bool> enable_thinking;
    std::optional<std::string> reasoning_effort;
};

class Tokenizer {
public:
    ~Tokenizer();
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;

    // The files a tokenizer is built from, already in memory. `tokenizer_json` is
    // required; the rest may be empty. This is what lets a caller that holds the
    // resources rather than a directory -- the serving engine reads them out of
    // its model artifact -- build the same tokenizer without staging temp files.
    struct Sources {
        std::string tokenizer_json;
        std::string model_config_json;      // config.json: names the pre-tokenizer family
        std::string tokenizer_config_json;  // specials, chat template
        std::string chat_template_jinja;    // standalone template, if the checkpoint ships one
    };

    // Load from a HuggingFace model directory (reads tokenizer.json + tokenizer_config.json)
    static Tokenizer from_pretrained(const std::string& model_dir);

    // Same, from sources already in memory.
    static Tokenizer from_sources(const Sources& sources);

    // Encode text to token IDs. Special tokens in the text are NOT encoded unless
    // they appear in allowed_special.
    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = false) const;

    // Encode text, treating all special tokens found in the text as special.
    std::vector<int32_t> encode_with_special_tokens(const std::string& text) const;

    // Encode without any special token handling (pure BPE on the full text).
    std::vector<int32_t> encode_ordinary(const std::string& text) const;

    // Batch encode for throughput.
    std::vector<std::vector<int32_t>> encode_batch(const std::vector<std::string>& texts,
                                                   bool add_special_tokens = false) const;

    // Decode token IDs back to text. `strip_leading_space` undoes the word mark
    // the normalizer prepends, which is only correct for a whole sequence -- a
    // caller decoding one token at a time must pass false or every word loses the
    // space in front of it.
    std::string decode(const std::vector<int32_t>& ids, bool strip_leading_space = true) const;

    // Single token encode/decode.
    int32_t encode_single_token(const std::string& token_bytes) const;
    std::string decode_single_token(int32_t id) const;

    // Vocabulary info.
    int32_t vocab_size() const;
    int32_t bos_token_id() const;
    // Whether this checkpoint opens a sequence with its BOS, from either
    // tokenizer_config's add_bos_token or the tokenizer.json post-processor.
    bool adds_bos() const;
    int32_t eos_token_id() const;
    int32_t pad_token_id() const;

    // Check if a token ID is a special token.
    bool is_special_token(int32_t id) const;

    // Get the string content of a special token by name (e.g. "eos_token").
    std::string special_token(const std::string& name) const;

    // Apply a chat template to a list of messages.
    // Returns the formatted string ready for encode().
    // If add_generation_prompt is true, appends the assistant header so the
    // model can continue generating.
    // enable_thinking, when set, is passed to the template (e.g. Qwen3 emits an
    // open "<think>\n" generation prompt when true vs. an empty
    // "<think>\n\n</think>\n\n" block when false); std::nullopt leaves it unset
    // so the template falls back to its own default.
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt = false,
                                    std::optional<bool> enable_thinking = std::nullopt) const;

    // Same, with every template variable this renderer can set. The overload above
    // is the thinking switch on its own, which is all the training paths ask for.
    std::string apply_chat_template(const std::vector<ChatMessage>& messages,
                                    bool add_generation_prompt,
                                    const ChatTemplateVariables& variables) const;

    // Convenience: apply_chat_template + encode_with_special_tokens in one call.
    std::vector<int32_t> apply_chat_template_and_encode(const std::vector<ChatMessage>& messages,
                                                        bool add_generation_prompt = false) const;

    // Encode a conversation for training with loss masking.
    //
    // Uses incremental chat template rendering to identify which tokens belong
    // to assistant responses vs. system/user/template chrome. Returns input_ids
    // and labels where labels[i] = -100 for masked tokens (no loss) and
    // labels[i] = token_id for trainable tokens.
    //
    // The strategy controls which tokens are trainable:
    //   DEFAULT    — all assistant response + suffix tokens
    //   LAST_ROUND — only the last assistant response + suffix
    //   THINKING_ONLY — reasoning tokens through </think>; direct/final tokens masked
    //   FINAL_ONLY — final tokens after </think>; reasoning/direct tokens masked
    //   ALL        — all tokens (system, user, assistant, template)
    //
    // labels[0] is always -100.
    TrainingEncoded encode_for_training(const std::vector<ChatMessage>& messages,
                                        LossStrategy strategy = LossStrategy::DEFAULT) const;

    // Batch version of encode_for_training (multi-threaded).
    std::vector<TrainingEncoded> encode_for_training_batch(const std::vector<std::vector<ChatMessage>>& batch,
                                                           LossStrategy strategy = LossStrategy::DEFAULT) const;

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tokenizer
