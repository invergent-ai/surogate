// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Regression tests for encode_for_training() thinking / no-think mode handling.
//
// Bug (fixed): render_chat_template() never set `enable_thinking`, so the
// generation-prompt render always emitted the no-think empty "<think>\n\n</think>\n\n"
// block. For a THINKING assistant turn this misaligned the byte-diff that defines
// the trainable span, producing a doubled </think>, corrupted reasoning, and a
// target that taught the model to reason AFTER an already-closed think block
// (i.e. to "think" in no-think mode). encode_for_training() now derives
// enable_thinking from the assistant content (a "</think>" block => thinking turn)
// and passes it to both renders consistently.
//
// Requires a real tokenizer dir with a Qwen3-style chat template; set
// SUROGATE_TEST_MODEL=<dir>. Skips cleanly when unset.

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>
#include <vector>

#include "tokenizer/tokenizer.h"

using tokenizer::ChatMessage;
using tokenizer::LossStrategy;
using tokenizer::Tokenizer;
using tokenizer::TrainingEncoded;

namespace {

size_t count_substr(const std::string& s, const std::string& sub) {
    size_t n = 0, pos = 0;
    while ((pos = s.find(sub, pos)) != std::string::npos) {
        ++n;
        pos += sub.size();
    }
    return n;
}

std::string decode_full(const Tokenizer& tok, const TrainingEncoded& enc) {
    return tok.decode(enc.input_ids);
}

std::string decode_trained(const Tokenizer& tok, const TrainingEncoded& enc) {
    std::vector<int32_t> ids;
    for (size_t i = 0; i < enc.input_ids.size(); ++i) {
        if (enc.labels[i] != -100) ids.push_back(enc.input_ids[i]);
    }
    return tok.decode(ids);
}

}  // namespace

TEST_CASE("encode_for_training separates thinking and no-think modes", "[tokenizer][chat-template]") {
    const char* model_dir = std::getenv("SUROGATE_TEST_MODEL");
    if (!model_dir) {
        SUCCEED("SUROGATE_TEST_MODEL not set; skipping chat-template regression test");
        return;
    }
    Tokenizer tok = Tokenizer::from_pretrained(model_dir);

    SECTION("thinking turn: reasoning kept, single think block, trained after open <think>") {
        std::vector<ChatMessage> msgs = {
            {"user", "Cat fac 2+2?"},
            {"assistant", "<think>\nAdun doi si doi.\n</think>\n\nPatru."},
        };
        auto enc = tok.encode_for_training(msgs, LossStrategy::DEFAULT);
        std::string full = decode_full(tok, enc);
        std::string trained = decode_trained(tok, enc);

        // Exactly one think block — the bug produced an empty block + a second </think>.
        REQUIRE(count_substr(full, "<think>") == 1);
        REQUIRE(count_substr(full, "</think>") == 1);
        // No empty (no-think) block should appear for a thinking turn.
        REQUIRE(full.find("<think>\n\n</think>") == std::string::npos);
        // Reasoning must be intact and trained (the bug sliced its first tokens off).
        REQUIRE(full.find("Adun doi si doi.") != std::string::npos);
        REQUIRE(trained.find("Adun doi si doi.") != std::string::npos);
        REQUIRE(trained.find("Patru.") != std::string::npos);
    }

    SECTION("no-think turn: empty block, only the answer is trained") {
        std::vector<ChatMessage> msgs = {
            {"user", "Cat fac 2+2?"},
            {"assistant", "Patru."},
        };
        auto enc = tok.encode_for_training(msgs, LossStrategy::DEFAULT);
        std::string full = decode_full(tok, enc);
        std::string trained = decode_trained(tok, enc);

        REQUIRE(count_substr(full, "<think>") == 1);
        REQUIRE(count_substr(full, "</think>") == 1);
        // The empty no-think block must be present...
        REQUIRE(full.find("<think>\n\n</think>") != std::string::npos);
        // ...and the model must NOT be trained to emit any reasoning/think tokens.
        REQUIRE(trained.find("<think>") == std::string::npos);
        REQUIRE(trained.find("</think>") == std::string::npos);
        REQUIRE(trained.find("Patru.") != std::string::npos);
    }
}

TEST_CASE("apply_chat_template honors enable_thinking generation prompt", "[tokenizer][chat-template]") {
    const char* model_dir = std::getenv("SUROGATE_TEST_MODEL");
    if (!model_dir) {
        SUCCEED("SUROGATE_TEST_MODEL not set; skipping chat-template regression test");
        return;
    }
    Tokenizer tok = Tokenizer::from_pretrained(model_dir);
    std::vector<ChatMessage> msgs = {{"user", "Salut"}};

    // enable_thinking=true => open "<think>\n" prompt (no close); =false => empty block.
    std::string thinking = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true, /*enable_thinking=*/true);
    std::string no_think = tok.apply_chat_template(msgs, /*add_generation_prompt=*/true, /*enable_thinking=*/false);

    REQUIRE(count_substr(thinking, "</think>") == 0);
    REQUIRE(count_substr(no_think, "</think>") == 1);
}
