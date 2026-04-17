// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tokenizer/tokenizer.h"
#include "tokenizer/unicode.h"
#include "tokenizer/bpe.h"

#include <filesystem>
#include <string>
#include <vector>

// ============================================================================
// Unicode tests
// ============================================================================

TEST_CASE("unicode_len_utf8", "[tokenizer][unicode]") {
    CHECK(unicode_len_utf8('A') == 1);     // ASCII
    CHECK(unicode_len_utf8('\xC3') == 2);  // 2-byte UTF-8 lead
    CHECK(unicode_len_utf8('\xE2') == 3);  // 3-byte UTF-8 lead
    CHECK(unicode_len_utf8('\xF0') == 4);  // 4-byte UTF-8 lead
}

TEST_CASE("unicode_cpt_to_utf8 roundtrip", "[tokenizer][unicode]") {
    // ASCII
    CHECK(unicode_cpt_to_utf8('A') == "A");
    CHECK(unicode_cpt_to_utf8(' ') == " ");

    // 2-byte: é (U+00E9)
    std::string e_acute = unicode_cpt_to_utf8(0x00E9);
    size_t off = 0;
    CHECK(unicode_cpt_from_utf8(e_acute, off) == 0x00E9);

    // 3-byte: 中 (U+4E2D)
    std::string zhong = unicode_cpt_to_utf8(0x4E2D);
    off = 0;
    CHECK(unicode_cpt_from_utf8(zhong, off) == 0x4E2D);

    // 4-byte: 😀 (U+1F600)
    std::string smile = unicode_cpt_to_utf8(0x1F600);
    off = 0;
    CHECK(unicode_cpt_from_utf8(smile, off) == 0x1F600);
}

TEST_CASE("unicode_cpt_flags classification", "[tokenizer][unicode]") {
    auto flags_A = unicode_cpt_flags_from_cpt('A');
    CHECK(flags_A.is_letter);
    CHECK(flags_A.is_uppercase);
    CHECK(!flags_A.is_number);

    auto flags_0 = unicode_cpt_flags_from_cpt('0');
    CHECK(flags_0.is_number);
    CHECK(!flags_0.is_letter);

    auto flags_sp = unicode_cpt_flags_from_cpt(' ');
    CHECK(flags_sp.is_whitespace);
}

TEST_CASE("unicode_byte_to_utf8 mapping", "[tokenizer][unicode]") {
    // Space (0x20) maps to Ġ (U+0120) in byte-level encoding
    std::string encoded_space = unicode_byte_to_utf8(0x20);
    CHECK(encoded_space == unicode_cpt_to_utf8(0x0120));

    // Printable ASCII maps to itself
    std::string encoded_A = unicode_byte_to_utf8('A');
    CHECK(encoded_A == "A");

    // Roundtrip
    CHECK(unicode_utf8_to_byte(encoded_space) == 0x20);
    CHECK(unicode_utf8_to_byte(encoded_A) == 'A');
}

TEST_CASE("unicode_regex_split llama3 pattern", "[tokenizer][unicode]") {
    // Use the expanded form (no (?i:) which std::regex doesn't support).
    // This is the form llama.cpp and our tokenizer actually use.
    std::string pattern = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                          "N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    auto pieces = unicode_regex_split("Hello, world! 123", {pattern});
    CHECK(!pieces.empty());

    // unicode_regex_split applies byte-level encoding, so space becomes Ġ
    bool found_hello = false;
    bool found_world = false;
    for (const auto& p : pieces) {
        if (p == "Hello") found_hello = true;
        // After byte-level encoding: " world" -> "Ġworld"
        if (p.find("world") != std::string::npos) found_world = true;
    }
    CHECK(found_hello);
    CHECK(found_world);
}

TEST_CASE("unicode_regex_split qwen pattern", "[tokenizer][unicode]") {
    // Qwen2/3 pattern: single digit matching (\p{N} instead of \p{N}{1,3})
    std::string pattern = "(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{"
                          "N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    auto pieces = unicode_regex_split("Hello 123", {pattern});
    CHECK(!pieces.empty());

    // Digits should be individual pieces (single digit matching)
    int digit_pieces = 0;
    for (const auto& p : pieces) {
        if (p == "1" || p == "2" || p == "3") digit_pieces++;
    }
    CHECK(digit_pieces == 3);
}

TEST_CASE("unicode_regex_split gpt-oss pattern", "[tokenizer][unicode]") {
    std::string pattern = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?"
                          "i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                          "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?"
                          "i:'s|'t|'re|'ve|'m|'ll|'d)?|"
                          "\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    auto pieces = unicode_regex_split(" NASA's GPT-OSS /path/to/file abc/\nfooBAR 123", {pattern});
    CHECK(pieces == std::vector<std::string>{
                        "ĠNASA's",
                        "ĠGPT",
                        "-OSS",
                        "Ġ/",
                        "path",
                        "/to",
                        "/file",
                        "Ġabc",
                        "/Ċ",
                        "foo",
                        "BAR",
                        "Ġ",
                        "123",
                    });
}

// ============================================================================
// BPE algorithm tests
// ============================================================================

TEST_CASE("bpe small piece encode", "[tokenizer][bpe]") {
    // Build a tiny vocab for testing
    tokenizer::Encoder enc;
    enc[{'a'}] = 0;
    enc[{'b'}] = 1;
    enc[{'c'}] = 2;
    enc[{'a', 'b'}] = 3;       // merge a+b -> ab (rank 3)
    enc[{'a', 'b', 'c'}] = 4;  // merge ab+c -> abc (rank 4)

    tokenizer::EncoderLookup lookup(enc);

    // Single byte
    auto r1 = tokenizer::byte_pair_encode(reinterpret_cast<const uint8_t*>("a"), 1, lookup);
    CHECK(r1.size() == 1);
    CHECK(r1[0] == 0);

    // "ab" should merge to rank 3
    auto r2 = tokenizer::byte_pair_encode(reinterpret_cast<const uint8_t*>("ab"), 2, lookup);
    CHECK(r2.size() == 1);
    CHECK(r2[0] == 3);

    // "abc" should merge to rank 4
    auto r3 = tokenizer::byte_pair_encode(reinterpret_cast<const uint8_t*>("abc"), 3, lookup);
    CHECK(r3.size() == 1);
    CHECK(r3[0] == 4);

    // "ac" cannot merge -> two tokens
    auto r4 = tokenizer::byte_pair_encode(reinterpret_cast<const uint8_t*>("ac"), 2, lookup);
    CHECK(r4.size() == 2);
    CHECK(r4[0] == 0);  // a
    CHECK(r4[1] == 2);  // c
}

// ============================================================================
// Full tokenizer integration test (requires model files)
// ============================================================================

TEST_CASE("tokenizer from_pretrained Qwen3", "[tokenizer][integration]") {
    // Look for a Qwen3 model in the HF cache
    std::string model_dir;
    namespace fs = std::filesystem;

    // Try common cache locations
    std::vector<std::string> candidates = {
        "/home/densemax2/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots",
        "/home/densemax2/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-FP8/snapshots",
    };

    for (const auto& base : candidates) {
        if (!fs::exists(base)) continue;
        for (auto& entry : fs::directory_iterator(base)) {
            if (fs::exists(entry.path() / "tokenizer.json")) {
                model_dir = entry.path().string();
                break;
            }
        }
        if (!model_dir.empty()) break;
    }

    if (model_dir.empty()) {
        SKIP("No Qwen3 model found in HF cache — skipping integration test");
    }

    SECTION("load and basic encode/decode") {
        auto tok = tokenizer::Tokenizer::from_pretrained(model_dir);

        CHECK(tok.vocab_size() > 100000);  // Qwen3 has ~151K vocab
        CHECK(tok.eos_token_id() >= 0);

        // Encode simple text
        auto ids = tok.encode_ordinary("Hello, world!");
        CHECK(!ids.empty());

        // Decode should roundtrip
        auto decoded = tok.decode(ids);
        CHECK(decoded == "Hello, world!");

        // Batch encode
        auto batch = tok.encode_batch({"Hello", "World"}, false);
        CHECK(batch.size() == 2);
        CHECK(!batch[0].empty());
        CHECK(!batch[1].empty());
    }

    SECTION("special tokens") {
        auto tok = tokenizer::Tokenizer::from_pretrained(model_dir);

        // Qwen3 uses <|im_end|> as EOS
        CHECK(tok.eos_token_id() >= 0);
        CHECK(tok.is_special_token(tok.eos_token_id()));

        // Encode with special tokens in text
        auto ids = tok.encode_with_special_tokens("Hello<|im_end|>");
        // Should contain the special token ID
        bool found_eos = false;
        for (int32_t id : ids) {
            if (id == tok.eos_token_id()) found_eos = true;
        }
        CHECK(found_eos);
    }

    SECTION("encode_ordinary ignores special tokens") {
        auto tok = tokenizer::Tokenizer::from_pretrained(model_dir);

        // encode_ordinary should NOT treat <|im_end|> as special
        auto ids = tok.encode_ordinary("<|im_end|>");
        // Should be multiple tokens (the literal bytes), not a single special token
        CHECK(ids.size() > 1);
    }
}
