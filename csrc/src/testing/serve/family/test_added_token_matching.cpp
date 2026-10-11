#include "tokenizer/tokenizer.h"
#include "family/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>
#include <cassert>
#include <future>
#include <iostream>
#include <string>
#include <vector>

int main() {
    using Json = nlohmann::json;
    Json fixture = {
        {"model", {{"type", "BPE"}, {"byte_fallback", true},
                   {"vocab", {{"a", 0}, {"b", 1}, {"c", 2}, {"x", 3}, {"z", 4}, {"<", 5}}},
                   {"merges", Json::array()}}},
        {"added_tokens", Json::array()}};
    const std::vector<std::string> tokens = {"ab", "aba", "bc", "é", "<image>", "<image>tail", "x"};
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        fixture["added_tokens"].push_back({{"content", tokens[i]}, {"id", 10 + i}, {"special", i != 6}});
    }
    // Many tokens share prefixes, as the reserved tokens in real checkpoints do.
    for (int i = 0; i < 7000; ++i) {
        fixture["added_tokens"].push_back({{"content", "<unused" + std::to_string(i) + ">"},
                                            {"id", 100 + i}, {"special", true}});
    }
    auto tokenizer = tokenizer::Tokenizer::from_sources({.tokenizer_json = fixture.dump()});
    Json byte_fixture = fixture;
    byte_fixture["model"]["byte_fallback"] = false;
    byte_fixture["model"]["vocab"].erase("x"); // the family loader requires one ID per spelling
    byte_fixture["normalizer"] = nullptr;
    for (auto& token : byte_fixture["added_tokens"]) {
        for (const char* flag : {"single_word", "lstrip", "rstrip", "normalized"}) { token[flag] = false; }
    }
    const Json byte_config = {{"added_tokens_decoder", {{"7100", {
        {"content", "<image>tailmore"}, {"special", true}, {"single_word", false},
        {"lstrip", false}, {"rstrip", false}, {"normalized", false}}}}}};
    sinfer::family::frontend_internal::Tokenizer byte_tokenizer(
        {byte_fixture.dump(), byte_config.dump(), R"({"eos_token_id":0})"});
    const auto check = [&](const std::string& text, std::vector<std::int32_t> expected) {
        assert(tokenizer.encode_with_special_tokens(text) == expected);
        assert(byte_tokenizer.encode(text) == expected);
    };
    check("", {});
    check("abac", {11, 2});                 // Longest match at the first position.
    check("abc", {10, 2});                  // An overlapping later match loses.
    check("abbc", {10, 12});                // Adjacent matches.
    check("xabc", {16, 10, 2});             // Non-special added tokens also match.
    check("éab", {13, 10});                 // UTF-8 matching preserves the token.
    check("z<image>tail<image>z", {4, 15, 14, 4});
    check("<<image>z", {5, 14, 4});
    check("<unused6999>", {7099});
    assert((byte_tokenizer.encode("z<image>tailmore<image>tail") == std::vector<int>{4, 7100, 15}));
    assert((tokenizer.encode_ordinary("xxx") == std::vector<std::int32_t>{3, 3, 3}));
    assert((byte_tokenizer.encode("aaa", {.parse_added_tokens=false}) == std::vector<int>{0, 0, 0}));

    std::string images;
    for (int i = 0; i < 256; ++i) { images += "<image>"; }
    const std::vector<std::int32_t> expected(256, 14);
    std::vector<std::future<void>> concurrent;
    for (int thread = 0; thread < 8; ++thread) {
        concurrent.push_back(std::async(std::launch::async, [&] {
            for (int i = 0; i < 30; ++i) { check(images, expected); }
        }));
    }
    for (auto& result : concurrent) { result.get(); }
    fixture["added_tokens"].push_back({{"content", ""}, {"id", 8000}});
    bool rejected = false;
    try { (void)tokenizer::Tokenizer::from_sources({.tokenizer_json = fixture.dump()}); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
    // encode_reusing_prefix() must return encode()'s ids exactly: agent-like prompts that drop a
    // short tail and grow every turn, prompts sharing only an opening, and several conversations
    // at once. The added tokens here overlap each other and ordinary text, the hard case for
    // reusing ids up to an added-token boundary.
    {
        // Ordinary pieces use the vocabulary's letters only (this tokenizer has no byte
        // fallback); added tokens meet across piece boundaries ("ab" + "a" is "aba").
        const std::vector<std::string> pieces = {"a", "b", "c", "z", "x", "<", "é", "<image>",
                                                 "<image>tail", "<image>tailmore", "ab", "aba",
                                                 "<unused42>", "bc"};
        // Prompts are joined pieces and are cut on piece boundaries, so every text stays
        // encodable (a cut inside "<image>tailmore" would leave letters outside the vocabulary).
        const auto conversation_run = [&](unsigned seed, int turns) {
            const auto next = [&seed] {
                seed = seed * 1664525U + 1013904223U;
                return seed >> 8;
            };
            std::vector<std::string> parts;
            std::size_t bytes = 0;
            const auto grow = [&](std::size_t more) {
                for (const std::size_t target = bytes + more; bytes < target;) {
                    parts.push_back(pieces[next() % pieces.size()]);
                    bytes += parts.back().size();
                }
            };
            const auto join = [&](std::size_t count) {
                std::string text;
                for (std::size_t i = 0; i < count; ++i) { text += parts[i]; }
                return text;
            };
            grow(20000);
            for (int turn = 0; turn < turns; ++turn) {
                const std::string conversation = join(parts.size());
                assert(byte_tokenizer.encode_reusing_prefix(conversation) ==
                       byte_tokenizer.encode(conversation));
                for (unsigned cut = next() % 6; cut != 0 && !parts.empty(); --cut) {
                    bytes -= parts.back().size();
                    parts.pop_back();
                }
                grow(64 + next() % 2048);
                if (turn % 7 == 3) {
                    std::string other = join(parts.size() * 3 / 4);
                    for (int i = 0; i < 400; ++i) { other += pieces[next() % pieces.size()]; }
                    assert(byte_tokenizer.encode_reusing_prefix(other) == byte_tokenizer.encode(other));
                }
            }
        };
        conversation_run(7U, 40);
        std::vector<std::future<void>> agents;
        for (unsigned agent = 0; agent < 6; ++agent) {
            agents.push_back(std::async(std::launch::async, [&, agent] { conversation_run(100U + agent, 20); }));
        }
        for (auto& agent : agents) { agent.get(); }
    }
    std::cout << "Added-token matching, concurrent image tokenization and prefix reuse passed\n";
}
