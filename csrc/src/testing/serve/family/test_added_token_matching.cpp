#include "tokenizer/tokenizer.h"

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
    const auto check = [&](const std::string& text, std::vector<std::int32_t> expected) {
        assert(tokenizer.encode_with_special_tokens(text) == expected);
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
    assert((tokenizer.encode_ordinary("xxx") == std::vector<std::int32_t>{3, 3, 3}));

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
    std::cout << "Added-token matching and concurrent image tokenization passed\n";
}
