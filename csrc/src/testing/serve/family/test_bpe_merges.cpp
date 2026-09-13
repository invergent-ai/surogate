#include "family/impl/frontend/tokenizer.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using sinfer::family::frontend_internal::Tokenizer;

std::vector<int> scan_reference(const std::string& text,
                                const std::unordered_map<std::string, int>& ranks,
                                const Json& vocab) {
    std::vector<std::string> symbols;
    for (const char c : text) { symbols.emplace_back(1, c); }
    while (symbols.size() > 1) {
        int rank = std::numeric_limits<int>::max();
        std::size_t best = symbols.size();
        for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
            const auto found = ranks.find(symbols[i] + ' ' + symbols[i + 1]);
            if (found != ranks.end() && found->second < rank) { rank = found->second; best = i; }
        }
        if (best == symbols.size()) { break; }
        symbols[best] += symbols[best + 1];
        symbols.erase(symbols.begin() + best + 1);
    }
    std::vector<int> result;
    for (const auto& symbol : symbols) { result.push_back(vocab.at(symbol)); }
    return result;
}

void test_merge_order() {
    Json vocab = Json::object();
    std::vector<std::string> words;
    for (int length = 1; length <= 5; ++length) {
        for (int bits = 0; bits < (1 << length); ++bits) {
            std::string word;
            for (int i = 0; i < length; ++i) { word += (bits & (1 << i)) ? 'b' : 'a'; }
            vocab[word] = words.size();
            words.push_back(word);
        }
    }
    std::vector<std::string> merges;
    for (const auto& left : words) {
        for (const auto& right : words) {
            if (left.size() + right.size() <= 5) { merges.push_back(left + ' ' + right); }
        }
    }
    std::mt19937 rng(106);
    for (int table = 0; table < 4; ++table) {
        std::shuffle(merges.begin(), merges.end(), rng);
        std::unordered_map<std::string, int> ranks;
        for (int i = 0; i < static_cast<int>(merges.size()); ++i) { ranks.emplace(merges[i], i); }
        const auto json = Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}},
                               {"normalizer", nullptr}, {"added_tokens", Json::array()}}.dump();
        Tokenizer tokenizer({json, R"({"added_tokens_decoder":{}})", R"({"eos_token_id":0})"});
        for (int trial = 0; trial < 1000; ++trial) {
            std::string input;
            const int length = trial % 64 + 1;
            for (int i = 0; i < length; ++i) { input += (rng() & 1) ? 'a' : 'b'; }
            if (tokenizer.encode(input) != scan_reference(input, ranks, vocab)) {
                throw std::runtime_error("BPE merge rank/leftmost ordering changed: " + input);
            }
        }
    }
}

void test_million_character_pretoken() {
    Json vocab = Json::object(), merges = Json::array();
    for (int power = 0; power <= 7; ++power) {
        const std::string word(1 << power, '!');
        vocab[word] = power;
        if (power != 0) {
            const std::string half(1 << (power - 1), '!');
            merges.push_back(half + ' ' + half);
        }
    }
    const auto json = Json{{"model", {{"type", "BPE"}, {"vocab", vocab}, {"merges", merges}}},
                           {"normalizer", nullptr}, {"added_tokens", Json::array()}}.dump();
    Tokenizer tokenizer({json, R"({"added_tokens_decoder":{}})", R"({"eos_token_id":0})"});
    if (tokenizer.encode(std::string(1 << 20, '!')) != std::vector<int>(1 << 13, 7)) {
        throw std::runtime_error("long pre-token merge result changed");
    }
}
} // namespace

int main() {
    try {
        test_merge_order();
        test_million_character_pretoken();
        std::cout << "BPE merge ordering and long pre-token: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
