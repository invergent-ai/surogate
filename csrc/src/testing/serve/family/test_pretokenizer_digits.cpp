// How many digits the pre-tokenizer keeps in one word, read from what the artifact declares.
//
// Every checkpoint this family served until GLM-5.3 declares `\p{N}` and gets one digit a word;
// GLM-5.3 declares `\p{N}{1,3}` and gets up to three. The rule was compiled in -- the splitter
// was named for the family it came from -- so GLM tokenised "3,344" as five separate digits
// where its own tokenizer makes four tokens, and "17" as "1" "7". A model asked how many sheep
// are left of seventeen cannot answer a question it was never shown.
//
// Two vocabularies, identical but for the declared pattern, and one input that separates them.
#include <api/family/frontend_resources.h>

#include "family/impl/frontend/test_access.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

/// A byte-level BPE vocabulary holding the single digits, the comma, and the merge `3`+`4`.
/// Ids are arbitrary; what the test reads is how many tokens come out and whether the merged
/// one appears, which is exactly what the digit run decides.
std::string tokenizer_json(std::string_view digit_pattern) {
    return std::string(R"({"model":{"type":"BPE","byte_fallback":false,"vocab":{)") +
           R"("1":0,"3":1,"4":2,"7":3,",":4,"34":5,"17":6},)" +
           R"("merges":["3 4","1 7"]},)" +
           R"("added_tokens":[],)" +
           R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[)" +
           R"({"type":"Split","pattern":{"Regex":")" + std::string(digit_pattern) +
           R"("},"behavior":"Isolated","invert":false},)" +
           R"({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}})";
}

/// The two patterns, verbatim as the two families write them.
constexpr std::string_view kSingleDigit =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
constexpr std::string_view kUpToThreeDigits =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

sinfer::family::FrontendResources resources(std::string_view digit_pattern) {
    sinfer::family::FrontendResources out;
    out.tokenizer_json         = tokenizer_json(digit_pattern);
    out.tokenizer_config_json  = R"({"added_tokens_decoder":{}})";
    out.generation_config_json = R"({"eos_token_id":0})";
    return out;
}

void expect_resources(const char* what, const sinfer::family::FrontendResources& assets,
                      std::string_view text, const std::vector<int>& want) {
    try {
        const std::vector<int> got =
            sinfer::family::FrontendTestAccess::encode_with(assets, text);
        if (got != want) {
            std::cerr << what << ": got";
            for (int id : got) { std::cerr << ' ' << id; }
            std::cerr << ", want";
            for (int id : want) { std::cerr << ' ' << id; }
            std::cerr << '\n';
            failures += 1;
        }
    } catch (const std::exception& error) {
        std::cerr << what << ": threw " << error.what() << '\n';
        failures += 1;
    }
}

void expect(const char* what, std::string_view pattern, std::string_view text,
            const std::vector<int>& want) {
    expect_resources(what, resources(pattern), text, want);
}

void check_split_sequence() {
    using Json = nlohmann::json;
    auto assets = resources(kUpToThreeDigits);
    auto root = Json::parse(assets.tokenizer_json);
    auto& stages = root["pre_tokenizer"]["pretokenizers"];
    std::string pattern = stages[0]["pattern"]["Regex"];
    pattern.replace(pattern.find("{1,3}"), 5, "+");
    stages[0]["pattern"]["Regex"] = pattern;
    stages.insert(stages.begin(), Json{{"type", "Split"}, {"pattern", {{"Regex", R"(\p{N}{1,3})"}}},
                                      {"behavior", "Isolated"}, {"invert", false}});
    root["model"]["vocab"]["Ġ"] = 7;
    root["model"]["vocab"]["ĠĠ"] = 8;
    root["model"]["merges"].push_back("Ġ Ġ");
    root["normalizer"] = nullptr;
    assets.tokenizer_json = root.dump();
    // Isolating numbers first leaves the preceding spaces as one trailing-space piece.
    expect_resources("ordered splits preserve spaces and digit groups", assets, "  1734", {8, 6, 1, 2});
    // A direct vocabulary hit bypasses merges only when the checkpoint requests it.
    root["model"]["vocab"]["134"] = 9;
    assets.tokenizer_json = root.dump();
    expect_resources("declared merges", assets, "134", {0, 5});
    root["model"]["ignore_merges"] = true;
    assets.tokenizer_json = root.dump();
    expect_resources("ignore merges", assets, "134", {9});
    // An absent normalizer must preserve decomposed text before byte encoding.
    root["model"]["vocab"]["e"] = 10;
    root["model"]["vocab"]["Ì"] = 11;
    root["model"]["vocab"]["ģ"] = 12;
    assets.tokenizer_json = root.dump();
    expect_resources("identity normalization", assets, "e\u0301", {10, 11, 12});
}

} // namespace

int main() {
    check_split_sequence();
    // `17`: one word of two digits merges to the vocabulary's `17`; two words of one digit
    // cannot, whatever merges exist, because a merge never crosses a word boundary.
    expect("17 with \\p{N}{1,3}", kUpToThreeDigits, "17", {6});
    expect("17 with \\p{N}", kSingleDigit, "17", {0, 3});

    // `3,344`: the comma splits regardless; what differs is the run after it.
    expect("3,344 with \\p{N}{1,3}", kUpToThreeDigits, "3,344", {1, 4, 5, 2});
    expect("3,344 with \\p{N}", kSingleDigit, "3,344", {1, 4, 1, 2, 2});

    // Four digits exceed even the wider run, so it splits after three -- the bound is a bound,
    // not "the whole number".
    expect("1734 with \\p{N}{1,3}", kUpToThreeDigits, "1734", {6, 1, 2});

    if (failures != 0) {
        std::cerr << failures << " pre-tokenizer digit check(s) failed\n";
        return 1;
    }
    std::cout << "pre-tokenizer digits: PASS\n";
    return 0;
}
