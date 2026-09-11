#include "runtime/contract/constraint.h"
#include <llguidance.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace sinfer {
namespace {
using Matcher = std::unique_ptr<LlgMatcher, decltype(&llg_free_matcher)>;

struct GrammarTokenizer {
    std::unique_ptr<LlgTokenizer, decltype(&llg_free_tokenizer)> tokenizer{nullptr, llg_free_tokenizer};
    std::function<std::vector<TokenId>(std::string_view)> encode;
    std::vector<TokenId> eos;
    static size_t tokenize(const void* self, const uint8_t* bytes, size_t size, uint32_t* out, size_t cap) noexcept {
        try {
            const auto tokens = static_cast<const GrammarTokenizer*>(self)->encode(
                std::string_view(reinterpret_cast<const char*>(bytes), size));
            std::copy_n(tokens.begin(), std::min(tokens.size(), cap), out);
            return tokens.size();
        } catch (...) { return 0; }
    }
    GrammarTokenizer(const std::vector<std::string>& vocab, const std::vector<TokenId>& stops,
                     std::function<std::vector<TokenId>(std::string_view)> encoder) : encode(std::move(encoder)), eos(stops) {
        if (stops.empty()) { throw std::invalid_argument("JSON constraints need an EOS token"); }
        std::vector<uint32_t> lengths;
        std::vector<uint8_t> bytes;
        for (const auto& token : vocab) {
            lengths.push_back(static_cast<uint32_t>(token.size()));
            bytes.insert(bytes.end(), token.begin(), token.end());
        }
        std::vector<uint32_t> eos(stops.begin(), stops.end());
        LlgTokenizerInitV2 init{};
        init.struct_size = sizeof(init);
        init.vocab_size = static_cast<uint32_t>(vocab.size());
        init.tok_eos = eos.front();
        init.tok_eos_extra = eos.data() + 1;
        init.tok_eos_extra_count = static_cast<uint32_t>(eos.size() - 1);
        init.token_lens = lengths.data();
        init.token_bytes = bytes.data();
        init.tokenize_assumes_string = true;
        init.tokenize_fn = encode ? &tokenize : nullptr;
        init.use_approximate_greedy_tokenize_fn = !encode; // byte-vocabulary unit fixtures
        init.tokenize_user_data = this;
        char error[2048]{};
        tokenizer.reset(llg_new_tokenizer_v2(&init, error, sizeof(error)));
        if (!tokenizer) { throw std::invalid_argument(std::string("constraint tokenizer: ") + error); }
    }
};

class ConstraintState final : public TokenConstraintState {
    std::shared_ptr<GrammarTokenizer> tokenizer_;
    Matcher matcher_;
    bool terminated_ = false;
public:
    ConstraintState(std::shared_ptr<GrammarTokenizer> tokenizer, const LlgMatcher* matcher)
        : tokenizer_(std::move(tokenizer)), matcher_(llg_clone_matcher(matcher), llg_free_matcher) {}
    bool stopped() const override { return terminated_; }
    bool try_accept(TokenId token) override {
        if (token < 0 || stopped()) { return false; }
        if (std::find(tokenizer_->eos.begin(), tokenizer_->eos.end(), token) != tokenizer_->eos.end()) {
            if (!llg_matcher_is_accepting(matcher_.get())) { return false; }
            terminated_ = true;
            return true;
        }
        uint32_t id = static_cast<uint32_t>(token);
        if (llg_matcher_validate_tokens(matcher_.get(), &id, 1) != 1) { return false; }
        return llg_matcher_consume_token(matcher_.get(), id) == 0;
    }
    void accept(TokenId token) override {
        if (!try_accept(token)) { throw std::logic_error("token violates JSON constraint: " + std::to_string(token)); }
    }
    std::unique_ptr<TokenConstraintState> fork() const override {
        auto result = std::make_unique<ConstraintState>(tokenizer_, matcher_.get());
        result->terminated_ = terminated_;
        return result;
    }
    void fill(std::span<int32_t> mask) override {
        if (llg_matcher_get_mask_byte_size(matcher_.get()) != mask.size_bytes()) {
            throw std::invalid_argument("JSON constraint vocabulary does not match sampling mask");
        }
        if (llg_matcher_compute_mask_into(matcher_.get(), reinterpret_cast<uint32_t*>(mask.data()), mask.size_bytes()) != 0) {
            const char* error = llg_matcher_get_error(matcher_.get());
            throw std::invalid_argument(std::string("JSON constraint: ") + (error ? error : "no continuation"));
        }
    }
};

// Keep date formats compatible with calendar validators, including leap years.
void normalize_formats(nlohmann::json& schema) {
    if (!schema.is_object()) { return; }
    for (const char* key : {"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"}) {
        if (schema.contains(key) && schema[key].is_object()) { for (auto& child : schema[key]) { normalize_formats(child); } }
    }
    for (const char* key : {"items", "additionalProperties", "contains", "not", "if", "then", "else"}) {
        if (schema.contains(key)) { normalize_formats(schema[key]); }
    }
    for (const char* key : {"allOf", "anyOf", "oneOf", "prefixItems"}) {
        if (schema.contains(key) && schema[key].is_array()) { for (auto& child : schema[key]) { normalize_formats(child); } }
    }
    if (schema.value("format", std::string{}) == "date") {
        const std::string year = "(000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})";
        const std::string leap = "([0-9]{2}(0[48]|[2468][048]|[13579][26])|(0[48]|[2468][048]|[13579][26])00)";
        const std::string pattern = "^(" + year + "-(01|03|05|07|08|10|12)-(0[1-9]|[12][0-9]|3[01])|" +
            year + "-(04|06|09|11)-(0[1-9]|[12][0-9]|30)|" + year + "-02-(0[1-9]|1[0-9]|2[0-8])|" + leap + "-02-29)$";
        if (!schema.contains("allOf")) { schema["allOf"] = nlohmann::json::array(); }
        if (!schema["allOf"].is_array()) { throw std::invalid_argument("allOf must be an array"); }
        schema["allOf"].push_back({{"pattern", pattern}});
    }
}

class Constraint final : public CompiledTokenConstraint {
    std::shared_ptr<GrammarTokenizer> tokenizer_;
    Matcher matcher_;
public:
    Constraint(std::shared_ptr<GrammarTokenizer> tokenizer, const std::string& schema)
        : tokenizer_(std::move(tokenizer)), matcher_(nullptr, llg_free_matcher) {
        LlgConstraintInit init;
        llg_constraint_init_set_defaults(&init, tokenizer_->tokenizer.get());
        matcher_.reset(llg_new_matcher(&init, "json", schema.c_str()));
        if (const char* error = llg_matcher_get_error(matcher_.get())) { throw std::invalid_argument(error); }
    }
    std::unique_ptr<TokenConstraintState> create_state() const override {
        return std::make_unique<ConstraintState>(tokenizer_, matcher_.get());
    }
};
} // namespace

void TokenConstraintState::fill_draft_masks(std::span<const TokenId> drafts, std::span<int32_t> masks) {
    const auto width = drafts.size() + 1;
    if (masks.empty() || masks.size() % width != 0) { throw std::logic_error("invalid speculative mask shape"); }
    const auto words = masks.size() / width;
    auto probe = fork();
    bool reachable = true;
    for (std::size_t col = 0; col < width; ++col) {
        auto mask = masks.subspan(col * words, words);
        if (reachable) { probe->fill(mask); }
        else { std::copy_n(masks.data() + (col - 1) * words, words, mask.data()); }
        if (col < drafts.size() && reachable) {
            reachable = probe->try_accept(drafts[col]) && !probe->stopped();
        }
    }
}

class JsonConstraintCompiler::Impl {
public:
    std::shared_ptr<GrammarTokenizer> tokenizer;
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<const CompiledTokenConstraint>> cache;
    Impl(const std::vector<std::string>& vocab, const std::vector<TokenId>& stops,
         std::function<std::vector<TokenId>(std::string_view)> encode)
        : tokenizer(std::make_shared<GrammarTokenizer>(vocab, stops, std::move(encode))) {}
};
JsonConstraintCompiler::JsonConstraintCompiler(std::vector<std::string> vocab, std::vector<TokenId> stops,
    std::function<std::vector<TokenId>(std::string_view)> encode)
    : impl_(std::make_unique<Impl>(vocab, stops, std::move(encode))) {}
JsonConstraintCompiler::~JsonConstraintCompiler() = default;
std::shared_ptr<const CompiledTokenConstraint> JsonConstraintCompiler::compile(const std::string& schema) {
    std::lock_guard lock(impl_->mutex);
    if (auto it = impl_->cache.find(schema); it != impl_->cache.end()) { return it->second; }
    try {
        auto normalized = nlohmann::json::parse(schema);
        normalize_formats(normalized);
        if (normalized.is_object()) {
            // Never enable approximations that silently drop schema assertions.
            normalized["x-guidance"] = {{"lenient", false}, {"coerce_one_of", false},
                {"whitespace_pattern", R"([\x20\x0A\x0D\x09]{1})"}};
        }
        auto result = std::make_shared<Constraint>(impl_->tokenizer, normalized.dump());
        if (impl_->cache.size() >= 32) { impl_->cache.clear(); }
        impl_->cache.emplace(schema, result);
        return result;
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("invalid or unsupported JSON schema: ") + error.what());
    }
}
} // namespace sinfer
