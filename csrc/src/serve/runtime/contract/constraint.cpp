#include "runtime/contract/constraint.h"
#include <llguidance.h>
#include <xgrammar/xgrammar.h>
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

namespace {
// What JsonConstraintCompiler::compile_after needs besides the schema, shared by every state.
struct ReasoningGate {
    TokenId close = -1;
    std::vector<TokenId> stops;
    std::vector<std::uint32_t> separators; // bitmask of the whitespace-only tokens
    bool stop(TokenId token) const { return std::find(stops.begin(), stops.end(), token) != stops.end(); }
    bool separator(TokenId token) const {
        const auto word = static_cast<std::size_t>(token) / 32;
        return word < separators.size() && ((separators[word] >> (static_cast<std::uint32_t>(token) % 32)) & 1U) != 0;
    }
};

class AfterReasoningState final : public TokenConstraintState {
    enum class Phase { Reasoning, Separator, Answer };
    std::shared_ptr<const ReasoningGate> gate_;
    std::shared_ptr<const CompiledTokenConstraint> schema_;
    std::unique_ptr<TokenConstraintState> answer_; // from the close on: a thought's drafts never fork a grammar
    Phase phase_ = Phase::Reasoning;
public:
    AfterReasoningState(std::shared_ptr<const ReasoningGate> gate, std::shared_ptr<const CompiledTokenConstraint> schema,
                        std::unique_ptr<TokenConstraintState> answer = nullptr, Phase phase = Phase::Reasoning)
        : gate_(std::move(gate)), schema_(std::move(schema)), answer_(std::move(answer)), phase_(phase) {}
    bool stopped() const override { return phase_ == Phase::Answer && answer_->stopped(); }
    bool try_accept(TokenId token) override {
        if (token < 0) { return false; }
        switch (phase_) {
        case Phase::Reasoning:
            if (gate_->stop(token)) { return false; }
            if (token == gate_->close) {
                answer_ = schema_->create_state();
                phase_  = Phase::Separator;
            }
            return true;
        case Phase::Separator:
            if (!gate_->separator(token) && !answer_->try_accept(token)) { return false; }
            phase_ = Phase::Answer;
            return true;
        case Phase::Answer: break;
        }
        return answer_->try_accept(token);
    }
    void accept(TokenId token) override {
        if (!try_accept(token)) { throw std::logic_error("token violates JSON constraint: " + std::to_string(token)); }
    }
    std::unique_ptr<TokenConstraintState> fork() const override {
        return std::make_unique<AfterReasoningState>(gate_, schema_, answer_ ? answer_->fork() : nullptr, phase_);
    }
    void fill(std::span<int32_t> mask) override {
        if (phase_ == Phase::Reasoning) {
            std::fill(mask.begin(), mask.end(), -1);
            for (const TokenId stop : gate_->stops) {
                const auto word = static_cast<std::size_t>(stop) / 32;
                if (stop >= 0 && word < mask.size()) {
                    mask[word] = static_cast<int32_t>(static_cast<std::uint32_t>(mask[word]) &
                                                      ~(1U << (static_cast<std::uint32_t>(stop) % 32)));
                }
            }
            return;
        }
        answer_->fill(mask);
        if (phase_ == Phase::Separator) {
            for (std::size_t word = 0; word < std::min(mask.size(), gate_->separators.size()); ++word) {
                mask[word] = static_cast<int32_t>(static_cast<std::uint32_t>(mask[word]) | gate_->separators[word]);
            }
        }
    }
};

class AfterReasoning final : public CompiledTokenConstraint {
    std::shared_ptr<const ReasoningGate> gate_;
    std::shared_ptr<const CompiledTokenConstraint> schema_;
public:
    AfterReasoning(std::shared_ptr<const ReasoningGate> gate, std::shared_ptr<const CompiledTokenConstraint> schema)
        : gate_(std::move(gate)), schema_(std::move(schema)) {}
    std::unique_ptr<TokenConstraintState> create_state() const override {
        return std::make_unique<AfterReasoningState>(gate_, schema_);
    }
};

std::vector<std::uint32_t> whitespace_tokens(const std::vector<std::string>& vocab) {
    std::vector<std::uint32_t> mask((vocab.size() + 31) / 32, 0U);
    for (std::size_t id = 0; id < vocab.size(); ++id) {
        const std::string& text = vocab[id];
        if (!text.empty() && std::all_of(text.begin(), text.end(),
                [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; })) {
            mask[id / 32] |= 1U << (id % 32);
        }
    }
    return mask;
}
} // namespace

class JsonConstraintCompiler::Impl {
public:
    std::shared_ptr<GrammarTokenizer> tokenizer;
    std::size_t vocabulary = 0;
    std::vector<std::uint32_t> separators;
    std::mutex mutex;
    std::unordered_map<std::string, std::shared_ptr<const CompiledTokenConstraint>> cache;
    std::shared_ptr<const ReasoningGate> gate; // compile_after's, for the one close token a model has
    Impl(const std::vector<std::string>& vocab, const std::vector<TokenId>& stops,
         std::function<std::vector<TokenId>(std::string_view)> encode)
        : tokenizer(std::make_shared<GrammarTokenizer>(vocab, stops, std::move(encode))),
          vocabulary(vocab.size()), separators(whitespace_tokens(vocab)) {}
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
std::shared_ptr<const CompiledTokenConstraint> JsonConstraintCompiler::compile_after(const std::string& schema, TokenId close) {
    auto answer = compile(schema);
    std::lock_guard lock(impl_->mutex);
    if (!impl_->gate || impl_->gate->close != close) {
        auto gate = std::make_shared<ReasoningGate>();
        gate->close = close;
        gate->stops = impl_->tokenizer->eos;
        gate->separators = impl_->separators;
        if (close < 0 || static_cast<std::size_t>(close) >= impl_->vocabulary || gate->stop(close)) {
            throw std::invalid_argument("the reasoning close token must be a non-stop token of the JSON vocabulary");
        }
        impl_->gate = std::move(gate);
    }
    return std::make_shared<AfterReasoning>(impl_->gate, std::move(answer));
}
namespace {
class ToolConstraintState final : public TokenConstraintState {
    xgrammar::GrammarMatcher matcher_;
public:
    explicit ToolConstraintState(xgrammar::GrammarMatcher matcher) : matcher_(std::move(matcher)) {}
    bool stopped() const override { return matcher_.IsTerminated(); }
    bool try_accept(TokenId token) override { return !stopped() && matcher_.AcceptToken(token); }
    void accept(TokenId token) override {
        if (!try_accept(token)) { throw std::logic_error("token violates tool constraint: " + std::to_string(token)); }
    }
    std::unique_ptr<TokenConstraintState> fork() const override {
        return std::make_unique<ToolConstraintState>(matcher_.Fork());
    }
    void fill(std::span<int32_t> mask) override {
        int64_t shape = static_cast<int64_t>(mask.size());
        DLTensor tensor{.data = mask.data(), .device = {kDLCPU, 0}, .ndim = 1,
            .dtype = {kDLInt, 32, 1}, .shape = &shape, .strides = nullptr, .byte_offset = 0};
        matcher_.FillNextTokenBitmask(&tensor);
        if (std::none_of(mask.begin(), mask.end(), [](auto word) { return word != 0; })) {
            throw std::invalid_argument("tool constraint has no continuation in this vocabulary");
        }
    }
};
class ToolConstraint final : public CompiledTokenConstraint {
    xgrammar::CompiledGrammar grammar_;
public:
    explicit ToolConstraint(xgrammar::CompiledGrammar grammar) : grammar_(std::move(grammar)) {}
    std::unique_ptr<TokenConstraintState> create_state() const override {
        return std::make_unique<ToolConstraintState>(xgrammar::GrammarMatcher(grammar_));
    }
};
} // namespace
class ToolConstraintCompiler::Impl {
public:
    xgrammar::GrammarCompiler compiler;
    std::mutex mutex;
    Impl(const std::vector<std::string>& vocab, const std::vector<TokenId>& stops)
        : compiler(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW,
            static_cast<int>(vocab.size()), stops), 1, true, 128LL << 20) {}
};
ToolConstraintCompiler::ToolConstraintCompiler(std::vector<std::string> vocab, std::vector<TokenId> stops)
    : impl_(std::make_unique<Impl>(vocab, stops)) {}
ToolConstraintCompiler::~ToolConstraintCompiler() = default;
std::shared_ptr<const CompiledTokenConstraint> ToolConstraintCompiler::compile(const std::string& tag) {
    std::lock_guard lock(impl_->mutex);
    try { return std::make_shared<ToolConstraint>(impl_->compiler.CompileStructuralTag(tag)); }
    catch (const std::exception& error) { throw std::invalid_argument(std::string("invalid tool constraint: ") + error.what()); }
}
} // namespace sinfer
