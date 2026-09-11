#include "runtime/contract/constraint.h"

#include <xgrammar/xgrammar.h>

#include <algorithm>
#include <mutex>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace sinfer {
namespace {

// Reject constraints the grammar compiler would otherwise ignore. Metadata is harmless,
// but an unsupported assertion must never silently weaken the requested language.
void validate_schema(const nlohmann::json& schema, unsigned depth = 0) {
    using Json = nlohmann::json;
    if (depth > 64) { throw std::invalid_argument("JSON schema nesting exceeds 64 levels"); }
    if (schema.is_boolean()) { return; }
    if (!schema.is_object()) { throw std::invalid_argument("JSON schema must be an object or boolean"); }
    static const std::set<std::string> metadata{
        "$schema", "$id", "$comment", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly"};
    static const std::set<std::string> assertions{
        "type", "enum", "const", "properties", "required", "additionalProperties", "items", "prefixItems",
        "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "anyOf", "$ref", "$defs", "definitions"};
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        const auto& key = it.key();
        if (!metadata.contains(key) && !assertions.contains(key)) {
            throw std::invalid_argument("unsupported JSON schema keyword: " + key);
        }
        if (key == "properties" || key == "$defs" || key == "definitions") {
            if (!it.value().is_object()) { throw std::invalid_argument(key + " must be an object"); }
            for (const auto& child : it.value()) { validate_schema(child, depth + 1); }
        } else if (key == "items" || key == "additionalProperties") {
            validate_schema(it.value(), depth + 1);
        } else if (key == "prefixItems" || key == "anyOf") {
            if (!it.value().is_array()) { throw std::invalid_argument(key + " must be an array"); }
            for (const auto& child : it.value()) { validate_schema(child, depth + 1); }
        }
    }
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) { throw std::invalid_argument("required must be an array"); }
        for (const auto& name : schema["required"]) {
            if (!name.is_string() || !schema.contains("properties") || !schema["properties"].contains(name.get<std::string>())) {
                throw std::invalid_argument("required properties must be declared in properties");
            }
        }
    }
    for (const char* exclusive : {"$ref", "anyOf", "enum", "const"}) {
        if (!schema.contains(exclusive)) { continue; }
        for (auto it = schema.begin(); it != schema.end(); ++it) {
            if (it.key() == exclusive || metadata.contains(it.key()) || it.key() == "$defs" || it.key() == "definitions") { continue; }
            // The conventional enum + type form is safe when every member has that type.
            if (it.key() == "type" && (std::string(exclusive) == "enum" || std::string(exclusive) == "const") && it.value().is_string()) {
                const auto type = it.value().get<std::string>();
                const auto matches = [&](const Json& v) {
                    return (type == "string" && v.is_string()) || (type == "boolean" && v.is_boolean()) ||
                           (type == "null" && v.is_null()) || (type == "integer" && v.is_number_integer()) ||
                           (type == "number" && v.is_number()) || (type == "object" && v.is_object()) ||
                           (type == "array" && v.is_array());
                };
                const auto values = std::string(exclusive) == "enum" ? schema[exclusive] : Json::array({schema[exclusive]});
                if (values.is_array() && std::all_of(values.begin(), values.end(), matches)) { continue; }
            }
            throw std::invalid_argument(std::string("unsupported JSON schema combination: ") + exclusive + " with " + it.key());
        }
    }
    if (schema.contains("$ref") && (!schema["$ref"].is_string() || !schema["$ref"].get<std::string>().starts_with("#"))) {
        throw std::invalid_argument("only local JSON schema references are supported");
    }
}

class ConstraintState final : public TokenConstraintState {
public:
    explicit ConstraintState(const xgrammar::CompiledGrammar& grammar) : matcher_(grammar) {}
    void accept(TokenId token) override {
        if (!matcher_.AcceptToken(token)) {
            throw std::logic_error("generated token " + std::to_string(token) + " violates the request's JSON constraint");
        }
    }
    void fill(std::span<std::int32_t> mask) override {
        int64_t shape = static_cast<int64_t>(mask.size());
        DLTensor tensor{.data = mask.data(), .device = {kDLCPU, 0}, .ndim = 1,
                        .dtype = {kDLInt, 32, 1}, .shape = &shape, .strides = nullptr, .byte_offset = 0};
        matcher_.FillNextTokenBitmask(&tensor);
        if (std::none_of(mask.begin(), mask.end(), [](auto word) { return word != 0; })) {
            throw std::invalid_argument("JSON constraint has no continuation in this model's vocabulary");
        }
    }
private:
    xgrammar::GrammarMatcher matcher_;
};

class Constraint final : public CompiledTokenConstraint {
public:
    explicit Constraint(xgrammar::CompiledGrammar grammar) : grammar_(std::move(grammar)) {}
    std::unique_ptr<TokenConstraintState> create_state() const override {
        return std::make_unique<ConstraintState>(grammar_);
    }
private:
    xgrammar::CompiledGrammar grammar_;
};

} // namespace

class JsonConstraintCompiler::Impl {
public:
    Impl(const std::vector<std::string>& vocab, const std::vector<TokenId>& stops)
        : compiler(xgrammar::TokenizerInfo(vocab, xgrammar::VocabType::RAW,
                                          static_cast<int>(vocab.size()), stops), 1, true, 128LL << 20) {}
    xgrammar::GrammarCompiler compiler;
    std::mutex mutex;
};

JsonConstraintCompiler::JsonConstraintCompiler(std::vector<std::string> vocabulary, std::vector<TokenId> stops)
    : impl_(std::make_unique<Impl>(vocabulary, stops)) {}
JsonConstraintCompiler::~JsonConstraintCompiler() = default;

std::shared_ptr<const CompiledTokenConstraint> JsonConstraintCompiler::compile(const std::string& schema) {
    std::lock_guard lock(impl_->mutex);
    try {
        validate_schema(nlohmann::json::parse(schema));
        // strict_mode=false preserves standard JSON Schema defaults for additional properties/items.
        return std::make_shared<Constraint>(impl_->compiler.CompileJSONSchema(
            schema, true, std::nullopt, std::nullopt, false, 1));
    } catch (const std::exception& error) {
        throw std::invalid_argument(std::string("invalid or unsupported JSON schema: ") + error.what());
    }
}

} // namespace sinfer
