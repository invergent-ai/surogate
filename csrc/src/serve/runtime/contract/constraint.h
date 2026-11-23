#pragma once

#include "api/types.h"

#include <span>

namespace sinfer {

class TokenConstraintState {
public:
    virtual ~TokenConstraintState() = default;
    virtual void accept(TokenId token) = 0;
    virtual bool try_accept(TokenId token) = 0;
    virtual bool stopped() const = 0;
    virtual std::unique_ptr<TokenConstraintState> fork() const = 0;
    void fill_draft_masks(std::span<const TokenId> drafts, std::span<std::int32_t> masks);
    virtual void fill(std::span<std::int32_t> mask) = 0;
};

class CompiledTokenConstraint {
public:
    virtual ~CompiledTokenConstraint() = default;
    [[nodiscard]] virtual std::unique_ptr<TokenConstraintState> create_state() const = 0;
};

class JsonConstraintCompiler {
public:
    JsonConstraintCompiler(std::vector<std::string> vocabulary, std::vector<TokenId> stops,
        std::function<std::vector<TokenId>(std::string_view)> encode = {});
    ~JsonConstraintCompiler();
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile(const std::string& schema);
    /// `compile(schema)` behind a reasoning region the prompt opened, which is the model's own: until it writes
    /// `close`, every token but a stop is allowed (so the region is closed before the answer ends). One
    /// whitespace-only token may then follow, the separator a chat template writes there, and the schema
    /// constrains the rest. vLLM's structured outputs do the same under a reasoning parser.
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile_after(const std::string& schema, TokenId close);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

class ToolConstraintCompiler {
public:
    ToolConstraintCompiler(std::vector<std::string> vocabulary, std::vector<TokenId> stops);
    ~ToolConstraintCompiler();
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile(const std::string& structural_tag);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer
