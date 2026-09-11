#pragma once

#include "api/types.h"

#include <span>

namespace sinfer {

class TokenConstraintState {
public:
    virtual ~TokenConstraintState() = default;
    virtual void accept(TokenId token) = 0;
    virtual void fill(std::span<std::int32_t> mask) = 0;
};

class CompiledTokenConstraint {
public:
    virtual ~CompiledTokenConstraint() = default;
    [[nodiscard]] virtual std::unique_ptr<TokenConstraintState> create_state() const = 0;
};

class JsonConstraintCompiler {
public:
    JsonConstraintCompiler(std::vector<std::string> vocabulary, std::vector<TokenId> stops);
    ~JsonConstraintCompiler();
    [[nodiscard]] std::shared_ptr<const CompiledTokenConstraint> compile(const std::string& schema);
private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sinfer
