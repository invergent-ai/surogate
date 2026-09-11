#include "runtime/contract/constraint.h"

#include <cassert>
#include <iostream>

using namespace sinfer;

int main() {
    std::vector<std::string> vocab(257);
    for (int i = 0; i < 256; ++i) { vocab[i] = std::string(1, static_cast<char>(i)); }
    JsonConstraintCompiler compiler(vocab, {256});
    const auto grammar = compiler.compile(R"({"type":"object","properties":{"ok":{"const":true}},"required":["ok"],"additionalProperties":false})");
    auto first = grammar->create_state();
    auto second = grammar->create_state();
    std::vector<std::int32_t> mask(9);
    const auto allowed = [&](int token) { return (static_cast<std::uint32_t>(mask[token / 32]) >> (token % 32)) & 1U; };
    first->fill(mask);
    assert(allowed('{') && !allowed('x') && !allowed(256));
    for (unsigned char token : std::string("{\"ok\":true}")) {
        first->fill(mask);
        assert(allowed(token));
        first->accept(token);
    }
    first->fill(mask);
    assert(allowed(256) && !allowed('x'));
    second->fill(mask);
    assert(allowed('{') && !allowed(256));
    first->accept(256);
    for (const auto* bad : {R"({"type":"integer","multipleOf":2})", R"({"not":{}})",
                            R"({"allOf":[{},{}]})", R"({"type":"string","enum":[1]})"}) {
        bool rejected = false;
        try { (void)compiler.compile(bad); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    const auto recursive = compiler.compile(R"({"$defs":{"node":{"anyOf":[{"type":"null"},{"type":"array","items":{"$ref":"#/$defs/node"}}]}},"$ref":"#/$defs/node"})");
    auto state = recursive->create_state();
    for (unsigned char token : std::string("[null,[]]")) {
        state->fill(mask);
        assert(allowed(token));
        state->accept(token);
    }
    state->fill(mask);
    assert(allowed(256));
    std::cout << "JSON masks, completion, isolation, local recursion, and unsupported schemas passed\n";
}
