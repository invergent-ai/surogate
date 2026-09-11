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
    for (const auto* bad : {R"({"not":{}})",
                            R"({"type":"string","enum":[1]})"}) {
        bool rejected = false;
        try { (void)compiler.compile(bad); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    const auto accepts = [&](const std::string& schema, const std::string& text) {
        auto probe = compiler.compile(schema)->create_state();
        for (unsigned char token : text) {
            probe->fill(mask);
            if (!allowed(token)) { return false; }
            probe->accept(token);
        }
        probe->fill(mask);
        return bool(allowed(256));
    };
    assert(accepts(R"({"type":"string","minLength":3,"maxLength":5,"pattern":"^[A-Z]+$"})", "\"ABC\""));
    assert(!accepts(R"({"type":"string","minLength":3,"maxLength":5,"pattern":"^[A-Z]+$"})", "\"AB\""));
    assert(accepts(R"({"type":"integer","minimum":2,"maximum":11,"multipleOf":3})", "9"));
    assert(!accepts(R"({"type":"integer","minimum":2,"maximum":11,"multipleOf":3})", "10"));
    assert(accepts(R"({"allOf":[{"type":"integer","minimum":3},{"maximum":5}]})", "4"));
    assert(!accepts(R"({"allOf":[{"type":"integer","minimum":3},{"maximum":5}]})", "6"));
    assert(accepts(R"({"oneOf":[{"type":"string"},{"type":"integer"}]})", "1"));
    assert(accepts(R"({"type":"string","format":"date"})", "\"2026-09-11\""));
    assert(!accepts(R"({"type":"string","format":"date"})", "\"not-a-date\""));
    assert(!accepts(R"({"type":"string","format":"date"})", "\"0000-01-01\""));
    assert(!accepts(R"({"type":"string","format":"date"})", "\"2025-02-29\""));
    assert(accepts(R"({"type":"string","format":"date"})", "\"2024-02-29\""));
    const std::vector<TokenId> drafts{'{','x', 'x'};
    auto probe = grammar->create_state();
    std::vector<int32_t> masks(9 * 4);
    probe->fill_draft_masks(drafts, masks);
    assert((static_cast<uint32_t>(masks['{' / 32]) >> ('{' % 32)) & 1U);
    assert(!((static_cast<uint32_t>(masks[9 + 'x' / 32]) >> ('x' % 32)) & 1U));
    probe->fill(mask);
    assert(allowed('{') && !allowed(256)); // draft exploration never advances the live state
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
