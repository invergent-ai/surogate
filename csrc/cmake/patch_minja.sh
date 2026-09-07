#!/usr/bin/env bash
# Two things real chat templates use that minja does not implement. Each patch is guarded on
# its own, so applying one does not skip the other, and re-running is a no-op: FetchContent
# re-runs this step whenever it re-populates the source.
set -eu
header="include/minja/minja.hpp"
[ -f "$header" ] || { echo "patch_minja: $header not found in $(pwd)" >&2; exit 1; }

# 1. Jinja2's `undefined` test. minja has `defined` but not its negation, and the Qwen3.5/3.6/3.8
#    templates open with `{%- if enable_thinking is undefined or ... %}`. minja already evaluates
#    `defined` as "not null", so this is the one missing line.
if ! grep -q 'name == "undefined"' "$header"; then
  sed -i '/if (name == "defined") return !l.is_null();/a\              if (name == "undefined") return l.is_null();' "$header"
  grep -q 'name == "undefined"' "$header" || {
    echo "patch_minja: the 'defined' anchor was not found" >&2; exit 1; }
fi

# 2. An integer attribute. Jinja2 reads `m.content.0.type` as `m.content[0].type`, and GLM-5.3's
#    template writes it that way. minja's `.` branch parses an identifier and nothing else, so
#    the integer case goes ahead of it -- where Jinja's own grammar puts it.
if ! grep -q 'attribute_index_tok' "$header"; then
  python3 - "$header" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = """        } else if (!consumeToken(".").empty()) {
            auto identifier = parseIdentifier();
            if (!identifier) throw std::runtime_error("Expected identifier in subscript");
"""
new = """        } else if (!consumeToken(".").empty()) {
            static std::regex attribute_index_tok(R"(\\d+)");
            auto attribute_index = consumeToken(attribute_index_tok, SpaceHandling::Keep);
            if (!attribute_index.empty()) {
              auto key = std::make_shared<LiteralExpr>(get_location(),
                                                       Value(static_cast<int64_t>(std::stoll(attribute_index))));
              value = std::make_shared<SubscriptExpr>(get_location(), std::move(value),
                                                      std::move(key));
              consumeSpaces();
              continue;
            }
            auto identifier = parseIdentifier();
            if (!identifier) throw std::runtime_error("Expected identifier in subscript");
"""
assert text.count(old) == 1, "patch_minja: the numeric-attribute anchor was not found"
open(path, "w").write(text.replace(old, new))
PY
  grep -q 'attribute_index_tok' "$header" || {
    echo "patch_minja: the numeric attribute patch did not apply" >&2; exit 1; }
fi

# 3. Adjacent string literals. Jinja2 concatenates `"a" "b"` the way Python does, and Gemma 4's
#    E-series template writes a long `raise_exception(...)` message that way -- three literals on
#    three lines inside one call. minja's constant parser takes the first and then finds a string
#    where it expects `,` or `)`, so the whole template fails to *parse*, at load, even for a
#    request that never renders it.
if ! grep -q 'adjacent string literal' "$header"; then
  python3 - "$header" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = """      if (*it == '"' || *it == '\\'') {
        auto str = parseString();
        if (str) return std::make_shared<Value>(*str);
      }
"""
new = """      if (*it == '"' || *it == '\\'') {
        auto str = parseString();
        if (str) {
          // Jinja2 concatenates an adjacent string literal, as Python does. Consume the run.
          for (;;) {
            auto mark = it;
            consumeSpaces();
            if (it == end || (*it != '"' && *it != '\\'')) { it = mark; break; }
            auto next = parseString();
            if (!next) { it = mark; break; }
            *str += *next;
          }
          return std::make_shared<Value>(*str);
        }
      }
"""
assert text.count(old) == 1, "patch_minja: the string-literal anchor was not found"
open(path, "w").write(text.replace(old, new))
PY
  grep -q 'adjacent string literal' "$header" || {
    echo "patch_minja: the adjacent string literal patch did not apply" >&2; exit 1; }
fi
