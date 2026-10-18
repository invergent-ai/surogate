#!/usr/bin/env bash
# Jinja features real chat templates use that minja does not implement. Each patch is guarded on
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

# 4. Integer string subscripts. GLM checks `m.content[0].output is defined` to distinguish
#    text tool results from structured output lists. Returning null for string[0] makes
#    the following attribute lookup throw. Jinja returns one Unicode character, whose
#    missing `output` attribute correctly fails the `defined` test.
if ! grep -q 'Jinja string subscripts index Unicode characters' "$header"; then
  python3 - "$header" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = """  Value get(const Value& key) {
    if (array_) {
"""
new = """  Value get(const Value& key) {
    if (is_string() && key.is_number_integer()) {
      // Jinja string subscripts index Unicode characters, including negative indices.
      // Stop at the requested character: GLM probes [0] even on long tool results.
      const auto& text = primitive_.get_ref<const std::string&>();
      auto index = key.get<int64_t>();
      auto continuation = [&](size_t pos) {
        return (static_cast<unsigned char>(text[pos]) & 0xc0) == 0x80;
      };
      size_t pos = index >= 0 ? 0 : text.size();
      while (index > 0 && pos < text.size()) {
        ++pos;
        while (pos < text.size() && continuation(pos)) ++pos;
        --index;
      }
      while (index < 0 && pos > 0) {
        --pos;
        while (pos > 0 && continuation(pos)) --pos;
        ++index;
      }
      if (index != 0 || pos >= text.size()) return Value();
      size_t end = pos + 1;
      while (end < text.size() && continuation(end)) ++end;
      return Value(text.substr(pos, end - pos));
    } else if (array_) {
"""
assert text.count(old) == 1, "patch_minja: the string-subscript anchor was not found"
open(path, "w").write(text.replace(old, new))
PY
fi

# 5. Python's whitespace for `trim` and argument-less `strip()`. Jinja's `trim` filter and the
#    `.strip()` / `.lstrip()` / `.rstrip()` methods are Python's str.strip(), which removes every
#    character str.isspace() accepts -- \v, \f, \x1c-\x1f, U+0085, U+00A0, U+1680, U+2000-U+200A,
#    U+2028, U+2029, U+202F, U+205F and U+3000 besides " \t\n\r". minja removed only those four, so
#    a message that starts or ends with, say, an ideographic space or a no-break space rendered
#    differently from transformers (and vLLM). Strips with explicit characters are unchanged.
if ! grep -q 'python_space_at' "$header"; then
  python3 - "$header" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
old = """static std::string strip(const std::string & s, const std::string & chars = "", bool left = true, bool right = true) {
  auto charset = chars.empty() ? " \\t\\n\\r" : chars;
"""
new = """// Byte length of the Python whitespace character (str.isspace) that starts at `pos`, else 0.
static size_t python_space_at(const std::string & s, size_t pos) {
  const auto at = [&](size_t i) { return i < s.size() ? static_cast<unsigned char>(s[i]) : 0u; };
  const unsigned b = at(pos);
  if (b == ' ' || (b >= 0x09 && b <= 0x0d) || (b >= 0x1c && b <= 0x1f)) return 1;
  if (b == 0xc2 && (at(pos + 1) == 0x85 || at(pos + 1) == 0xa0)) return 2;
  if (b == 0xe1 && at(pos + 1) == 0x9a && at(pos + 2) == 0x80) return 3;
  if (b == 0xe2 && at(pos + 1) == 0x80 && ((at(pos + 2) >= 0x80 && at(pos + 2) <= 0x8a) ||
                                           at(pos + 2) == 0xa8 || at(pos + 2) == 0xa9 || at(pos + 2) == 0xaf)) return 3;
  if (b == 0xe2 && at(pos + 1) == 0x81 && at(pos + 2) == 0x9f) return 3;
  if (b == 0xe3 && at(pos + 1) == 0x80 && at(pos + 2) == 0x80) return 3;
  return 0;
}

static std::string strip(const std::string & s, const std::string & chars = "", bool left = true, bool right = true) {
  if (chars.empty()) {
    size_t start = 0, end = s.size();
    while (left && start < end) {
      const size_t n = python_space_at(s, start);
      if (n == 0) break;
      start += n;
    }
    while (right && end > start) {
      size_t n = 0;
      for (size_t width = 1; width <= 3 && width <= end - start && n == 0; ++width) {
        if (python_space_at(s, end - width) == width) n = width;
      }
      if (n == 0) break;
      end -= n;
    }
    return s.substr(start, end - start);
  }
  auto charset = chars;
"""
assert text.count(old) == 1, "patch_minja: the strip anchor was not found"
open(path, "w").write(text.replace(old, new))
PY
  grep -q 'python_space_at' "$header" || {
    echo "patch_minja: the Python whitespace patch did not apply" >&2; exit 1; }
fi

# 6. Python's float repr. A template prints a float the way Python does: `tojson` through
#    json.dumps and `{{ x }}` / `| string` through str(), both repr() -- the shortest digits that
#    round-trip, correctly rounded, in fixed notation from 1e-4 up to 1e16 and with an exponent
#    outside it. nlohmann's grisu2 digits are sometimes not the correctly rounded ones and it
#    switches to an exponent a decade early (1e15 is "1e+15" there, "1000000000000000.0" in
#    Python); minja's str() used std::to_string ("1.000000").
if ! grep -q 'python_float_repr' "$header"; then
  python3 - "$header" <<'PY'
import sys
path = sys.argv[1]
text = open(path).read()
anchor = """  /* Python-style string repr */
  static void dump_string("""
helper = """  /* Python's repr() of a float. */
  static std::string python_float_repr(double value) {
    if (!std::isfinite(value)) return json(value).dump();
    char buffer[64];
    const auto written = std::to_chars(buffer, buffer + sizeof buffer, value, std::chars_format::scientific);
    const std::string scientific(buffer, written.ptr);
    const bool negative = scientific[0] == '-';
    const size_t exponent_at = scientific.find('e');
    std::string digits;
    for (size_t k = negative ? 1 : 0; k < exponent_at; ++k) {
      if (scientific[k] != '.') digits += scientific[k];
    }
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int exponent = std::stoi(scientific.substr(exponent_at + 1));
    std::string out = negative ? "-" : "";
    if (exponent >= 16 || exponent < -4) {
      out += digits.substr(0, 1);
      if (digits.size() > 1) out += "." + digits.substr(1);
      char tail[16];
      std::snprintf(tail, sizeof tail, "e%c%02d", exponent < 0 ? '-' : '+', exponent < 0 ? -exponent : exponent);
      return out + tail;
    }
    if (exponent < 0) return out + "0." + std::string(static_cast<size_t>(-exponent - 1), '0') + digits;
    const size_t whole = static_cast<size_t>(exponent) + 1;
    std::string integer = digits.substr(0, std::min(whole, digits.size()));
    integer.resize(whole, '0');
    return out + integer + "." + (digits.size() > whole ? digits.substr(whole) : "0");
  }

"""
assert text.count(anchor) == 1, "patch_minja: the float-repr anchor was not found"
text = text.replace(anchor, helper + anchor)
old_dump = """    } else {
      out << primitive_.dump();
    }
  }
"""
new_dump = """    } else if (primitive_.is_number_float()) {
      out << python_float_repr(primitive_.get<double>());
    } else {
      out << primitive_.dump();
    }
  }
"""
assert text.count(old_dump) == 1, "patch_minja: the dump anchor was not found"
text = text.replace(old_dump, new_dump)
old_str = """    if (is_number_float()) return std::to_string(get<double>());
"""
new_str = """    if (is_number_float()) return python_float_repr(get<double>());
"""
assert text.count(old_str) == 1, "patch_minja: the to_str anchor was not found"
text = text.replace(old_str, new_str)
include_anchor = "#include <algorithm>\n"
assert include_anchor in text, "patch_minja: the include anchor was not found"
text = text.replace(include_anchor, include_anchor + "#include <charconv>\n#include <cmath>\n", 1)
open(path, "w").write(text)
PY
  grep -q 'python_float_repr' "$header" || {
    echo "patch_minja: the float repr patch did not apply" >&2; exit 1; }
fi
