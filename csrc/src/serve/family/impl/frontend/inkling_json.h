#pragma once

// Inkling-2 frontend (ENGINE-INKLING lane f2): JSON with Python's semantics.
//
// vLLM's Inkling renderer and parser are Python and lean on Python's json module and number types: a request body is
// json.loads'ed (integers of any size kept exactly, NaN / Infinity accepted, a repeated key keeps its first position and
// last value), tool-call arguments are re-spelled with the reference's own float form (`_tml_json_dumps`), and a
// coerced tool call is re-serialised with json.dumps's default separators and float repr. Token-exact rendering and
// parse-exact output need those spellings byte for byte, which nlohmann::json (doubles for big integers, its own float
// form, sorted or unordered objects) cannot give. This file is that subset of Python, nothing more.

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::family::frontend_internal::inkling {

/// A request the reference answers with an error (HTTP 400): its ValueError / TypeError / pydantic ValidationError.
class RequestError : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/// A JSON value as Python holds it after json.loads.
struct PyValue {
    enum class Kind : std::uint8_t { Null, Bool, Int, Float, String, Array, Object };
    Kind kind = Kind::Null;
    bool boolean = false;
    double number = 0.0;
    /// Int: canonical decimal ("0", "-12", no leading zeros, any length). String: UTF-8 bytes; a lone surrogate from a
    /// \uD8xx escape is kept as its 3-byte (WTF-8) form, as Python keeps it in a str.
    std::string text;
    std::vector<PyValue> items;      // Array items; Object values, parallel to keys
    std::vector<std::string> keys;   // Object keys in insertion order (unique)

    static PyValue null() { return {}; }
    static PyValue from_bool(bool value);
    static PyValue from_int_text(std::string decimal);
    static PyValue from_float(double value);
    static PyValue from_string(std::string value);
    static PyValue array();
    static PyValue object();

    [[nodiscard]] bool is_null() const noexcept { return kind == Kind::Null; }
    [[nodiscard]] bool is_string() const noexcept { return kind == Kind::String; }
    [[nodiscard]] bool is_object() const noexcept { return kind == Kind::Object; }
    [[nodiscard]] bool is_array() const noexcept { return kind == Kind::Array; }
    /// Python truthiness: None, False, 0, 0.0, "", [] and {} are false.
    [[nodiscard]] bool truthy() const noexcept;

    /// Object lookup; nullptr when absent (Python dict.get(key) is None only for a missing key or a null value).
    [[nodiscard]] const PyValue* find(std::string_view key) const noexcept;
    [[nodiscard]] PyValue* find(std::string_view key) noexcept;
    /// dict[key] = value: replaces in place, or appends.
    void set(std::string key, PyValue value);
    /// dict.pop(key, None); returns whether it was present.
    bool erase(std::string_view key);
};

/// Python's `json.loads(text)` with default arguments (NaN / Infinity / -Infinity accepted; integers exact; floats by
/// correctly rounded conversion; strict strings: no raw control characters). Throws RequestError on invalid JSON.
[[nodiscard]] PyValue py_json_loads(std::string_view text);

/// The reference's `_tml_json_loads(json.dumps(value, allow_nan=False))`: integers outside [-2**63, 2**64) become floats,
/// non-finite floats are refused. Throws RequestError.
[[nodiscard]] PyValue tml_roundtrip(const PyValue& value);

struct PyDumpStyle {
    bool ensure_ascii = true;
    std::string_view item_separator = ", ";
    std::string_view key_separator = ": ";
    bool sort_keys = false;
    bool allow_nan = true;
};
/// Python's `json.dumps(value, ...)`.
[[nodiscard]] std::string py_json_dumps(const PyValue& value, const PyDumpStyle& style = {});
/// json.dumps(value, ensure_ascii=False) -- the spelling vLLM's tool-call paths use.
[[nodiscard]] std::string py_json_dumps_unicode(const PyValue& value);
/// The reference renderer's `_tml_json_dumps(value, legacy_floats=..., sort_keys=...)`.
[[nodiscard]] std::string tml_json_dumps(const PyValue& value, bool legacy_floats = true, bool sort_keys = true);
/// JSON string literal as json.dumps writes it.
void py_json_quote(std::string& out, std::string_view utf8, bool ensure_ascii);

/// Python's repr(float).
[[nodiscard]] std::string py_float_repr(double value);
/// Python's str(int(value)) for a finite double (the exact integer part, any size).
[[nodiscard]] std::string py_int_text_of_double(double value);

/// Python's int(text) (base 10): surrounding Unicode whitespace, a sign, Unicode decimal digits, single underscores
/// between digits, at most 4300 digits. nullopt where Python raises ValueError.
[[nodiscard]] std::optional<std::string> py_parse_int(std::string_view text);
/// Python's float(text): as int() plus a fraction, an exponent, inf / infinity / nan in any case. nullopt on ValueError.
[[nodiscard]] std::optional<double> py_parse_float(std::string_view text);
/// Python's str.strip() (Unicode whitespace as str.isspace) and str.rstrip().
[[nodiscard]] std::string_view py_strip(std::string_view text);
[[nodiscard]] std::string_view py_rstrip(std::string_view text);
/// Python's str.lower() for the ASCII letters (the only ones the parser compares against).
[[nodiscard]] std::string ascii_lower(std::string_view text);
/// True when the text holds a lone surrogate (WTF-8), which Python cannot hand to the tokenizer.
[[nodiscard]] bool has_lone_surrogate(std::string_view utf8) noexcept;

} // namespace sinfer::family::frontend_internal::inkling
