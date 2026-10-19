// Tool-call argument values exactly as vLLM 0.31's Qwen3 XML parser returns them (serve/vllm_tool_arguments.h).

#include "serve/vllm_tool_arguments.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <set>
#include <system_error>

namespace sinfer::serve::vllm_args {
namespace {

using Json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------------------------------------------
// Unicode, as CPython 3.12 (Unicode 15.0) classifies it.

struct CodePoint {
    char32_t value = 0;
    std::size_t size = 1;
    bool valid = false;
};

/// One UTF-8 code point at `pos`. A byte that does not start a valid sequence reads as one invalid unit; no
/// Python string holds one, so every conversion treats it as a character it does not accept.
CodePoint decode_utf8(std::string_view text, std::size_t pos) {
    const auto lead = static_cast<unsigned char>(text[pos]);
    if (lead < 0x80) { return {lead, 1, true}; }
    const std::size_t size = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : lead >= 0xC0 ? 2 : 0;
    if (size == 0 || lead > 0xF4 || pos + size > text.size()) { return {lead, 1, false}; }
    char32_t value = lead & (0x7F >> size);
    for (std::size_t i = 1; i < size; ++i) {
        const auto next = static_cast<unsigned char>(text[pos + i]);
        if ((next & 0xC0) != 0x80) { return {lead, 1, false}; }
        value = (value << 6) | (next & 0x3F);
    }
    static constexpr std::array<char32_t, 5> kSmallest{0, 0, 0x80, 0x800, 0x10000};
    if (value < kSmallest[size] || value > 0x10FFFF || (value >= 0xD800 && value <= 0xDFFF)) {
        return {lead, 1, false};
    }
    return {value, size, true};
}

/// `str.isspace()`: what `str.strip()` removes and what `int()` / `float()` read as a space.
bool python_isspace(char32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

/// `\s` of the `regex` module the converter's pattern is compiled with: `str.isspace()` without U+001C..U+001F.
bool regex_isspace(char32_t c) { return python_isspace(c) && !(c >= 0x1C && c <= 0x1F); }

/// First code point of each run of ten decimal digits (Unicode category Nd), which `int()` and `float()` read as
/// 0..9.
constexpr std::array<char32_t, 68> kDecimalRuns{
    0x30,    0x660,   0x6f0,   0x7c0,   0x966,   0x9e6,   0xa66,   0xae6,   0xb66,   0xbe6,   0xc66,   0xce6,
    0xd66,   0xde6,   0xe50,   0xed0,   0xf20,   0x1040,  0x1090,  0x17e0,  0x1810,  0x1946,  0x19d0,  0x1a80,
    0x1a90,  0x1b50,  0x1bb0,  0x1c40,  0x1c50,  0xa620,  0xa8d0,  0xa900,  0xa9d0,  0xa9f0,  0xaa50,  0xabf0,
    0xff10,  0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0,
    0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0, 0x16b50, 0x1d7ce, 0x1d7d8,
    0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950, 0x1fbf0};

int decimal_value(char32_t c) {
    const auto run = std::upper_bound(kDecimalRuns.begin(), kDecimalRuns.end(), c);
    if (run == kDecimalRuns.begin()) { return -1; }
    const char32_t start = *(run - 1);
    return c - start < 10 ? static_cast<int>(c - start) : -1;
}

bool ascii_digit(char c) { return c >= '0' && c <= '9'; }

/// `Py_ISSPACE`: the ASCII spaces `int()` and `float()` skip once the text is ASCII.
bool ascii_space(char c) { return c == ' ' || (c >= '\t' && c <= '\r'); }

char ascii_lower(char c) { return c >= 'A' && c <= 'Z' ? static_cast<char>(c - 'A' + 'a') : c; }

bool ascii_iequals(std::string_view a, std::string_view b) {
    return a.size() == b.size() &&
           std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return ascii_lower(x) == ascii_lower(y); });
}

/// `_PyUnicode_TransformDecimalAndSpaceToASCII`, which `int()` and `float()` parse: other decimal digits become
/// ASCII digits, other whitespace a space, and anything else non-ASCII ends the text with a character neither
/// accepts.
std::string numeric_ascii(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t pos = 0; pos < text.size();) {
        const CodePoint cp = decode_utf8(text, pos);
        if (cp.valid && cp.value < 127) {
            out += static_cast<char>(cp.value);
        } else if (cp.valid && python_isspace(cp.value)) {
            out += ' ';
        } else if (const int digit = cp.valid ? decimal_value(cp.value) : -1; digit >= 0) {
            out += static_cast<char>('0' + digit);
        } else {
            out += '?';
            break;
        }
        pos += cp.size;
    }
    return out;
}

// ---------------------------------------------------------------------------------------------------------------
// Numbers.

constexpr std::size_t kMaxIntDigits = 4300; // sys.get_int_max_str_digits(): int() refuses longer decimal text

std::string canonical_integer(bool negative, std::string_view digits) {
    const std::size_t first = digits.find_first_not_of('0');
    if (first == std::string_view::npos) { return "0"; }
    return (negative ? "-" : "") + std::string(digits.substr(first));
}

/// A decimal number already validated as `[-]digits[.digits][e[sign]digits]` (either digit run may be empty, not
/// both), read with correct rounding as Python reads it: beyond the range of a double it is an infinity, below it
/// a signed zero.
double decimal_to_double(bool negative, std::string_view integer, std::string_view fraction, bool exponent_negative,
                         std::string_view exponent) {
    std::string text = negative ? "-" : "";
    text += integer.empty() ? "0" : integer;
    text += '.';
    text += fraction.empty() ? "0" : fraction;
    if (!exponent.empty()) {
        text += exponent_negative ? "e-" : "e";
        text += exponent;
    }
    double value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec == std::errc::result_out_of_range) {
        // Out of range either way; the leading digit's power of ten says which way.
        long long magnitude = 0;
        const std::size_t lead = integer.find_first_not_of('0');
        if (lead != std::string_view::npos) {
            magnitude = static_cast<long long>(integer.size() - lead) - 1;
        } else {
            const std::size_t first = fraction.find_first_not_of('0');
            magnitude = first == std::string_view::npos ? 0 : -static_cast<long long>(first) - 1;
        }
        long long shift = 0;
        for (const char c : exponent) { shift = std::min<long long>(shift * 10 + (c - '0'), 1LL << 40); }
        magnitude += exponent_negative ? -shift : shift;
        value = magnitude > 0 ? std::numeric_limits<double>::infinity() : 0.0;
        return negative ? -value : value;
    }
    return value;
}

/// A JSON number token, as Python's json scanner reads it: an integer when it has neither fraction nor exponent.
struct JsonNumber {
    bool is_float = false;
    double value = 0;
    std::string integer;
};

// ---------------------------------------------------------------------------------------------------------------
// Python values: what `json.loads` and the converter produce, and what `json.dumps` writes back.

struct PyValue {
    enum class Kind { Null, False, True, Int, Float, String, Array, Object };
    Kind kind = Kind::Null;
    std::string text;   // Int: decimal digits; String: UTF-8 (WTF-8 for a lone surrogate from a \u escape)
    double number = 0;  // Float
    std::vector<PyValue> items;
    std::vector<std::pair<std::string, PyValue>> members; // dict order: first insertion, last value

    static PyValue string(std::string value) {
        PyValue out;
        out.kind = Kind::String;
        out.text = std::move(value);
        return out;
    }
    static PyValue integer(std::string digits) {
        PyValue out;
        out.kind = Kind::Int;
        out.text = std::move(digits);
        return out;
    }
    static PyValue real(double value) {
        PyValue out;
        out.kind   = Kind::Float;
        out.number = value;
        return out;
    }
    static PyValue boolean(bool value) {
        PyValue out;
        out.kind = value ? Kind::True : Kind::False;
        return out;
    }
};

void set_member(std::vector<std::pair<std::string, PyValue>>& members, std::string key, PyValue value) {
    for (auto& member : members) {
        if (member.first == key) {
            member.second = std::move(value);
            return;
        }
    }
    members.emplace_back(std::move(key), std::move(value));
}

void append_utf8(std::string& out, char32_t cp) {
    if (cp < 0x80) {
        out += static_cast<char>(cp);
    } else if (cp < 0x800) {
        out += static_cast<char>(0xC0 | (cp >> 6));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) { // a lone surrogate lands here too, as WTF-8
        out += static_cast<char>(0xE0 | (cp >> 12));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        out += static_cast<char>(0xF0 | (cp >> 18));
        out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        out += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

/// Python's `json.loads` (the C scanner, strict): ASCII-only whitespace and numbers, no control characters in
/// strings, NaN and the infinities refused here since the converter refuses non-finite results, a repeated key
/// keeping its first position with its last value.
class PythonJsonReader {
public:
    explicit PythonJsonReader(std::string_view text) : text_(text) {}

    bool read_document(PyValue& out) {
        if (text_.starts_with("\xEF\xBB\xBF")) { return false; } // "Unexpected UTF-8 BOM"
        skip_whitespace();
        if (!read_value(out, 0)) { return false; }
        skip_whitespace();
        return pos_ == text_.size();
    }

private:
    // Python raises RecursionError near its recursion limit; no model writes arguments this deep.
    static constexpr unsigned kMaxDepth = 512;

    std::string_view text_;
    std::size_t pos_ = 0;

    bool at_end() const { return pos_ >= text_.size(); }
    char peek() const { return text_[pos_]; }

    void skip_whitespace() {
        while (!at_end() && (peek() == ' ' || peek() == '\t' || peek() == '\n' || peek() == '\r')) { ++pos_; }
    }

    bool literal(std::string_view word) {
        if (text_.substr(pos_, word.size()) != word) { return false; }
        pos_ += word.size();
        return true;
    }

    bool read_value(PyValue& out, unsigned depth) {
        if (at_end() || depth > kMaxDepth) { return false; }
        switch (peek()) {
        case '"':
            out.kind = PyValue::Kind::String;
            return read_string(out.text);
        case '{':
            return read_object(out, depth);
        case '[':
            return read_array(out, depth);
        case 'n':
            out.kind = PyValue::Kind::Null;
            return literal("null");
        case 't':
            out.kind = PyValue::Kind::True;
            return literal("true");
        case 'f':
            out.kind = PyValue::Kind::False;
            return literal("false");
        default:
            // NaN, Infinity and -Infinity parse, but are not finite: the converter refuses them.
            return read_number(out);
        }
    }

    bool read_number(PyValue& out) {
        const bool negative = !at_end() && peek() == '-';
        if (negative) { ++pos_; }
        const std::size_t integer_start = pos_;
        if (!at_end() && peek() >= '1' && peek() <= '9') {
            while (!at_end() && ascii_digit(peek())) { ++pos_; }
        } else if (!at_end() && peek() == '0') {
            ++pos_;
        } else {
            return false;
        }
        const std::string_view integer = text_.substr(integer_start, pos_ - integer_start);
        std::string_view fraction;
        std::string_view exponent;
        bool exponent_negative = false;
        bool is_float          = false;
        if (pos_ + 1 < text_.size() && peek() == '.' && ascii_digit(text_[pos_ + 1])) {
            const std::size_t fraction_start = ++pos_;
            while (!at_end() && ascii_digit(peek())) { ++pos_; }
            fraction = text_.substr(fraction_start, pos_ - fraction_start);
            is_float = true;
        }
        if (!at_end() && (peek() == 'e' || peek() == 'E')) {
            const std::size_t mark = pos_++;
            if (!at_end() && (peek() == '+' || peek() == '-')) { exponent_negative = text_[pos_++] == '-'; }
            const std::size_t exponent_start = pos_;
            while (!at_end() && ascii_digit(peek())) { ++pos_; }
            if (pos_ > exponent_start) {
                exponent = text_.substr(exponent_start, pos_ - exponent_start);
                is_float = true;
            } else {
                pos_ = mark; // not part of the number
            }
        }
        if (!is_float) {
            if (integer.size() > kMaxIntDigits) { return false; }
            out = PyValue::integer(canonical_integer(negative, integer));
            return true;
        }
        const double value = decimal_to_double(negative, integer, fraction, exponent_negative, exponent);
        if (!std::isfinite(value)) { return false; }
        out = PyValue::real(value);
        return true;
    }

    bool read_hex4(char32_t& out) {
        if (pos_ + 4 > text_.size()) { return false; }
        out = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = text_[pos_ + i];
            int digit    = -1;
            if (c >= '0' && c <= '9') { digit = c - '0'; }
            if (c >= 'a' && c <= 'f') { digit = c - 'a' + 10; }
            if (c >= 'A' && c <= 'F') { digit = c - 'A' + 10; }
            if (digit < 0) { return false; }
            out = (out << 4) | static_cast<char32_t>(digit);
        }
        pos_ += 4;
        return true;
    }

    bool read_string(std::string& out) {
        ++pos_; // opening quote
        out.clear();
        for (;;) {
            if (at_end()) { return false; }
            const auto c = static_cast<unsigned char>(peek());
            if (c == '"') {
                ++pos_;
                return true;
            }
            if (c < 0x20) { return false; } // "Invalid control character"
            if (c != '\\') {
                out += static_cast<char>(c);
                ++pos_;
                continue;
            }
            ++pos_;
            if (at_end()) { return false; }
            const char escape = text_[pos_++];
            switch (escape) {
            case '"': out += '"'; break;
            case '\\': out += '\\'; break;
            case '/': out += '/'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'n': out += '\n'; break;
            case 'r': out += '\r'; break;
            case 't': out += '\t'; break;
            case 'u': {
                char32_t cp = 0;
                if (!read_hex4(cp)) { return false; }
                if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 < text_.size() && text_[pos_] == '\\' &&
                    text_[pos_ + 1] == 'u') {
                    const std::size_t mark = pos_;
                    pos_ += 2;
                    char32_t low = 0;
                    if (!read_hex4(low)) { return false; }
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else {
                        pos_ = mark; // read as its own escape next
                    }
                }
                append_utf8(out, cp);
                break;
            }
            default:
                return false; // "Invalid \escape"
            }
        }
    }

    bool read_object(PyValue& out, unsigned depth) {
        ++pos_;
        out = PyValue{};
        out.kind = PyValue::Kind::Object;
        skip_whitespace();
        if (!at_end() && peek() == '}') {
            ++pos_;
            return true;
        }
        for (;;) {
            if (at_end() || peek() != '"') { return false; }
            std::string key;
            if (!read_string(key)) { return false; }
            skip_whitespace();
            if (at_end() || peek() != ':') { return false; }
            ++pos_;
            skip_whitespace();
            PyValue value;
            if (!read_value(value, depth + 1)) { return false; }
            set_member(out.members, std::move(key), std::move(value));
            skip_whitespace();
            if (at_end()) { return false; }
            if (peek() == '}') {
                ++pos_;
                return true;
            }
            if (peek() != ',') { return false; }
            ++pos_;
            skip_whitespace();
        }
    }

    bool read_array(PyValue& out, unsigned depth) {
        ++pos_;
        out = PyValue{};
        out.kind = PyValue::Kind::Array;
        skip_whitespace();
        if (!at_end() && peek() == ']') {
            ++pos_;
            return true;
        }
        for (;;) {
            PyValue value;
            if (!read_value(value, depth + 1)) { return false; }
            out.items.push_back(std::move(value));
            skip_whitespace();
            if (at_end()) { return false; }
            if (peek() == ']') {
                ++pos_;
                return true;
            }
            if (peek() != ',') { return false; }
            ++pos_;
            skip_whitespace();
        }
    }
};

std::optional<PyValue> python_json_loads(std::string_view text) {
    PyValue value;
    if (!PythonJsonReader(text).read_document(value)) { return std::nullopt; }
    return value;
}

/// A JSON string as `json.dumps(..., ensure_ascii=False)` writes it: only `"`, `\` and control characters
/// escaped. A lone surrogate (WTF-8 from a `\u` escape) is written as its escape: Python would hold the code point,
/// which UTF-8 cannot carry.
void write_string(std::string& out, std::string_view text) {
    out += '"';
    for (std::size_t i = 0; i < text.size(); ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        switch (c) {
        case '"': out += "\\\""; continue;
        case '\\': out += "\\\\"; continue;
        case '\n': out += "\\n"; continue;
        case '\r': out += "\\r"; continue;
        case '\t': out += "\\t"; continue;
        case '\b': out += "\\b"; continue;
        case '\f': out += "\\f"; continue;
        default: break;
        }
        if (c < 0x20) {
            std::array<char, 8> buffer{};
            std::snprintf(buffer.data(), buffer.size(), "\\u%04x", c);
            out += buffer.data();
        } else if (c == 0xED && i + 2 < text.size() && (static_cast<unsigned char>(text[i + 1]) & 0xE0) == 0xA0) {
            const char32_t cp = 0xD000 | ((static_cast<unsigned char>(text[i + 1]) & 0x3F) << 6) |
                                (static_cast<unsigned char>(text[i + 2]) & 0x3F);
            std::array<char, 8> buffer{};
            std::snprintf(buffer.data(), buffer.size(), "\\u%04x", static_cast<unsigned>(cp));
            out += buffer.data();
            i += 2;
        } else {
            out += static_cast<char>(c);
        }
    }
    out += '"';
}

void write_value(std::string& out, const PyValue& value) {
    switch (value.kind) {
    case PyValue::Kind::Null: out += "null"; return;
    case PyValue::Kind::False: out += "false"; return;
    case PyValue::Kind::True: out += "true"; return;
    case PyValue::Kind::Int: out += value.text; return;
    case PyValue::Kind::Float: out += python_float_repr(value.number); return;
    case PyValue::Kind::String: write_string(out, value.text); return;
    case PyValue::Kind::Array:
        out += '[';
        for (std::size_t i = 0; i < value.items.size(); ++i) {
            if (i != 0) { out += ", "; }
            write_value(out, value.items[i]);
        }
        out += ']';
        return;
    case PyValue::Kind::Object:
        out += '{';
        for (std::size_t i = 0; i < value.members.size(); ++i) {
            if (i != 0) { out += ", "; }
            write_string(out, value.members[i].first);
            out += ": ";
            write_value(out, value.members[i].second);
        }
        out += '}';
        return;
    }
}

// ---------------------------------------------------------------------------------------------------------------
// vllm/tool_parsers/utils.py: extract_types_from_schema and coerce_to_schema_type.

void schema_types(const Json& schema, std::set<std::string>& out) {
    std::set<std::string> types;
    if (schema.is_object()) {
        if (const auto type = schema.find("type"); type != schema.end()) {
            if (type->is_string()) { types.insert(type->get<std::string>()); }
            if (type->is_array()) {
                for (const auto& item : *type) {
                    if (item.is_string()) { types.insert(item.get<std::string>()); }
                }
            }
        }
        if (const auto members = schema.find("enum");
            members != schema.end() && members->is_array() && !members->empty()) {
            for (const auto& member : *members) {
                types.insert(member.is_null()             ? "null"
                             : member.is_boolean()        ? "boolean"
                             : member.is_number_integer() ? "integer"
                             : member.is_number_float()   ? "number"
                             : member.is_string()         ? "string"
                             : member.is_array()          ? "array"
                                                          : "object");
            }
        }
        for (const char* choice : {"anyOf", "oneOf", "allOf"}) {
            if (const auto branches = schema.find(choice); branches != schema.end() && branches->is_array()) {
                for (const auto& branch : *branches) { schema_types(branch, types); }
            }
        }
    }
    if (types.empty()) { types.insert("string"); }
    out.insert(types.begin(), types.end());
}

std::string normalized_type(const std::string& type) {
    std::string name = python_strip(type);
    std::transform(name.begin(), name.end(), name.begin(), ascii_lower);
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 23> kAliases{{
        {"str", "string"},     {"text", "string"},      {"varchar", "string"}, {"char", "string"},
        {"enum", "string"},    {"int", "integer"},      {"int32", "integer"},  {"int64", "integer"},
        {"uint", "integer"},   {"uint32", "integer"},   {"uint64", "integer"}, {"long", "integer"},
        {"short", "integer"},  {"unsigned", "integer"}, {"float", "number"},   {"float32", "number"},
        {"float64", "number"}, {"double", "number"},    {"bool", "boolean"},   {"dict", "object"},
        {"arr", "array"},      {"list", "array"},       {"sequence", "array"},
    }};
    for (const auto& [alias, canonical] : kAliases) {
        if (name == alias) { return std::string(canonical); }
    }
    return name;
}

/// `coerce_to_schema_type(value, extract_types_from_schema(schema))`. `identical` (when given) is set to whether
/// Python would return the very `value` object: the string type, or nothing accepting it.
PyValue coerce(const std::string& value, const Json& schema, bool* identical = nullptr) {
    if (identical != nullptr) { *identical = false; }
    std::set<std::string> declared;
    schema_types(schema, declared);
    std::set<std::string> types;
    for (const auto& type : declared) { types.insert(normalized_type(type)); }
    for (const std::string_view candidate : {"null", "integer", "number", "boolean", "object", "array", "string"}) {
        if (!types.contains(std::string(candidate))) { continue; }
        if (candidate == "null") {
            if (ascii_iequals(value, "null")) { return PyValue{}; }
        } else if (candidate == "string") {
            if (identical != nullptr) { *identical = true; }
            return PyValue::string(value);
        } else if (candidate == "integer") {
            if (auto digits = python_int(value)) { return PyValue::integer(std::move(*digits)); }
        } else if (candidate == "number") {
            if (const auto number = python_float(value); number && std::isfinite(*number)) {
                if (*number == std::trunc(*number)) { return PyValue::integer(python_int_of_float(*number)); }
                return PyValue::real(*number);
            }
        } else if (candidate == "boolean") {
            const std::string word = python_strip(value);
            if (ascii_iequals(word, "true") || word == "1") { return PyValue::boolean(true); }
            if (ascii_iequals(word, "false") || word == "0") { return PyValue::boolean(false); }
        } else if (auto parsed = python_json_loads(value)) { // object, array
            return std::move(*parsed);
        }
    }
    if (auto parsed = python_json_loads(value)) { return std::move(*parsed); }
    if (identical != nullptr) { *identical = true; }
    return PyValue::string(value);
}

// ---------------------------------------------------------------------------------------------------------------
// vllm/parser/engine/parser_engine.py: ParserEngine._coerce_value / _coerce_dict, for arguments that came as JSON.

bool python_same_type(const PyValue& a, const PyValue& b) {
    const auto group = [](PyValue::Kind kind) {
        return kind == PyValue::Kind::True ? PyValue::Kind::False : kind; // both are bool
    };
    return group(a.kind) == group(b.kind);
}

bool python_equal(const PyValue& a, const PyValue& b) {
    if (!python_same_type(a, b)) { return false; }
    switch (a.kind) {
    case PyValue::Kind::Null: return true;
    case PyValue::Kind::False:
    case PyValue::Kind::True: return a.kind == b.kind;
    case PyValue::Kind::Int:
    case PyValue::Kind::String: return a.text == b.text;
    case PyValue::Kind::Float: return a.number == b.number;
    case PyValue::Kind::Array:
        if (a.items.size() != b.items.size()) { return false; }
        for (std::size_t i = 0; i < a.items.size(); ++i) {
            if (!python_equal(a.items[i], b.items[i])) { return false; }
        }
        return true;
    case PyValue::Kind::Object:
        if (a.members.size() != b.members.size()) { return false; }
        for (const auto& [key, value] : a.members) {
            const auto other = std::find_if(b.members.begin(), b.members.end(),
                                            [&](const auto& member) { return member.first == key; });
            if (other == b.members.end() || !python_equal(value, other->second)) { return false; }
        }
        return true;
    }
    return false;
}

bool coerce_members(std::vector<std::pair<std::string, PyValue>>& members, const Json& properties);

/// `_coerce_value`: whether `value` changed; it is replaced in place when it did.
bool coerce_value(PyValue& value, const Json& schema) {
    if (value.kind == PyValue::Kind::String) {
        bool identical = false;
        PyValue coerced = coerce(value.text, schema, &identical);
        if (identical) { return false; }
        value = std::move(coerced);
        return true;
    }
    if (value.kind == PyValue::Kind::Object) {
        const auto nested = schema.find("properties");
        return nested != schema.end() && nested->is_object() && coerce_members(value.members, *nested);
    }
    if (value.kind == PyValue::Kind::Array) {
        const auto items = schema.find("items");
        if (items == schema.end() || !items->is_object()) { return false; }
        bool changed = false;
        for (auto& item : value.items) { changed = coerce_value(item, *items) || changed; }
        return changed;
    }
    std::string as_text;
    write_value(as_text, value);
    PyValue coerced = coerce(as_text, schema);
    if (python_same_type(coerced, value) && python_equal(coerced, value)) { return false; }
    value = std::move(coerced);
    return true;
}

bool coerce_members(std::vector<std::pair<std::string, PyValue>>& members, const Json& properties) {
    bool changed = false;
    for (auto& [key, value] : members) {
        const auto property = properties.find(key);
        if (property == properties.end() || !property->is_object()) { continue; }
        changed = coerce_value(value, *property) || changed;
    }
    return changed;
}

// ---------------------------------------------------------------------------------------------------------------
// vllm/parser/qwen3.py: _PARAM_RE = <\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))

void skip_regex_space(std::string_view text, std::size_t& pos) {
    while (pos < text.size()) {
        const CodePoint cp = decode_utf8(text, pos);
        if (!cp.valid || !regex_isspace(cp.value)) { return; }
        pos += cp.size;
    }
}

bool take(std::string_view text, std::size_t& pos, std::string_view word) {
    if (text.substr(pos, word.size()) != word) { return false; }
    pos += word.size();
    return true;
}

/// `<\s*parameter\s*=` at `pos` (a '<'); on success `pos` is past the '='.
bool opening_tag(std::string_view text, std::size_t& pos) {
    std::size_t at = pos + 1;
    skip_regex_space(text, at);
    if (!take(text, at, "parameter")) { return false; }
    skip_regex_space(text, at);
    if (!take(text, at, "=")) { return false; }
    pos = at;
    return true;
}

/// `<\s*/\s*parameter\s*>` at `pos` (a '<'); on success `pos` is past the '>'.
bool closing_tag(std::string_view text, std::size_t& pos) {
    std::size_t at = pos + 1;
    skip_regex_space(text, at);
    if (!take(text, at, "/")) { return false; }
    skip_regex_space(text, at);
    if (!take(text, at, "parameter")) { return false; }
    skip_regex_space(text, at);
    if (!take(text, at, ">")) { return false; }
    pos = at;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------

std::string python_strip(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const CodePoint cp = decode_utf8(text, begin);
        if (!cp.valid || !python_isspace(cp.value)) { break; }
        begin += cp.size;
    }
    std::size_t end = text.size();
    while (end > begin) {
        std::size_t start = end - 1; // back to the lead byte of the last code point
        while (start > begin && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80 && end - start < 4) {
            --start;
        }
        const CodePoint cp = decode_utf8(text, start);
        if (!cp.valid || start + cp.size != end || !python_isspace(cp.value)) { break; }
        end = start;
    }
    return std::string(text.substr(begin, end - begin));
}

std::optional<std::string> python_int(std::string_view text) {
    const std::string s = numeric_ascii(text);
    std::size_t i = 0;
    while (i < s.size() && ascii_space(s[i])) { ++i; }
    bool negative = false;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) { negative = s[i++] == '-'; }
    if (i < s.size() && s[i] == '_') { return std::nullopt; } // may not start with an underscore
    const std::size_t start = i;
    std::string digits;
    char previous = 0;
    while (i < s.size() && (ascii_digit(s[i]) || s[i] == '_')) {
        if (s[i] == '_') {
            if (previous == '_') { return std::nullopt; }
        } else {
            digits += s[i];
        }
        previous = s[i++];
    }
    if (previous == '_' || i == start) { return std::nullopt; }
    while (i < s.size() && ascii_space(s[i])) { ++i; }
    if (i != s.size() || digits.size() > kMaxIntDigits) { return std::nullopt; }
    return canonical_integer(negative, digits);
}

std::optional<double> python_float(std::string_view text) {
    std::string s = numeric_ascii(text);
    if (s.find('_') != std::string::npos) {
        // _Py_string_to_number_with_underscores: an underscore only between two digits
        std::string kept;
        char previous = 0;
        for (const char c : s) {
            if (c == '\0') { return std::nullopt; }
            if (c == '_') {
                if (!ascii_digit(previous)) { return std::nullopt; }
            } else {
                if (previous == '_' && !ascii_digit(c)) { return std::nullopt; }
                kept += c;
            }
            previous = c;
        }
        if (previous == '_') { return std::nullopt; }
        s = std::move(kept);
    }
    std::size_t begin = 0;
    std::size_t end   = s.size();
    while (begin < end && ascii_space(s[begin])) { ++begin; }
    while (end > begin && ascii_space(s[end - 1])) { --end; }
    const std::string_view t(s.data() + begin, end - begin);
    std::size_t i       = 0;
    const bool negative = !t.empty() && t[0] == '-';
    if (!t.empty() && (t[0] == '+' || t[0] == '-')) { ++i; }
    const std::string_view word = t.substr(i);
    if (ascii_iequals(word, "inf") || ascii_iequals(word, "infinity")) {
        return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    }
    if (ascii_iequals(word, "nan")) { return std::numeric_limits<double>::quiet_NaN(); }
    const std::size_t integer_start = i;
    while (i < t.size() && ascii_digit(t[i])) { ++i; }
    const std::string_view integer = t.substr(integer_start, i - integer_start);
    std::string_view fraction;
    if (i < t.size() && t[i] == '.') {
        const std::size_t fraction_start = ++i;
        while (i < t.size() && ascii_digit(t[i])) { ++i; }
        fraction = t.substr(fraction_start, i - fraction_start);
    }
    if (integer.empty() && fraction.empty()) { return std::nullopt; }
    std::string_view exponent;
    bool exponent_negative = false;
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
        std::size_t at = i + 1;
        bool sign_negative = false;
        if (at < t.size() && (t[at] == '+' || t[at] == '-')) { sign_negative = t[at++] == '-'; }
        const std::size_t exponent_start = at;
        while (at < t.size() && ascii_digit(t[at])) { ++at; }
        if (at > exponent_start) {
            exponent          = t.substr(exponent_start, at - exponent_start);
            exponent_negative = sign_negative;
            i                 = at;
        }
    }
    if (i != t.size()) { return std::nullopt; }
    return decimal_to_double(negative, integer, fraction, exponent_negative, exponent);
}

std::string python_float_repr(double value) {
    if (value == 0) { return std::signbit(value) ? "-0.0" : "0.0"; }
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::scientific);
    const std::string_view text(buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data()));
    // "[-]d[.ddd]e(+|-)XX": the shortest digits that round-trip, as Python's repr chooses them
    const bool negative     = text.front() == '-';
    const std::size_t e     = text.find('e');
    std::string digits;
    for (const char c : text.substr(negative ? 1 : 0, e - (negative ? 1 : 0))) {
        if (c != '.') { digits += c; }
    }
    int exponent = 0;
    std::from_chars(text.data() + e + (text[e + 1] == '+' ? 2 : 1), text.data() + text.size(), exponent);
    const int point = exponent + 1; // digits read as 0.DIGITS x 10^point
    std::string out = negative ? "-" : "";
    if (point <= -4 || point > 16) {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        std::array<char, 16> tail{};
        std::snprintf(tail.data(), tail.size(), "e%+.02d", exponent);
        out += tail.data();
    } else if (point <= 0) {
        out += "0.";
        out.append(static_cast<std::size_t>(-point), '0');
        out += digits;
    } else if (static_cast<std::size_t>(point) >= digits.size()) {
        out += digits;
        out.append(static_cast<std::size_t>(point) - digits.size(), '0');
        out += ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(point));
        out += '.';
        out += digits.substr(static_cast<std::size_t>(point));
    }
    return out;
}

std::string python_int_of_float(double value) {
    if (std::fabs(value) < 9.0e18) { return std::to_string(static_cast<long long>(value)); }
    // value = mantissa * 2^shift exactly, shift > 0 here; written out in base 10^9 limbs
    int exponent           = 0;
    const double fraction  = std::frexp(std::fabs(value), &exponent);
    auto mantissa          = static_cast<std::uint64_t>(std::ldexp(fraction, 53));
    int shift              = exponent - 53;
    std::vector<std::uint32_t> limbs; // little endian, base 1e9
    while (mantissa != 0) {
        limbs.push_back(static_cast<std::uint32_t>(mantissa % 1000000000U));
        mantissa /= 1000000000U;
    }
    for (; shift > 0; --shift) {
        std::uint32_t carry = 0;
        for (auto& limb : limbs) {
            const std::uint64_t doubled = static_cast<std::uint64_t>(limb) * 2 + carry;
            limb  = static_cast<std::uint32_t>(doubled % 1000000000U);
            carry = static_cast<std::uint32_t>(doubled / 1000000000U);
        }
        if (carry != 0) { limbs.push_back(carry); }
    }
    std::string out = value < 0 ? "-" : "";
    out += std::to_string(limbs.back());
    for (std::size_t i = limbs.size() - 1; i-- > 0;) {
        std::array<char, 16> buffer{};
        std::snprintf(buffer.data(), buffer.size(), "%09u", limbs[i]);
        out += buffer.data();
    }
    return out;
}

Parameters extract_qwen3_parameters(std::string_view body) {
    Parameters out;
    std::size_t search = 0;
    while ((search = body.find('<', search)) != std::string_view::npos) {
        const std::size_t open = search;
        std::size_t at         = open;
        if (!opening_tag(body, at)) {
            search = open + 1;
            continue;
        }
        skip_regex_space(body, at);
        const std::size_t name_end = body.find('>', at);
        if (name_end == std::string_view::npos) { break; } // no tag can close any more
        const std::size_t value_begin = name_end + 1;
        // The value is the shortest text followed by a closing tag or by the next opening tag.
        std::size_t value_end = std::string_view::npos;
        std::size_t next      = value_begin;
        for (std::size_t candidate = value_begin;
             (candidate = body.find('<', candidate)) != std::string_view::npos; ++candidate) {
            std::size_t past = candidate;
            if (closing_tag(body, past)) {
                value_end = candidate;
                next      = past;
                break;
            }
            past = candidate;
            if (opening_tag(body, past)) {
                value_end = candidate;
                next      = candidate;
                break;
            }
        }
        if (value_end == std::string_view::npos) {
            search = open + 1; // unterminated: no match here, and none further on
            continue;
        }
        std::string_view value = body.substr(value_begin, value_end - value_begin);
        if (value.starts_with('\n')) { value.remove_prefix(1); }
        if (value.ends_with('\n')) { value.remove_suffix(1); }
        std::string name(body.substr(at, name_end - at));
        bool replaced = false;
        for (auto& [key, existing] : out) {
            if (key == name) {
                existing = std::string(value);
                replaced = true;
                break;
            }
        }
        if (!replaced) { out.emplace_back(std::move(name), std::string(value)); }
        search = next;
    }
    return out;
}

std::string qwen3_arguments_json(const Parameters& parameters, const Json& properties) {
    PyValue arguments;
    arguments.kind = PyValue::Kind::Object;
    for (const auto& [name, raw] : parameters) {
        const auto property = properties.is_object() ? properties.find(name) : properties.end();
        if (properties.is_object() && property != properties.end() && property->is_object()) {
            arguments.members.emplace_back(name, coerce(raw, *property));
        } else {
            arguments.members.emplace_back(name, PyValue::string(raw));
        }
    }
    std::string out;
    write_value(out, arguments);
    return out;
}

std::optional<std::string> python_json_normalize(std::string_view text) {
    const auto value = python_json_loads(text);
    if (!value) { return std::nullopt; }
    std::string out;
    write_value(out, *value);
    return out;
}

std::string python_json_dumps(const Json& value) {
    std::string out;
    switch (value.type()) {
    case Json::value_t::null: return "null";
    case Json::value_t::boolean: return value.get<bool>() ? "true" : "false";
    case Json::value_t::number_integer: return std::to_string(value.get<std::int64_t>());
    case Json::value_t::number_unsigned: return std::to_string(value.get<std::uint64_t>());
    case Json::value_t::number_float: {
        const double number = value.get<double>();
        return std::isfinite(number) ? python_float_repr(number) : "null";
    }
    case Json::value_t::string: write_string(out, value.get_ref<const std::string&>()); return out;
    case Json::value_t::array: {
        out += '[';
        bool first = true;
        for (const auto& item : value) {
            if (!first) { out += ", "; }
            first = false;
            out += python_json_dumps(item);
        }
        out += ']';
        return out;
    }
    case Json::value_t::object: {
        out += '{';
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) { out += ", "; }
            first = false;
            write_string(out, it.key());
            out += ": ";
            out += python_json_dumps(it.value());
        }
        out += '}';
        return out;
    }
    default: return "null";
    }
}

std::string fix_arg_types(std::string_view args_json, const Json& properties) {
    if (!properties.is_object() || properties.empty()) { return std::string(args_json); }
    std::optional<PyValue> args = python_json_loads(args_json);
    if (!args || args->kind != PyValue::Kind::Object) { return std::string(args_json); }
    if (!coerce_members(args->members, properties)) { return std::string(args_json); }
    std::string out;
    write_value(out, *args);
    return out;
}

std::pair<std::optional<std::string>, std::string> extract_json_name_and_args(std::string_view raw_body) {
    const std::string body = python_strip(raw_body);
    std::optional<PyValue> parsed = python_json_loads(body);
    if (!parsed || parsed->kind != PyValue::Kind::Object) { return {std::string(), body}; }
    std::optional<std::string> name = std::string();
    for (const auto& [key, value] : parsed->members) {
        if (key == "name") { name = value.kind == PyValue::Kind::String ? std::optional(value.text) : std::nullopt; }
    }
    for (const char* field : {"arguments", "parameters"}) {
        for (const auto& [key, value] : parsed->members) {
            if (key != field) { continue; }
            if (value.kind == PyValue::Kind::String) { return {name, value.text}; }
            std::string out;
            write_value(out, value);
            return {name, out};
        }
    }
    PyValue rest;
    rest.kind = PyValue::Kind::Object;
    for (const auto& member : parsed->members) {
        if (member.first != "name") { rest.members.push_back(member); }
    }
    std::string out;
    write_value(out, rest);
    return {name, out};
}

std::size_t regex_space_size(std::string_view text, std::size_t pos) {
    if (pos >= text.size()) { return 0; }
    const CodePoint cp = decode_utf8(text, pos);
    return cp.valid && regex_isspace(cp.value) ? cp.size : 0;
}

} // namespace sinfer::serve::vllm_args
