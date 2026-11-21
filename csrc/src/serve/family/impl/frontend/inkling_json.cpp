#include "family/impl/frontend/inkling_json.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace sinfer::family::frontend_internal::inkling {
namespace {

// ---------------------------------------------------------------- UTF-8 (lenient: WTF-8 surrogates decode as is)

struct Cp {
    std::uint32_t value = 0;
    std::size_t size = 1;
};

Cp decode_at(std::string_view s, std::size_t i) noexcept {
    const auto b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) { return {b0, 1}; }
    const auto cont = [&](std::size_t k) -> int {
        if (i + k >= s.size()) { return -1; }
        const auto b = static_cast<unsigned char>(s[i + k]);
        return (b & 0xC0) == 0x80 ? (b & 0x3F) : -1;
    };
    if ((b0 & 0xE0) == 0xC0) {
        const int c1 = cont(1);
        if (c1 >= 0) { return {static_cast<std::uint32_t>(((b0 & 0x1F) << 6) | c1), 2}; }
    } else if ((b0 & 0xF0) == 0xE0) {
        const int c1 = cont(1), c2 = cont(2);
        if (c1 >= 0 && c2 >= 0) { return {static_cast<std::uint32_t>(((b0 & 0x0F) << 12) | (c1 << 6) | c2), 3}; }
    } else if ((b0 & 0xF8) == 0xF0) {
        const int c1 = cont(1), c2 = cont(2), c3 = cont(3);
        if (c1 >= 0 && c2 >= 0 && c3 >= 0) {
            return {static_cast<std::uint32_t>(((b0 & 0x07) << 18) | (c1 << 12) | (c2 << 6) | c3), 4};
        }
    }
    return {0xFFFD, 1};
}

void append_utf8(std::string& out, std::uint32_t cp) {
    if (cp < 0x80) {
        out.push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {  // surrogates included (WTF-8), as Python keeps them in a str
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

// Python 3.14 (Unicode 16.0) str.isspace(), read from the reference image.
bool py_isspace(std::uint32_t cp) noexcept {
    if (cp <= 0x20) { return (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x20); }
    switch (cp) {
    case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029: case 0x202F: case 0x205F: case 0x3000: return true;
    default: return cp >= 0x2000 && cp <= 0x200A;
    }
}

// The zero of every Unicode 16.0 decimal-digit (Nd) run, read from the reference image's unicodedata: each run is ten
// consecutive code points, and these are exactly the characters Python's int() / float() accept as digits.
constexpr std::array<std::uint32_t, 76> kDigitZeros = {
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66, 0xde6, 0xe50, 0xed0,
    0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90, 0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620,
    0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10, 0x104a0, 0x10d30, 0x10d40, 0x11066, 0x110f0, 0x11136,
    0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650, 0x116c0, 0x116d0, 0x116da, 0x11730, 0x118e0, 0x11950, 0x11bf0,
    0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16130, 0x16a60, 0x16ac0, 0x16b50, 0x16d70, 0x1ccf0, 0x1d7ce, 0x1d7d8,
    0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e5f1, 0x1e950, 0x1fbf0};

int decimal_value(std::uint32_t cp) noexcept {
    if (cp >= '0' && cp <= '9') { return static_cast<int>(cp - '0'); }
    if (cp < 0x660) { return -1; }
    for (const std::uint32_t zero : kDigitZeros) {
        if (cp >= zero && cp < zero + 10) { return static_cast<int>(cp - zero); }
    }
    return -1;
}

// ---------------------------------------------------------------- numbers

std::string canonical_int(std::string_view sign_and_digits) {
    bool negative = false;
    std::size_t i = 0;
    if (i < sign_and_digits.size() && (sign_and_digits[i] == '-' || sign_and_digits[i] == '+')) {
        negative = sign_and_digits[i] == '-';
        ++i;
    }
    while (i + 1 < sign_and_digits.size() && sign_and_digits[i] == '0') { ++i; }
    std::string out(sign_and_digits.substr(i));
    if (out.empty() || out == "0") { return "0"; }
    return negative ? "-" + out : out;
}

// Compare two non-negative decimal strings without leading zeros.
int compare_magnitude(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) { return a.size() < b.size() ? -1 : 1; }
    const int c = a.compare(b);
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

double strtod_c(const std::string& ascii) {
    // std::from_chars is correctly rounded and locale-free, as CPython's float parser is.
    double value = 0.0;
    const char* begin = ascii.data();
    const char* end = begin + ascii.size();
    if (begin != end && *begin == '+') { ++begin; }
    const auto result = std::from_chars(begin, end, value, std::chars_format::general);
    if (result.ec == std::errc::result_out_of_range) {
        // from_chars reports overflow / underflow without a value; Python gives inf or the rounded tiny value.
        return std::strtod(ascii.c_str(), nullptr);
    }
    return value;
}

// ---------------------------------------------------------------- json.loads

class Loader {
public:
    explicit Loader(std::string_view text) : s_(text) {}

    PyValue run() {
        skip_ws();
        PyValue v = value();
        skip_ws();
        if (pos_ != s_.size()) { fail("Extra data"); }
        return v;
    }

private:
    std::string_view s_;
    std::size_t pos_ = 0;
    int depth_ = 0;

    [[noreturn]] void fail(const char* what) const {
        throw RequestError(std::string(what) + " (char " + std::to_string(pos_) + ")");
    }
    void skip_ws() {
        while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\t' || s_[pos_] == '\n' || s_[pos_] == '\r')) { ++pos_; }
    }
    bool literal(std::string_view word) {
        if (s_.substr(pos_, word.size()) == word) {
            pos_ += word.size();
            return true;
        }
        return false;
    }

    PyValue value() {
        if (pos_ >= s_.size()) { fail("Expecting value"); }
        const char c = s_[pos_];
        if (c == '"') { return PyValue::from_string(string_body()); }
        if (c == '{') { return object(); }
        if (c == '[') { return array(); }
        if (c == 'n' && literal("null")) { return PyValue::null(); }
        if (c == 't' && literal("true")) { return PyValue::from_bool(true); }
        if (c == 'f' && literal("false")) { return PyValue::from_bool(false); }
        if (c == 'N' && literal("NaN")) { return PyValue::from_float(std::nan("")); }
        if (c == 'I' && literal("Infinity")) { return PyValue::from_float(HUGE_VAL); }
        if (c == '-' && s_.substr(pos_, 9) == "-Infinity") {
            pos_ += 9;
            return PyValue::from_float(-HUGE_VAL);
        }
        return number();
    }

    PyValue number() {
        const std::size_t start = pos_;
        std::size_t i = pos_;
        if (i < s_.size() && s_[i] == '-') { ++i; }
        if (i >= s_.size() || s_[i] < '0' || s_[i] > '9') { fail("Expecting value"); }
        if (s_[i] == '0') {
            ++i;
        } else {
            while (i < s_.size() && s_[i] >= '0' && s_[i] <= '9') { ++i; }
        }
        bool is_float = false;
        if (i + 1 < s_.size() && s_[i] == '.' && s_[i + 1] >= '0' && s_[i + 1] <= '9') {
            is_float = true;
            i += 2;
            while (i < s_.size() && s_[i] >= '0' && s_[i] <= '9') { ++i; }
        }
        if (i < s_.size() && (s_[i] == 'e' || s_[i] == 'E')) {
            std::size_t j = i + 1;
            if (j < s_.size() && (s_[j] == '+' || s_[j] == '-')) { ++j; }
            if (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') {
                is_float = true;
                while (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') { ++j; }
                i = j;
            }
        }
        pos_ = i;
        const std::string lexeme(s_.substr(start, i - start));
        if (is_float) { return PyValue::from_float(strtod_c(lexeme)); }
        const std::size_t digits = lexeme.size() - (lexeme[0] == '-' ? 1 : 0);
        if (digits > 4300) { fail("Exceeds the limit (4300 digits) for integer string conversion"); }
        return PyValue::from_int_text(canonical_int(lexeme));
    }

    unsigned hex4() {
        if (pos_ + 4 > s_.size()) { fail("Invalid \\uXXXX escape"); }
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s_[pos_ + static_cast<std::size_t>(k)];
            v <<= 4;
            if (c >= '0' && c <= '9') { v |= static_cast<unsigned>(c - '0'); }
            else if (c >= 'a' && c <= 'f') { v |= static_cast<unsigned>(c - 'a' + 10); }
            else if (c >= 'A' && c <= 'F') { v |= static_cast<unsigned>(c - 'A' + 10); }
            else { fail("Invalid \\uXXXX escape"); }
        }
        pos_ += 4;
        return v;
    }

    std::string string_body() {
        ++pos_;  // opening quote
        std::string out;
        while (true) {
            if (pos_ >= s_.size()) { fail("Unterminated string starting at"); }
            const char c = s_[pos_];
            if (c == '"') {
                ++pos_;
                return out;
            }
            if (static_cast<unsigned char>(c) < 0x20) { fail("Invalid control character at"); }
            if (c != '\\') {
                out.push_back(c);
                ++pos_;
                continue;
            }
            ++pos_;
            if (pos_ >= s_.size()) { fail("Unterminated string starting at"); }
            const char e = s_[pos_++];
            switch (e) {
            case '"': out.push_back('"'); break;
            case '\\': out.push_back('\\'); break;
            case '/': out.push_back('/'); break;
            case 'b': out.push_back('\b'); break;
            case 'f': out.push_back('\f'); break;
            case 'n': out.push_back('\n'); break;
            case 'r': out.push_back('\r'); break;
            case 't': out.push_back('\t'); break;
            case 'u': {
                std::uint32_t cp = hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF && pos_ + 6 <= s_.size() && s_[pos_] == '\\' && s_[pos_ + 1] == 'u') {
                    const std::size_t save = pos_;
                    pos_ += 2;
                    const std::uint32_t low = hex4();
                    if (low >= 0xDC00 && low <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                    } else {
                        pos_ = save;
                    }
                }
                append_utf8(out, cp);
                break;
            }
            default: fail("Invalid \\escape");
            }
        }
    }

    PyValue object() {
        if (++depth_ > 1000) { fail("maximum recursion depth exceeded"); }
        ++pos_;
        PyValue obj = PyValue::object();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            --depth_;
            return obj;
        }
        while (true) {
            if (pos_ >= s_.size() || s_[pos_] != '"') { fail("Expecting property name enclosed in double quotes"); }
            std::string key = string_body();
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':') { fail("Expecting ':' delimiter"); }
            ++pos_;
            skip_ws();
            PyValue v = value();
            obj.set(std::move(key), std::move(v));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                --depth_;
                return obj;
            }
            if (pos_ >= s_.size() || s_[pos_] != ',') { fail("Expecting ',' delimiter"); }
            ++pos_;
            skip_ws();
        }
    }

    PyValue array() {
        if (++depth_ > 1000) { fail("maximum recursion depth exceeded"); }
        ++pos_;
        PyValue arr = PyValue::array();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            --depth_;
            return arr;
        }
        while (true) {
            arr.items.push_back(value());
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                --depth_;
                return arr;
            }
            if (pos_ >= s_.size() || s_[pos_] != ',') { fail("Expecting ',' delimiter"); }
            ++pos_;
            skip_ws();
        }
    }
};

// ---------------------------------------------------------------- dumps

void dump(std::string& out, const PyValue& v, const PyDumpStyle& style) {
    switch (v.kind) {
    case PyValue::Kind::Null: out += "null"; return;
    case PyValue::Kind::Bool: out += v.boolean ? "true" : "false"; return;
    case PyValue::Kind::Int: out += v.text; return;
    case PyValue::Kind::Float:
        if (!std::isfinite(v.number)) {
            if (!style.allow_nan) { throw RequestError("Out of range float values are not JSON compliant"); }
            out += std::isnan(v.number) ? "NaN" : (v.number > 0 ? "Infinity" : "-Infinity");
            return;
        }
        out += py_float_repr(v.number);
        return;
    case PyValue::Kind::String: py_json_quote(out, v.text, style.ensure_ascii); return;
    case PyValue::Kind::Array:
        out.push_back('[');
        for (std::size_t i = 0; i < v.items.size(); ++i) {
            if (i) { out += style.item_separator; }
            dump(out, v.items[i], style);
        }
        out.push_back(']');
        return;
    case PyValue::Kind::Object: {
        std::vector<std::size_t> order(v.keys.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        if (style.sort_keys) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v.keys[a] < v.keys[b]; });
        }
        out.push_back('{');
        for (std::size_t n = 0; n < order.size(); ++n) {
            if (n) { out += style.item_separator; }
            py_json_quote(out, v.keys[order[n]], style.ensure_ascii);
            out += style.key_separator;
            dump(out, v.items[order[n]], style);
        }
        out.push_back('}');
        return;
    }
    }
}

std::string tml_float(double value, bool legacy_floats) {
    if (!std::isfinite(value)) { throw RequestError("JSON numbers must be finite"); }
    std::string result = py_float_repr(value);
    const std::size_t e = result.find('e');
    if (e == std::string::npos) { return result; }
    const std::string mantissa = result.substr(0, e);
    const int exponent = std::stoi(result.substr(e + 1));
    if (exponent >= -5 && exponent < 16) {
        const bool negative = !mantissa.empty() && mantissa[0] == '-';
        std::string digits;
        for (const char c : mantissa) {
            if (c != '-' && c != '.') { digits.push_back(c); }
        }
        const int decimal = exponent + 1;
        std::string fixed;
        if (decimal <= 0) {
            fixed = "0." + std::string(static_cast<std::size_t>(-decimal), '0') + digits;
        } else if (static_cast<std::size_t>(decimal) >= digits.size()) {
            fixed = digits + std::string(static_cast<std::size_t>(decimal) - digits.size(), '0') + ".0";
        } else {
            fixed = digits.substr(0, static_cast<std::size_t>(decimal)) + "." + digits.substr(static_cast<std::size_t>(decimal));
        }
        return (negative ? "-" : "") + fixed;
    }
    const std::string sign = exponent >= 0 && !legacy_floats ? "+" : "";
    return mantissa + "e" + sign + std::to_string(exponent);
}

void tml_dump(std::string& out, const PyValue& v, bool legacy_floats, bool sort_keys) {
    switch (v.kind) {
    case PyValue::Kind::Object: {
        std::vector<std::size_t> order(v.keys.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        if (sort_keys) {
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return v.keys[a] < v.keys[b]; });
        }
        out.push_back('{');
        for (std::size_t n = 0; n < order.size(); ++n) {
            if (n) { out.push_back(','); }
            py_json_quote(out, v.keys[order[n]], false);
            out.push_back(':');
            tml_dump(out, v.items[order[n]], legacy_floats, sort_keys);
        }
        out.push_back('}');
        return;
    }
    case PyValue::Kind::Array:
        out.push_back('[');
        for (std::size_t i = 0; i < v.items.size(); ++i) {
            if (i) { out.push_back(','); }
            tml_dump(out, v.items[i], legacy_floats, sort_keys);
        }
        out.push_back(']');
        return;
    case PyValue::Kind::Float: out += tml_float(v.number, legacy_floats); return;
    default: {
        const PyDumpStyle compact{.ensure_ascii = false, .item_separator = ",", .key_separator = ":",
                                  .sort_keys = false, .allow_nan = false};
        dump(out, v, compact);
        return;
    }
    }
}

// Exact decimal of a non-negative integer held as base-1e9 limbs (little endian).
std::string limbs_to_decimal(const std::vector<std::uint32_t>& limbs) {
    std::string out;
    for (std::size_t i = limbs.size(); i-- > 0;) {
        std::string part = std::to_string(limbs[i]);
        if (!out.empty()) { part.insert(0, 9 - part.size(), '0'); }
        out += part;
    }
    return out.empty() ? "0" : out;
}

} // namespace

// ---------------------------------------------------------------- PyValue

PyValue PyValue::from_bool(bool value) {
    PyValue v;
    v.kind = Kind::Bool;
    v.boolean = value;
    return v;
}
PyValue PyValue::from_int_text(std::string decimal) {
    PyValue v;
    v.kind = Kind::Int;
    v.text = std::move(decimal);
    return v;
}
PyValue PyValue::from_float(double value) {
    PyValue v;
    v.kind = Kind::Float;
    v.number = value;
    return v;
}
PyValue PyValue::from_string(std::string value) {
    PyValue v;
    v.kind = Kind::String;
    v.text = std::move(value);
    return v;
}
PyValue PyValue::array() {
    PyValue v;
    v.kind = Kind::Array;
    return v;
}
PyValue PyValue::object() {
    PyValue v;
    v.kind = Kind::Object;
    return v;
}

bool PyValue::truthy() const noexcept {
    switch (kind) {
    case Kind::Null: return false;
    case Kind::Bool: return boolean;
    case Kind::Int: return text != "0";
    case Kind::Float: return number != 0.0;
    case Kind::String: return !text.empty();
    case Kind::Array: return !items.empty();
    case Kind::Object: return !keys.empty();
    }
    return false;
}

const PyValue* PyValue::find(std::string_view key) const noexcept {
    if (kind != Kind::Object) { return nullptr; }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key) { return &items[i]; }
    }
    return nullptr;
}

PyValue* PyValue::find(std::string_view key) noexcept {
    return const_cast<PyValue*>(static_cast<const PyValue&>(*this).find(key));
}

void PyValue::set(std::string key, PyValue value) {
    if (PyValue* existing = find(key)) {
        *existing = std::move(value);
        return;
    }
    keys.push_back(std::move(key));
    items.push_back(std::move(value));
}

bool PyValue::erase(std::string_view key) {
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (keys[i] == key) {
            keys.erase(keys.begin() + static_cast<std::ptrdiff_t>(i));
            items.erase(items.begin() + static_cast<std::ptrdiff_t>(i));
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------- public functions

PyValue py_json_loads(std::string_view text) { return Loader(text).run(); }

PyValue tml_roundtrip(const PyValue& value) {
    switch (value.kind) {
    case PyValue::Kind::Float:
        if (!std::isfinite(value.number)) { throw RequestError("Out of range float values are not JSON compliant"); }
        return value;
    case PyValue::Kind::Int: {
        // bounded_int: -(1 << 63) <= n < 1 << 64 stays an int; otherwise finite_float(value).
        const bool negative = value.text[0] == '-';
        const std::string_view magnitude = std::string_view(value.text).substr(negative ? 1 : 0);
        const bool in_range = negative ? compare_magnitude(magnitude, "9223372036854775808") <= 0
                                       : compare_magnitude(magnitude, "18446744073709551616") < 0;
        if (in_range) { return value; }
        const double d = strtod_c(value.text);
        if (!std::isfinite(d)) { throw RequestError("JSON number out of range"); }
        return PyValue::from_float(d);
    }
    case PyValue::Kind::Array: {
        PyValue out = PyValue::array();
        for (const PyValue& item : value.items) { out.items.push_back(tml_roundtrip(item)); }
        return out;
    }
    case PyValue::Kind::Object: {
        PyValue out = PyValue::object();
        for (std::size_t i = 0; i < value.keys.size(); ++i) { out.set(value.keys[i], tml_roundtrip(value.items[i])); }
        return out;
    }
    default: return value;
    }
}

std::string py_json_dumps(const PyValue& value, const PyDumpStyle& style) {
    std::string out;
    dump(out, value, style);
    return out;
}

std::string py_json_dumps_unicode(const PyValue& value) {
    return py_json_dumps(value, PyDumpStyle{.ensure_ascii = false});
}

std::string tml_json_dumps(const PyValue& value, bool legacy_floats, bool sort_keys) {
    std::string out;
    tml_dump(out, value, legacy_floats, sort_keys);
    return out;
}

void py_json_quote(std::string& out, std::string_view s, bool ensure_ascii) {
    static constexpr char kHex[] = "0123456789abcdef";
    const auto u4 = [&](std::uint32_t c) {
        out += "\\u";
        out.push_back(kHex[(c >> 12) & 0xF]);
        out.push_back(kHex[(c >> 8) & 0xF]);
        out.push_back(kHex[(c >> 4) & 0xF]);
        out.push_back(kHex[c & 0xF]);
    };
    out.push_back('"');
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b = static_cast<unsigned char>(s[i]);
        if (b < 0x80) {
            switch (b) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (b < 0x20 || (ensure_ascii && b == 0x7F)) {
                    u4(b);
                } else {
                    out.push_back(static_cast<char>(b));
                }
            }
            ++i;
            continue;
        }
        const Cp cp = decode_at(s, i);
        if (!ensure_ascii) {
            out.append(s.substr(i, cp.size));
        } else if (cp.value >= 0x10000) {
            const std::uint32_t v = cp.value - 0x10000;
            u4(0xD800 + (v >> 10));
            u4(0xDC00 + (v & 0x3FF));
        } else {
            u4(cp.value);
        }
        i += cp.size;
    }
    out.push_back('"');
}

std::string py_float_repr(double value) {
    if (std::isnan(value)) { return "nan"; }
    if (std::isinf(value)) { return value > 0 ? "inf" : "-inf"; }
    if (value == 0.0) { return std::signbit(value) ? "-0.0" : "0.0"; }
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
    const std::string sci(buf, res.ptr);
    const std::size_t e = sci.find('e');
    const bool negative = sci[0] == '-';
    std::string digits;
    for (std::size_t i = negative ? 1 : 0; i < e; ++i) {
        if (sci[i] != '.') { digits.push_back(sci[i]); }
    }
    const int exp10 = std::stoi(sci.substr(e + 1));
    const int decpt = exp10 + 1;
    std::string out = negative ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) { out += "." + digits.substr(1); }
        out += exp10 < 0 ? "e-" : "e+";
        std::string mag = std::to_string(exp10 < 0 ? -exp10 : exp10);
        if (mag.size() < 2) { mag.insert(0, "0"); }
        out += mag;
    } else if (decpt <= 0) {
        out += "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (static_cast<std::size_t>(decpt) >= digits.size()) {
        out += digits + std::string(static_cast<std::size_t>(decpt) - digits.size(), '0') + ".0";
    } else {
        out += digits.substr(0, static_cast<std::size_t>(decpt)) + "." + digits.substr(static_cast<std::size_t>(decpt));
    }
    return out;
}

std::string py_int_text_of_double(double value) {
    const double t = std::trunc(value);
    if (t == 0.0) { return "0"; }
    const bool negative = t < 0;
    int exponent = 0;
    const double frac = std::frexp(std::fabs(t), &exponent);  // |t| = frac * 2^exponent, frac in [0.5, 1)
    auto mantissa = static_cast<std::uint64_t>(std::ldexp(frac, 53));
    int shift = exponent - 53;
    std::string digits;
    if (shift <= 0) {
        digits = std::to_string(mantissa >> (-shift));
    } else {
        std::vector<std::uint32_t> limbs;
        while (mantissa) {
            limbs.push_back(static_cast<std::uint32_t>(mantissa % 1000000000u));
            mantissa /= 1000000000u;
        }
        while (shift-- > 0) {
            std::uint64_t carry = 0;
            for (std::uint32_t& limb : limbs) {
                const std::uint64_t x = static_cast<std::uint64_t>(limb) * 2 + carry;
                limb = static_cast<std::uint32_t>(x % 1000000000u);
                carry = x / 1000000000u;
            }
            if (carry) { limbs.push_back(static_cast<std::uint32_t>(carry)); }
        }
        digits = limbs_to_decimal(limbs);
    }
    return negative ? "-" + digits : digits;
}

std::string_view py_strip(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size()) {
        const Cp cp = decode_at(text, begin);
        if (!py_isspace(cp.value)) { break; }
        begin += cp.size;
    }
    return py_rstrip(text.substr(begin));
}

std::string_view py_rstrip(std::string_view text) {
    std::size_t end = text.size();
    while (end > 0) {
        std::size_t start = end - 1;
        while (start > 0 && (static_cast<unsigned char>(text[start]) & 0xC0) == 0x80) { --start; }
        const Cp cp = decode_at(text, start);
        if (start + cp.size != end || !py_isspace(cp.value)) { break; }
        end = start;
    }
    return text.substr(0, end);
}

std::string ascii_lower(std::string_view text) {
    std::string out(text);
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') { c = static_cast<char>(c - 'A' + 'a'); }
    }
    return out;
}

bool has_lone_surrogate(std::string_view s) noexcept {
    for (std::size_t i = 0; i + 2 < s.size(); ++i) {
        if (static_cast<unsigned char>(s[i]) == 0xED && (static_cast<unsigned char>(s[i + 1]) & 0xE0) == 0xA0) { return true; }
    }
    return false;
}

namespace {

// Unicode digits -> ASCII, the transformation CPython applies before parsing a number. nullopt on any other non-ASCII.
std::optional<std::string> ascii_number_text(std::string_view text) {
    std::string out;
    std::size_t i = 0;
    while (i < text.size()) {
        const Cp cp = decode_at(text, i);
        if (cp.value < 0x80) {
            out.push_back(static_cast<char>(cp.value));
        } else {
            const int d = decimal_value(cp.value);
            if (d < 0) { return std::nullopt; }
            out.push_back(static_cast<char>('0' + d));
        }
        i += cp.size;
    }
    return out;
}

// Underscores only between two digits, then removed (PEP 515).
std::optional<std::string> drop_underscores(std::string_view s) {
    std::string out;
    for (std::size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '_') {
            const bool ok = i > 0 && i + 1 < s.size() && std::isdigit(static_cast<unsigned char>(s[i - 1])) &&
                            std::isdigit(static_cast<unsigned char>(s[i + 1]));
            if (!ok) { return std::nullopt; }
            continue;
        }
        out.push_back(s[i]);
    }
    return out;
}

} // namespace

std::optional<std::string> py_parse_int(std::string_view text) {
    const auto ascii = ascii_number_text(py_strip(text));
    if (!ascii) { return std::nullopt; }
    std::string_view s = *ascii;
    std::string sign;
    if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
        sign = s[0] == '-' ? "-" : "";
        s.remove_prefix(1);
    }
    if (s.empty() || !std::isdigit(static_cast<unsigned char>(s[0]))) { return std::nullopt; }
    const auto digits = drop_underscores(s);
    if (!digits || digits->empty()) { return std::nullopt; }
    for (const char c : *digits) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { return std::nullopt; }
    }
    if (digits->size() > 4300) { return std::nullopt; }
    return canonical_int(sign + *digits);
}

std::optional<double> py_parse_float(std::string_view text) {
    const auto ascii = ascii_number_text(py_strip(text));
    if (!ascii) { return std::nullopt; }
    std::string_view s = *ascii;
    bool negative = false;
    std::string_view body = s;
    if (!body.empty() && (body[0] == '+' || body[0] == '-')) {
        negative = body[0] == '-';
        body.remove_prefix(1);
    }
    const std::string lower = ascii_lower(body);
    if (lower == "inf" || lower == "infinity") { return negative ? -HUGE_VAL : HUGE_VAL; }
    if (lower == "nan") { return negative ? -std::nan("") : std::nan(""); }
    const auto cleaned = drop_underscores(s);
    if (!cleaned) { return std::nullopt; }
    // [sign] (digits [. digits*] | . digits) [(e|E) [sign] digits]
    const std::string& c = *cleaned;
    std::size_t i = 0;
    if (i < c.size() && (c[i] == '+' || c[i] == '-')) { ++i; }
    std::size_t mantissa_digits = 0;
    while (i < c.size() && std::isdigit(static_cast<unsigned char>(c[i]))) { ++i, ++mantissa_digits; }
    if (i < c.size() && c[i] == '.') {
        ++i;
        while (i < c.size() && std::isdigit(static_cast<unsigned char>(c[i]))) { ++i, ++mantissa_digits; }
    }
    if (mantissa_digits == 0) { return std::nullopt; }
    if (i < c.size() && (c[i] == 'e' || c[i] == 'E')) {
        ++i;
        if (i < c.size() && (c[i] == '+' || c[i] == '-')) { ++i; }
        std::size_t exp_digits = 0;
        while (i < c.size() && std::isdigit(static_cast<unsigned char>(c[i]))) { ++i, ++exp_digits; }
        if (exp_digits == 0) { return std::nullopt; }
    }
    if (i != c.size()) { return std::nullopt; }
    return strtod_c(c);
}

} // namespace sinfer::family::frontend_internal::inkling
