// Copyright (c) 2026 Invergent SA. SPDX-License-Identifier: Apache-2.0
// Romanian normalization adapted from the Surogate Romanian text frontend 2.2.2.
// Sentence splitting adapted from NVIDIA NeMo (Apache-2.0). See NOTICE.md.
#include "frontend.h"
#include <unicode/locid.h>
#include <unicode/normalizer2.h>
#include <unicode/regex.h>
#include <unicode/uchar.h>
#include <unicode/unistr.h>
#include <algorithm>
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>

namespace sinfer::tts {
namespace {
using U       = icu::UnicodeString;
using Match   = icu::RegexMatcher;
using Replace = std::function<std::string(Match&)>;

U unicode(const std::string& s) { return U::fromUTF8(s); }

std::string utf8(const U& s) {
    std::string r;
    s.toUTF8String(r);
    return r;
}

void check(UErrorCode status) {
    if (U_FAILURE(status))
        throw std::invalid_argument(std::string("Text processing failed: ") + u_errorName(status));
}

std::string group(Match& m, int i = 0) {
    UErrorCode e = U_ZERO_ERROR;
    auto r       = m.group(i, e);
    check(e);
    return utf8(r);
}

std::string sub(const std::string& s, const std::string& pattern, const Replace& fn,
                bool insensitive = false) {
    UErrorCode e = U_ZERO_ERROR;
    auto input   = unicode(s);
    Match m(unicode(pattern), insensitive ? UREGEX_CASE_INSENSITIVE : 0, e);
    check(e);
    m.reset(input);
    m.setTimeLimit(1000, e);
    U output;
    int32_t end = 0;
    while (m.find(e)) {
        output.append(input, end, m.start(e) - end);
        output.append(unicode(fn(m)));
        end = m.end(e);
    }
    output.append(input, end, input.length() - end);
    check(e);
    return utf8(output);
}

std::string sub(const std::string& s, const std::string& pattern, const std::string& replacement,
                bool ic = false) {
    return sub(s, pattern, [&](Match&) { return replacement; }, ic);
}

bool full(const std::string& s, const std::string& pattern) {
    UErrorCode e = U_ZERO_ERROR;
    auto input   = unicode(s);
    Match m(unicode(pattern), 0, e);
    check(e);
    m.reset(input);
    m.setTimeLimit(1000, e);
    bool result = m.matches(e);
    check(e);
    return result;
}

std::string literal(std::string s) {
    return sub(s, R"([.\\+*?\[\](){}^$|])", [](Match& m) { return "\\" + group(m); });
}

void replace(std::string& s, const std::string& a, const std::string& b) {
    size_t pos = 0;
    while ((pos = s.find(a, pos)) != std::string::npos) {
        s.replace(pos, a.size(), b);
        pos += b.size();
    }
}

std::string trim(const std::string& s) { return sub(sub(s, R"(^\s+)", ""), R"(\s+$)", ""); }

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> parts;
    size_t start = 0, end;
    while ((end = s.find(sep, start)) != std::string::npos) {
        parts.push_back(s.substr(start, end - start));
        start = end + 1;
    }
    parts.push_back(s.substr(start));
    return parts;
}

std::string join(const std::vector<std::string>& parts, const std::string& separator = " ") {
    std::string result;
    for (const auto& part : parts) {
        if (!result.empty()) result += separator;
        result += part;
    }
    return result;
}

int64_t integer(const std::string& s) {
    auto u = unicode(s);
    std::string ascii;
    for (int32_t i = 0; i < u.length();) {
        auto c = u.char32At(i);
        i += U16_LENGTH(c);
        if (c == '+' || c == '-')
            ascii += char(c);
        else {
            int d = u_charDigitValue(c);
            if (d < 0) throw std::invalid_argument("Invalid number");
            ascii += char('0' + d);
        }
    }
    try {
        // Keep recursive number expansion bounded and avoid signed overflow.
        if (ascii.size() > 18) throw std::out_of_range("number");
        size_t used = 0;
        auto n      = std::stoll(ascii, &used);
        if (used != ascii.size()) throw std::out_of_range("number");
        return n;
    } catch (...) {
        throw std::invalid_argument("Number is too large to speak; use a digit identifier");
    }
}

const std::vector<std::string> small = {
    "zero",          "unu",         "doi",           "trei",         "patru",
    "cinci",         "șase",        "șapte",         "opt",          "nouă",
    "zece",          "unsprezece",  "doisprezece",   "treisprezece", "paisprezece",
    "cincisprezece", "șaisprezece", "șaptesprezece", "optsprezece",  "nouăsprezece"};
const std::vector<std::string> tens = {"",          "",        "douăzeci",  "treizeci", "patruzeci",
                                       "cincizeci", "șaizeci", "șaptezeci", "optzeci",  "nouăzeci"};

bool needs_de(int64_t n) {
    n = std::abs(n);
    return n >= 20 && (n % 100 == 0 || n % 100 >= 20);
}

std::string feminine(std::string s, bool one = true) {
    s = sub(s, "doi(?=(?: de)?$)", "două");
    s = sub(s, "doisprezece(?=(?: de)?$)", "douăsprezece");
    return one ? sub(s, "unu(?=(?: de)?$)", "una") : s;
}

std::string words(int64_t n) {
    if (n < 0) return "minus " + words(-n);

    struct Scale {
        int64_t n;
        const char* singular;
        const char* plural;
    };

    for (auto scale :
         {Scale{1000000000000LL, "trilion", "trilioane"}, Scale{1000000000, "miliard", "miliarde"},
          Scale{1000000, "milion", "milioane"}, Scale{1000, "mie", "mii"}}) {
        if (n < scale.n) continue;
        auto count = n / scale.n, rest = n % scale.n;
        std::string leading = count == 1
                                  ? std::string(scale.n == 1000 ? "o " : "un ") + scale.singular
                                  : feminine(words(count), scale.n == 1000) +
                                        (needs_de(count) ? " de " : " ") + scale.plural;
        return leading + (rest ? " " + words(rest) : "");
    }
    if (n < 20) return small[n];
    if (n < 100) return tens[n / 10] + (n % 10 ? " și " + small[n % 10] : "");
    return (n / 100 == 1 ? std::string("o sută")
                         : (n / 100 == 2 ? "două" : small[n / 100]) + " sute") +
           (n % 100 ? " " + words(n % 100) : "");
}

std::string digits(const std::string& s) {
    std::vector<std::string> parts;
    auto u = unicode(s);
    for (int32_t i = 0; i < u.length();) {
        auto c = u.char32At(i);
        i += U16_LENGTH(c);
        if (u_charType(c) == U_DECIMAL_DIGIT_NUMBER) parts.push_back(words(u_charDigitValue(c)));
    }
    return join(parts);
}

std::pair<int64_t, std::string> number(std::string s) {
    replace(s, ".", "");
    auto parts = split(s, ',');
    return {integer(parts[0]), parts.size() > 1 ? parts[1] : ""};
}

std::string sign(const std::string& s) {
    return s.starts_with('-') ? "minus " : s.starts_with('+') ? "plus " : "";
}

std::string decimal(const std::string& s) {
    auto [n, fraction] = number(s);
    return sign(s) + words(std::abs(n)) + (fraction.empty() ? "" : " virgulă " + digits(fraction));
}

std::string quantity(int64_t n, const std::string& singular, const std::string& plural,
                     bool female = false) {
    if (n == 1) return (female ? "o " : "un ") + singular;
    auto s = words(n) + (needs_de(n) ? " de" : "");
    return (female ? feminine(s) : s) + " " + plural;
}

const std::string NUMBER = "[+-]?(?:\\d{1,3}(?:\\.\\d{3})+|\\d+)(?:,\\d+)?";
const std::vector<std::pair<std::string, std::string>> abbrev_expand = {
    {"str.", "strada"},        {"bd.", "bulevardul"},   {"nr.", "numărul"},
    {"et.", "etajul"},         {"ap.", "apartamentul"}, {"sec.", "secolul"},
    {"dl.", "domnul"},         {"dna.", "doamna"},      {"î.Hr.", "înainte de Hristos"},
    {"d.Hr.", "după Hristos"}, {"ex.", "de exemplu"},   {"etc.", "et cetera"},
    {"mp", "metri pătrați"},   {"km", "kilometri"},     {"cm", "centimetri"},
    {"kg", "kilograme"},       {"ml", "mililitri"}};
const std::vector<std::pair<std::string, std::string>> abbrev = {
    {"SRL", "S R L"}, {"SA", "S A"},       {"CUI", "C U I"},    {"TVA", "T V A"},
    {"CNP", "C N P"}, {"IBAN", "iban"},    {"BCR", "B C R"},    {"BRD", "B R D"},
    {"CEC", "cec"},   {"ING", "I N G"},    {"CNAS", "C N A S"}, {"UE", "U E"},
    {"SUA", "S U A"}, {"HR", "H R"},       {"IT", "I T"},       {"RON", "lei"},
    {"EUR", "euro"},  {"GB", "gigabaiți"}, {"MB", "megabaiți"}, {"TB", "terabaiți"}};
const std::vector<std::pair<std::string, std::string>> symbol = {
    {"%", " la sută"}, {"°C", " grade Celsius"}, {"°F", " grade Fahrenheit"},
    {"€", " euro"},    {"$", " dolari"},         {"£", " lire"},
    {"+", " plus "},   {"=", " egal "},          {"&", " și "},
    {"@", " arond "},  {"#", " diez "}};
const std::vector<std::pair<std::string, std::string>> letters = {
    {"a", "a"},     {"b", "be"},  {"c", "ce"}, {"d", "de"}, {"e", "e"},        {"f", "ef"},
    {"g", "ge"},    {"h", "haș"}, {"i", "i"},  {"j", "je"}, {"k", "ca"},       {"l", "el"},
    {"m", "em"},    {"n", "en"},  {"o", "o"},  {"p", "pe"}, {"q", "chiu"},     {"r", "er"},
    {"s", "es"},    {"t", "te"},  {"u", "u"},  {"v", "ve"}, {"w", "dublu ve"}, {"x", "ics"},
    {"y", "igrec"}, {"z", "zet"}};
const std::vector<std::pair<std::string, std::string>> translation = {
    {"ş", "ș"},  {"Ş", "Ș"},  {"ţ", "ț"},  {"Ţ", "Ț"},   {"„", "\""},
    {"”", "\""}, {"«", "\""}, {"»", "\""}, {"’", "'"},   {"‘", "'"},
    {"–", "-"},  {"—", "-"},  {"―", "-"},  {"…", "..."}, {" ", " "}};
const std::set<std::string> acronyms = {
    "AI",   "ANAF",  "API", "ASE",  "ASF",  "ATM",    "BCR",  "BIC", "BNR", "BRD",
    "CASS", "CCR",   "CEC", "CNAS", "CNP",  "CPU",    "CSS",  "CUI", "DAE", "DNA",
    "EUR",  "GPS",   "GPU", "HTML", "HTTP", "HTTPS",  "IBAN", "ING", "ISU", "IT",
    "JSON", "LASER", "LED", "NATO", "ONU",  "OTP",    "PC",   "PDF", "PIN", "POS",
    "QR",   "RADAR", "RAM", "RON",  "SEPA", "SIDA",   "SIM",  "SQL", "SRI", "SSD",
    "SUA",  "SWIFT", "TV",  "TVA",  "UE",   "UNESCO", "URL",  "USB", "USD", "XML"};
const std::map<std::string, std::vector<std::string>> currencies = {
    {"lei", {"leu", "lei", "ban", "bani"}},
    {"leu", {"leu", "lei", "ban", "bani"}},
    {"ron", {"leu", "lei", "ban", "bani"}},
    {"euro", {"euro", "euro", "cent", "cenți"}},
    {"eur", {"euro", "euro", "cent", "cenți"}},
    {"€", {"euro", "euro", "cent", "cenți"}},
    {"dolar", {"dolar", "dolari", "cent", "cenți"}},
    {"dolari", {"dolar", "dolari", "cent", "cenți"}},
    {"usd", {"dolar", "dolari", "cent", "cenți"}},
    {"$", {"dolar", "dolari", "cent", "cenți"}}};
const std::vector<std::string> months = {
    "",      "ianuarie", "februarie",  "martie",    "aprilie",   "mai",      "iunie",
    "iulie", "august",   "septembrie", "octombrie", "noiembrie", "decembrie"};
const std::map<std::string, std::string> ordinals = {{"I", "întâi"},
                                                     {"II", "al doilea"},
                                                     {"III", "al treilea"},
                                                     {"IV", "al patrulea"},
                                                     {"V", "al cincilea"},
                                                     {"VI", "al șaselea"},
                                                     {"VII", "al șaptelea"},
                                                     {"VIII", "al optulea"},
                                                     {"IX", "al nouălea"},
                                                     {"X", "al zecelea"},
                                                     {"XI", "al unsprezecelea"},
                                                     {"XII", "al doisprezecelea"},
                                                     {"XIII", "al treisprezecelea"},
                                                     {"XIV", "al paisprezecelea"},
                                                     {"XV", "al cincisprezecelea"},
                                                     {"XVI", "al șaisprezecelea"},
                                                     {"XVII", "al șaptesprezecelea"},
                                                     {"XVIII", "al optsprezecelea"},
                                                     {"XIX", "al nouăsprezecelea"},
                                                     {"XX", "al douăzecilea"},
                                                     {"XXI", "al douăzeci și unulea"}};

std::string money(const std::string& amount, const std::string& currency) {
    auto [n, fraction] = number(amount);
    auto forms         = currencies.at(casefold(currency));
    if (fraction.size() > 2) return decimal(amount) + " " + forms[1];
    auto result = sign(amount) + quantity(std::abs(n), forms[0], forms[1]);
    if (!fraction.empty()) {
        if (fraction.size() == 1) fraction += '0';
        auto cents = integer(fraction);
        if (cents) result += " și " + quantity(cents, forms[2], forms[3]);
    }
    return result;
}

std::string date(const std::string& d, const std::string& m, const std::string& y) {
    int day = integer(d), month = integer(m), year = integer(y);
    if (year < 1 || year > 9999 ||
        !std::chrono::year_month_day{std::chrono::year{year}, std::chrono::month{unsigned(month)},
                                     std::chrono::day{unsigned(day)}}
             .ok())
        throw std::invalid_argument("Invalid date");
    return (day == 1 ? "întâi" : words(day)) + " " + months.at(month) + " " + words(year);
}

std::string identifier(const std::string& s) {
    auto u = unicode(s);
    std::vector<std::string> parts;
    for (int32_t i = 0; i < u.length();) {
        auto c = u.char32At(i);
        i += U16_LENGTH(c);
        if (u_charType(c) == U_DECIMAL_DIGIT_NUMBER)
            parts.push_back(words(u_charDigitValue(c)));
        else if (u_isalpha(c)) {
            auto lower = utf8(U(u_tolower(c)));
            for (auto& [a, b] : letters)
                if (a == lower) {
                    lower = b;
                    break;
                }
            parts.push_back(lower);
        } else if (c == '+')
            parts.push_back("plus");
    }
    return join(parts);
}
} // namespace

std::string casefold(const std::string& s) { return utf8(unicode(s).foldCase()); }

std::string normalize(const std::string& text) {
    UErrorCode e = U_ZERO_ERROR;
    U normalized;
    icu::Normalizer2::getNFCInstance(e)->normalize(unicode(text), normalized, e);
    check(e);
    auto t = utf8(normalized);
    for (auto& [a, b] : translation) replace(t, a, b);
    if (full(t, "\\s*" + NUMBER + "(?:\\s*,\\s+" + NUMBER + "){2,}\\s*[.!?]?\\s*")) {
        auto items = split(sub(sub(trim(t), R"([.!?]+$)", ""), R"(,\s+)", "|"), '|');
        for (auto& item : items) item = decimal(trim(item));
        t = join(items, ". ") + ".";
    }
    std::vector<std::string> protected_values;
    auto protect = [&](const std::string& value) {
        if (protected_values.size() >= 4096)
            throw std::invalid_argument("Too many structured values");
        auto key = utf8(U(UChar32(0xE100 + protected_values.size())));
        protected_values.push_back(value);
        return key;
    };
    t = sub(t, R"(https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+)", [&](Match& m) {
        auto original = group(m), value = sub(original, R"([.,;!?]+$)", "");
        auto tail = original.substr(value.size());
        value     = sub(value, R"(^https?://)", "");
        replace(value, "www.", "ve ve ve punct ");
        for (auto& [a, b] :
             std::vector<std::pair<std::string, std::string>>{{"@", " arond "},
                                                              {".", " punct "},
                                                              {"/", " slash "},
                                                              {"-", " liniuță "},
                                                              {"_", " underscore "},
                                                              {"?", " semnul întrebării "},
                                                              {"=", " egal "}})
            replace(value, a, b);
        value = sub(value, R"(\d+)", [](Match& x) { return digits(group(x)); });
        return protect(value) + tail;
    });
    t = sub(t, R"((?<!\w)(\d{4})-(\d{2})-(\d{2})(?!\w))",
            [&](Match& m) { return protect(date(group(m, 3), group(m, 2), group(m, 1))); });
    t = sub(t, R"((?<!\w)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\w))",
            [&](Match& m) { return protect(date(group(m, 1), group(m, 2), group(m, 3))); });
    std::vector<std::string> month_names(months.begin() + 1, months.end());
    t = sub(t, "\\b(\\d{1,2})\\s+(" + join(month_names, "|") + ")\\s+(\\d{4})\\b", [&](Match& m) {
        return protect(words(integer(group(m, 1))) + " " + group(m, 2) + " " +
                       words(integer(group(m, 3))));
    });
    t = sub(t, R"((\bora\s+)?\b(\d{1,2}):(\d{2})\b)", [&](Match& m) {
        auto hour = integer(group(m, 2)), minute = integer(group(m, 3));
        if (hour > 23 || minute > 59) throw std::invalid_argument("Invalid time");
        return protect((group(m, 1).empty() ? "ora " : group(m, 1)) + words(hour) +
                       (minute == 0 ? " fix" : " și " + words(minute)));
    });
    t = sub(t, R"(\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){11,30}\b)",
            [&](Match& m) { return protect(identifier(group(m))); });
    t = sub(
        t, R"(\b(CUI|CNP|cod(?:ul)?(?: de verificare)?|comanda)\b([ :#-]{0,3})(\d(?:[\d -]*\d)?))",
        [&](Match& m) { return group(m, 1) + group(m, 2) + protect(digits(group(m, 3))); }, true);
    t = sub(
        t, R"((se termină în\s+)(\d{4})\b)",
        [&](Match& m) { return group(m, 1) + protect(digits(group(m, 2))); }, true);
    t = sub(t, R"((?<!\w)(?:\+\d[\d -]{7,}\d|0\d[\d -]{6,}\d)(?!\w))",
            [&](Match& m) { return protect(identifier(group(m))); });
    std::vector<std::string> units;
    for (auto& [k, v] : currencies) units.push_back(literal(k));
    std::stable_sort(units.begin(), units.end(),
                     [](const auto& a, const auto& b) { return a.size() > b.size(); });
    t = sub(
        t, "(?<![\\w.,])(" + NUMBER + ")\\s*(?:de\\s+)?(" + join(units, "|") + ")(?!\\w)",
        [&](Match& m) { return protect(money(group(m, 1), group(m, 2))); }, true);
    t = sub(t, R"((?<![\w.,])(\d+)\s*(?:de\s+)?(ban|bani|cent|cenți)(?!\w))", [&](Match& m) {
        bool ban = group(m, 2) == "ban" || group(m, 2) == "bani";
        return protect(
            quantity(integer(group(m, 1)), ban ? "ban" : "cent", ban ? "bani" : "cenți"));
    });
    t = sub(t, "(?<![\\w.,])(" + NUMBER + ")\\s*%",
            [&](Match& m) { return protect(decimal(group(m, 1)) + " la sută"); });
    t = sub(t, "(?<![\\w.,])(" + NUMBER + ")\\s*°([CF])\\b", [&](Match& m) {
        auto n = group(m, 1);
        return protect(decimal(n) + (n == "1" || n == "+1" || n == "-1" ? " grad " : " grade ") +
                       (group(m, 2) == "C" ? "Celsius" : "Fahrenheit"));
    });
    t = sub(t, R"(\bsec\.?\s*([IVXL]+)\b)", [&](Match& m) {
        auto it = ordinals.find(group(m, 1));
        return it == ordinals.end() ? group(m) : protect("secolul " + it->second);
    });
    auto dotted = [&](const std::string& s) {
        auto parts = split(s, '.');
        for (auto& part : parts) part = words(integer(part));
        return join(parts, " punct ");
    };
    t = sub(
        t, R"((\b(?:versiunea|versiune|version|v)\s*)(\d+(?:\.\d+){2,})\b)",
        [&](Match& m) { return group(m, 1) + protect(dotted(group(m, 2))); }, true);
    t = sub(t, R"(\b\d+(?:\.\d+){2,}\b)", [&](Match& m) {
        auto v = group(m);
        return protect(full(v, R"(\d{1,3}(?:\.\d{3})+)") ? decimal(v) : dotted(v));
    });
    for (auto& [k, v] : abbrev_expand) t = sub(t, "(?<![\\w.])" + literal(k) + "(?!\\w)", v);

    struct Unit {
        const char* name;
        const char* singular;
        const char* plural;
        bool feminine;
    };

    for (auto unit :
         {Unit{"GB", "gigabait", "gigabaiți", false}, Unit{"MB", "megabait", "megabaiți", false},
          Unit{"TB", "terabait", "terabaiți", false},
          Unit{"kilograme", "kilogram", "kilograme", true},
          Unit{"kilometri", "kilometru", "kilometri", false},
          Unit{"centimetri", "centimetru", "centimetri", false},
          Unit{"metri pătrați", "metru pătrat", "metri pătrați", false},
          Unit{"mililitri", "mililitru", "mililitri", false}}) {
        t = sub(t, "\\b(\\d+)\\s+" + literal(unit.name) + "\\b", [&](Match& m) {
            return protect(
                quantity(integer(group(m, 1)), unit.singular, unit.plural, unit.feminine));
        });
    }
    t = sub(t, R"(\b[A-ZȘȚĂÎÂ]{2,6}\b)", [&](Match& m) {
        auto v = group(m);
        if (acronyms.contains(v)) return v;
        for (auto& [a, b] : abbrev)
            if (a == v) return b;
        auto u = unicode(v);
        std::vector<std::string> chars;
        for (int32_t i = 0; i < u.length();) {
            auto c = u.char32At(i);
            i += U16_LENGTH(c);
            chars.push_back(utf8(U(c)));
        }
        return join(chars);
    });
    t = sub(t, "(?<![\\w.,])" + NUMBER + "(?!\\w|\\.\\d)",
            [&](Match& m) { return protect(decimal(group(m))); });
    for (auto& [a, b] : symbol) replace(t, a, b);
    for (size_t i = 0; i < protected_values.size(); ++i)
        replace(t, utf8(U(UChar32(0xE100 + i))), protected_values[i]);
    return trim(sub(t, R"(\s+)", " "));
}

std::size_t input_characters(const std::string& text) {
    return static_cast<std::size_t>(unicode(text).countChar32());
}

std::vector<std::vector<int32_t>> tokenize(const std::string& text, std::size_t max_characters) {
    auto input = unicode(text);
    if (input_characters(text) > max_characters)
        throw std::invalid_argument("input exceeds " + std::to_string(max_characters) + " characters");
    for (int32_t i = 0; i < input.length();) {
        auto c = input.char32At(i);
        i += U16_LENGTH(c);
        if (c == 0 || (c >= 0xE000 && c <= 0xF8FF))
            throw std::invalid_argument("input contains a reserved character");
    }
    for (auto reserved : {"<extra_id_", "<pad>", "</s>", "<unk>"})
        if (text.find(reserved) != std::string::npos)
            throw std::invalid_argument("input contains a reserved tokenizer symbol");
    auto t = normalize(sub(text, R"(\[[^\]]*\]|\*[^*]*\*)", " "));
    if (t.empty()) throw std::invalid_argument("input must contain spoken text");
    const std::size_t max_spoken = std::min<std::size_t>(4 * max_characters, max_spoken_characters);
    if (static_cast<std::size_t>(unicode(t).countChar32()) > max_spoken)
        throw std::invalid_argument("Expanded input exceeds " + std::to_string(max_spoken) +
                                    " characters");
    std::vector<std::string> pieces;
    if (split(t, ' ').size() < 45)
        pieces.push_back(t);
    else {
        // Preserve the validated NeMo sentence boundaries and capitalization.
        replace(t, "-", " ");
        replace(t, "*", "");
        auto u                             = unicode(t);
        const std::set<std::string> titles = {"mr",  "mrs", "ms",  "dr",  "prof", "sr",  "jr",
                                              "rev", "gov", "gen", "col", "lt",   "sgt", "capt"};
        int32_t start                      = 0;
        for (int32_t i = 0; i < u.length(); ++i) {
            auto c = u.charAt(i);
            if (c != '.' && c != '?' && c != '!') continue;
            if (i + 1 < u.length() && !u_isUWhiteSpace(u.char32At(i + 1))) continue;
            if (c == '.') {
                auto prefix = trim(utf8(u.tempSubStringBetween(start, i)));
                auto words  = split(prefix, ' ');
                if (titles.contains(casefold(words.back()))) continue;
            }
            auto piece = trim(utf8(u.tempSubStringBetween(start, i + 1)));
            if (!piece.empty()) pieces.push_back(piece);
            start = i + 2;
        }
        if (start < u.length()) {
            auto tail = trim(utf8(u.tempSubString(start)));
            if (!tail.empty()) pieces.push_back(tail);
        }
        for (auto& piece : pieces) {
            auto s = unicode(piece);
            auto c = s.char32At(0);
            if (!u_isupper(c)) {
                auto initial = s.tempSubString(0, U16_LENGTH(c));
                initial.toUpper(icu::Locale::getRoot());
                piece = utf8(initial) + utf8(s.tempSubString(U16_LENGTH(c)));
            }
        }
    }
    std::vector<std::vector<int32_t>> chunks;
    for (const auto& piece : pieces) {
        std::vector<int32_t> tokens;
        for (unsigned char c : piece) tokens.push_back(c + 99);
        tokens.push_back(97);
        tokens.push_back(3358);
        if (tokens.size() > 4096) throw std::invalid_argument("Input exceeds the text-chunk limit");
        chunks.push_back(std::move(tokens));
    }
    if (chunks.empty() || chunks.size() > std::max<std::size_t>(64, max_characters / 16))
        throw std::invalid_argument("Input exceeds the text-chunk limit");
    return chunks;
}
} // namespace sinfer::tts
