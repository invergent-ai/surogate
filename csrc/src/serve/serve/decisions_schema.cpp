#include "serve/decisions_schema.h"

#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"
#include "serve/decisions_thinking.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

namespace sinfer::serve {

// The floor is the engine's dispatch threshold, not a number of this layer's own.
static_assert(kDecisionMinPrefillTokens == static_cast<std::size_t>(ops::detail::kSparseMoePrefillGgmlKMin));
static_assert(kDecisionMinPrefillTokens >= static_cast<std::size_t>(ops::detail::kSparseMoePrefillQ4Q6Min));
static_assert(kDecisionMinPrefillTokens >= static_cast<std::size_t>(ops::detail::kSparseMoePrefillQ4Q5Min));
static_assert(kDecisionMinPrefillTokens >= static_cast<std::size_t>(ops::detail::kSparseMoePrefillW8W8Min));

namespace {

/// RFC 6901 escaping of one reference token.
std::string pointer_token(const std::string& key) {
    std::string out;
    for (const char c : key) {
        if (c == '~') { out += "~0"; } else if (c == '/') { out += "~1"; } else { out.push_back(c); }
    }
    return out;
}

[[noreturn]] void invalid(std::string message, std::string param) {
    throw ApiException(ApiError{.status  = 400,
                                .type    = "invalid_request_error",
                                .message = std::move(message),
                                .param   = std::move(param),
                                .code    = "invalid_decisions_request"});
}

/// Python's `json.dumps` string escaping with `ensure_ascii=False`: only the quote, the
/// backslash and the C0 controls are escaped; everything else, non-ASCII included, is raw.
void dump_string(const std::string& value, std::string& out) {
    out.push_back('"');
    for (const unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        case '\b': out += "\\b"; break;
        case '\f': out += "\\f"; break;
        default:
            if (c < 0x20) {
                std::array<char, 8> buffer{};
                std::snprintf(buffer.data(), buffer.size(), "\\u%04x", static_cast<unsigned>(c));
                out += buffer.data();
            } else {
                out.push_back(static_cast<char>(c));
            }
        }
    }
    out.push_back('"');
}

void dump_value(const OrderedJson& value, std::string& out, const JsonLiterals& literals,
                const std::string& pointer) {
    switch (value.type()) {
    case OrderedJson::value_t::null: out += "null"; return;
    case OrderedJson::value_t::boolean: out += value.get<bool>() ? "true" : "false"; return;
    case OrderedJson::value_t::number_integer:
        out += std::to_string(value.get<std::int64_t>());
        return;
    case OrderedJson::value_t::number_unsigned:
        out += std::to_string(value.get<std::uint64_t>());
        return;
    case OrderedJson::value_t::number_float: {
        // An integer literal too wide for 64 bits arrives here as a double; Python kept it.
        if (!literals.empty()) {
            const auto found = literals.find(pointer);
            if (found != literals.end()) {
                out += found->second;
                return;
            }
        }
        out += python_float_repr(value.get<double>());
        return;
    }
    case OrderedJson::value_t::string: dump_string(value.get_ref<const std::string&>(), out); return;
    case OrderedJson::value_t::array: {
        out.push_back('[');
        std::size_t index = 0;
        for (const auto& item : value) {
            if (index > 0) { out += ", "; }
            dump_value(item, out, literals, literals.empty() ? pointer : pointer + "/" + std::to_string(index));
            ++index;
        }
        out.push_back(']');
        return;
    }
    case OrderedJson::value_t::object: {
        out.push_back('{');
        bool first = true;
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (!first) { out += ", "; }
            first = false;
            dump_string(it.key(), out);
            out += ": ";
            dump_value(it.value(), out, literals, literals.empty() ? pointer : pointer + "/" + pointer_token(it.key()));
        }
        out.push_back('}');
        return;
    }
    case OrderedJson::value_t::binary:
    case OrderedJson::value_t::discarded:
        break;
    }
    throw std::invalid_argument("value cannot be rendered as JSON text");
}

/// Walks the body once before it is parsed into a value, for two things the DOM loses:
/// nlohmann keeps the last value of a repeated key, silently, and the two objects whose keys
/// *are* the contract -- the question set and a choice question's criteria -- must refuse a
/// repeat rather than lose one (other objects keep Python's last-wins reading of the state);
/// and an integer literal too wide for 64 bits, which nlohmann turns into a double while
/// Python keeps it exact, is recorded here with its raw text and the JSON pointer of its place.
class DuplicateKeyScanner final : public nlohmann::json_sax<OrderedJson> {
public:
    std::string duplicate; // the first offending repeated key
    std::string scope;
    JsonLiterals literals;

    bool null() override { return scalar(); }
    bool boolean(bool) override { return scalar(); }
    bool number_integer(number_integer_t) override { return scalar(); }
    bool number_unsigned(number_unsigned_t) override { return scalar(); }
    bool number_float(number_float_t, const string_t& raw) override {
        // nlohmann only reports an integer-looking literal as a float when it overflowed.
        if (raw.find_first_of(".eE") == std::string::npos) { literals[pointer()] = raw; }
        return scalar();
    }
    bool string(string_t&) override { return scalar(); }
    bool binary(binary_t&) override { return scalar(); }
    bool start_object(std::size_t) override {
        Frame frame;
        frame.object = true;
        frames_.push_back(std::move(frame));
        return true;
    }
    bool key(string_t& value) override {
        Frame& frame = frames_.back();
        frame.key    = value;
        if (!frame.seen.insert(value).second) {
            if (in_questions()) {
                scope     = "questions";
                duplicate = value;
                return false;
            }
            if (in_criteria()) {
                scope     = "questions." + frames_[1].key + ".criteria";
                duplicate = value;
                return false;
            }
            // Anywhere else Python keeps the last value of a repeated key, and so does the DOM:
            // forget any wide literal the earlier value left at this key or below it, or its
            // text would be written in place of the value that won.
            const std::string at = pointer();
            std::erase_if(literals, [&](const auto& entry) {
                return entry.first == at || entry.first.starts_with(at + "/");
            });
        }
        return true;
    }
    bool end_object() override {
        frames_.pop_back();
        return scalar();
    }
    bool start_array(std::size_t) override {
        frames_.push_back(Frame{});
        return true;
    }
    bool end_array() override {
        frames_.pop_back();
        return scalar();
    }
    bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override {
        return false;
    }

private:
    struct Frame {
        bool object = false;
        std::string key;   // the key whose value is being parsed (objects)
        std::size_t index = 0; // the index of the value being parsed (arrays)
        std::set<std::string> seen;
    };
    /// A value has completed: the enclosing array moves to its next index.
    bool scalar() {
        if (!frames_.empty() && !frames_.back().object) { ++frames_.back().index; }
        return true;
    }
    [[nodiscard]] std::string pointer() const {
        std::string out;
        for (const Frame& frame : frames_) {
            out += "/";
            out += frame.object ? pointer_token(frame.key) : std::to_string(frame.index);
        }
        return out;
    }
    // frames_[i].key is the key whose value frames_[i + 1] is.
    [[nodiscard]] bool in_questions() const {
        return frames_.size() == 2 && frames_[0].object && frames_[0].key == "questions";
    }
    [[nodiscard]] bool in_criteria() const {
        return frames_.size() == 4 && frames_[0].object && frames_[0].key == "questions" &&
               frames_[1].object && frames_[2].object && frames_[2].key == "criteria";
    }
    std::vector<Frame> frames_;
};

/// The text a question or option is shown as: strings as they are, anything else as the
/// same JSON text the state is rendered with.
std::string rendered_text(const OrderedJson& value, const JsonLiterals& literals, const std::string& pointer) {
    if (value.is_string()) { return value.get<std::string>(); }
    return python_json_dumps(value, literals, pointer);
}

void require_text_value(const OrderedJson& value, const std::string& what, const std::string& param) {
    if (value.is_null() || value.is_binary() || value.is_discarded()) {
        invalid(what + " must be a string (recommended), an object or an array", param);
    }
}

sinfer::product::media_acquire::Source parse_image_source(const OrderedJson& item, std::size_t index) {
    const std::string param = "images";
    std::string url;
    if (item.is_string()) {
        url = item.get<std::string>();
    } else if (item.is_object() && item.contains("url") && item.at("url").is_string()) {
        url = item.at("url").get<std::string>();
    } else {
        invalid("images[" + std::to_string(index) + "] must be a data URL string or an object with a string url", param);
    }
    if (url.empty()) { invalid("images[" + std::to_string(index) + "] URL must not be empty", param); }
    sinfer::product::media_acquire::Source source;
    source.value = std::move(url);
    if (source.value.starts_with("data:")) {
        source.kind = sinfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = sinfer::product::media_acquire::SourceKind::Url;
    } else {
        invalid("images[" + std::to_string(index) + "] must be a data URL or an HTTP(S) URL", param);
    }
    return source;
}

DecisionQuestion parse_question(const std::string& name, const OrderedJson& spec, const JsonLiterals& literals) {
    const std::string param   = "questions." + name;
    const std::string pointer = "/questions/" + pointer_token(name);
    if (!spec.is_object()) { invalid("question '" + name + "' must be an object", param); }
    if (!spec.contains("type") || !spec.at("type").is_string()) {
        invalid("question '" + name + "' needs a string type: choice, noul or score", param + ".type");
    }
    const std::string type = spec.at("type").get<std::string>();
    DecisionQuestion question;
    question.name = name;
    if (type == "choice") {
        question.kind = DecisionKind::Choice;
    } else if (type == "noul") {
        question.kind = DecisionKind::Noul;
    } else if (type == "score") {
        question.kind = DecisionKind::Score;
    } else {
        invalid("question '" + name + "' has unknown type '" + type + "'; expected choice, noul or score",
                param + ".type");
    }
    if (!spec.contains("instructions")) {
        invalid("question '" + name + "' needs instructions", param + ".instructions");
    }
    require_text_value(spec.at("instructions"), "question '" + name + "' instructions", param + ".instructions");
    question.instructions = rendered_text(spec.at("instructions"), literals, pointer + "/instructions");
    if (!spec.contains("criteria")) { invalid("question '" + name + "' needs criteria", param + ".criteria"); }
    const OrderedJson& criteria = spec.at("criteria");
    const std::string criteria_param = param + ".criteria";
    switch (question.kind) {
    case DecisionKind::Choice: {
        if (!criteria.is_object()) {
            invalid("choice question '" + name + "' criteria must be an object of option key to description",
                    criteria_param);
        }
        for (auto it = criteria.begin(); it != criteria.end(); ++it) {
            require_text_value(it.value(), "option '" + it.key() + "' of question '" + name + "'",
                               criteria_param + "." + it.key());
            question.option_keys.push_back(it.key());
            question.option_texts.push_back(rendered_text(it.value(), literals, pointer + "/criteria/" + pointer_token(it.key())));
            question.option_values.push_back(it.value());
        }
        break;
    }
    case DecisionKind::Noul: {
        if (!criteria.is_object() || criteria.size() != 2 || !criteria.contains("true") ||
            !criteria.contains("false")) {
            invalid("noul question '" + name + "' criteria must be an object with exactly the keys true and false",
                    criteria_param);
        }
        for (const char* key : {"false", "true"}) {
            require_text_value(criteria.at(key), std::string("criteria.") + key + " of question '" + name + "'",
                               criteria_param + "." + key);
            question.option_keys.emplace_back(key);
            question.option_texts.push_back(rendered_text(criteria.at(key), literals, pointer + "/criteria/" + key));
            question.option_values.push_back(criteria.at(key));
        }
        break;
    }
    case DecisionKind::Score: {
        if (!criteria.is_array()) {
            invalid("score question '" + name + "' criteria must be an array of level descriptions", criteria_param);
        }
        std::size_t index = 0;
        for (const auto& level : criteria) {
            require_text_value(level, "level " + std::to_string(index) + " of question '" + name + "'",
                               criteria_param + "[" + std::to_string(index) + "]");
            question.option_keys.push_back(std::to_string(index));
            question.option_texts.push_back(rendered_text(level, literals, pointer + "/criteria/" + std::to_string(index)));
            question.option_values.push_back(level);
            ++index;
        }
        break;
    }
    }
    if (question.option_count() < kDecisionMinOptions || question.option_count() > kDecisionMaxOptions) {
        invalid("question '" + name + "' has " + std::to_string(question.option_count()) +
                    " options; between " + std::to_string(kDecisionMinOptions) + " and " +
                    std::to_string(kDecisionMaxOptions) + " are supported",
                criteria_param);
    }
    return question;
}

/// TypeSafe's `_normalize`: divide by the sum, or uniform when the sum is zero.
std::vector<double> normalize(const std::vector<double>& probabilities) {
    double total = 0.0;
    for (const double p : probabilities) { total += p; }
    std::vector<double> out(probabilities.size());
    if (total == 0.0) {
        std::fill(out.begin(), out.end(), 1.0 / static_cast<double>(probabilities.size()));
        return out;
    }
    for (std::size_t i = 0; i < probabilities.size(); ++i) { out[i] = probabilities[i] / total; }
    return out;
}

std::size_t first_argmax(const std::vector<double>& values) {
    return static_cast<std::size_t>(std::max_element(values.begin(), values.end()) - values.begin());
}

/// A choice question's answer at a temperature: the first option that is most probable both
/// untempered and tempered. One always exists -- the largest logit's exponent is exactly 0 at
/// every T -- so the choice is always a maximum of the probabilities it is returned with. With
/// T >= 1 it is the untempered choice: tempering cannot separate options that tied by rounding
/// at T = 1, and a tie that a large T creates by rounding is settled by the untempered order.
/// Only a T below 1 can move it, and only between options whose logits were so close (within
/// about 5e-17) that they tied by rounding at T = 1.
std::size_t tempered_choice(const std::vector<double>& untempered, const std::vector<double>& tempered) {
    const double top  = *std::max_element(untempered.begin(), untempered.end());
    const double peak = *std::max_element(tempered.begin(), tempered.end());
    for (std::size_t i = 0; i < untempered.size(); ++i) {
        if (untempered[i] == top && tempered[i] == peak) { return i; }
    }
    return first_argmax(untempered); // unreachable: the largest logit is maximal on both sides
}

} // namespace

const char* decision_kind_name(DecisionKind kind) noexcept {
    switch (kind) {
    case DecisionKind::Choice: return "choice";
    case DecisionKind::Noul: return "noul";
    case DecisionKind::Score: return "score";
    }
    return "choice";
}

bool DecisionQuestion::extended() const noexcept { return option_count() > kDecisionLetterOptions; }

const std::string& decision_boundary_marker() {
    static const std::string marker = [] {
        std::string value;
        value.push_back('\0');
        value += "JEV_QUESTION_BOUNDARY";
        value.push_back('\0');
        return value;
    }();
    return marker;
}

std::string python_float_repr(double value) {
    if (!std::isfinite(value)) { throw std::invalid_argument("non-finite float has no JSON text"); }
    if (value == 0.0) { return std::signbit(value) ? "-0.0" : "0.0"; }
    // The shortest digit string that round-trips, in scientific form so the digits and the
    // decimal exponent are easy to read back; Python then chooses the layout from the
    // position of the decimal point, exactly as `float_repr_style == 'short'` does.
    std::array<char, 64> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                      std::chars_format::scientific);
    if (result.ec != std::errc()) { throw std::logic_error("float repr overflowed its buffer"); }
    std::string text(buffer.data(), result.ptr);
    const bool negative = text.front() == '-';
    if (negative) { text.erase(0, 1); }
    const auto e = text.find('e');
    std::string digits;
    for (const char c : text.substr(0, e)) {
        if (c != '.') { digits.push_back(c); }
    }
    while (digits.size() > 1 && digits.back() == '0') { digits.pop_back(); }
    const int exponent = std::stoi(text.substr(e + 1));
    const int decpt    = exponent + 1; // value = 0.d1d2..dn * 10^decpt
    const auto n       = static_cast<int>(digits.size());
    std::string out;
    if (decpt <= -4 || decpt > 16) {
        out = digits.substr(0, 1);
        if (n > 1) { out += "." + digits.substr(1); }
        const int shown = decpt - 1;
        out += 'e';
        out += shown < 0 ? '-' : '+';
        const int magnitude = std::abs(shown);
        if (magnitude < 10) { out += '0'; }
        out += std::to_string(magnitude);
    } else if (decpt <= 0) {
        out = "0." + std::string(static_cast<std::size_t>(-decpt), '0') + digits;
    } else if (decpt >= n) {
        out = digits + std::string(static_cast<std::size_t>(decpt - n), '0') + ".0";
    } else {
        out = digits.substr(0, static_cast<std::size_t>(decpt)) + "." +
              digits.substr(static_cast<std::size_t>(decpt));
    }
    return negative ? "-" + out : out;
}

std::string python_json_dumps(const OrderedJson& value, const JsonLiterals& literals, const std::string& pointer) {
    std::string out;
    dump_value(value, out, literals, pointer);
    return out;
}

std::string decision_state_text(const OrderedJson& state, const JsonLiterals& literals, const std::string& pointer) {
    return "SHARED STATE (JSON string):\n" + python_json_dumps(state, literals, pointer) + "\n\n";
}

DecisionsRequest parse_decisions_request(std::string_view body) {
    DuplicateKeyScanner scanner;
    if (!OrderedJson::sax_parse(body, &scanner)) {
        if (scanner.scope == "questions") {
            invalid("question '" + scanner.duplicate + "' is named twice", "questions");
        }
        if (!scanner.scope.empty()) {
            invalid("option label '" + scanner.duplicate + "' appears twice in " + scanner.scope, scanner.scope);
        }
        invalid("request body is not valid JSON", "");
    }
    const OrderedJson root = OrderedJson::parse(body, nullptr, false);
    if (root.is_discarded()) { invalid("request body is not valid JSON", ""); }
    if (!root.is_object()) { invalid("request body must be a JSON object", ""); }

    DecisionsRequest request;
    if (!root.contains("model") || !root.at("model").is_string() || root.at("model").get<std::string>().empty()) {
        invalid("model must be a non-empty string", "model");
    }
    request.model = root.at("model").get<std::string>();

    if (!root.contains("state")) { invalid("state is required", "state"); }
    const OrderedJson& state = root.at("state");
    if (!state.is_string() && !state.is_object() && !state.is_array()) {
        invalid("state must be a string, an object or an array", "state");
    }
    request.state      = state;
    // The dumper escapes every control character, so the boundary marker (two NUL bytes)
    // cannot appear in this text; the split of the rendered prompt is checked where it is made.
    request.state_text = decision_state_text(state, scanner.literals, "/state");

    if (!root.contains("questions") || !root.at("questions").is_object()) {
        invalid("questions must be an object of question name to question", "questions");
    }
    const OrderedJson& questions = root.at("questions");
    if (questions.empty()) { invalid("questions must contain at least one question", "questions"); }
    for (auto it = questions.begin(); it != questions.end(); ++it) {
        request.questions.push_back(parse_question(it.key(), it.value(), scanner.literals));
    }

    if (root.contains("images") && !root.at("images").is_null()) {
        const OrderedJson& images = root.at("images");
        if (!images.is_array()) { invalid("images must be an array of data URLs", "images"); }
        std::size_t index = 0;
        for (const auto& item : images) {
            ContentPart part;
            part.kind     = ContentKind::Image;
            part.type_raw = "image_url";
            part.source   = parse_image_source(item, index++);
            request.images.push_back(std::move(part));
        }
    }
    // Our extension (decisions_thinking.h): absent, null and "none" are v1, and nothing below
    // reads the field then.
    request.thinking = parse_decision_thinking_level(root);
    // provider, session_id, user and trace are accepted and ignored, as are unknown fields.
    return request;
}

std::vector<std::string> decision_codebook(
    const std::function<std::vector<TokenId>(std::string_view)>& encode,
    const std::function<std::string(TokenId)>& decode) {
    std::vector<std::string> codes;
    std::set<TokenId> seen;
    const auto consider = [&](const std::string& code) {
        const std::vector<TokenId> ids = encode(code);
        if (ids.size() != 1 || seen.contains(ids.front())) { return; }
        if (decode(ids.front()) != code) { return; }
        codes.push_back(code);
        seen.insert(ids.front());
    };
    for (char first = 'A'; first <= 'Z'; ++first) { consider(std::string(1, first)); }
    for (char first = 'A'; first <= 'Z'; ++first) {
        for (char second = 'A'; second <= 'Z'; ++second) { consider(std::string{first, second}); }
    }
    return codes;
}

std::vector<std::string> decision_labels(std::size_t count, const std::vector<std::string>& codebook) {
    if (count < kDecisionMinOptions || count > kDecisionMaxOptions) {
        invalid("expected between " + std::to_string(kDecisionMinOptions) + " and " +
                    std::to_string(kDecisionMaxOptions) + " options",
                "questions");
    }
    std::vector<std::string> labels;
    if (count <= kDecisionLetterOptions) {
        for (std::size_t i = 0; i < count; ++i) { labels.emplace_back(1, static_cast<char>('A' + i)); }
        return labels;
    }
    if (codebook.size() < count) {
        invalid("the tokenizer supplies only " + std::to_string(codebook.size()) +
                    " single-token option codes; this question needs " + std::to_string(count),
                "questions");
    }
    labels.assign(codebook.begin(), codebook.begin() + static_cast<std::ptrdiff_t>(count));
    return labels;
}

RenderedDecisionQuestion render_decision_question(const DecisionQuestion& question,
                                                  const std::vector<std::string>& codebook) {
    RenderedDecisionQuestion rendered;
    rendered.extended = question.extended();
    rendered.system   = rendered.extended ? kDecisionExtendedSystemPrompt : kDecisionSystemPrompt;
    rendered.labels   = decision_labels(question.option_count(), codebook);
    std::string branch = "QUESTION:\n" + question.instructions + "\nOPTIONS:\n";
    for (std::size_t i = 0; i < question.option_count(); ++i) {
        if (i > 0) { branch += "\n"; }
        branch += rendered.labels[i] + ": " + question.option_texts[i];
    }
    branch += rendered.extended ? "\nAnswer with one option code only." : "\nAnswer with one option letter only.";
    rendered.branch = std::move(branch);
    return rendered;
}

std::size_t decision_shared_prefix(const std::vector<std::vector<TokenId>>& prefixes,
                                   const std::vector<std::vector<TokenId>>& full) {
    if (prefixes.empty() || full.empty()) { return 0; }
    const std::vector<TokenId>& prefix = prefixes.front();
    std::size_t length = prefix.size();
    for (const auto& candidate : prefixes) { length = std::min(length, candidate.size()); }
    for (const auto& ids : full) {
        length = std::min(length, ids.empty() ? std::size_t{0} : ids.size() - 1);
        for (std::size_t i = 0; i < length; ++i) {
            if (prefix[i] != ids[i]) {
                length = i;
                break;
            }
        }
    }
    return length;
}

std::size_t decision_shared_prefix_floor(std::size_t shared, const std::vector<std::size_t>& full_lengths,
                                         std::size_t minimum) {
    for (const std::size_t length : full_lengths) {
        const std::size_t cap = length > minimum + 1 ? length - minimum : 1;
        shared                = std::min(shared, cap);
    }
    return shared < minimum ? 0 : shared;
}

double decision_choice_confidence(const std::vector<double>& probabilities) {
    if (probabilities.size() == 1) { return 1.0; }
    const std::vector<double> normalized = normalize(probabilities);
    const double uniform = 1.0 / static_cast<double>(normalized.size());
    return (*std::max_element(normalized.begin(), normalized.end()) - uniform) / (1.0 - uniform);
}

double decision_score_confidence(const std::vector<double>& probabilities) {
    if (probabilities.size() == 1) { return 1.0; }
    const std::vector<double> normalized = normalize(probabilities);
    const std::size_t mode               = first_argmax(normalized);
    double distance_from_mode            = 0.0;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        distance_from_mode += normalized[i] * std::fabs(static_cast<double>(i) - static_cast<double>(mode));
    }
    const double uniform_center = static_cast<double>(normalized.size() - 1) / 2.0;
    double uniform_mad          = 0.0;
    for (std::size_t i = 0; i < normalized.size(); ++i) {
        uniform_mad += std::fabs(static_cast<double>(i) - uniform_center);
    }
    uniform_mad /= static_cast<double>(normalized.size());
    return std::max(0.0, 1.0 - distance_from_mode / uniform_mad);
}

bool valid_decision_temperature(double temperature) noexcept {
    return std::isfinite(temperature) && temperature > 0.0;
}

std::vector<double> decision_probabilities(const std::vector<float>& logits, double temperature) {
    if (!valid_decision_temperature(temperature)) {
        throw std::logic_error("decision temperature must be finite and greater than zero");
    }
    if (logits.empty()) { throw std::runtime_error("decision readout is empty"); }
    if (std::any_of(logits.begin(), logits.end(), [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error("model returned non-finite logits");
    }
    // Shift by the maximum, then temper: every exponent is (z_i - max) / T <= 0 and the
    // maximum's is exactly 0, so the sum is at least 1 whatever T is. At T == 1 the division
    // is exact and this is the untempered softmax, bit for bit.
    const double maximum = *std::max_element(logits.begin(), logits.end());
    std::vector<double> probabilities(logits.size());
    double sum = 0.0;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp((static_cast<double>(logits[i]) - maximum) / temperature);
        sum += probabilities[i];
    }
    for (double& p : probabilities) { p /= sum; }
    return probabilities;
}

OrderedJson resolve_decision_answer(const DecisionQuestion& question, const std::vector<float>& logits,
                                    double temperature) {
    const std::size_t n = question.option_count();
    if (logits.size() != n || n == 0) { throw std::runtime_error("decision readout does not match its options"); }
    // The one tempered distribution every number of the answer is computed from.
    const std::vector<double> probabilities = decision_probabilities(logits, temperature);

    OrderedJson answer;
    answer["type"] = decision_kind_name(question.kind);
    switch (question.kind) {
    case DecisionKind::Choice: {
        // Dividing by T > 0 keeps the order of the logits, so the choice is the untempered one
        // (see tempered_choice). At T == 1 both sides are the same vector and this is the first
        // argmax, exactly as before the temperature existed.
        const std::size_t best = temperature == kDecisionDefaultTemperature
                                     ? first_argmax(probabilities)
                                     : tempered_choice(decision_probabilities(logits), probabilities);
        answer["choice"]       = question.option_keys[best];
        answer["confidence"]   = decision_choice_confidence(probabilities);
        OrderedJson table      = OrderedJson::object();
        for (std::size_t i = 0; i < n; ++i) { table[question.option_keys[i]] = probabilities[i]; }
        answer["probabilities"] = std::move(table);
        break;
    }
    case DecisionKind::Noul:
        answer["noul"] = probabilities[1];
        break;
    case DecisionKind::Score: {
        double score = 0.0;
        for (std::size_t i = 0; i < n; ++i) { score += static_cast<double>(i) * probabilities[i]; }
        answer["score"]      = score;
        answer["confidence"] = decision_score_confidence(probabilities);
        OrderedJson legend   = OrderedJson::object();
        OrderedJson table    = OrderedJson::object();
        for (std::size_t i = 0; i < n; ++i) {
            legend[question.option_keys[i]] = question.option_values[i];
            table[question.option_keys[i]]  = probabilities[i];
        }
        answer["legend"]        = std::move(legend);
        answer["probabilities"] = std::move(table);
        break;
    }
    }
    return answer;
}

OrderedJson resolve_decision_answers(const DecisionsRequest& request, const std::vector<std::vector<float>>& logits,
                                     double temperature) {
    if (logits.size() != request.questions.size()) {
        throw std::runtime_error("decision readouts do not match the questions");
    }
    OrderedJson answers = OrderedJson::object();
    for (std::size_t i = 0; i < request.questions.size(); ++i) {
        answers[request.questions[i].name] = resolve_decision_answer(request.questions[i], logits[i], temperature);
    }
    return answers;
}

DecisionsFault classify_decisions_fault(const std::exception& fault, DecisionsFaultStage stage) {
    // `InvalidRequest` is the engine saying the request itself is wrong -- a prompt that does
    // not fit, a token outside the domain, an option readout it cannot take -- so it keeps the
    // 400 it has always had, wherever it was raised. Preparation is the caller's side of the
    // call by construction: their body, the chat template and the tokenizer, nothing else.
    if (stage == DecisionsFaultStage::Preparation ||
        dynamic_cast<const sinfer::InvalidRequest*>(&fault) != nullptr) {
        return DecisionsFault{.error = ApiError{.status  = 400,
                                                .message = fault.what(),
                                                .param   = "questions",
                                                .code    = "invalid_decisions_request"}};
    }
    // The engine broke on its own state. The caller cannot fix it and must not be told they
    // can, so the detail goes to the log and the answer carries a status a client retries.
    return DecisionsFault{
        .error           = ApiError{.status  = 500,
                                    .type    = "server_error",
                                    .message = "the inference engine failed while answering this "
                                               "request; the request is unchanged and may be retried",
                                    .code    = "internal_error"},
        .internal_detail = fault.what()};
}

std::string new_decision_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%016llx", static_cast<unsigned long long>(dist(rng)));
    return "dec-" + std::string(buffer.data());
}

} // namespace sinfer::serve
