#include "serve/decisions_system15.h"

#include <stdexcept>

namespace sinfer::serve {

namespace {

/// Python str() of a JSON scalar as json.loads would have produced it.
std::string scalar_text(const OrderedJson& v) {
    if (v.is_string()) { return v.get<std::string>(); }
    if (v.is_boolean()) { return v.get<bool>() ? "True" : "False"; }
    if (v.is_number_integer()) { return std::to_string(v.get<std::int64_t>()); }
    if (v.is_number_unsigned()) { return std::to_string(v.get<std::uint64_t>()); }
    if (v.is_number_float()) { return python_float_repr(v.get<double>()); }
    throw std::logic_error("system15: not a scalar");
}

std::string lstrip(const std::string& s) {
    std::size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r' || s[i] == '\f' || s[i] == '\v')) {
        ++i;
    }
    return s.substr(i);
}

/// `_opt_text(name, desc)`: name if desc is None or "", else "name: " + _render(desc).
std::string option_text(const std::string& name, const OrderedJson& desc) {
    if (desc.is_null() || (desc.is_string() && desc.get<std::string>().empty())) { return name; }
    return name + ": " + system15_render(desc);
}

void append(std::vector<TokenId>& out, const std::vector<TokenId>& more) { out.insert(out.end(), more.begin(), more.end()); }

} // namespace

std::string system15_render(const OrderedJson& v, int indent) {
    const std::string pad(static_cast<std::size_t>(indent) * 2, ' ');
    if (v.is_null()) { return ""; }
    if (v.is_primitive()) { return scalar_text(v); }
    std::string out;
    bool first = true;
    if (v.is_array()) {
        for (const auto& x : v) {
            if (!first) { out += "\n"; }
            first = false;
            out += pad + "- " + lstrip(system15_render(x, indent + 1));
        }
        return out;
    }
    for (auto it = v.begin(); it != v.end(); ++it) {
        if (!first) { out += "\n"; }
        first = false;
        const OrderedJson& x = it.value();
        if (x.is_object() || x.is_array()) {
            out += pad + it.key() + ":\n" + system15_render(x, indent + 1);
        } else {
            out += pad + it.key() + ": " + system15_render(x);
        }
    }
    return out;
}

std::string system15_escape(std::string_view text) {
    // re.sub(r"<\|([A-Za-z0-9_]+)\|>", r"<¦\1¦>", text)
    static const std::string broken_bar = "\xC2\xA6";
    std::string out;
    std::size_t i = 0;
    while (i < text.size()) {
        if (text.compare(i, 2, "<|") == 0) {
            std::size_t j = i + 2;
            while (j < text.size() && (std::isalnum(static_cast<unsigned char>(text[j])) != 0 || text[j] == '_')) { ++j; }
            if (j > i + 2 && text.compare(j, 2, "|>") == 0) {
                out += "<" + broken_bar;
                out.append(text.substr(i + 2, j - i - 2));
                out += broken_bar + ">";
                i = j + 2;
                continue;
            }
        }
        out += text[i++];
    }
    return out;
}

std::vector<TokenId> system15_state_tokens(const OrderedJson& state,
                                           const std::function<std::vector<TokenId>(std::string_view)>& encode,
                                           const std::vector<TokenId>& markers, std::size_t max_state) {
    if (markers.size() != 5) { throw std::logic_error("system15: five marker tokens"); }
    std::vector<TokenId> out{markers[0]};
    std::vector<TokenId> body = encode(system15_escape(system15_render(state)));
    if (max_state > 0 && body.size() > max_state - 1) { body.resize(max_state - 1); }
    append(out, body);
    return out;
}

System15Prompt system15_question_prompt(const std::vector<TokenId>& state_tokens, const DecisionQuestion& question,
                                        const std::function<std::vector<TokenId>(std::string_view)>& encode,
                                        const std::vector<TokenId>& markers) {
    if (markers.size() != 5) { throw std::logic_error("system15: five marker tokens"); }
    const TokenId q_id = markers[1], o_id = markers[2], c_id = markers[3], d_id = markers[4];
    System15Prompt p;
    std::vector<std::string> opts;
    const OrderedJson& cr = question.criteria_value;
    switch (question.kind) {
    case DecisionKind::Noul: {
        const OrderedJson c = cr.is_object() ? cr : OrderedJson::object();
        p.keys = {"false", "true"};
        opts   = {option_text("no", c.contains("false") ? c.at("false") : OrderedJson()),
                  option_text("yes", c.contains("true") ? c.at("true") : OrderedJson())};
        break;
    }
    case DecisionKind::Choice:
        for (auto it = cr.begin(); it != cr.end(); ++it) {
            p.keys.push_back(it.key());
            opts.push_back(option_text(it.key(), it.value()));
        }
        break;
    case DecisionKind::Score: {
        int i = 0;
        for (const auto& x : cr) {
            p.keys.push_back(std::to_string(i++));
            opts.push_back(system15_render(x));
        }
        break;
    }
    }
    std::vector<TokenId> instr{q_id};
    append(instr, encode(system15_escape(system15_render(question.instructions_value))));
    p.ids = state_tokens;
    append(p.ids, instr);
    std::int32_t cur = static_cast<std::int32_t>(state_tokens.size() + instr.size());
    for (const std::string& o : opts) {
        std::vector<TokenId> span{o_id};
        append(span, encode(system15_escape(o)));
        span.push_back(c_id);
        cur += static_cast<std::int32_t>(span.size());
        p.option_idx.push_back(cur - 1);
        append(p.ids, span);
    }
    p.ids.push_back(d_id);
    p.decide_idx = static_cast<std::int32_t>(p.ids.size()) - 1;
    return p;
}

} // namespace sinfer::serve
