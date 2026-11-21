#include "family/impl/frontend/inkling_render.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <utility>

namespace sinfer::family::frontend_internal::inkling {
namespace {

// vllm/renderers/inkling.py _NAMED_REASONING_EFFORT
constexpr std::pair<std::string_view, double> kNamedEffort[] = {
    {"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7}, {"high", 0.9}, {"xhigh", 0.99}, {"max", 0.99}};
// The effort a request that sets none is rendered with. vLLM's renderer default (inkling_encoding.DEFAULT_REASONING_EFFORT)
// is 0.9; OWNER RULE 2026-11-21 17:40 (rl/engine-inkling-2026-11-21/LANE-RULES.md 20): "always use inkling at 0.99
// reasoning effort" -- our default is 0.99 (= 'xhigh' / 'max'). A request that names an effort (top-level
// reasoning_effort, chat_template_kwargs.reasoning_effort: a level, a number, a numeric string) keeps it; absent, null and
// "auto" (dropped by the kwargs merge) take this default. Test: sinfer_inkling_frontend_test over vectors/renderer-099
// (the default-effort requests expect vLLM's ids for the same body with chat_template_kwargs.reasoning_effort = 0.99).
constexpr double kDefaultEffort = 0.99;
constexpr double kMaxEffort = 0.99;

std::optional<double> named_effort(std::string_view name) {
    for (const auto& [n, v] : kNamedEffort) {
        if (n == name) { return v; }
    }
    return std::nullopt;
}

bool is_none(const PyValue* v) noexcept { return v == nullptr || v->is_null(); }

// Python repr() of a str, enough for error messages.
std::string py_repr(const PyValue& v) {
    if (v.is_string()) { return "'" + v.text + "'"; }
    if (v.kind == PyValue::Kind::Bool) { return v.boolean ? "True" : "False"; }
    if (v.is_null()) { return "None"; }
    return py_json_dumps_unicode(v);
}

// merge_kwargs(defaults, overrides): overrides win unless they are None or "auto".
PyValue merge_kwargs(const PyValue& defaults, const PyValue& overrides) {
    PyValue out = defaults.is_object() ? defaults : PyValue::object();
    for (std::size_t i = 0; i < overrides.keys.size(); ++i) {
        const PyValue& v = overrides.items[i];
        if (v.is_null() || (v.is_string() && v.text == "auto")) { continue; }
        out.set(overrides.keys[i], v);
    }
    return out;
}

bool is_image_part(std::string_view type) {
    return type == "image" || type == "input_image" || type == "image_url";
}

// vLLM's parse_chat_message_content_part for a part type outside its parser map (renderers/chat_utils/chat_message.py):
// sorted(MM_PARSER_MAP.keys() | {"text", "refusal"}).
constexpr std::string_view kSupportedPartTypes =
    "audio_embeds, audio_url, image_embeds, image_pil, image_url, input_audio, input_image, input_text, output_text, "
    "prompt_embeds, refusal, text, thinking, tool_reference, video_embeds, video_url";

// Python's repr() of a JSON value (str quoting as CPython: single quotes unless the text has ' and no ").
void python_repr(std::string& out, const PyValue& v) {
    switch (v.kind) {
    case PyValue::Kind::Null: out += "None"; return;
    case PyValue::Kind::Bool: out += v.boolean ? "True" : "False"; return;
    case PyValue::Kind::Int: out += v.text; return;
    case PyValue::Kind::Float: out += py_float_repr(v.number); return;
    case PyValue::Kind::String: {
        const bool has_single = v.text.find('\'') != std::string::npos;
        const char quote = has_single && v.text.find('"') == std::string::npos ? '"' : '\'';
        out += quote;
        for (const char ch : v.text) {
            const auto c = static_cast<unsigned char>(ch);
            if (ch == '\\' || ch == quote) { out += '\\'; out += ch; }
            else if (ch == '\n') { out += "\\n"; }
            else if (ch == '\r') { out += "\\r"; }
            else if (ch == '\t') { out += "\\t"; }
            else if (c < 0x20 || c == 0x7f) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\x%02x", c);
                out += buf;
            } else { out += ch; }
        }
        out += quote;
        return;
    }
    case PyValue::Kind::Array:
        out += '[';
        for (std::size_t i = 0; i < v.items.size(); ++i) {
            if (i != 0) { out += ", "; }
            python_repr(out, v.items[i]);
        }
        out += ']';
        return;
    case PyValue::Kind::Object:
        out += '{';
        for (std::size_t i = 0; i < v.keys.size(); ++i) {
            if (i != 0) { out += ", "; }
            python_repr(out, PyValue::from_string(v.keys[i]));
            out += ": ";
            python_repr(out, v.items[i]);
        }
        out += '}';
        return;
    }
}

// vLLM parses an input_image part with TypeAdapter(ResponseInputImageParam), whose `detail` is required: without it
// pydantic refuses the request with this text (pydantic 2.14 in the reference image; input_value is the part's repr,
// cut to its first 25 and last 24 characters when longer than 50).
[[noreturn]] void refuse_input_image_without_detail(const PyValue& part) {
    std::string repr;
    python_repr(repr, part);
    if (repr.size() > 50) {
        // pydantic-core cuts the repr by bytes (first 25, last 24); the cut points move to UTF-8 character boundaries
        // so a non-ASCII value can never make the error body invalid UTF-8.
        const auto continuation = [&](std::size_t i) { return (static_cast<unsigned char>(repr[i]) & 0xC0U) == 0x80U; };
        std::size_t head = 25, tail = repr.size() - 24;
        while (head > 0 && continuation(head)) { --head; }
        while (tail < repr.size() && continuation(tail)) { ++tail; }
        repr = repr.substr(0, head) + "..." + repr.substr(tail);
    }
    throw RequestError("1 validation error for ResponseInputImageParam\ndetail\n  Field required [type=missing, input_value=" +
                       repr + ", input_type=dict]\n    For further information visit https://errors.pydantic.dev/2.14/v/missing");
}

// The image source of an image_url / input_image part, as vLLM's parser map reads it: image_url.url (the OpenAI form)
// or a plain URL string (the simple form vLLM also accepts); input_image's image_url string (Responses spelling).
std::string image_source(const PyValue& part, std::string_view type) {
    const PyValue* value = part.find("image_url");
    if (value != nullptr && type == "image_url" && value->is_object()) { value = value->find("url"); }
    if (value == nullptr || !value->is_string() || value->text.empty()) {
        throw RequestError("image part has no image URL: expected " +
                           std::string(type == "image_url" ? "image_url.url" : "image_url") + " to be a URL string");
    }
    return value->text;
}

// Images in the request (vision on): vLLM's MultiModalItemTracker counts the image items in message order and refuses
// the first one over the limit; the sources are what the serving layer fetches.
struct ImageScan {
    std::uint64_t limit = 0;
    std::vector<std::string>* urls = nullptr;
};

void take_image_part(const PyValue& part, std::string_view type, ImageScan& scan) {
    if (scan.limit == 0) { throw RequestError(image_count_message(0)); }
    if (type == "image") {
        throw RequestError("Unsupported chat content part type: 'image'. Supported types: " +
                           std::string(kSupportedPartTypes) + ".");
    }
    if (type == "input_image" && part.find("detail") == nullptr) { refuse_input_image_without_detail(part); }
    if (scan.urls->size() >= scan.limit) { throw RequestError(image_count_message(scan.limit)); }
    scan.urls->push_back(image_source(part, type));
}

// What the request model leaves of a message's content: the shapes that reach the renderer as a lazy iterator instead
// of a list are refused there ("message content must be a string or a sequence of parts").
void check_content_shape(const PyValue& message, ImageScan& images) {
    const PyValue* content = message.find("content");
    if (content == nullptr || content->is_null() || content->is_string()) { return; }
    if (!content->is_array()) { throw RequestError("message content must be a string or a sequence of parts"); }
    if (content->items.empty()) { throw RequestError("message content must be a string or a sequence of parts"); }
    for (const PyValue& part : content->items) {
        if (part.is_string()) { continue; }
        if (!part.is_object()) { throw RequestError("content part must be mapping"); }
        const PyValue* type = part.find("type");
        if (is_none(type)) { continue; }
        if (!type->is_string()) { throw RequestError("message content must be a string or a sequence of parts"); }
        const std::string& t = type->text;
        if (t == "text" || t == "input_text" || t == "thinking") { continue; }
        if (is_image_part(t)) {
            take_image_part(part, t, images);
            continue;
        }
        if (t == "tool_error") { throw RequestError("message content must be a string or a sequence of parts"); }
        throw RequestError("unsupported Inkling2 content part type: '" + t + "'");
    }
}

struct Part {
    std::string kind;  // text | thinking | tool_error | image
    std::string text;
};

// _iter_inkling2_parts
std::vector<Part> iter_parts(const PyValue* content) {
    std::vector<Part> parts;
    if (is_none(content)) { return parts; }
    if (content->is_string()) {
        parts.push_back({"text", content->text});
        return parts;
    }
    if (!content->is_array()) { throw RequestError("message content must be a string or a sequence of parts"); }
    for (const PyValue& part : content->items) {
        if (part.is_string()) {
            parts.push_back({"text", part.text});
            continue;
        }
        if (!part.is_object()) { throw RequestError("content part must be mapping"); }
        const PyValue* type = part.find("type");
        const auto text_of = [&](std::string_view key) -> std::string {
            const PyValue* v = part.find(key);
            if (v == nullptr || !v->truthy()) { return ""; }  // part.get(key) or ""
            if (!v->is_string()) { throw RequestError("content part text must be a string"); }
            return v->text;
        };
        if (is_none(type) || (type->is_string() && (type->text == "text" || type->text == "input_text"))) {
            parts.push_back({"text", text_of("text")});
        } else if (type->is_string() && type->text == "thinking") {
            parts.push_back({"thinking", text_of("thinking")});
        } else if (type->is_string() && type->text == "tool_error") {
            parts.push_back({"tool_error", text_of("text")});
        } else if (type->is_string() && is_image_part(type->text)) {
            // Accepted by parse_chat_request only with vision on; the marker and its pads are written by Writer.
            parts.push_back({"image", ""});
        } else {
            throw RequestError("unsupported Inkling2 content part type: " + py_repr(*type));
        }
    }
    return parts;
}

class Writer {
public:
    Writer(const TextEncoder& encode, const SpecialTokens& sp, std::span<const std::uint64_t> image_tokens)
        : encode_(encode), sp_(sp), image_tokens_(image_tokens) {}

    void text(std::string_view s) {
        if (s.empty()) { return; }
        if (has_lone_surrogate(s)) { throw RequestError("text holds a lone surrogate the tokenizer cannot encode"); }
        const std::vector<int> ids = encode_(s);
        ids_.insert(ids_.end(), ids.begin(), ids.end());
    }
    void token(int id) { ids_.push_back(id); }

    int role_token(std::string_view role) const {
        if (role == "user") { return sp_.message_user; }
        if (role == "assistant") { return sp_.message_model; }
        if (role == "system" || role == "developer") { return sp_.message_system; }
        return sp_.message_tool;
    }

    // _append_message
    void message(std::string_view role, std::string_view kind, std::string_view payload, std::string_view author) {
        token(role_token(role));
        if (!author.empty()) { text(author); }
        if (kind == "image") {
            // vLLM: the renderer's bare <|content_image|>, which the multimodal processor replaces with itself + N pads.
            if (next_image_ >= image_tokens_.size()) {
                throw std::logic_error("Inkling image part without a prepared image");
            }
            if (sp_.image_pad < 0) { throw std::invalid_argument("Inkling tokenizer has no <|unused_200054|> image pad"); }
            token(sp_.content_image);
            ids_.insert(ids_.end(), static_cast<std::size_t>(image_tokens_[next_image_++]), sp_.image_pad);
            token(sp_.end_message);
            return;
        }
        int kind_token = -1;
        if (kind == "text") { kind_token = sp_.content_text; }
        else if (kind == "thinking") { kind_token = sp_.content_thinking; }
        else if (kind == "tool_error") { kind_token = sp_.content_tool_error; }
        else if (kind == "xml") { kind_token = sp_.content_xml; }
        else if (kind == "invoke_tool_json") { kind_token = sp_.content_invoke_tool_json; }
        if (kind_token < 0) { throw RequestError("unsupported Inkling render part kind: '" + std::string(kind) + "'"); }
        token(kind_token);
        text(payload);
        token(sp_.end_message);
    }

    std::vector<int> take() {
        if (next_image_ != image_tokens_.size()) {
            throw std::logic_error("Inkling render met fewer image parts than prepared images");
        }
        return std::move(ids_);
    }

private:
    const TextEncoder& encode_;
    const SpecialTokens& sp_;
    std::span<const std::uint64_t> image_tokens_;
    std::size_t next_image_ = 0;
    std::vector<int> ids_;
};

const PyValue& as_mapping(const PyValue& v) {
    if (!v.is_object()) { throw RequestError("expected mapping"); }
    return v;
}

// _inkling2_tool_declare_json
std::string tool_declare_json(const std::vector<const PyValue*>& tools) {
    PyValue specs = PyValue::array();
    for (const PyValue* tool_value : tools) {
        const PyValue& tool = as_mapping(*tool_value);
        const PyValue* fn_value = tool.find("function");
        PyValue function = PyValue::object();
        if (fn_value != nullptr) {
            const PyValue& fn = as_mapping(*fn_value);
            for (std::size_t i = 0; i < fn.keys.size(); ++i) {
                if (!fn.items[i].is_null()) { function.set(fn.keys[i], fn.items[i]); }
            }
        }
        const PyValue* parameters = function.find("parameters");
        PyValue params = parameters ? *parameters : PyValue::object();
        if (!params.is_object()) { throw RequestError("tool parameters must be a JSON object"); }
        const PyValue* name = function.find("name");
        if (name == nullptr) { throw RequestError("'name'"); }
        const PyValue* description = function.find("description");
        const PyValue* type = tool.find("type");
        PyValue spec = PyValue::object();
        spec.set("description", description ? *description : PyValue::from_string(""));
        spec.set("name", *name);
        spec.set("parameters", tml_roundtrip(params));
        spec.set("type", (type != nullptr && type->truthy()) ? *type : PyValue::from_string("function"));
        specs.items.push_back(std::move(spec));
    }
    return tml_json_dumps(specs, /*legacy_floats=*/false, /*sort_keys=*/true);
}

std::string tool_name_of(const PyValue& tool) {
    const PyValue* fn = as_mapping(tool).find("function");
    if (fn == nullptr) { return {}; }
    const PyValue* name = as_mapping(*fn).find("name");
    return (name != nullptr && name->is_string()) ? name->text : std::string{};
}

// _with_appended_text
PyValue with_appended_text(const PyValue& message, const std::string& text) {
    PyValue out = message;
    const PyValue* content = message.find("content");
    if (content != nullptr && content->is_string() && !content->text.empty()) {
        out.set("content", PyValue::from_string(content->text + "\n\n" + text));
    } else if (content == nullptr || !content->truthy()) {
        out.set("content", PyValue::from_string(text));
    } else {
        PyValue list = *content;
        PyValue part = PyValue::object();
        part.set("type", PyValue::from_string("text"));
        part.set("text", PyValue::from_string(text));
        list.items.push_back(std::move(part));
        out.set("content", std::move(list));
    }
    return out;
}

std::string str_of(const PyValue& v) {
    if (v.is_string()) { return v.text; }
    if (v.kind == PyValue::Kind::Int) { return v.text; }
    if (v.kind == PyValue::Kind::Float) { return py_float_repr(v.number); }
    if (v.kind == PyValue::Kind::Bool) { return v.boolean ? "True" : "False"; }
    if (v.is_null()) { return "None"; }
    return py_json_dumps_unicode(v);
}

// json.dumps(name) with the default ensure_ascii=True, as the tool-choice instruction spells the name.
std::string ascii_json_string(const std::string& s) {
    std::string out;
    py_json_quote(out, s, /*ensure_ascii=*/true);
    return out;
}

} // namespace

std::string image_count_message(std::uint64_t limit) {
    return "At most " + std::to_string(limit) + " image(s) may be provided in one prompt.";
}

SpecialTokens SpecialTokens::resolve(const std::function<std::optional<int>(std::string_view)>& token_id) {
    SpecialTokens t;
    std::string missing;
    const auto need = [&](int& slot, std::initializer_list<std::string_view> spellings, bool optional) {
        for (const std::string_view s : spellings) {
            if (const auto id = token_id(s)) {
                slot = *id;
                return;
            }
        }
        if (!optional) {
            missing += missing.empty() ? "" : ", ";
            missing += *spellings.begin();
        }
    };
    need(t.message_user, {"<|message_user|>"}, false);
    need(t.message_model, {"<|message_model|>"}, false);
    need(t.message_system, {"<|message_system|>"}, false);
    need(t.message_tool, {"<|message_tool|>"}, false);
    need(t.content_text, {"<|content_text|>"}, false);
    need(t.content_image, {"<|content_image|>"}, false);
    need(t.content_model_end_sampling, {"<|content_model_end_sampling|>"}, false);
    need(t.content_thinking, {"<|content_thinking|>"}, false);
    need(t.content_tool_error, {"<|content_tool_error|>"}, true);
    need(t.content_xml, {"<|content_xml|>", "<|unused_200024|>"}, false);
    need(t.content_invoke_tool_json, {"<|content_invoke_tool_json|>"}, false);
    need(t.content_invoke_tool_text, {"<|content_invoke_tool_text|>"}, true);
    need(t.end_message, {"<|end_message|>"}, false);
    need(t.image_pad, {"<|unused_200054|>"}, true);
    if (!missing.empty()) { throw std::invalid_argument("Inkling tokenizer is missing special tokens: " + missing); }
    return t;
}

ChatRequest parse_chat_request(std::string_view body_json, std::uint64_t image_limit) {
    ChatRequest r;
    ImageScan images{.limit = image_limit, .urls = &r.image_urls};
    PyValue body = py_json_loads(body_json);
    if (!body.is_object()) { throw RequestError("request body must be a JSON object"); }

    // messages: the request model's normalisation (_normalize_messages_before)
    const PyValue* messages = body.find("messages");
    if (messages == nullptr || !messages->is_array()) { throw RequestError("messages: Input should be a valid list"); }
    if (messages->items.empty()) { throw RequestError("messages must be a non-empty list."); }
    r.messages = *messages;
    for (PyValue& m : r.messages.items) {
        if (!m.is_object()) { throw RequestError("messages: each message must be an object"); }
        const PyValue* role = m.find("role");
        if (role == nullptr || !role->is_string()) { throw RequestError("messages: role must be a string"); }
        const PyValue* tool_calls = m.find("tool_calls");
        if (tool_calls != nullptr && !tool_calls->is_null() && !tool_calls->is_array()) {
            throw RequestError("messages: tool_calls must be a list");
        }
        PyValue reasoning_content;
        const bool had = m.find("reasoning_content") != nullptr;
        if (had) {
            reasoning_content = *m.find("reasoning_content");
            m.erase("reasoning_content");
        }
        if (had && !reasoning_content.is_null() && is_none(m.find("reasoning"))) {
            m.set("reasoning", std::move(reasoning_content));
        }
        check_content_shape(m, images);
    }

    // tools (check_tool_usage: [] means none) and their model dump
    const PyValue* tools = body.find("tools");
    if (!is_none(tools)) {
        if (!tools->is_array()) { throw RequestError("tools: Input should be a valid list"); }
        if (!tools->items.empty()) {
            PyValue dumped = PyValue::array();
            for (const PyValue& t : tools->items) {
                if (!t.is_object()) { throw RequestError("tools: each tool must be an object"); }
                const PyValue* type = t.find("type");
                if (!is_none(type) && !(type->is_string() && type->text == "function")) {
                    throw RequestError("tools: type should be 'function'");
                }
                const PyValue* fn = t.find("function");
                if (fn == nullptr || !fn->is_object()) { throw RequestError("tools: function is required"); }
                const PyValue* name = fn->find("name");
                if (name == nullptr || !name->is_string()) { throw RequestError("tools: function.name must be a string"); }
                const PyValue* description = fn->find("description");
                if (!is_none(description) && !description->is_string()) {
                    throw RequestError("tools: function.description must be a string");
                }
                const PyValue* parameters = fn->find("parameters");
                if (!is_none(parameters) && !parameters->is_object()) {
                    throw RequestError("tools: function.parameters must be an object");
                }
                PyValue function = *fn;
                if (function.find("description") == nullptr) { function.set("description", PyValue::null()); }
                if (function.find("parameters") == nullptr) { function.set("parameters", PyValue::null()); }
                PyValue tool = PyValue::object();
                tool.set("type", PyValue::from_string("function"));
                tool.set("function", std::move(function));
                dumped.items.push_back(std::move(tool));
            }
            r.tools = std::move(dumped);
        }
    }

    // tool_choice (check_tool_usage + the named-choice model)
    const PyValue* tc = body.find("tool_choice");
    PyValue choice = tc ? *tc : PyValue::null();
    if (r.tools) {
        if (choice.is_null()) { choice = PyValue::from_string("auto"); }
    } else if (choice.is_null() || (choice.is_string() && choice.text == "auto")) {
        choice = PyValue::from_string("none");
    }
    if (choice.is_string() && choice.text == "none") {
        r.tool_choice = ToolChoice::None;
    } else if (!r.tools) {
        throw RequestError("When using `tool_choice`, `tools` must be set.");
    } else if (choice.is_string() && choice.text == "required") {
        r.tool_choice = ToolChoice::Required;
    } else if (choice.is_string() && choice.text == "auto") {
        r.tool_choice = ToolChoice::Auto;
    } else if (!choice.is_object()) {
        throw RequestError("Invalid value for `tool_choice`: " + str_of(choice) + "!");
    } else {
        const PyValue* function = choice.find("function");
        if (function == nullptr || !function->is_object()) { throw RequestError("Invalid value for `function` in `tool_choice`"); }
        const PyValue* name = function->find("name");
        if (name == nullptr) { throw RequestError("Expected field `name` in `function` in `tool_choice`"); }
        if (!name->is_string() || name->text.empty()) { throw RequestError("Invalid `name` in `function` in `tool_choice`"); }
        bool found = false;
        for (const PyValue& t : tools->items) {
            const PyValue* fn = t.find("function");
            const PyValue* n = fn ? fn->find("name") : nullptr;
            if (n != nullptr && n->is_string() && n->text == name->text) { found = true; }
        }
        if (!found) {
            throw RequestError("The tool `" + name->text + "` specified in `tool_choice` does not match any of the specified `tools`");
        }
        const PyValue* type = choice.find("type");
        if (!is_none(type) && !(type->is_string() && type->text == "function")) {
            throw RequestError("tool_choice: type should be 'function'");
        }
        r.tool_choice = ToolChoice::Named;
        r.tool_choice_name = name->text;
    }

    const auto boolean_field = [&](std::string_view key, bool fallback) {
        const PyValue* v = body.find(key);
        if (is_none(v)) { return fallback; }
        if (v->kind != PyValue::Kind::Bool) { throw RequestError(std::string(key) + ": Input should be a valid boolean"); }
        return v->boolean;
    };
    r.parallel_tool_calls = boolean_field("parallel_tool_calls", true);
    r.include_reasoning = boolean_field("include_reasoning", true);
    const bool add_generation_prompt = boolean_field("add_generation_prompt", true);
    const bool continue_final_message = boolean_field("continue_final_message", false);
    {
        const PyValue* agp = body.find("add_generation_prompt");
        const PyValue* cfm = body.find("continue_final_message");
        if (cfm != nullptr && cfm->truthy() && agp != nullptr && agp->truthy()) {
            throw RequestError("Cannot set both `continue_final_message` and `add_generation_prompt` to True.");
        }
    }

    // top-level reasoning_effort: the request model's literal
    const PyValue* top_effort = body.find("reasoning_effort");
    if (!is_none(top_effort) && !(top_effort->is_string() && named_effort(top_effort->text))) {
        throw RequestError("reasoning_effort: Input should be 'none', 'minimal', 'low', 'medium', 'high', 'xhigh' or 'max'");
    }

    // build_chat_params(...).with_defaults(tools=tool_dicts, tokenize=False)
    const PyValue* ctk_value = body.find("chat_template_kwargs");
    if (!is_none(ctk_value) && !ctk_value->is_object()) { throw RequestError("chat_template_kwargs: Input should be a valid dictionary"); }
    const PyValue ctk = is_none(ctk_value) ? PyValue::object() : *ctk_value;
    PyValue extra = PyValue::object();
    extra.set("add_generation_prompt", PyValue::from_bool(add_generation_prompt));
    extra.set("continue_final_message", PyValue::from_bool(continue_final_message));
    extra.set("documents", PyValue::null());
    extra.set("reasoning_effort", is_none(top_effort) ? PyValue::null() : *top_effort);
    if (!is_none(top_effort) && ctk.find("enable_thinking") == nullptr) {
        extra.set("enable_thinking", PyValue::from_bool(top_effort->text != "none"));
    }
    const PyValue merged = merge_kwargs(ctk, extra);
    PyValue defaults_in = PyValue::object();
    defaults_in.set("tools", r.tools ? *r.tools : PyValue::null());
    defaults_in.set("tokenize", PyValue::from_bool(false));
    const PyValue defaults = merge_kwargs(PyValue::object(), defaults_in);
    r.template_kwargs = merge_kwargs(defaults, merged);
    return r;
}

double resolve_reasoning_effort(const ChatRequest& request) {
    const PyValue* found = request.template_kwargs.find("reasoning_effort");
    PyValue value = found ? *found : PyValue::null();
    if (value.is_string() && !named_effort(value.text)) {
        if (const auto f = py_parse_float(value.text)) { value = PyValue::from_float(*f); }
    }
    switch (value.kind) {
    case PyValue::Kind::Null: return kDefaultEffort;
    case PyValue::Kind::String:
        if (const auto v = named_effort(value.text)) { return *v; }
        break;
    case PyValue::Kind::Int: {
        const double d = std::strtod(value.text.c_str(), nullptr);
        if (!std::isfinite(d)) { throw RequestError("int too large to convert to float"); }
        return d;
    }
    case PyValue::Kind::Float: return value.number;
    default: break;
    }
    throw RequestError("invalid Inkling reasoning_effort: " + py_repr(found ? *found : value));
}

std::string effort_text(double effort) {
    if (!(effort >= 0.0 && effort <= kMaxEffort)) {
        throw RequestError("Inkling reasoning_effort must be in [0.0, 0.99], got " + py_float_repr(effort));
    }
    // round() is round-half-even on the exact binary value (nearbyint in the default rounding mode) and returns an
    // int, so -0.0 comes out as 0 ("0", never "-0").
    const auto millieffort = static_cast<long long>(std::nearbyint(effort * 1000.0));
    const auto hundredths = static_cast<long long>(std::nearbyint((static_cast<double>(millieffort) / 1000.0) * 100.0));
    const double rounded = static_cast<double>(hundredths) / 100.0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%g", rounded);
    return std::string("Thinking effort level: ") + buf;
}

std::vector<int> render_chat(const ChatRequest& request, const TextEncoder& encode_text, const SpecialTokens& sp,
                             std::span<const std::uint64_t> image_tokens) {
    if (image_tokens.size() != request.image_urls.size()) {
        throw std::logic_error("Inkling render needs one token count per image part");
    }
    const PyValue& kw = request.template_kwargs;
    const PyValue* kw_tools = kw.find("tools");
    const PyValue* cfm = kw.find("continue_final_message");
    const bool continue_final_message = cfm != nullptr && cfm->truthy();
    const double effort = resolve_reasoning_effort(request);

    // ChatParams.tool_choice = request.tool_choice if request.tools else None (None renders as "auto")
    ToolChoice choice = request.tools ? request.tool_choice : ToolChoice::Auto;

    std::vector<PyValue> messages = request.messages.items;

    // continue_final_message: a final plain-string assistant message without tool calls or reasoning stays open
    std::optional<std::string> assistant_prefix;
    if (continue_final_message && !messages.empty()) {
        const PyValue& last = messages.back();
        const PyValue* role = last.find("role");
        const PyValue* content = last.find("content");
        const PyValue* tool_calls = last.find("tool_calls");
        const PyValue* reasoning = last.find("reasoning");
        if (role && role->is_string() && role->text == "assistant" && content && content->is_string() &&
            !(tool_calls && tool_calls->truthy()) && !(reasoning && reasoning->truthy())) {
            assistant_prefix = content->text;
            messages.pop_back();
        }
    }

    // _collect_tools(messages, None if choice == "none" else tools)
    std::vector<const PyValue*> declared;
    if (choice != ToolChoice::None && !is_none(kw_tools)) {
        if (!kw_tools->is_array()) { throw RequestError("tools must be a list"); }
        for (const PyValue& t : kw_tools->items) { declared.push_back(&t); }
    }
    for (const PyValue& m : messages) {
        const PyValue* role = m.find("role");
        if (role && role->is_string() && role->text == "developer") {
            const PyValue* dev_tools = m.find("tools");
            if (dev_tools != nullptr && dev_tools->truthy()) {
                if (!dev_tools->is_array()) { throw RequestError("developer tools must be a list"); }
                for (const PyValue& t : dev_tools->items) { declared.push_back(&t); }
            }
        }
    }
    if (choice == ToolChoice::Named) {
        std::vector<const PyValue*> kept;
        for (const PyValue* t : declared) {
            if (tool_name_of(*t) == request.tool_choice_name) { kept.push_back(t); }
        }
        declared = std::move(kept);
    }

    Writer w(encode_text, sp, image_tokens);
    if (!declared.empty()) { w.message("system", "xml", tool_declare_json(declared), "tool_declare"); }

    std::optional<std::string> pending_effort = effort_text(effort);
    std::optional<std::string> instruction;
    if (!declared.empty()) {
        if (choice == ToolChoice::Named) {
            instruction = "Call the function named " + ascii_json_string(request.tool_choice_name) +
                          (request.parallel_tool_calls ? "" : " exactly once") + " in your next response.";
        } else if (choice == ToolChoice::Required) {
            instruction = std::string("Call ") + (request.parallel_tool_calls ? "at least one" : "exactly one") +
                          " of the provided functions in your next response.";
        } else if (choice == ToolChoice::Auto && !request.parallel_tool_calls) {
            instruction = "If you call a function, call at most one function in your next response.";
        }
    }
    if (instruction) {
        const PyValue* role0 = messages.empty() ? nullptr : messages[0].find("role");
        if (role0 && role0->is_string() && (role0->text == "system" || role0->text == "developer")) {
            messages[0] = with_appended_text(messages[0], *instruction);
        } else {
            *pending_effort += "\n\n" + *instruction;
        }
    }

    std::vector<std::pair<std::string, std::string>> call_names;  // tool_call_id -> name
    const auto name_for_call = [&](const std::string& id) -> std::string {
        for (auto it = call_names.rbegin(); it != call_names.rend(); ++it) {
            if (it->first == id) { return it->second; }
        }
        return {};
    };

    for (const PyValue& message : messages) {
        const PyValue* role_value = message.find("role");
        const std::string role = (role_value && role_value->is_string()) ? role_value->text : std::string{};
        if (role != "user" && role != "assistant" && role != "system" && role != "developer" && role != "tool") {
            throw RequestError("unsupported Inkling message role " +
                               (role_value ? py_repr(*role_value) : std::string("None")) +
                               "; expected one of ['assistant', 'developer', 'system', 'tool', 'user']");
        }
        if (pending_effort && role != "system" && role != "developer") {
            w.message("system", "text", *pending_effort, "");
            pending_effort.reset();
        }

        std::string author;
        if (role == "tool") {
            const PyValue* name = message.find("name");
            if (name != nullptr && name->truthy()) {
                author = str_of(*name);
            } else {
                const PyValue* id = message.find("tool_call_id");
                author = name_for_call((id != nullptr && id->truthy()) ? str_of(*id) : std::string{});
            }
        }

        struct Block {
            std::string kind, text, author;
        };
        std::vector<Block> blocks;
        const PyValue* content = message.find("content");
        if (role != "assistant") {
            for (Part& p : iter_parts(content)) { blocks.push_back({std::move(p.kind), std::move(p.text), {}}); }
        } else {
            const PyValue* reasoning = message.find("reasoning");
            if (!is_none(reasoning) && !reasoning->is_string()) {
                throw RequestError("assistant reasoning_content must be a string");
            }
            const PyValue* tool_calls = message.find("tool_calls");
            const bool has_calls = tool_calls != nullptr && tool_calls->truthy();
            const bool has_reasoning = reasoning != nullptr && reasoning->truthy();
            if (content && content->is_string() && content->text.empty() && (has_calls || has_reasoning)) {
                content = nullptr;
            }
            if (has_reasoning) { blocks.push_back({"thinking", reasoning->text, {}}); }
            for (Part& p : iter_parts(content)) { blocks.push_back({std::move(p.kind), std::move(p.text), {}}); }
            if (has_calls) {
                for (const PyValue& call_value : tool_calls->items) {
                    const PyValue& call = as_mapping(call_value);
                    const PyValue* fn_value = call.find("function");
                    const PyValue function = fn_value ? as_mapping(*fn_value) : PyValue::object();
                    const PyValue* name = function.find("name");
                    if (name == nullptr || !name->is_string()) { throw RequestError("tool call function name must be a string"); }
                    const PyValue* raw = function.find("arguments");
                    PyValue args = PyValue::object();
                    if (raw != nullptr && raw->truthy()) {
                        if (raw->is_string()) {
                            if (!py_strip(raw->text).empty()) { args = py_json_loads(raw->text); }
                        } else {
                            args = *raw;
                        }
                    }
                    if (!args.is_object()) { throw RequestError("tool call function arguments must decode to an object"); }
                    const PyValue* id = call.find("id");
                    if (id != nullptr && id->truthy()) { call_names.emplace_back(str_of(*id), name->text); }
                    const PyValue normalised = tml_roundtrip(args);
                    // {key: args[key] for key in sorted(args)}, nested objects keep their order
                    std::vector<std::size_t> order(normalised.keys.size());
                    for (std::size_t i = 0; i < order.size(); ++i) { order[i] = i; }
                    std::stable_sort(order.begin(), order.end(),
                                     [&](std::size_t a, std::size_t b) { return normalised.keys[a] < normalised.keys[b]; });
                    PyValue sorted_args = PyValue::object();
                    for (const std::size_t i : order) { sorted_args.set(normalised.keys[i], normalised.items[i]); }
                    const std::string payload = "{\"name\":" + tml_json_dumps(*name) + ",\"args\":" +
                                                tml_json_dumps(sorted_args, /*legacy_floats=*/true, /*sort_keys=*/false) + "}";
                    blocks.push_back({"invoke_tool_json", payload, name->text});
                }
            }
        }
        for (const Block& b : blocks) {
            w.message(role, b.kind, b.text, !b.author.empty() ? b.author : author);
        }
        if (role == "assistant" && !blocks.empty()) { w.token(sp.content_model_end_sampling); }
    }
    if (pending_effort) { w.message("system", "text", *pending_effort, ""); }
    if (assistant_prefix) {
        w.token(sp.message_model);
        w.token(sp.content_text);
        w.text(*assistant_prefix);
    }
    return w.take();
}

} // namespace sinfer::family::frontend_internal::inkling
