#include "serve/openai_schema.h"

#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <string>

namespace sinfer::serve {
namespace {

using Json = nlohmann::json;

namespace {
/// Serialize a response body, substituting U+FFFD for any byte sequence that is
/// not valid UTF-8.
///
/// A token is a sequence of bytes, not a character: a Chinese character or an
/// emoji is spread over several, and generation stops on a token boundary rather
/// than a character one. So a completion truncated at max_tokens, and every
/// per-token string in a `logprobs` list, can end mid-character. nlohmann::json
/// throws on that, which turned a rollout into a 500 and lost the whole request --
/// 281 of them in one training run. The bytes themselves are not lost: a logprobs
/// entry carries them separately in `bytes`, which is what that field is for.
[[nodiscard]] std::string dump_lossy(const Json& payload) {
    return payload.dump(-1, ' ', false, Json::error_handler_t::replace);
}
} // namespace

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

ChatRole parse_message_role(const std::string& role) {
    if (role == "system") { return ChatRole::System; }
    if (role == "developer") { return ChatRole::Developer; }
    if (role == "user") { return ChatRole::User; }
    if (role == "assistant") { return ChatRole::Assistant; }
    if (role == "tool") { return ChatRole::Tool; }
    if (role == "function") {
        bad_request("role 'function' is not supported yet", "messages", "unsupported_role");
    }
    bad_request("unsupported role: " + role, "messages", "unsupported_role");
}

const Json& require_object(const Json& body) {
    if (!body.is_object()) { bad_request("request body must be a JSON object"); }
    return body;
}

bool get_bool(const Json& obj, const char* key, bool fallback) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return fallback; }
    if (!obj.at(key).is_boolean()) { bad_request(std::string(key) + " must be a boolean", key); }
    return obj.at(key).get<bool>();
}

std::optional<double> get_number(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number()) { bad_request(std::string(key) + " must be a number", key); }
    return obj.at(key).get<double>();
}

std::optional<int> get_int(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be an integer", key);
    }
    if (obj.at(key).is_number_unsigned()) {
        const std::uint64_t value = obj.at(key).get<std::uint64_t>();
        if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            bad_request(std::string(key) + " is out of range", key);
        }
        return static_cast<int>(value);
    }
    const std::int64_t value = obj.at(key).get<std::int64_t>();
    if (value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max()) {
        bad_request(std::string(key) + " is out of range", key);
    }
    return static_cast<int>(value);
}

std::optional<std::uint64_t> get_u64(const Json& obj, const char* key) {
    if (!obj.contains(key) || obj.at(key).is_null()) { return std::nullopt; }
    if (!obj.at(key).is_number_integer()) {
        bad_request(std::string(key) + " must be a nonnegative integer", key);
    }
    if (obj.at(key).is_number_unsigned()) { return obj.at(key).get<std::uint64_t>(); }
    const std::int64_t value = obj.at(key).get<std::int64_t>();
    if (value < 0) { bad_request(std::string(key) + " must be nonnegative", key); }
    return static_cast<std::uint64_t>(value);
}

bool is_valid_function_name(const std::string& name) {
    if (name.empty() || name.size() > 64) { return false; }
    for (const unsigned char c : name) {
        if (std::isalnum(c) == 0 && c != '_' && c != '-' && c != '.') { return false; }
    }
    return true;
}

std::string require_function_name(const Json& obj, const char* param) {
    if (!obj.contains("name") || !obj.at("name").is_string()) {
        bad_request("function name must be a string", param);
    }
    std::string name = obj.at("name").get<std::string>();
    if (!is_valid_function_name(name)) {
        bad_request("function name must match [A-Za-z0-9_.-]{1,64}", param);
    }
    return name;
}

bool has_tool_named(const GenerationRequest& req, const std::string& name) {
    for (const ToolDefinition& tool : req.tools) {
        if (tool.name == name) { return true; }
    }
    return false;
}

sinfer::product::media_acquire::Source parse_media_url(const Json& part, const char* field) {
    if (!part.contains(field)) {
        bad_request(std::string(field) + " content part must contain " + field, "messages");
    }
    const Json& value = part.at(field);
    std::string url;
    if (value.is_string()) {
        url = value.get<std::string>();
    } else if (value.is_object() && value.contains("url") && value.at("url").is_string()) {
        url = value.at("url").get<std::string>();
    } else {
        bad_request(std::string(field) + " must be a URL string or object containing url",
                    "messages");
    }
    if (url.empty()) { bad_request(std::string(field) + " URL must not be empty", "messages"); }
    sinfer::product::media_acquire::Source source;
    source.value = std::move(url);
    if (source.value.starts_with("data:")) {
        source.kind = sinfer::product::media_acquire::SourceKind::Data;
    } else if (source.value.starts_with("http://") || source.value.starts_with("https://")) {
        source.kind = sinfer::product::media_acquire::SourceKind::Url;
    } else {
        bad_request(std::string(field) + " must use HTTP(S) or a data URI", "messages");
    }
    return source;
}

void parse_content_parts(const Json& content, ChatTurn& turn, std::size_t index) {
    if (content.is_string()) {
        turn.content.push_back(ContentPart{ContentKind::Text, content.get<std::string>(), "text"});
        return;
    }
    if (!content.is_array()) {
        bad_request("message " + std::to_string(index) + " content must be a string or array",
                    "messages");
    }
    for (const Json& part : content) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string()) {
            bad_request("message " + std::to_string(index) +
                            " content parts must be objects with a string 'type'",
                        "messages");
        }
        const std::string type = part.at("type").get<std::string>();
        ContentPart out;
        out.type_raw = type;
        if (type == "text") {
            if (!part.contains("text") || !part.at("text").is_string()) {
                bad_request("text content part must contain a string 'text'", "messages");
            }
            out.kind = ContentKind::Text;
            out.text = part.at("text").get<std::string>();
        } else if (type == "image_url") {
            out.kind   = ContentKind::Image;
            out.source = parse_media_url(part, "image_url");
        } else if (type == "video_url") {
            out.kind   = ContentKind::Video;
            out.source = parse_media_url(part, "video_url");
        } else if (type == "input_audio") {
            out.kind = ContentKind::InputAudio;
        } else {
            out.kind = ContentKind::Unsupported;
        }
        turn.content.push_back(std::move(out));
    }
    if (turn.content.empty()) {
        bad_request("message " + std::to_string(index) + " content must not be empty", "messages");
    }
}

std::vector<ToolCall> parse_assistant_tool_calls(const Json& item, std::size_t index) {
    std::vector<ToolCall> calls;
    if (!item.contains("tool_calls") || item.at("tool_calls").is_null()) { return calls; }
    const Json& tool_calls = item.at("tool_calls");
    if (!tool_calls.is_array() || tool_calls.empty()) {
        bad_request("assistant message " + std::to_string(index) +
                        " tool_calls must be a non-empty array",
                    "messages");
    }
    calls.reserve(tool_calls.size());
    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
        const Json& call = tool_calls.at(i);
        if (!call.is_object()) { bad_request("tool_calls entries must be objects", "messages"); }
        if (!call.contains("id") || !call.at("id").is_string() ||
            call.at("id").get<std::string>().empty()) {
            bad_request("tool_calls entries must contain a string id", "messages");
        }
        if (!call.contains("type") || !call.at("type").is_string() ||
            call.at("type").get<std::string>() != "function") {
            bad_request("only function tool_calls are supported", "messages",
                        "tool_type_not_supported");
        }
        if (!call.contains("function") || !call.at("function").is_object()) {
            bad_request("tool_calls entries must contain a function object", "messages");
        }
        const Json& fn = call.at("function");
        ToolCall out;
        out.id   = call.at("id").get<std::string>();
        out.name = require_function_name(fn, "messages");
        if (!fn.contains("arguments") || !fn.at("arguments").is_string()) {
            bad_request("function tool_calls must contain string arguments", "messages");
        }
        out.arguments_json = fn.at("arguments").get<std::string>();
        const Json parsed  = Json::parse(out.arguments_json, nullptr, false);
        if (parsed.is_discarded() || !parsed.is_object()) {
            bad_request("function tool_call arguments must be a JSON object string", "messages");
        }
        calls.push_back(std::move(out));
    }
    return calls;
}

void parse_messages(const Json& body, GenerationRequest& out) {
    if (!body.contains("messages")) { bad_request("missing required field: messages", "messages"); }
    const Json& messages = body.at("messages");
    if (!messages.is_array() || messages.empty()) {
        bad_request("messages must be a non-empty array", "messages");
    }
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const Json& item = messages.at(i);
        if (!item.is_object()) {
            bad_request("message " + std::to_string(i) + " must be an object", "messages");
        }
        if (!item.contains("role") || !item.at("role").is_string()) {
            bad_request("message " + std::to_string(i) + " must have a string role", "messages");
        }
        const std::string role     = item.at("role").get<std::string>();
        const ChatRole parsed_role = parse_message_role(role);
        if (item.contains("function_call") && !item.at("function_call").is_null()) {
            bad_request("message function_call is not supported", "messages",
                        "tools_not_supported");
        }
        ChatTurn turn;
        turn.role = parsed_role;
        if (parsed_role == ChatRole::Tool) {
            if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
                bad_request("tool messages must not contain tool_calls", "messages");
            }
            if (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string() ||
                item.at("tool_call_id").get<std::string>().empty()) {
                bad_request("tool messages must contain a string tool_call_id", "messages");
            }
            if (!item.contains("content") || !item.at("content").is_string()) {
                bad_request("tool messages must contain string content", "messages");
            }
            turn.tool_call_id = item.at("tool_call_id").get<std::string>();
            turn.content.push_back(
                ContentPart{ContentKind::Text, item.at("content").get<std::string>(), "text"});
            out.messages.push_back(std::move(turn));
            continue;
        }
        if (item.contains("tool_call_id") && !item.at("tool_call_id").is_null()) {
            bad_request("tool_call_id is only valid on tool messages", "messages");
        }
        if (parsed_role == ChatRole::Assistant) {
            turn.tool_calls = parse_assistant_tool_calls(item, i);
            if (item.contains("content") && !item.at("content").is_null()) {
                parse_content_parts(item.at("content"), turn, i);
            } else if (turn.tool_calls.empty()) {
                bad_request("assistant message " + std::to_string(i) +
                                " must have content or tool_calls",
                            "messages");
            }
            if (item.contains("reasoning_content") && !item.at("reasoning_content").is_null()) {
                if (!item.at("reasoning_content").is_string()) {
                    bad_request("assistant message " + std::to_string(i) +
                                    " reasoning_content must be a string",
                                "messages");
                }
                turn.reasoning_content = item.at("reasoning_content").get<std::string>();
            }
            out.messages.push_back(std::move(turn));
            continue;
        }
        if (item.contains("tool_calls") && !item.at("tool_calls").is_null()) {
            bad_request("tool_calls are only valid on assistant messages", "messages");
        }
        if (!item.contains("content") || item.at("content").is_null()) {
            bad_request("message " + std::to_string(i) + " must have content", "messages");
        }
        parse_content_parts(item.at("content"), turn, i);
        out.messages.push_back(std::move(turn));
    }
}

void parse_tools(const Json& body, GenerationRequest& out) {
    if (!body.contains("tools") || body.at("tools").is_null()) { return; }
    const Json& tools = body.at("tools");
    if (!tools.is_array()) { bad_request("tools must be an array", "tools"); }
    out.tools.reserve(tools.size());
    for (std::size_t i = 0; i < tools.size(); ++i) {
        const Json& item = tools.at(i);
        if (!item.is_object()) { bad_request("tools entries must be objects", "tools"); }
        if (!item.contains("type") || !item.at("type").is_string()) {
            bad_request("tools entries must contain a string type", "tools");
        }
        if (item.at("type").get<std::string>() != "function") {
            bad_request("only function tools are supported", "tools", "tool_type_not_supported");
        }
        if (!item.contains("function") || !item.at("function").is_object()) {
            bad_request("function tools must contain a function object", "tools");
        }
        Json normalized = item;
        Json& fn        = normalized["function"];
        ToolDefinition tool;
        tool.name = require_function_name(fn, "tools");
        if (fn.contains("description") && !fn.at("description").is_null()) {
            if (!fn.at("description").is_string()) {
                bad_request("function description must be a string", "tools");
            }
            tool.description = fn.at("description").get<std::string>();
        }
        if (!fn.contains("parameters") || fn.at("parameters").is_null()) {
            fn["parameters"] = Json{{"type", "object"}, {"properties", Json::object()}};
        }
        if (!fn.at("parameters").is_object()) {
            bad_request("function parameters must be a JSON object", "tools");
        }
        tool.parameters_json = fn.at("parameters").dump();
        if (fn.contains("strict") && !fn.at("strict").is_null()) {
            if (!fn.at("strict").is_boolean()) {
                bad_request("function strict must be a boolean", "tools");
            }
            tool.strict = fn.at("strict").get<bool>();
            tool.strict_set = true;
        } else {
            fn["strict"] = false;
        }
        tool.definition_json = normalized.dump();
        out.tools.push_back(std::move(tool));
    }
}

void parse_tool_choice(const Json& body, GenerationRequest& out) {
    if (!body.contains("tool_choice") || body.at("tool_choice").is_null()) { return; }
    const Json& choice = body.at("tool_choice");
    if (choice.is_string()) {
        const std::string value = choice.get<std::string>();
        if (value == "none") {
            out.tool_choice.mode = ToolChoiceMode::None;
        } else if (value == "auto") {
            out.tool_choice.mode = ToolChoiceMode::Auto;
        } else if (value == "required") {
            out.tool_choice.mode = ToolChoiceMode::Required;
        } else {
            bad_request("tool_choice must be 'none', 'auto', 'required', or a function choice",
                        "tool_choice");
        }
    } else if (choice.is_object()) {
        if (!choice.contains("type") || !choice.at("type").is_string() ||
            choice.at("type").get<std::string>() != "function") {
            bad_request("only function tool_choice objects are supported", "tool_choice",
                        "tool_type_not_supported");
        }
        if (!choice.contains("function") || !choice.at("function").is_object()) {
            bad_request("function tool_choice must contain a function object", "tool_choice");
        }
        out.tool_choice.mode = ToolChoiceMode::Named;
        out.tool_choice.name = require_function_name(choice.at("function"), "tool_choice");
    } else {
        bad_request("tool_choice must be a string or object", "tool_choice");
    }
    if (out.tool_choice.mode != ToolChoiceMode::None && out.tools.empty()) {
        bad_request("tool_choice requires tools", "tool_choice");
    }
    if (out.tool_choice.mode == ToolChoiceMode::Named &&
        !has_tool_named(out, out.tool_choice.name)) {
        bad_request("tool_choice references unknown function: " + out.tool_choice.name,
                    "tool_choice");
    }
}

void parse_stop(const Json& body, GenerationRequest& out) {
    if (body.contains("ignore_eos") && body.at("ignore_eos").is_boolean()) {
        out.ignore_eos = body.at("ignore_eos").get<bool>();
    }
    if (const std::optional<int> minimum = get_int(body, "min_tokens")) {
        if (*minimum < 0) { bad_request("min_tokens must be nonnegative", "min_tokens"); }
        out.min_tokens = *minimum;
    }
    // `logprobs` is OpenAI's own; `return_token_ids` is vLLM's extension, and an
    // RL client sends both. They were accepted and ignored before, which is worse
    // than refusing them: the rollout came back looking complete and carried
    // nothing to train on.
    if (body.contains("logprobs") && body.at("logprobs").is_boolean()) {
        out.want_logprobs = body.at("logprobs").get<bool>();
    }
    if (body.contains("return_token_ids") && body.at("return_token_ids").is_boolean()) {
        out.return_token_ids = body.at("return_token_ids").get<bool>();
    }
    // Fields this engine does not implement. Accepting them and generating anyway
    // is the failure this whole endpoint has been bitten by twice: the caller gets
    // a response that looks complete and is not what it asked for. A value that
    // asks for nothing -- the defaults every client sends -- is still accepted.
    if (body.contains("prompt_logprobs") && !body.at("prompt_logprobs").is_null() &&
        !(body.at("prompt_logprobs").is_boolean() && !body.at("prompt_logprobs").get<bool>())) {
        bad_request("prompt_logprobs is not implemented: scoring every prompt position needs the "
                    "logits of a whole prefill, which this engine does not retain",
                    "prompt_logprobs");
    }
    if (body.contains("add_generation_prompt") &&
        body.at("add_generation_prompt").is_boolean()) {
        out.add_generation_prompt = body.at("add_generation_prompt").get<bool>();
    }
    if (body.contains("tokens") && !body.at("tokens").is_null()) {
        const Json& tokens = body.at("tokens");
        if (!tokens.is_array()) { bad_request("tokens must be an array of integers", "tokens"); }
        out.prompt_token_ids.reserve(tokens.size());
        for (const Json& item : tokens) {
            if (!item.is_number_integer()) {
                bad_request("tokens must be an array of integers", "tokens");
            }
            out.prompt_token_ids.push_back(static_cast<sinfer::TokenId>(item.get<std::int64_t>()));
        }
        if (out.prompt_token_ids.empty()) { bad_request("tokens must not be empty", "tokens"); }
    }
    if (!body.contains("stop") || body.at("stop").is_null()) { return; }
    const Json& stop = body.at("stop");
    if (stop.is_string()) {
        if (!stop.get<std::string>().empty()) {
            out.stop_strings.push_back(stop.get<std::string>());
        }
        return;
    }
    if (stop.is_array()) {
        for (const Json& s : stop) {
            if (!s.is_string()) { bad_request("stop entries must be strings", "stop"); }
            if (!s.get<std::string>().empty()) { out.stop_strings.push_back(s.get<std::string>()); }
        }
        return;
    }
    bad_request("stop must be a string or array of strings", "stop");
}

void parse_sampling(const Json& body, GenerationRequest& out) {
    SamplingParams& s   = out.sampling;
    s.temperature       = get_number(body, "temperature");
    s.top_p             = get_number(body, "top_p");
    s.top_k             = get_int(body, "top_k");
    s.min_p             = get_number(body, "min_p");
    s.repetition_penalty = get_number(body, "repetition_penalty");
    s.presence_penalty  = get_number(body, "presence_penalty");
    s.frequency_penalty = get_number(body, "frequency_penalty");
    s.seed              = get_u64(body, "seed");
    if (body.contains("logit_bias") && !body.at("logit_bias").is_null()) {
        const Json& bias = body.at("logit_bias");
        if (!bias.is_object()) { bad_request("logit_bias must be an object", "logit_bias"); }
        for (auto it = bias.begin(); it != bias.end(); ++it) {
            if (!it.value().is_number()) {
                bad_request("logit_bias values must be numbers", "logit_bias");
            }
            try {
                std::size_t consumed = 0;
                const int token = std::stoi(it.key(), &consumed);
                if (consumed != it.key().size() || token < 0 ||
                    it.key().find_first_not_of("0123456789") != std::string::npos) {
                    throw std::invalid_argument("invalid token id");
                }
                s.logit_bias.emplace(token, it.value().get<double>());
            } catch (const std::exception&) {
                bad_request("logit_bias keys must be integer token ids", "logit_bias");
            }
        }
    }
    if (const std::optional<int> n = get_int(body, "n")) {
        s.n = *n;
        if (s.n != 1) {
            ApiError error;
            error.message = "n>1 is not supported yet";
            error.param   = "n";
            error.code    = "n_not_supported";
            throw ApiException(std::move(error));
        }
    }
}

void parse_output_features(const Json& body, GenerationRequest& out) {
    for (const char* key : {"functions", "function_call"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            ApiError error;
            error.message = std::string(key) + " is not supported yet";
            error.param   = key;
            error.code    = "tools_not_supported";
            throw ApiException(std::move(error));
        }
    }
    if (body.contains("response_format") && !body.at("response_format").is_null()) {
        out.json_schema = parse_json_response_format(body.at("response_format"), "response_format");
    }
}

Json completion_base_chunk(const std::string& id, const std::string& model,
                           std::int64_t created) {
    return Json{
        {"id", id}, {"object", "text_completion"}, {"created", created}, {"model", model}};
}

Json base_chunk(const std::string& id, const std::string& model, std::int64_t created) {
    return Json{
        {"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model}};
}

Json tool_calls_json(const std::vector<ToolCall>& tool_calls, bool include_index) {
    Json out = Json::array();
    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
        const ToolCall& call = tool_calls[i];
        Json item            = {{"id", call.id},
                                {"type", "function"},
                                {"function", Json{{"name", call.name}, {"arguments", call.arguments_json}}}};
        if (include_index) { item["index"] = static_cast<int>(i); }
        out.push_back(std::move(item));
    }
    return out;
}

std::string sse_event(const Json& payload) { return "data: " + dump_lossy(payload) + "\n\n"; }

} // namespace

std::optional<bool> parse_openai_preserve_thinking(const Json& body) {
    std::optional<bool> top_level;
    if (body.contains("preserve_thinking") && !body.at("preserve_thinking").is_null()) {
        if (!body.at("preserve_thinking").is_boolean()) {
            bad_request("preserve_thinking must be a boolean or null", "preserve_thinking");
        }
        top_level = body.at("preserve_thinking").get<bool>();
    }

    std::optional<bool> template_value;
    if (body.contains("chat_template_kwargs")) {
        const Json& kwargs = body.at("chat_template_kwargs");
        if (!kwargs.is_object()) {
            bad_request("chat_template_kwargs must be an object", "chat_template_kwargs");
        }
        for (auto it = kwargs.begin(); it != kwargs.end(); ++it) {
            // `enable_thinking` is where every other server takes Qwen3's thinking
            // switch, and this engine has always had the switch -- it just read it
            // only from the top level, so a client that put it where the model card
            // says to got its whole request refused.
            if (it.key() != "preserve_thinking" && it.key() != "enable_thinking" &&
                !it.value().is_null()) {
                bad_request("chat_template_kwargs." + it.key() + " is not supported",
                            "chat_template_kwargs", "chat_template_option_not_supported");
            }
        }
        if (kwargs.contains("preserve_thinking") && !kwargs.at("preserve_thinking").is_null()) {
            if (!kwargs.at("preserve_thinking").is_boolean()) {
                bad_request("chat_template_kwargs.preserve_thinking must be a boolean or null",
                            "chat_template_kwargs");
            }
            template_value = kwargs.at("preserve_thinking").get<bool>();
        }
    }

    if (top_level && template_value && *top_level != *template_value) {
        bad_request("conflicting preserve_thinking values", "preserve_thinking",
                    "conflicting_template_option");
    }
    return template_value ? template_value : top_level;
}

void parse_openai_reasoning_effort(const Json& body, GenerationRequest& out) {
    if (!body.contains("reasoning_effort") || body.at("reasoning_effort").is_null()) { return; }
    if (!body.at("reasoning_effort").is_string()) {
        bad_request("reasoning_effort must be a string or null", "reasoning_effort");
    }
    const std::string value = body.at("reasoning_effort").get<std::string>();
    const std::optional<RequestedReasoningEffort> effort = parse_requested_reasoning_effort(value);
    if (!effort) {
        bad_request("reasoning_effort must be one of none, minimal, low, medium, high, xhigh, or "
                    "max",
                    "reasoning_effort");
    }
    out.reasoning_effort       = *effort;
    out.reasoning_effort_param = "reasoning_effort";
}

// The prompt a /v1/completions body carries. OpenAI allows a string, a list of strings, a
// token list, or a list of token lists; the batch forms are one request per prompt, and this
// server answers one prompt per request, so a single-element list is accepted as the string it
// wraps and anything longer is refused rather than silently answering only the first.
std::string parse_completion_prompt(const Json& body) {
    if (!body.contains("prompt")) { bad_request("missing required field: prompt", "prompt"); }
    const Json& prompt = body.at("prompt");
    if (prompt.is_string()) { return prompt.get<std::string>(); }
    if (prompt.is_array()) {
        if (prompt.size() == 1 && prompt.front().is_string()) {
            return prompt.front().get<std::string>();
        }
        if (!prompt.empty() && prompt.front().is_number_integer()) {
            bad_request("prompt must be text; pre-tokenized prompts are not supported", "prompt",
                        "token_prompt_not_supported");
        }
        bad_request("prompt must be a string or a one-element list; this server answers one "
                    "prompt per request",
                    "prompt", "prompt_batch_not_supported");
    }
    bad_request("prompt must be a string", "prompt");
}

void reject_unsupported_completion_features(const Json& body) {
    // Each of these changes what the response means, so answering without them would be
    // answering a different question than the one asked.
    for (const char* key : {"echo", "logprobs", "suffix", "best_of"}) {
        if (body.contains(key) && !body.at(key).is_null()) {
            const Json& value = body.at(key);
            const bool inert  = (value.is_boolean() && !value.get<bool>()) ||
                               (value.is_number_integer() && value.get<int>() <= 1 &&
                                std::string_view(key) == "best_of");
            if (inert) { continue; }
            ApiError error;
            error.message = std::string(key) + " is not supported";
            error.param   = key;
            error.code    = "completion_option_not_supported";
            throw ApiException(std::move(error));
        }
    }
}

std::string parse_json_response_format(const Json& format, const std::string& param, bool flat_schema) {
    if (!format.is_object() || !format.contains("type") || !format["type"].is_string()) {
        bad_request(param + " must contain a string type", param);
    }
    const auto type = format["type"].get<std::string>();
    if (type == "text") { return {}; }
    if (type == "json_object") { return R"({"type":"object"})"; }
    if (type != "json_schema") { bad_request(param + " type must be text, json_object, or json_schema", param); }
    if (!flat_schema && (!format.contains("json_schema") || !format["json_schema"].is_object())) {
        bad_request(param + ".json_schema must be an object", param);
    }
    const auto& spec = flat_schema ? format : format["json_schema"];
    if (!spec.contains("schema") || !(spec["schema"].is_object() || spec["schema"].is_boolean())) {
        bad_request(param + " schema must be an object or boolean", param);
    }
    if (spec.contains("strict") && !spec["strict"].is_boolean()) {
        bad_request(param + " strict must be boolean", param);
    }
    if (spec.contains("name") && (!spec["name"].is_string() || spec["name"].get<std::string>().empty())) {
        bad_request(param + " name must be a nonempty string", param);
    }
    return spec["schema"].dump();
}

GenerationRequest parse_chat_completion_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    GenerationRequest out;
    parse_output_features(body, out);
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.model = body.at("model").get<std::string>();

    parse_tools(body, out);
    parse_tool_choice(body, out);
    out.parallel_tool_calls = get_bool(body, "parallel_tool_calls", true);
    parse_messages(body, out);
    parse_stop(body, out);
    parse_sampling(body, out);

    out.stream = get_bool(body, "stream", false);
    if (body.contains("stream_options") && body.at("stream_options").is_object()) {
        out.include_usage = get_bool(body.at("stream_options"), "include_usage", false);
    }
    if (body.contains("chat_template_kwargs") && body.at("chat_template_kwargs").is_object() &&
        body.at("chat_template_kwargs").contains("enable_thinking") &&
        !body.at("chat_template_kwargs").at("enable_thinking").is_null()) {
        const Json& value = body.at("chat_template_kwargs").at("enable_thinking");
        if (!value.is_boolean()) {
            bad_request("chat_template_kwargs.enable_thinking must be a boolean or null",
                        "chat_template_kwargs");
        }
        out.enable_thinking       = value.get<bool>();
        out.enable_thinking_param = "chat_template_kwargs.enable_thinking";
    }
    if (body.contains("enable_thinking") && !body.at("enable_thinking").is_null()) {
        out.enable_thinking       = get_bool(body, "enable_thinking", false);
        out.enable_thinking_param = "enable_thinking";
    }
    parse_openai_reasoning_effort(body, out);
    out.preserve_thinking = parse_openai_preserve_thinking(body);

    std::optional<int> max_tokens = get_int(body, "max_completion_tokens");
    if (!max_tokens) { max_tokens = get_int(body, "max_tokens"); }
    if (max_tokens) {
        if (*max_tokens <= 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        out.max_tokens     = *max_tokens;
        out.max_tokens_set = true;
    } else {
        out.max_tokens     = limits.default_max_tokens;
        out.max_tokens_set = false;
    }
    return out;
}

static Json chat_completion_payload(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage,
                                          const TokenDetail& detail) {
    Json message = {{"role", "assistant"}, {"content", content}};
    if (!reasoning.empty()) { message["reasoning_content"] = reasoning; }
    Json choice = {{"index", 0}, {"message", std::move(message)},
                   {"finish_reason", finish_reason}};
    if (detail.include_logprobs) {
        // OpenAI's shape: one entry per generated token, in order. `top_logprobs`
        // stays empty -- nothing asks this engine for alternatives, and an empty
        // array is what the schema says when none were requested.
        Json entries = Json::array();
        for (std::size_t i = 0; i < detail.logprobs.size(); ++i) {
            const std::string& text = i < detail.texts.size() ? detail.texts[i] : std::string{};
            // `bytes` carries the token exactly as the model emitted it. `token` is
            // the same bytes as a string and may not be valid UTF-8 on its own -- a
            // character wider than one byte is split across tokens -- so the dump
            // substitutes U+FFFD there while these stay exact.
            Json raw = Json::array();
            for (const unsigned char byte : text) { raw.push_back(static_cast<int>(byte)); }
            entries.push_back(Json{{"token", text},
                                   {"logprob", detail.logprobs[i]},
                                   {"bytes", std::move(raw)},
                                   {"top_logprobs", Json::array()}});
        }
        choice["logprobs"] = Json{{"content", std::move(entries)}};
    }
    if (detail.include_token_ids) { choice["token_ids"] = detail.completion_token_ids; }
    Json payload = {
        {"id", id},
        {"object", "chat.completion"},
        {"created", created},
        {"model", model},
        {"choices", Json::array({std::move(choice)})},
        {"usage", Json{{"prompt_tokens", usage.prompt_tokens},
                       {"completion_tokens", usage.completion_tokens},
                       {"total_tokens", usage.prompt_tokens + usage.completion_tokens}}}};
    // The prompt's ids sit at the top level, where vLLM puts them, because they
    // belong to the request rather than to any one choice.
    if (detail.include_token_ids) { payload["prompt_token_ids"] = detail.prompt_token_ids; }
    return payload;
}

std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage, const TokenDetail& detail) {
    return dump_lossy(chat_completion_payload(id, model, created, content, reasoning, finish_reason, usage, detail));
}

std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const CompletionUsage& usage, const TokenDetail& detail) {
    Json payload = chat_completion_payload(id, model, created, content, reasoning, "tool_calls", usage, detail);
    Json& message = payload["choices"][0]["message"];
    if (content.empty()) { message["content"] = nullptr; }
    message["tool_calls"] = tool_calls_json(tool_calls, false);
    return dump_lossy(payload);
}

std::string make_chat_chunk_token_detail(const std::string& id, const std::string& model,
                                         std::int64_t created, const TokenDetail& detail, bool include_usage) {
    Json payload = chat_completion_payload(id, model, created, "", "", "", {}, detail);
    payload["object"] = "chat.completion.chunk";
    Json& choice = payload["choices"][0];
    choice.erase("message");
    choice["delta"] = Json::object();
    choice["finish_reason"] = nullptr;
    payload.erase("usage");
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"delta", Json{{"role", "assistant"}, {"content", ""}}},
                                           {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"delta", Json{{"reasoning_content", delta_text}}},
                                           {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0}, {"delta", Json{{"content", delta_text}}}, {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_tool_calls(const std::string& id, const std::string& model,
                                       std::int64_t created,
                                       const std::vector<ToolCall>& tool_calls,
                                       bool include_usage) {
    Json payload = base_chunk(id, model, created);
    payload["choices"] =
        Json::array({Json{{"index", 0},
                          {"delta", Json{{"tool_calls", tool_calls_json(tool_calls, true)}}},
                          {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                  std::int64_t created, const char* finish_reason,
                                  bool include_usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array(
        {Json{{"index", 0}, {"delta", Json::object()}, {"finish_reason", finish_reason}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage) {
    Json payload       = base_chunk(id, model, created);
    payload["choices"] = Json::array();
    payload["usage"]   = Json{{"prompt_tokens", usage.prompt_tokens},
                              {"completion_tokens", usage.completion_tokens},
                              {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
    return sse_event(payload);
}

std::string sse_done() { return "data: [DONE]\n\n"; }

std::string make_models_list(const std::string& model_id, std::int64_t created,
                             const std::vector<std::string>& adapters) {
    Json data = Json::array({Json{{"id", model_id},
                                  {"object", "model"},
                                  {"created", created},
                                  {"owned_by", "sinfer"}}});
    // Adapters are selectable model ids, so they belong in the listing: a client
    // discovers what it may put in `model`, and an adapter that is loaded but
    // invisible here cannot be found by anything that enumerates first.
    for (const std::string& adapter : adapters) {
        data.push_back(Json{{"id", adapter},
                            {"object", "model"},
                            {"created", created},
                            {"owned_by", "sinfer"},
                            {"parent", model_id}});
    }
    const Json payload = {{"object", "list"}, {"data", std::move(data)}};
    return dump_lossy(payload);
}

std::string make_model_object(const std::string& model_id, std::int64_t created) {
    const Json payload = {
        {"id", model_id}, {"object", "model"}, {"created", created}, {"owned_by", "sinfer"}};
    return dump_lossy(payload);
}

std::string make_error_body(const ApiError& error) {
    Json err     = {{"message", error.message}, {"type", error.type}};
    err["param"] = error.param.empty() ? Json(nullptr) : Json(error.param);
    err["code"]  = error.code.empty() ? Json(nullptr) : Json(error.code);
    return Json{{"error", err}}.dump();
}

GenerationRequest parse_completion_request(const Json& body, const RequestLimits& limits) {
    require_object(body);
    reject_unsupported_completion_features(body);

    GenerationRequest out;
    if (!body.contains("model") || !body.at("model").is_string() ||
        body.at("model").get<std::string>().empty()) {
        bad_request("missing required field: model", "model");
    }
    out.model      = body.at("model").get<std::string>();
    out.raw_prompt = parse_completion_prompt(body);

    parse_stop(body, out);
    parse_sampling(body, out);

    out.stream = get_bool(body, "stream", false);
    if (body.contains("stream_options") && body.at("stream_options").is_object()) {
        out.include_usage = get_bool(body.at("stream_options"), "include_usage", false);
    }

    const std::optional<int> max_tokens = get_int(body, "max_tokens");
    if (max_tokens) {
        if (*max_tokens <= 0) { bad_request("max_tokens must be positive", "max_tokens"); }
        out.max_tokens     = *max_tokens;
        out.max_tokens_set = true;
    } else {
        out.max_tokens     = limits.default_max_tokens;
        out.max_tokens_set = false;
    }
    return out;
}

std::string make_completion_response(const std::string& id, const std::string& model,
                                     std::int64_t created, const std::string& text,
                                     const char* finish_reason, const CompletionUsage& usage) {
    const Json payload = {
        {"id", id},
        {"object", "text_completion"},
        {"created", created},
        {"model", model},
        {"choices", Json::array({Json{{"index", 0},
                                      {"text", text},
                                      {"logprobs", nullptr},
                                      {"finish_reason", finish_reason}}})},
        {"usage", Json{{"prompt_tokens", usage.prompt_tokens},
                       {"completion_tokens", usage.completion_tokens},
                       {"total_tokens", usage.prompt_tokens + usage.completion_tokens}}}};
    return dump_lossy(payload);
}

std::string make_completion_chunk_text(const std::string& id, const std::string& model,
                                       std::int64_t created, const std::string& delta_text,
                                       bool include_usage) {
    Json payload       = completion_base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"text", delta_text},
                                           {"logprobs", nullptr},
                                           {"finish_reason", nullptr}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_completion_chunk_final(const std::string& id, const std::string& model,
                                        std::int64_t created, const char* finish_reason,
                                        bool include_usage) {
    Json payload       = completion_base_chunk(id, model, created);
    payload["choices"] = Json::array({Json{{"index", 0},
                                           {"text", ""},
                                           {"logprobs", nullptr},
                                           {"finish_reason", finish_reason}}});
    if (include_usage) { payload["usage"] = nullptr; }
    return sse_event(payload);
}

std::string make_completion_chunk_usage(const std::string& id, const std::string& model,
                                        std::int64_t created, const CompletionUsage& usage) {
    Json payload       = completion_base_chunk(id, model, created);
    payload["choices"] = Json::array();
    payload["usage"]   = Json{{"prompt_tokens", usage.prompt_tokens},
                              {"completion_tokens", usage.completion_tokens},
                              {"total_tokens", usage.prompt_tokens + usage.completion_tokens}};
    return sse_event(payload);
}

std::string new_completion_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%016llx", static_cast<unsigned long long>(dist(rng)));
    return "cmpl-" + std::string(buf.data());
}

std::string new_chat_completion_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<std::uint64_t> dist;
    std::array<char, 32> buf{};
    std::snprintf(buf.data(), buf.size(), "%016llx", static_cast<unsigned long long>(dist(rng)));
    return "chatcmpl-" + std::string(buf.data());
}

std::int64_t unix_time_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

} // namespace sinfer::serve
