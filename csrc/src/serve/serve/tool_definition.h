#pragma once

// The tool object a chat template is handed (`{{ tool | tojson }}`), shared by every wire
// adapter that accepts function tools (Chat Completions, Responses, Anthropic Messages).

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <utility>

namespace sinfer::serve {

/// The fields of a function tool that reach the prompt.
struct TemplateToolFields {
    std::string name;
    std::optional<std::string> description;
    /// The parameters schema exactly as the client wrote it, key order included; null when the
    /// client sent none.
    nlohmann::ordered_json parameters;
    std::optional<bool> strict;
    /// `defer_loading` on the function object, and on the tool object around it.
    std::optional<bool> defer_loading;
    std::optional<bool> tool_defer_loading;
};

/// The tool as vLLM's OpenAI-compatible server renders it: `ChatCompletionToolsParam.model_dump()`
/// (vLLM 0.30/0.31, the server the reference measurements were made on), which is
///
///   {"type": "function", "function": {"name": N, "description": D|null, "parameters": P|null
///    [, "strict": S][, "defer_loading": L]}[, "defer_loading": L]}
///
/// -- declared fields only, in declaration order, whatever order the client sent them in. An absent
/// description or parameters schema is null (the pydantic default), `strict` and `defer_loading`
/// appear only when the client sent them, and keys the client added beside them are dropped. A
/// tool-level `defer_loading` is also copied onto the function when that has none, as vLLM's
/// validator does. The parameters schema keeps the client's own key order: properties are listed
/// to the model in the order the client wrote them, which is also the order the model is expected
/// to write the arguments back in.
///
/// Rendering the client's request any other way changes the prompt the model was trained and
/// measured on. Re-serialising it through a sorted map put the keys in alphabetical order, and an
/// injected `"strict": false` added a field no client sent; both cost tool-calling accuracy.
[[nodiscard]] inline std::string template_tool_json(const TemplateToolFields& fields) {
    using OrderedJson         = nlohmann::ordered_json;
    OrderedJson function      = OrderedJson::object();
    function["name"]          = fields.name;
    function["description"]   = fields.description ? OrderedJson(*fields.description) : OrderedJson(nullptr);
    function["parameters"]    = fields.parameters.is_object() ? fields.parameters : OrderedJson(nullptr);
    if (fields.strict) { function["strict"] = *fields.strict; }
    if (const std::optional<bool> defer = fields.defer_loading ? fields.defer_loading
                                                               : fields.tool_defer_loading) {
        function["defer_loading"] = *defer;
    }
    OrderedJson tool = OrderedJson::object();
    tool["type"]     = "function";
    tool["function"] = std::move(function);
    if (fields.tool_defer_loading) { tool["defer_loading"] = *fields.tool_defer_loading; }
    return tool.dump();
}

} // namespace sinfer::serve
