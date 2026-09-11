#include "serve/tool_constraints.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <set>
namespace sinfer::serve {
namespace {
using Json = nlohmann::json;
// Reject constraints the grammar compiler would otherwise ignore. Metadata is harmless,
// but an unsupported assertion must never silently weaken the requested language.
void validate_schema(const nlohmann::json& schema, unsigned depth = 0) {
    using Json = nlohmann::json;
    if (depth > 64) { throw std::invalid_argument("JSON schema nesting exceeds 64 levels"); }
    if (schema.is_boolean()) { return; }
    if (!schema.is_object()) { throw std::invalid_argument("JSON schema must be an object or boolean"); }
    static const std::set<std::string> metadata{
        "$schema", "$id", "$comment", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly"};
    static const std::set<std::string> assertions{
        "type", "enum", "const", "properties", "required", "additionalProperties", "items", "prefixItems",
        "minItems", "maxItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "anyOf", "$ref", "$defs", "definitions"};
    for (auto it = schema.begin(); it != schema.end(); ++it) {
        const auto& key = it.key();
        if (!metadata.contains(key) && !assertions.contains(key)) {
            throw std::invalid_argument("unsupported tool argument schema keyword: " + key);
        }
        if (key == "properties" || key == "$defs" || key == "definitions") {
            if (!it.value().is_object()) { throw std::invalid_argument(key + " must be an object"); }
            for (const auto& child : it.value()) { validate_schema(child, depth + 1); }
        } else if (key == "items" || key == "additionalProperties") {
            validate_schema(it.value(), depth + 1);
        } else if (key == "prefixItems" || key == "anyOf") {
            if (!it.value().is_array()) { throw std::invalid_argument(key + " must be an array"); }
            for (const auto& child : it.value()) { validate_schema(child, depth + 1); }
        }
    }
    if (schema.contains("required")) {
        if (!schema["required"].is_array()) { throw std::invalid_argument("required must be an array"); }
        for (const auto& name : schema["required"]) {
            if (!name.is_string() || !schema.contains("properties") || !schema["properties"].contains(name.get<std::string>())) {
                throw std::invalid_argument("required properties must be declared in properties");
            }
        }
    }
    for (const char* exclusive : {"$ref", "anyOf", "enum", "const"}) {
        if (!schema.contains(exclusive)) { continue; }
        for (auto it = schema.begin(); it != schema.end(); ++it) {
            if (it.key() == exclusive || metadata.contains(it.key()) || it.key() == "$defs" || it.key() == "definitions") { continue; }
            // The conventional enum + type form is safe when every member has that type.
            if (it.key() == "type" && (std::string(exclusive) == "enum" || std::string(exclusive) == "const") && it.value().is_string()) {
                const auto type = it.value().get<std::string>();
                const auto matches = [&](const Json& v) {
                    return (type == "string" && v.is_string()) || (type == "boolean" && v.is_boolean()) ||
                           (type == "null" && v.is_null()) || (type == "integer" && v.is_number_integer()) ||
                           (type == "number" && v.is_number()) || (type == "object" && v.is_object()) ||
                           (type == "array" && v.is_array());
                };
                const auto values = std::string(exclusive) == "enum" ? schema[exclusive] : Json::array({schema[exclusive]});
                if (values.is_array() && std::all_of(values.begin(), values.end(), matches)) { continue; }
            }
            throw std::invalid_argument(std::string("unsupported tool argument schema combination: ") + exclusive + " with " + it.key());
        }
    }
    if (schema.contains("$ref") && (!schema["$ref"].is_string() || !schema["$ref"].get<std::string>().starts_with("#"))) {
        throw std::invalid_argument("only local JSON schema references are supported");
    }
}


bool matches(const Json& value, const Json& schema, const Json& root, unsigned depth = 0) {
    if (depth > 256) { return false; }
    if (schema.is_boolean()) { return schema.get<bool>(); }
    if (schema.contains("$ref")) {
        const auto ref = schema["$ref"].get<std::string>();
        return matches(value, root.at(Json::json_pointer(ref.substr(1))), root, depth + 1);
    }
    if (schema.contains("const") && value != schema["const"]) { return false; }
    if (schema.contains("enum") && std::find(schema["enum"].begin(), schema["enum"].end(), value) == schema["enum"].end()) { return false; }
    if (schema.contains("anyOf") && std::none_of(schema["anyOf"].begin(), schema["anyOf"].end(),
        [&](const Json& branch) { return matches(value, branch, root, depth + 1); })) { return false; }
    const auto has_type = [&](const Json& type) {
        return (type == "object" && value.is_object()) || (type == "array" && value.is_array()) ||
            (type == "string" && value.is_string()) || (type == "boolean" && value.is_boolean()) ||
            (type == "null" && value.is_null()) || (type == "number" && value.is_number()) ||
            (type == "integer" && value.is_number() && std::floor(value.get<double>()) == value.get<double>());
    };
    if (schema.contains("type")) {
        const auto& types = schema["type"];
        if (types.is_array() ? std::none_of(types.begin(), types.end(), has_type) : !has_type(types)) { return false; }
    }
    if (value.is_number()) {
        const double number = value.get<double>();
        if (!std::isfinite(number)) { return false; }
        if (schema.contains("minimum") && number < schema["minimum"].get<double>()) { return false; }
        if (schema.contains("maximum") && number > schema["maximum"].get<double>()) { return false; }
        if (schema.contains("exclusiveMinimum") && number <= schema["exclusiveMinimum"].get<double>()) { return false; }
        if (schema.contains("exclusiveMaximum") && number >= schema["exclusiveMaximum"].get<double>()) { return false; }
    }
    if (value.is_array()) {
        if (schema.contains("minItems") && value.size() < schema["minItems"].get<size_t>()) { return false; }
        if (schema.contains("maxItems") && value.size() > schema["maxItems"].get<size_t>()) { return false; }
        const auto prefix = schema.value("prefixItems", Json::array());
        for (size_t i = 0; i < value.size(); ++i) {
            if (i < prefix.size()) {
                if (!matches(value[i], prefix[i], root, depth + 1)) { return false; }
            } else if (schema.contains("items") && !matches(value[i], schema["items"], root, depth + 1)) { return false; }
        }
    }
    if (value.is_object()) {
        for (const auto& required : schema.value("required", Json::array())) {
            if (!value.contains(required.get<std::string>())) { return false; }
        }
        const auto properties = schema.value("properties", Json::object());
        for (auto it = value.begin(); it != value.end(); ++it) {
            if (properties.contains(it.key())) {
                if (!matches(it.value(), properties[it.key()], root, depth + 1)) { return false; }
            } else if (schema.contains("additionalProperties") && !matches(it.value(), schema["additionalProperties"], root, depth + 1)) { return false; }
        }
    }
    return true;
}

Json schema_format(const Json& schema, const std::string& style) {
    return {{"type", "json_schema"}, {"json_schema", schema}, {"style", style},
            {"any_order", false}, {"max_whitespace_cnt", 1}};
}
Json tag(std::string begin, Json content, std::string end) {
    return {{"type", "tag"}, {"begin", std::move(begin)}, {"content", std::move(content)}, {"end", std::move(end)}};
}
Json sequence(Json elements) { return {{"type", "sequence"}, {"elements", std::move(elements)}}; }
} // namespace
bool tool_arguments_match_schema(const ToolDefinition& tool, const std::string& arguments) {
    if (tool.strict_set && !tool.strict) { return true; }
    try {
        const auto schema = Json::parse(tool.parameters_json);
        const auto value = Json::parse(arguments);
        return value.is_object() && matches(value, schema, schema);
    } catch (const std::exception&) { return false; }
}
std::string make_tool_constraint(const GenerationRequest& request, ToolCallFormat format, bool thinking) {
    if (!request.uses_tools()) { return {}; }
    const bool forced = request.tool_choice.mode == ToolChoiceMode::Named;
    const bool required = forced || request.tool_choice.mode == ToolChoiceMode::Required;
    const bool strict = std::any_of(request.tools.begin(), request.tools.end(), [](const auto& tool) { return tool.strict; });
    if (!required && !strict) { return {}; }
    if (format == ToolCallFormat::None) {
        throw ApiException({.message="constrained tool calls require an enabled tool parser", .param="tools"});
    }
    if (request.ignore_eos || request.min_tokens != 0 || !request.stop_strings.empty()) {
        throw ApiException({.message="constrained tools require ignore_eos=false, min_tokens=0, and no custom stops", .param="tools"});
    }
    Json tags = Json::array();
    for (const auto& tool : request.tools) {
        if (forced && tool.name != request.tool_choice.name) { continue; }
        Json schema = true;
        if (tool.strict || !tool.strict_set) {
            schema = Json::parse(tool.parameters_json);
            try { validate_schema(schema); }
            catch (const std::invalid_argument& error) {
                throw ApiException({.message=tool.name + ": " + error.what(), .param="tools", .code="tool_schema_not_supported"});
            }
        }
        const auto quoted = Json(tool.name).dump();
        if (format == ToolCallFormat::QwenXml) {
            for (const auto& newline : {std::string{}, std::string{"\n"}}) {
                tags.push_back(tag("<tool_call>" + newline + "{\"name\": " + quoted + ", \"arguments\": ",
                    schema_format(schema, "json"), "}" + newline + "</tool_call>"));
                tags.push_back(tag("<tool_call>" + newline + "<function=" + tool.name + ">" + newline,
                    schema_format(schema, "qwen_xml"), newline + "</function>" + newline + "</tool_call>"));
            }
        } else if (format == ToolCallFormat::Spark25) {
            for (const auto& newline : {std::string{}, std::string{"\n"}}) {
                tags.push_back(tag("<tool_call>" + newline + tool.name,
                    schema_format(schema, "glm_xml"), "</tool_call>"));
            }
        } else {
            for (const auto& prefix : {std::string{}, std::string{"<|python_tag|>"}}) {
                tags.push_back(tag(prefix + "{\"name\": " + quoted + ", \"parameters\": ",
                    schema_format(schema, "json"), "}"));
            }
        }
    }
    const bool llama = format == ToolCallFormat::Llama3Json;
    Json calls = {{"type", "tags_with_separator"}, {"tags", tags}, {"separator", llama ? "; " : "\n"},
                  {"at_least_one", true}, {"stop_after_first", forced || !request.parallel_tool_calls}};
    Json suffix = calls;
    if (!required) {
        suffix = {{"type", "triggered_tags"}, {"tags", tags},
                  {"triggers", llama ? Json::array({"<|python_tag|>", "{\"name\":"}) : Json::array({"<tool_call>"})},
                  {"at_least_one", false}, {"stop_after_first", false}};
    }
    if (thinking) {
        Json reasoning = tag("", Json{{"type", "any_text"}, {"excludes", Json::array({"</think>"})}}, "</think>");
        Json optional = {{"type", "optional"}, {"content", reasoning}};
        suffix = sequence(Json::array({optional, Json{{"type", "regex"}, {"pattern", R"([ \t\r\n]*)"}}, suffix}));
    }
    return Json{{"type", "structural_tag"}, {"format", suffix}}.dump();
}
} // namespace sinfer::serve
