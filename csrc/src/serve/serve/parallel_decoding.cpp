#include "serve/parallel_decoding.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

namespace sinfer::serve {
namespace {
using Json = nlohmann::json;
[[noreturn]] void invalid(const std::string& message) {
    throw ApiException({.message = "parallel_decoding: " + message, .param = "parallel_decoding"});
}
void keys(const Json& object, const std::set<std::string>& allowed) {
    for (auto it = object.begin(); it != object.end(); ++it) {
        if (!allowed.contains(it.key())) invalid("unsupported schema keyword '" + it.key() + "'");
    }
}
bool matches_type(const Json& value, const std::string& type) {
    if (type == "boolean") return value.is_boolean();
    if (type == "string") return value.is_string();
    if (type == "number") return value.is_number();
    if (type == "integer") return value.is_number_integer() ||
        (value.is_number_float() && std::isfinite(value.get<double>()) && std::trunc(value.get<double>()) == value.get<double>());
    if (type == "null") return value.is_null();
    return false;
}
}

void validate_parallel_request(const GenerationRequest& r) {
    if (r.json_schema.empty()) invalid("requires response_format.type=json_schema");
    if (r.raw_prompt || !r.prompt_token_ids.empty() || r.media_item_count() ||
        !r.tools.empty() || r.has_tool_history()) invalid("requires text chat messages without tools or explicit tokens");
    if (r.want_logprobs || r.prompt_logprobs >= 0 || r.return_token_ids)
        invalid("token logprobs and token IDs are unavailable for assembled JSON; use field probabilities");
    if (r.min_tokens || r.ignore_eos || !r.stop_strings.empty()) invalid("custom stopping rules are unavailable");
    const auto& s = r.sampling;
    if ((s.top_p && *s.top_p != 1) || (s.top_k && *s.top_k != 0 && *s.top_k != -1) ||
        (s.min_p && *s.min_p != 0) || (s.repetition_penalty && *s.repetition_penalty != 1) ||
        (s.presence_penalty && *s.presence_penalty != 0) ||
        (s.frequency_penalty && *s.frequency_penalty != 0) || !s.logit_bias.empty()) {
        invalid("classification accepts temperature, but not sampling filters, penalties, or logit_bias");
    }
}

std::vector<ParallelField> parse_parallel_schema(const std::string& schema) {
    const auto root = Json::parse(schema, nullptr, false);
    if (!root.is_object()) invalid("schema must be an object");
    keys(root, {"type", "properties", "required", "additionalProperties", "description", "title", "$schema"});
    if (root.value("type", Json()) != "object" || !root.contains("properties") ||
        !root["properties"].is_object() || root["properties"].empty() || root["properties"].size() > 64)
        invalid("requires an object with 1 to 64 properties");
    if (root.value("additionalProperties", Json()) != false)
        invalid("set additionalProperties to false");
    if (!root.contains("required") || !root["required"].is_array()) invalid("all properties must be required");
    std::set<std::string> required;
    for (const auto& name : root["required"]) {
        if (!name.is_string() || !required.insert(name.get<std::string>()).second)
            invalid("required must contain distinct property names");
    }
    if (required.size() != root["properties"].size()) invalid("all properties must be required");
    std::vector<ParallelField> fields;
    for (auto it = root["properties"].begin(); it != root["properties"].end(); ++it) {
        if (!required.contains(it.key())) invalid("all properties must be required");
        const auto& spec = it.value();
        if (!spec.is_object()) invalid("each property must be a boolean or scalar enum schema");
        keys(spec, {"type", "enum", "description", "title"});
        if (spec.contains("type") && (!spec["type"].is_string() ||
            !std::set<std::string>{"boolean", "string", "number", "integer", "null"}.contains(spec["type"].get<std::string>())))
            invalid("property type must be boolean, string, number, integer, or null");
        ParallelField field;
        field.name = it.key();
        if (spec.contains("enum")) {
            if (!spec["enum"].is_array() || spec["enum"].empty() || spec["enum"].size() > 256)
                invalid("each enum must contain 1 to 256 choices");
            for (const auto& value : spec["enum"]) {
                if (value.is_structured() || (spec.contains("type") && !matches_type(value, spec["type"].get<std::string>())))
                    invalid("enum choices must be scalars matching their property type");
                if (std::find(field.values.begin(), field.values.end(), value) != field.values.end())
                    invalid("enum choices must be distinct");
                field.values.push_back(value);
            }
        } else if (spec.value("type", Json()) == "boolean") {
            field.values = {false, true};
        } else {
            invalid("each non-boolean property needs an enum");
        }
        fields.push_back(std::move(field));
    }
    return fields;
}

ParallelDecodingPlan compile_parallel_plan(std::vector<ParallelField> fields,
    const std::function<std::vector<TokenId>(std::string_view)>& encode) {
    ParallelDecodingPlan plan;
    plan.fields = std::move(fields);
    std::size_t total_nodes = 0;
    for (auto& field : plan.fields) {
        for (std::size_t choice = 0; choice < field.values.size(); ++choice) {
            // Include a terminator: e.g. 1 versus 10 and "a" versus "ab" must
            // diverge before reaching a leaf. Tokenize the whole suffix so BPE
            // merges across quotes, property names, and values stay consistent.
            const auto ids = encode("{\n" + Json(field.name).dump() + ": " + field.values[choice].dump() + "\n");
            if (ids.empty() || ids.size() > 256) invalid("field continuations must fit in 256 tokens");
            std::size_t node = 0;
            for (auto id : ids) {
                auto [it, inserted] = field.nodes[node].children.emplace(id, field.nodes.size());
                const auto child = it->second;
                if (inserted) {
                    if (++total_nodes > 32768) invalid("schema token expansion exceeds 32768 nodes");
                    field.nodes.emplace_back();
                }
                node = child;
            }
            if (field.nodes[node].choice >= 0) invalid("distinct choices have identical token encodings");
            field.nodes[node].choice = static_cast<int>(choice);
        }
        std::vector<TokenId> suffix;
        std::function<void(std::size_t)> visit = [&](std::size_t index) {
            auto& node = field.nodes[index];
            if (node.choice >= 0 && !node.children.empty()) invalid("choice token encoding is a prefix of another choice");
            if (node.children.size() > 1) {
                if (plan.queries.size() >= 1024) invalid("schema requires more than 1024 branching decisions");
                node.query = static_cast<int>(plan.queries.size());
                ParallelQuery query{.suffix = suffix};
                for (const auto& [id, child] : node.children) query.candidates.push_back(id);
                plan.queries.push_back(std::move(query));
            }
            for (const auto& [id, child] : node.children) {
                suffix.push_back(id);
                visit(child);
                suffix.pop_back();
            }
        };
        visit(0);
    }
    return plan;
}

ParallelDecodingResult resolve_parallel_plan(const ParallelDecodingPlan& plan,
    const std::vector<std::vector<float>>& logits, double temperature) {
    if (logits.size() != plan.queries.size() || !std::isfinite(temperature) || temperature < 0)
        throw std::invalid_argument("invalid parallel classification readouts");
    const double scale = temperature > 0 ? 1 / temperature : 1;
    ParallelDecodingResult result{Json::object(), Json::object()};
    for (const auto& field : plan.fields) {
        std::vector<double> scores(field.values.size(), -std::numeric_limits<double>::infinity());
        std::function<void(std::size_t, double)> visit = [&](std::size_t index, double score) {
            const auto& node = field.nodes[index];
            if (node.choice >= 0) { scores[node.choice] = score; return; }
            if (node.children.size() == 1) { visit(node.children.begin()->second, score); return; }
            const auto& row = logits.at(node.query);
            if (row.size() != node.children.size() ||
                std::any_of(row.begin(), row.end(), [](float value) { return !std::isfinite(value); }))
                throw std::runtime_error("missing or non-finite parallel classification logits");
            const double maximum = *std::max_element(row.begin(), row.end());
            double sum = 0;
            for (auto value : row) sum += std::exp((value - maximum) * scale);
            std::size_t i = 0;
            for (const auto& [id, child] : node.children)
                visit(child, score + (row[i++] - maximum) * scale - std::log(sum));
        };
        visit(0, 0);
        const auto best = std::max_element(scores.begin(), scores.end()) - scores.begin();
        Json probabilities = Json::array();
        double sum = 0;
        for (auto score : scores) sum += std::exp(score);
        for (std::size_t i = 0; i < scores.size(); ++i)
            probabilities.push_back({{"value", field.values[i]}, {"probability", std::exp(scores[i]) / sum}});
        result.content[field.name] = field.values[best];
        result.fields[field.name] = {{"value", field.values[best]}, {"probabilities", probabilities}};
    }
    return result;
}
} // namespace sinfer::serve
