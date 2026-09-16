#include "serve/parallel_decoding.h"
#include "serve/openai_schema.h"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace sinfer::serve;
using Json = nlohmann::json;

static Json schema(Json properties) {
    Json required = Json::array();
    for (auto it = properties.begin(); it != properties.end(); ++it) required.push_back(it.key());
    return {{"type", "object"}, {"properties", properties}, {"required", required}, {"additionalProperties", false}};
}
static void refused(const std::function<void()>& f) {
    bool caught = false;
    try { f(); } catch (const ApiException& e) { caught = e.error().status == 400; }
    assert(caught);
}
static std::vector<sinfer::TokenId> bytes(std::string_view s) {
    std::vector<sinfer::TokenId> ids;
    for (unsigned char c : s) ids.push_back(c);
    return ids;
}
int main() {
    auto valid = schema({{"flag", {{"type", "boolean"}}},
                         {"label", {{"type", "string"}, {"enum", {"a", "ab", "b"}}}}});
    auto plan = compile_parallel_plan(parse_parallel_schema(valid.dump()), bytes);
    assert(plan.queries.size() == 3);
    std::vector<std::vector<float>> logits;
    for (auto& query : plan.queries) {
        std::vector<float> row;
        for (auto token : query.candidates) {
            row.push_back(token == 'a' ? std::log(.6f) : token == 'b' && query.candidates[0] == 'a' ?
                          std::log(.4f) : 0);
        }
        logits.push_back(row);
    }
    auto result = resolve_parallel_plan(plan, logits, 1);
    assert(result.content["label"] == "b"); // Greedy token decoding would choose a/ab.
    assert(result.content["flag"] == false); // Stable tie in schema order.
    auto probabilities = result.fields["label"]["probabilities"];
    for (int i = 0; i < 3; ++i)
        assert(std::abs(probabilities[i]["probability"].get<double>() - (i == 2 ? .4 : .3)) < 1e-6);
    assert(resolve_parallel_plan(plan, logits, 0).content == result.content);
    auto sharp = resolve_parallel_plan(plan, logits, .1);
    assert(sharp.content["label"] == "a");
    auto single = compile_parallel_plan(parse_parallel_schema(schema({
        {"constant", {{"enum", {"quoted\"日本語\n"}}}}, {"number", {{"type", "integer"}, {"enum", {10}}}}
    }).dump()), bytes);
    assert(single.queries.empty());
    assert(resolve_parallel_plan(single, {}, 1).content["number"] == 10);
    assert(resolve_parallel_plan(single, {}, 1).fields["constant"]["probabilities"][0]["probability"] == 1);
    for (const auto& bad : {Json(true), Json::object(), schema({{"x", {{"type", "string"}}}}),
        schema({{"x", {{"enum", Json::array()}}}}), schema({{"x", {{"enum", {1, 1.0}}}}}),
        schema({{"x", {{"type", "integer"}, {"enum", {"one"}}}}}),
        schema({{"x", {{"type", "boolean"}, {"const", true}}}}),
        schema({{"x", {{"type", "array"}, {"enum", {Json::array({1})}}}}})})
        refused([&] { parse_parallel_schema(bad.dump()); });
    for (const auto* keyword : {"allOf", "if", "dependentRequired", "$ref", "minProperties"}) {
        auto bad = valid; bad[keyword] = Json::object();
        refused([&] { parse_parallel_schema(bad.dump()); });
    }
    auto optional = valid; optional["required"] = {"flag"};
    refused([&] { parse_parallel_schema(optional.dump()); });
    auto open = valid; open["additionalProperties"] = true;
    refused([&] { parse_parallel_schema(open.dump()); });
    refused([&] { compile_parallel_plan(parse_parallel_schema(valid.dump()),
        [](std::string_view) { return std::vector<sinfer::TokenId>{1}; }); });
    auto numbers = compile_parallel_plan(parse_parallel_schema(schema({
        {"number", {{"type", "integer"}, {"enum", {1, 10, 100}}}}
    }).dump()), bytes);
    assert(numbers.queries.size() == 2);
    assert(numbers.queries[0].parent == -1);
    assert(numbers.queries[1].parent == 0);
    assert(std::equal(numbers.queries[0].suffix.begin(), numbers.queries[0].suffix.end(),
                      numbers.queries[1].suffix.begin()));
    Json choices = Json::array();
    for (int i = 0; i < 256; ++i) choices.push_back(i);
    auto large = parse_parallel_schema(schema({{"x", {{"enum", choices}}}}).dump());
    assert(large[0].values.size() == 256);
    choices.push_back(256);
    refused([&] { parse_parallel_schema(schema({{"x", {{"enum", choices}}}}).dump()); });

    Json body{{"model", "test"}, {"messages", {{{"role", "user"}, {"content", "Classify this"}}}},
              {"parallel_decoding", true}, {"response_format", {{"type", "json_schema"},
              {"json_schema", {{"name", "test"}, {"schema", valid}}}}}};
    auto request = parse_chat_completion_request(body, {});
    assert(request.parallel_decoding);
    validate_parallel_request(request);
    body["parallel_decoding"] = "yes";
    refused([&] { parse_chat_completion_request(body, {}); });
    body["parallel_decoding"] = true; body["response_format"] = {{"type", "json_object"}};
    refused([&] { parse_chat_completion_request(body, {}); });
    body["response_format"] = nullptr;
    refused([&] { parse_chat_completion_request(body, {}); });
    request.want_logprobs = true;
    refused([&] { validate_parallel_request(request); });
    request.want_logprobs = false; request.sampling.top_p = .9;
    refused([&] { validate_parallel_request(request); });
    const auto event = make_chat_chunk_final("id", "model", 0, "stop", true, "{\"fields\":{}}");
    assert(event.find("\"parallel_decoding\":{\"fields\":{}}") != std::string::npos);
    std::cout << "parallel decoding schema, trie probabilities, collisions and API checks passed\n";
}
