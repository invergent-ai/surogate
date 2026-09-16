#pragma once

#include "api/engine.h"
#include "serve/request.h"
#include <nlohmann/json.hpp>
#include <functional>
#include <map>

namespace sinfer::serve {

// A finite trie of complete JSON field continuations. Only branching nodes need
// a model readout; unary paths are forced by the constraint.
struct ParallelField {
    struct Node {
        std::map<TokenId, std::size_t> children;
        int choice = -1;
        int query = -1;
    };
    std::string name;
    std::vector<nlohmann::json> values;
    std::vector<Node> nodes{1};
};
struct ParallelQuery {
    int parent = -1;
    std::vector<TokenId> suffix;
    std::vector<TokenId> candidates;
};
struct ParallelDecodingPlan {
    std::vector<ParallelField> fields;
    std::vector<ParallelQuery> queries;
};
struct ParallelDecodingResult {
    nlohmann::json content;
    nlohmann::json fields;
};

void validate_parallel_request(const GenerationRequest& request);
std::vector<ParallelField> parse_parallel_schema(const std::string& schema);
ParallelDecodingPlan compile_parallel_plan(std::vector<ParallelField> fields,
    const std::function<std::vector<TokenId>(std::string_view)>& encode);
ParallelDecodingResult resolve_parallel_plan(const ParallelDecodingPlan& plan,
    const std::vector<std::vector<float>>& logits, double temperature);

struct ParallelDecodingRequest {
    ParallelDecodingPlan plan;
    std::vector<TokenId> prefix;
    RequestOptions options;
    std::shared_ptr<void> adapter;
    std::size_t max_tokens = 0;
};

} // namespace sinfer::serve
