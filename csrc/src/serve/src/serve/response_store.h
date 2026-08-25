#pragma once

// Process-local bounded storage for OpenAI Responses objects and their
// previous_response_id context DAG. The Engine remains stateless; stored
// contexts are flattened only when a continuation is submitted.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace ninfer::serve {

struct ResponseContextNode {
    std::shared_ptr<const ResponseContextNode> parent;
    std::vector<ChatTurn> turns;
    std::size_t owned_bytes = 0;
};

using ResponseContext = std::shared_ptr<const ResponseContextNode>;

ResponseContext append_response_context(ResponseContext parent, std::vector<ChatTurn> turns);
std::vector<ChatTurn> flatten_response_context(const ResponseContext& context);

struct StoredResponse {
    std::string id;
    nlohmann::json response;
    std::vector<nlohmann::json> input_items;
    ResponseContext context;
    bool preserve_thinking = false;
};

class ResponseStore {
public:
    ResponseStore(std::size_t max_records, std::size_t max_bytes);

    // get() refreshes LRU recency. Returned immutable records remain valid if
    // another request evicts or deletes their public store entry.
    std::shared_ptr<const StoredResponse> get(const std::string& id);
    void put(StoredResponse response);
    bool erase(const std::string& id);

    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::size_t bytes() const;

private:
    struct Entry {
        std::shared_ptr<const StoredResponse> response;
        std::list<std::string>::iterator lru;
    };

    [[nodiscard]] std::size_t recompute_bytes_locked() const;
    void erase_locked(const std::string& id);

    std::size_t max_records_ = 0;
    std::size_t max_bytes_   = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> records_;
    std::list<std::string> lru_;
    std::size_t current_bytes_ = 0;
};

} // namespace ninfer::serve
