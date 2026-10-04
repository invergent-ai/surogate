#include "serve/replica_router.h"

#include "serve/request.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace sinfer::serve {
namespace {

constexpr std::size_t kNone = std::numeric_limits<std::size_t>::max();
constexpr std::size_t kRawPromptChunkBytes = 4096;
constexpr std::size_t kTokenPromptChunk    = 1024;

/// FNV-1a over everything fed so far; `mark()` finalizes the running state without ending it,
/// so each boundary's hash covers the whole prefix before it.
class PrefixHasher {
public:
    void bytes(const void* data, std::size_t size) {
        const auto* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < size; ++i) {
            state_ ^= p[i];
            state_ *= 1099511628211ULL;
        }
    }
    void number(std::uint64_t value) { bytes(&value, sizeof value); }
    /// Length-prefixed, so ("ab", "c") and ("a", "bc") differ.
    void text(std::string_view value) {
        number(value.size());
        bytes(value.data(), value.size());
    }
    [[nodiscard]] std::uint64_t mark() const {
        // splitmix64's finalizer: FNV's low bits are weak, and the router compares whole values.
        std::uint64_t x = state_;
        x ^= x >> 30;
        x *= 0xbf58476d1ce4e5b9ULL;
        x ^= x >> 27;
        x *= 0x94d049bb133111ebULL;
        x ^= x >> 31;
        return x;
    }

private:
    std::uint64_t state_ = 14695981039346656037ULL;
};

} // namespace

ReplicaRouter::ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available)
    : ReplicaRouter(replicas, std::move(load), std::move(available), Options{}) {}

ReplicaRouter::ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available,
                             Options options)
    : load_(std::move(load)),
      available_(std::move(available)),
      options_(options),
      replicas_(replicas),
      reserved_(std::make_shared<std::vector<std::atomic<std::size_t>>>(replicas)) {
    if (replicas == 0) { throw std::invalid_argument("ReplicaRouter needs at least one replica"); }
    if (!load_ || !available_) { throw std::invalid_argument("ReplicaRouter needs load and availability"); }
    if (options_.remembered_prefixes == 0) { options_.remembered_prefixes = 1; }
    if (!(options_.load_slack >= 1.0)) { options_.load_slack = 1.0; }
}

std::size_t ReplicaRouter::longest_match(const Replica& replica,
                                         const std::vector<std::uint64_t>& prefixes) {
    for (std::size_t length = prefixes.size(); length > 0; --length) {
        if (replica.seen.count(prefixes[length - 1]) != 0) { return length; }
    }
    return 0;
}

void ReplicaRouter::remember(Replica& replica, const std::vector<std::uint64_t>& prefixes) {
    for (const std::uint64_t hash : prefixes) {
        const auto found = replica.seen.find(hash);
        if (found != replica.seen.end()) {
            replica.order.splice(replica.order.begin(), replica.order, found->second);
            continue;
        }
        replica.order.push_front(hash);
        replica.seen.emplace(hash, replica.order.begin());
    }
    while (replica.order.size() > options_.remembered_prefixes) {
        replica.seen.erase(replica.order.back());
        replica.order.pop_back();
    }
}

ReplicaRouter::Route ReplicaRouter::reserve(std::size_t replica) {
    class Hold {
    public:
        Hold(std::shared_ptr<std::vector<std::atomic<std::size_t>>> counts, std::size_t replica)
            : counts_(std::move(counts)), replica_(replica) {
            (*counts_)[replica_].fetch_add(1, std::memory_order_relaxed);
        }
        Hold(const Hold&)            = delete;
        Hold& operator=(const Hold&) = delete;
        ~Hold() { (*counts_)[replica_].fetch_sub(1, std::memory_order_relaxed); }

    private:
        std::shared_ptr<std::vector<std::atomic<std::size_t>>> counts_;
        std::size_t replica_;
    };
    return Route{.replica = replica, .reservation = std::make_shared<Hold>(reserved_, replica)};
}

ReplicaRouter::Route ReplicaRouter::pick(const std::vector<std::uint64_t>& prefixes) {
    const std::lock_guard lock(mutex_);
    const std::size_t count = replicas_.size();
    std::vector<std::size_t> loads(count);
    std::vector<bool> candidate(count);
    std::size_t candidates = 0;
    for (std::size_t i = 0; i < count; ++i) {
        loads[i] = load_(i) + (*reserved_)[i].load(std::memory_order_relaxed);
        candidate[i] = available_(i);
        candidates += candidate[i] ? 1 : 0;
    }
    if (candidates == 0) {
        candidate.assign(count, true);
        candidates = count;
    }

    // Scan from a rotating start: the first replica met at the lowest load wins a tie, so
    // requests that arrive together go round the replicas instead of piling on the first.
    const std::size_t start = cursor_;
    cursor_ = (cursor_ + 1) % count;
    std::size_t least = kNone;
    std::size_t preferred = kNone;
    std::size_t best_match = 0;
    std::size_t total = 0;
    for (std::size_t step = 0; step < count; ++step) {
        const std::size_t i = (start + step) % count;
        if (!candidate[i]) { continue; }
        total += loads[i];
        if (least == kNone || loads[i] < loads[least]) { least = i; }
        const std::size_t match = longest_match(replicas_[i], prefixes);
        if (match == 0) { continue; }
        if (match > best_match || (match == best_match && loads[i] < loads[preferred])) {
            best_match = match;
            preferred  = i;
        }
    }

    std::size_t chosen = least;
    if (preferred != kNone) {
        // Bounded loads: the replica holding the prefix keeps the conversation while it carries
        // at most `load_slack` times the mean, counting this request.
        const double mean  = static_cast<double>(total + 1) / static_cast<double>(candidates);
        const auto bound   = static_cast<std::size_t>(std::ceil(options_.load_slack * mean));
        if (loads[preferred] + 1 <= bound) { chosen = preferred; }
    }
    remember(replicas_[chosen], prefixes);
    return reserve(chosen);
}

ReplicaRouter::Route ReplicaRouter::pin(std::size_t replica,
                                        const std::vector<std::uint64_t>& prefixes) {
    if (replica >= replicas_.size()) {
        throw std::out_of_range("replica " + std::to_string(replica) + " does not exist; this server has " +
                                std::to_string(replicas_.size()));
    }
    const std::lock_guard lock(mutex_);
    remember(replicas_[replica], prefixes);
    return reserve(replica);
}

std::size_t ReplicaRouter::reserved(std::size_t replica) const {
    return (*reserved_).at(replica).load(std::memory_order_relaxed);
}

std::vector<std::uint64_t> conversation_prefix_hashes(const GenerationRequest& request) {
    std::vector<std::uint64_t> prefixes;
    PrefixHasher hasher;
    if (!request.messages.empty()) {
        // The tools render into the head of the prompt, so they belong to the first boundary.
        hasher.number(request.tools.size());
        for (const ToolDefinition& tool : request.tools) {
            hasher.text(tool.name);
            hasher.text(tool.definition_json.empty() ? tool.parameters_json : tool.definition_json);
        }
        prefixes.reserve(request.messages.size());
        for (const ChatTurn& turn : request.messages) {
            hasher.number(static_cast<std::uint64_t>(turn.role));
            hasher.number(turn.content.size());
            for (const ContentPart& part : turn.content) {
                hasher.number(static_cast<std::uint64_t>(part.kind));
                hasher.text(part.kind == ContentKind::Text ? std::string_view(part.text)
                                                           : std::string_view(part.type_raw));
            }
            hasher.text(turn.reasoning_content);
            hasher.number(turn.tool_calls.size());
            for (const ToolCall& call : turn.tool_calls) {
                hasher.text(call.name);
                hasher.text(call.arguments_json);
            }
            hasher.text(turn.tool_call_id);
            prefixes.push_back(hasher.mark());
        }
        return prefixes;
    }
    // No messages: cut at fixed sizes, so a prompt that extends another shares its boundaries.
    if (request.raw_prompt.has_value()) {
        const std::string& prompt = *request.raw_prompt;
        for (std::size_t end = kRawPromptChunkBytes; end <= prompt.size(); end += kRawPromptChunkBytes) {
            hasher.bytes(prompt.data() + end - kRawPromptChunkBytes, kRawPromptChunkBytes);
            prefixes.push_back(hasher.mark());
        }
        return prefixes;
    }
    const auto& tokens = request.prompt_token_ids;
    for (std::size_t end = kTokenPromptChunk; end <= tokens.size(); end += kTokenPromptChunk) {
        hasher.bytes(tokens.data() + end - kTokenPromptChunk, kTokenPromptChunk * sizeof(tokens[0]));
        prefixes.push_back(hasher.mark());
    }
    return prefixes;
}

} // namespace sinfer::serve
