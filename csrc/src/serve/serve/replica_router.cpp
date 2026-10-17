#include "serve/replica_router.h"

#include "serve/request.h"

#include <algorithm>
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
/// Placement scores this close are a tie (the KV fractions are ratios of page counts).
constexpr double kScoreTie = 1e-9;

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
        fed_ += size;
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
    /// Bytes hashed so far: the size of the prefix a boundary covers, near enough.
    [[nodiscard]] std::size_t fed() const noexcept { return fed_; }

private:
    std::uint64_t state_ = 14695981039346656037ULL;
    std::size_t fed_     = 0;
};

} // namespace

ReplicaRouter::ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available)
    : ReplicaRouter(replicas, std::move(load), std::move(available), Options{}) {}

ReplicaRouter::ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available,
                             Options options)
    : ReplicaRouter(replicas, std::move(load), std::move(available), KvFn{}, options) {}

ReplicaRouter::ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available, KvFn kv,
                             Options options)
    : load_(std::move(load)),
      available_(std::move(available)),
      kv_(std::move(kv)),
      options_(options),
      replicas_(replicas),
      reserved_(std::make_shared<std::vector<std::atomic<std::size_t>>>(replicas)) {
    if (replicas == 0) { throw std::invalid_argument("ReplicaRouter needs at least one replica"); }
    if (!load_ || !available_) { throw std::invalid_argument("ReplicaRouter needs load and availability"); }
    if (options_.remembered_prefixes == 0) { options_.remembered_prefixes = 1; }
    if (!(options_.load_slack >= 1.0)) { options_.load_slack = 1.0; }
    if (!(options_.kv_weight >= 0.0)) { options_.kv_weight = 0.0; }
}

std::size_t ReplicaRouter::longest_match(const Replica& replica,
                                         const std::vector<std::uint64_t>& prefixes) {
    for (std::size_t length = prefixes.size(); length > 0; --length) {
        if (replica.seen.count(prefixes[length - 1]) != 0) { return length; }
    }
    return 0;
}

bool ReplicaRouter::owned(const std::vector<std::uint64_t>& prefixes, std::size_t length) const {
    // The whole prompt was seen before (a retry): nothing in it can tell the conversation apart.
    if (length >= prefixes.size()) { return true; }
    const auto found = forks_.find(prefixes[length - 1]);
    // Forgotten, or the last request of this conversation ended exactly there: its own turn.
    if (found == forks_.end() || !found->second.has_next) { return true; }
    // Continued in two ways already, or about to be: a head other conversations share.
    return !found->second.shared && found->second.next == prefixes[length];
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

void ReplicaRouter::note_continuations(const std::vector<std::uint64_t>& prefixes) {
    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const auto [found, inserted] = forks_.try_emplace(prefixes[i]);
        Fork& fork = found->second;
        if (inserted) {
            fork_order_.push_front(prefixes[i]);
            fork.at = fork_order_.begin();
        } else {
            fork_order_.splice(fork_order_.begin(), fork_order_, fork.at);
        }
        if (i + 1 == prefixes.size()) { continue; }
        if (!fork.has_next) {
            fork.next     = prefixes[i + 1];
            fork.has_next = true;
        } else if (fork.next != prefixes[i + 1]) {
            fork.shared = true;
        }
    }
    while (fork_order_.size() > options_.remembered_prefixes) {
        forks_.erase(fork_order_.back());
        fork_order_.pop_back();
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
    return pick(RoutingKey{.prefixes = prefixes});
}

ReplicaRouter::Route ReplicaRouter::pick(const RoutingKey& key) {
    const std::vector<std::uint64_t>& prefixes = key.prefixes;
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
    const auto full = [&](std::size_t i) {
        return options_.replica_capacity != 0 && loads[i] >= options_.replica_capacity;
    };

    // Scan from a rotating start: the first replica met at the best score wins a tie, so
    // requests that arrive together go round the replicas instead of piling on the first.
    const std::size_t start = cursor_;
    cursor_ = (cursor_ + 1) % count;

    // The replica that served this conversation: the longest match that is its own.
    std::vector<std::size_t> match(count, 0);
    std::size_t home = kNone;
    std::size_t home_match = 0;
    std::size_t total = 0;
    bool room = false;
    for (std::size_t step = 0; step < count; ++step) {
        const std::size_t i = (start + step) % count;
        if (!candidate[i]) { continue; }
        total += loads[i];
        room = room || !full(i);
        match[i] = longest_match(replicas_[i], prefixes);
        if (match[i] == 0 || !owned(prefixes, match[i])) { continue; }
        if (match[i] > home_match || (match[i] == home_match && loads[i] < loads[home])) {
            home_match = match[i];
            home       = i;
        }
    }

    std::size_t chosen = kNone;
    if (home != kNone && !full(home)) {
        const std::size_t cached =
            key.bytes.size() == prefixes.size() ? key.bytes[home_match - 1] : 0;
        if (cached >= options_.sticky_bytes) {
            chosen = home;
        } else {
            // Bounded loads: a small cached prefix keeps the conversation while its replica
            // carries at most `load_slack` times the mean, counting this request.
            const double mean = static_cast<double>(total + 1) / static_cast<double>(candidates);
            const auto bound  = static_cast<std::size_t>(std::ceil(options_.load_slack * mean));
            if (loads[home] + 1 <= bound) { chosen = home; }
        }
    }
    if (chosen == kNone) {
        // Placement: the least load plus KV in use; a full replica only when every one is.
        double best = 0.0;
        for (std::size_t step = 0; step < count; ++step) {
            const std::size_t i = (start + step) % count;
            if (!candidate[i] || (room && full(i))) { continue; }
            const double kv    = kv_ ? std::clamp(kv_(i), 0.0, 1.0) : 0.0;
            const double score = static_cast<double>(loads[i]) + options_.kv_weight * kv;
            if (chosen == kNone || score < best - kScoreTie ||
                (score <= best + kScoreTie && match[i] > match[chosen])) {
                chosen = i;
                best   = score;
            }
        }
    }
    note_continuations(prefixes);
    remember(replicas_[chosen], prefixes);
    return reserve(chosen);
}

ReplicaRouter::Route ReplicaRouter::pin(std::size_t replica,
                                        const std::vector<std::uint64_t>& prefixes) {
    return pin(replica, RoutingKey{.prefixes = prefixes});
}

ReplicaRouter::Route ReplicaRouter::pin(std::size_t replica, const RoutingKey& key) {
    if (replica >= replicas_.size()) {
        throw std::out_of_range("replica " + std::to_string(replica) + " does not exist; this server has " +
                                std::to_string(replicas_.size()));
    }
    const std::lock_guard lock(mutex_);
    note_continuations(key.prefixes);
    remember(replicas_[replica], key.prefixes);
    return reserve(replica);
}

std::size_t ReplicaRouter::reserved(std::size_t replica) const {
    return (*reserved_).at(replica).load(std::memory_order_relaxed);
}

RoutingKey conversation_routing_key(const GenerationRequest& request) {
    RoutingKey key;
    PrefixHasher hasher;
    const auto mark = [&] {
        key.prefixes.push_back(hasher.mark());
        key.bytes.push_back(hasher.fed());
    };
    if (!request.messages.empty()) {
        // The tools render into the head of the prompt, so they belong to the first boundary.
        hasher.number(request.tools.size());
        for (const ToolDefinition& tool : request.tools) {
            hasher.text(tool.name);
            hasher.text(tool.definition_json.empty() ? tool.parameters_json : tool.definition_json);
        }
        key.prefixes.reserve(request.messages.size());
        key.bytes.reserve(request.messages.size());
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
            mark();
        }
        return key;
    }
    // No messages: cut at fixed sizes, so a prompt that extends another shares its boundaries.
    if (request.raw_prompt.has_value()) {
        const std::string& prompt = *request.raw_prompt;
        for (std::size_t end = kRawPromptChunkBytes; end <= prompt.size(); end += kRawPromptChunkBytes) {
            hasher.bytes(prompt.data() + end - kRawPromptChunkBytes, kRawPromptChunkBytes);
            mark();
        }
        return key;
    }
    const auto& tokens = request.prompt_token_ids;
    for (std::size_t end = kTokenPromptChunk; end <= tokens.size(); end += kTokenPromptChunk) {
        hasher.bytes(tokens.data() + end - kTokenPromptChunk, kTokenPromptChunk * sizeof(tokens[0]));
        mark();
    }
    return key;
}

std::vector<std::uint64_t> conversation_prefix_hashes(const GenerationRequest& request) {
    return conversation_routing_key(request).prefixes;
}

} // namespace sinfer::serve
