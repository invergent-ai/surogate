#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace sinfer::serve {

struct GenerationRequest;

/// What the router keys a request on (`conversation_routing_key`).
struct RoutingKey {
    /// Cumulative hashes of the prompt, one per boundary, shortest first (see
    /// `conversation_prefix_hashes`).
    std::vector<std::uint64_t> prefixes;
    /// Prompt bytes up to each boundary, as many as `prefixes` (empty when unknown): how much a
    /// replica that holds that prefix has in cache, so how much leaving it would cost.
    std::vector<std::size_t> bytes;
};

/// Spreads one served model's requests over its data-parallel replicas (`--data-parallel`):
/// independent engines, one per GPU, behind a single model id.
///
/// A multi-turn conversation is worth keeping on one replica. Each turn's prompt extends the
/// previous one, and only the replica that served that turn holds its prefix in cache; sent
/// elsewhere, an agentic turn of 20k-260k tokens is prefilled again from scratch. So the router
/// remembers, per replica, the prompt prefixes it was sent (hashes at message boundaries, see
/// `conversation_prefix_hashes`) and keeps a conversation on the replica that saw the longest one.
///
/// Only the conversation's OWN prefix counts. Every conversation of one agent opens with the same
/// tools and system prompt, so that head is in the cache of every replica that ever served one;
/// preferring it drew each new task onto a replica already hosting another, and two 200k-token
/// conversations on a 460k-token cache evicted each other on every turn (2026-10-17, gpubox1/2:
/// 48 and 71 full re-reads of 214k-262k prompts, 76-80% of all re-read tokens). The router tells a
/// head from a conversation by what followed it: a prefix that requests continued in two
/// different ways is shared (`Fork`), and a match that ends on one is no reason to stay.
///
/// A conversation whose cached prefix is large (`sticky_bytes`) stays on its replica until that
/// replica is full (`replica_capacity`): a re-read of that much prompt costs more than any
/// imbalance. A smaller one keeps the old "bounded loads" rule -- it stays while its replica
/// carries at most `load_slack` times the mean load. Everything else (a new conversation, or one
/// that leaves) goes to the replica with the lowest load plus `kv_weight` times the fraction of
/// its KV cache in use: requests come and go between an agent's tool calls, and the KV that idle
/// conversations hold is what a new one would evict. Ties go to the replica holding the longest
/// shared head (its prefill is cheaper there), then rotate so a burst spreads evenly.
///
/// Load is the replica's own count of admitted requests plus the routes still on their way to
/// it (`Route::reservation`), so requests routed in the same instant do not all see the same
/// counts.
class ReplicaRouter {
public:
    struct Options {
        /// Prefix hashes remembered per replica; the least recently routed go first. The fork
        /// record (which prefixes were continued in more than one way) has the same bound.
        std::size_t remembered_prefixes = std::size_t{1} << 17;
        /// How far above the mean load the replica holding a SMALL cached prefix may go before
        /// the request goes elsewhere.
        double load_slack = 1.25;
        /// Prompt bytes in cache from which a conversation stays on its replica whatever the
        /// balance (~16k tokens: 1-2 s of prefill on one GPU, the cost of one move).
        std::size_t sticky_bytes = std::size_t{64} << 10;
        /// Requests a replica can run at once (its --max-num-seqs); 0 = no limit known. A
        /// sticky conversation leaves only a replica that is this full.
        std::size_t replica_capacity = 0;
        /// Weight of the KV fraction in use (0..1) against admitted requests when placing.
        double kv_weight = 10.0;
    };

    /// Requests a replica has admitted and not yet finished.
    using LoadFn = std::function<std::size_t(std::size_t replica)>;
    /// Whether a replica may take requests now (awake and healthy). When none is, every
    /// replica is a candidate, so the request reaches one and gets its refusal from there.
    using AvailableFn = std::function<bool(std::size_t replica)>;
    /// The fraction of a replica's KV cache in use (0..1), cached conversations included.
    using KvFn = std::function<double(std::size_t replica)>;

    struct Route {
        std::size_t replica = 0;
        /// Counts the request against its replica until the replica admits it (from then on
        /// `LoadFn` covers it). Releasing is resetting or destroying it.
        std::shared_ptr<void> reservation;
    };

    ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available);
    ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available, Options options);
    ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available, KvFn kv, Options options);
    ReplicaRouter(const ReplicaRouter&)            = delete;
    ReplicaRouter& operator=(const ReplicaRouter&) = delete;

    /// Routes a request (see the class comment).
    [[nodiscard]] Route pick(const RoutingKey& key);
    /// The same with prefix hashes alone: no cached size is known, so a conversation keeps the
    /// bounded-loads rule however long it is.
    [[nodiscard]] Route pick(const std::vector<std::uint64_t>& prefixes);
    /// Routes to the replica the caller named (vLLM's `X-data-parallel-rank`). The prefixes
    /// are remembered as having been sent there.
    /// \throws std::out_of_range If `replica` is not below size().
    [[nodiscard]] Route pin(std::size_t replica, const RoutingKey& key);
    [[nodiscard]] Route pin(std::size_t replica, const std::vector<std::uint64_t>& prefixes);

    [[nodiscard]] std::size_t size() const noexcept { return replicas_.size(); }
    /// Routes still on their way to `replica` (for tests and diagnostics).
    [[nodiscard]] std::size_t reserved(std::size_t replica) const;

private:
    struct Replica {
        /// Most recently routed first.
        std::list<std::uint64_t> order;
        std::unordered_map<std::uint64_t, std::list<std::uint64_t>::iterator> seen;
    };
    /// What followed a prefix in the requests routed so far: the next boundary's hash, or that
    /// requests continued it in more than one way (a head shared by several conversations).
    struct Fork {
        std::uint64_t next = 0;
        bool has_next      = false;
        bool shared        = false;
        std::list<std::uint64_t>::iterator at;
    };

    [[nodiscard]] static std::size_t longest_match(const Replica& replica,
                                                   const std::vector<std::uint64_t>& prefixes);
    /// Whether a match of the first `length` prefixes belongs to this conversation alone: the
    /// prefix it ends on was not continued in a way this request does not continue it.
    [[nodiscard]] bool owned(const std::vector<std::uint64_t>& prefixes, std::size_t length) const;
    void remember(Replica& replica, const std::vector<std::uint64_t>& prefixes);
    void note_continuations(const std::vector<std::uint64_t>& prefixes);
    [[nodiscard]] Route reserve(std::size_t replica);

    LoadFn load_;
    AvailableFn available_;
    KvFn kv_;
    Options options_;
    mutable std::mutex mutex_;
    std::vector<Replica> replicas_;
    std::list<std::uint64_t> fork_order_;
    std::unordered_map<std::uint64_t, Fork> forks_;
    /// Routes on their way to each replica. Shared with the reservations, which a request
    /// holds and may release after the router is gone (server shutdown).
    std::shared_ptr<std::vector<std::atomic<std::size_t>>> reserved_;
    /// Where the next tie between equally placed replicas starts, so ties rotate.
    std::size_t cursor_ = 0;
};

/// Cumulative hashes of the request's prompt, one per message boundary (the tools, then each
/// turn): two turns of one conversation share every hash up to where they diverge. A raw
/// `/v1/completions` prompt is cut every 4 KiB instead, and a token-id prompt with no messages
/// every 1024 tokens. Empty when there is no prompt.
[[nodiscard]] std::vector<std::uint64_t> conversation_prefix_hashes(const GenerationRequest& request);

/// The hashes above together with the prompt bytes up to each boundary.
[[nodiscard]] RoutingKey conversation_routing_key(const GenerationRequest& request);

} // namespace sinfer::serve
