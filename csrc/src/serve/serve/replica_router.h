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

/// Spreads one served model's requests over its data-parallel replicas (`--data-parallel`):
/// independent engines, one per GPU, behind a single model id.
///
/// A multi-turn conversation is worth keeping on one replica. Each turn's prompt extends the
/// previous one, and only the replica that served that turn holds its prefix in cache; sent
/// elsewhere, an agentic turn of 20k-100k tokens is prefilled again from scratch. So the router
/// remembers, per replica, the prompt prefixes it was sent (hashes at message boundaries, see
/// `conversation_prefix_hashes`) and prefers the replica that saw the longest one -- unless that
/// replica already carries more than its share of the load (`load_slack` times the mean, the
/// "bounded loads" rule), in which case the least loaded replica takes the request. Without any
/// remembered prefix the least loaded replica takes it, ties rotating so a burst spreads evenly.
///
/// Load is the replica's own count of admitted requests plus the routes still on their way to
/// it (`Route::reservation`), so requests routed in the same instant do not all see the same
/// counts.
class ReplicaRouter {
public:
    struct Options {
        /// Prefix hashes remembered per replica; the least recently routed go first.
        std::size_t remembered_prefixes = std::size_t{1} << 17;
        /// How far above the mean load the replica holding a prefix may go before the
        /// request goes to the least loaded replica instead.
        double load_slack = 1.25;
    };

    /// Requests a replica has admitted and not yet finished.
    using LoadFn = std::function<std::size_t(std::size_t replica)>;
    /// Whether a replica may take requests now (awake and healthy). When none is, every
    /// replica is a candidate, so the request reaches one and gets its refusal from there.
    using AvailableFn = std::function<bool(std::size_t replica)>;

    struct Route {
        std::size_t replica = 0;
        /// Counts the request against its replica until the replica admits it (from then on
        /// `LoadFn` covers it). Releasing is resetting or destroying it.
        std::shared_ptr<void> reservation;
    };

    ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available);
    ReplicaRouter(std::size_t replicas, LoadFn load, AvailableFn available, Options options);

    ReplicaRouter(const ReplicaRouter&)            = delete;
    ReplicaRouter& operator=(const ReplicaRouter&) = delete;

    /// Routes a request whose prompt has the cumulative prefix hashes `prefixes`, shortest
    /// first (empty when the request has nothing to key on).
    [[nodiscard]] Route pick(const std::vector<std::uint64_t>& prefixes);
    /// Routes to the replica the caller named (vLLM's `X-data-parallel-rank`). The prefixes
    /// are remembered as having been sent there.
    /// \throws std::out_of_range If `replica` is not below size().
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

    [[nodiscard]] static std::size_t longest_match(const Replica& replica,
                                                   const std::vector<std::uint64_t>& prefixes);
    void remember(Replica& replica, const std::vector<std::uint64_t>& prefixes);
    [[nodiscard]] Route reserve(std::size_t replica);

    LoadFn load_;
    AvailableFn available_;
    Options options_;
    mutable std::mutex mutex_;
    std::vector<Replica> replicas_;
    /// Routes on their way to each replica. Shared with the reservations, which a request
    /// holds and may release after the router is gone (server shutdown).
    std::shared_ptr<std::vector<std::atomic<std::size_t>>> reserved_;
    /// Where the next tie between equally loaded replicas starts, so ties rotate.
    std::size_t cursor_ = 0;
};

/// Cumulative hashes of the request's prompt, one per message boundary (the tools, then each
/// turn): two turns of one conversation share every hash up to where they diverge. A raw
/// `/v1/completions` prompt is cut every 4 KiB instead, and a token-id prompt with no messages
/// every 1024 tokens. Empty when there is no prompt.
[[nodiscard]] std::vector<std::uint64_t> conversation_prefix_hashes(const GenerationRequest& request);

} // namespace sinfer::serve
