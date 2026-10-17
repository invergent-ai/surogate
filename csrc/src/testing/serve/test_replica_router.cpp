// Data-parallel routing (--data-parallel, #261): which replica a request goes to. No model or GPU.
#include "serve/replica_router.h"
#include "serve/request.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using sinfer::ChatRole;
using sinfer::serve::ChatTurn;
using sinfer::serve::ContentKind;
using sinfer::serve::ContentPart;
using sinfer::serve::GenerationRequest;
using sinfer::serve::ReplicaRouter;
using sinfer::serve::RoutingKey;
using sinfer::serve::conversation_prefix_hashes;
using sinfer::serve::conversation_routing_key;

namespace {

/// Admitted requests per replica, as GenerationService::active_requests would report them, and
/// the share of each replica's KV cache in use.
struct Fleet {
    std::vector<std::size_t> active;
    std::vector<bool> up;
    std::vector<double> kv;
    explicit Fleet(std::size_t replicas) : active(replicas, 0), up(replicas, true), kv(replicas, 0.0) {}
    ReplicaRouter router(ReplicaRouter::Options options = {}) {
        return ReplicaRouter(
            active.size(), [this](std::size_t i) { return active[i]; },
            [this](std::size_t i) { return static_cast<bool>(up[i]); },
            [this](std::size_t i) { return kv[i]; }, options);
    }
};

/// Routes and admits: the reservation is dropped once the replica counts the request itself.
std::size_t admit(ReplicaRouter& router, Fleet& fleet, const std::vector<std::uint64_t>& prefixes) {
    auto route = router.pick(prefixes);
    ++fleet.active[route.replica];
    return route.replica;
}
std::size_t admit(ReplicaRouter& router, Fleet& fleet, const RoutingKey& key) {
    auto route = router.pick(key);
    ++fleet.active[route.replica];
    return route.replica;
}

ChatTurn turn(ChatRole role, std::string text) {
    ChatTurn result;
    result.role = role;
    ContentPart part;
    part.kind = ContentKind::Text;
    part.text = std::move(text);
    result.content.push_back(std::move(part));
    return result;
}

GenerationRequest conversation(const std::vector<std::string>& texts,
                               const std::string& system = "you are a tool-using agent") {
    GenerationRequest request;
    request.messages.push_back(turn(ChatRole::System, system));
    for (std::size_t i = 0; i < texts.size(); ++i) {
        request.messages.push_back(turn(i % 2 == 0 ? ChatRole::User : ChatRole::Assistant, texts[i]));
    }
    return request;
}

void conversation_hashes_share_their_common_prefix() {
    const auto first  = conversation_prefix_hashes(conversation({"task A"}));
    const auto second = conversation_prefix_hashes(conversation({"task A", "call tool", "result"}));
    const auto other  = conversation_prefix_hashes(conversation({"task B"}));
    assert(first.size() == 2 && second.size() == 4);
    assert(second[0] == first[0] && second[1] == first[1]);
    assert(other[0] == first[0]);  // the same system prompt
    assert(other[1] != first[1]);
    // Splitting text differently across turns is a different conversation.
    const auto split = conversation_prefix_hashes(conversation({"task", "A"}));
    assert(split[1] != first[1]);

    GenerationRequest raw;
    raw.raw_prompt = std::string(10000, 'x');
    const auto raw_hashes = conversation_prefix_hashes(raw);
    assert(raw_hashes.size() == 2);  // two whole 4 KiB chunks
    raw.raw_prompt->append(5000, 'y');
    const auto longer = conversation_prefix_hashes(raw);
    assert(longer.size() == 3 && longer[0] == raw_hashes[0] && longer[1] == raw_hashes[1]);

    GenerationRequest tokens;
    tokens.prompt_token_ids.assign(2500, 7);
    assert(conversation_prefix_hashes(tokens).size() == 2);
    assert(conversation_prefix_hashes(GenerationRequest{}).empty());

    // The routing key carries the same hashes and the prompt bytes up to each boundary.
    const RoutingKey key = conversation_routing_key(conversation({"task A", "call tool", "result"}));
    assert(key.prefixes == second && key.bytes.size() == key.prefixes.size());
    for (std::size_t i = 1; i < key.bytes.size(); ++i) { assert(key.bytes[i] > key.bytes[i - 1]); }
    assert(conversation_routing_key(raw).bytes.back() == 12288);  // three 4 KiB chunks
}

void requests_without_history_spread_evenly() {
    Fleet fleet(4);
    auto router = fleet.router();
    std::vector<std::size_t> per_replica(4, 0);
    for (int i = 0; i < 40; ++i) { ++per_replica[admit(router, fleet, std::vector<std::uint64_t>{})]; }
    for (const std::size_t count : per_replica) { assert(count == 10); }
}

void a_burst_spreads_before_anything_is_admitted() {
    // Routed in the same instant: none admitted yet, only the reservations tell them apart.
    Fleet fleet(3);
    auto router = fleet.router();
    std::vector<ReplicaRouter::Route> held;
    std::set<std::size_t> used;
    for (int i = 0; i < 3; ++i) {
        held.push_back(router.pick(std::vector<std::uint64_t>{}));
        used.insert(held.back().replica);
    }
    assert(used.size() == 3);
    for (std::size_t i = 0; i < 3; ++i) { assert(router.reserved(i) == 1); }
    held.clear();
    for (std::size_t i = 0; i < 3; ++i) { assert(router.reserved(i) == 0); }
}

void a_conversation_stays_on_its_replica() {
    Fleet fleet(4);
    auto router = fleet.router();
    // Four unrelated conversations in flight, one per replica.
    std::vector<std::vector<std::string>> chats{{"a"}, {"b"}, {"c"}, {"d"}};
    const auto request = [&](std::size_t c) {
        return conversation_prefix_hashes(conversation(chats[c], "agent " + std::to_string(c)));
    };
    std::vector<std::size_t> home;
    for (std::size_t c = 0; c < chats.size(); ++c) { home.push_back(admit(router, fleet, request(c))); }
    assert(std::set<std::size_t>(home.begin(), home.end()).size() == 4);
    // Later turns extend each conversation and go back where it was served.
    for (int round = 0; round < 5; ++round) {
        for (std::size_t c = 0; c < chats.size(); ++c) {
            --fleet.active[home[c]];  // the previous turn finished
            chats[c].push_back("reply " + std::to_string(round));
            chats[c].push_back("tool result " + std::to_string(round));
            assert(admit(router, fleet, request(c)) == home[c]);
        }
    }
}

void a_shared_system_prompt_does_not_gather_tasks() {
    // Every task starts with the same system prompt, which every replica that served one holds in
    // cache. That head is no reason to send a NEW task there: tasks spread as if it were not
    // cached (it used to draw each one onto a replica hosting another, up to 1.25x the mean).
    Fleet fleet(4);
    auto router = fleet.router();
    std::vector<std::size_t> per_replica(4, 0);
    for (int task = 0; task < 40; ++task) {
        ++per_replica[admit(router, fleet, conversation_routing_key(conversation({"task " + std::to_string(task)})))];
    }
    for (const std::size_t count : per_replica) { assert(count == 10); }
}

void a_second_task_is_not_drawn_to_the_first() {
    // The first task of an agent is idle between tool calls, its replica's cache holding it. The
    // second, the same system prompt but another task, goes where the cache is free -- before any
    // third conversation has shown that the system prompt is a shared head.
    Fleet fleet(2);
    auto router = fleet.router();
    const std::size_t first = admit(router, fleet, conversation_routing_key(conversation({"task A"})));
    --fleet.active[first];   // its turn finished; the agent runs a tool
    fleet.kv[first] = 0.5;   // ...and its 230k tokens stay cached
    const std::size_t second = admit(router, fleet, conversation_routing_key(conversation({"task B"})));
    assert(second != first);
    // The first task's next turn still goes home: its own prefix is there.
    --fleet.active[second];
    assert(admit(router, fleet, conversation_routing_key(conversation({"task A", "reply", "result"}))) == first);
}

void a_new_task_avoids_a_full_cache() {
    // Load alone said replica 0 was free; its cache is 90% held by idle conversations a new one
    // would evict. Two requests running elsewhere cost less than that.
    Fleet fleet(3);
    auto router = fleet.router();
    fleet.active = {0, 2, 2};
    fleet.kv     = {0.9, 0.0, 0.1};
    assert(admit(router, fleet, conversation_routing_key(conversation({"new task"}))) == 1);
}

void a_long_conversation_stays_on_a_busy_replica() {
    // 70 KB of prompt in its replica's cache: a re-read elsewhere costs more than the imbalance,
    // so it stays where the bounded-loads rule would have moved it (6 + 1 > ceil(1.25 x 7/4)).
    Fleet fleet(4);
    auto router = fleet.router({.replica_capacity = 8});
    const std::string task(70000, 't');
    const std::size_t home = admit(router, fleet, conversation_routing_key(conversation({task})));
    fleet.active.assign(4, 0);
    fleet.active[home] = 6;
    assert(admit(router, fleet, conversation_routing_key(conversation({task, "reply", "result"}))) == home);
    // ...until its replica is full: then it goes, and to the replica with room.
    fleet.active[home] = 8;
    const std::size_t moved =
        admit(router, fleet, conversation_routing_key(conversation({task, "reply", "result", "reply 2", "result 2"})));
    assert(moved != home);
}

void an_overloaded_replica_gives_up_the_conversation() {
    Fleet fleet(2);
    auto router = fleet.router();
    const auto first = conversation_prefix_hashes(conversation({"long task"}));
    const std::size_t home = admit(router, fleet, first);
    // Its replica fills up with other work: past the bound, the next turn goes elsewhere.
    fleet.active[home] = 10;
    const auto next = conversation_prefix_hashes(conversation({"long task", "reply", "result"}));
    const std::size_t moved = admit(router, fleet, next);
    assert(moved != home);
    // Within the bound it stays: 1.25 x the mean (3 + 2 + 1) / 2 rounds up to 4, and 3 + 1 fits.
    fleet.active = {0, 0};
    fleet.active[moved] = 3;
    fleet.active[1 - moved] = 2;
    const auto third = conversation_prefix_hashes(
        conversation({"long task", "reply", "result", "reply 2", "result 2"}));
    assert(admit(router, fleet, third) == moved);
}

void unavailable_replicas_are_skipped() {
    Fleet fleet(3);
    auto router = fleet.router();
    const auto prefixes = conversation_prefix_hashes(conversation({"x"}));
    const std::size_t home = admit(router, fleet, prefixes);
    fleet.up[home] = false;  // asleep or failed
    for (int i = 0; i < 6; ++i) { assert(admit(router, fleet, prefixes) != home); }
    // With every replica down, a request still reaches one and is refused there.
    fleet.up.assign(3, false);
    (void)admit(router, fleet, prefixes);
}

void a_pinned_request_goes_where_it_was_told() {
    Fleet fleet(2);
    auto router = fleet.router();
    const auto prefixes = conversation_prefix_hashes(conversation({"pinned"}));
    fleet.active[1] = 50;
    auto route = router.pin(1, prefixes);
    assert(route.replica == 1 && router.reserved(1) == 1);
    route.reservation.reset();
    assert(router.reserved(1) == 0);
    // ...and the prefix is remembered there.
    fleet.active[1] = 0;
    assert(router.pick(prefixes).replica == 1);
    bool refused = false;
    try {
        (void)router.pin(2, prefixes);
    } catch (const std::out_of_range&) { refused = true; }
    assert(refused);
}

void memory_is_bounded() {
    Fleet fleet(1);
    auto router = fleet.router({.remembered_prefixes = 4, .load_slack = 1.25});
    for (std::uint64_t i = 0; i < 100; ++i) { (void)router.pick({i, i + 1000}); }
    // The oldest prefixes are forgotten; only the four most recent are matched. A one-replica
    // router still routes everything to it.
    assert(router.pick({1}).replica == 0);
}

} // namespace

int main() {
    conversation_hashes_share_their_common_prefix();
    requests_without_history_spread_evenly();
    a_burst_spreads_before_anything_is_admitted();
    a_conversation_stays_on_its_replica();
    a_shared_system_prompt_does_not_gather_tasks();
    a_second_task_is_not_drawn_to_the_first();
    a_new_task_avoids_a_full_cache();
    a_long_conversation_stays_on_a_busy_replica();
    an_overloaded_replica_gives_up_the_conversation();
    unavailable_replicas_are_skipped();
    a_pinned_request_goes_where_it_was_told();
    memory_is_bounded();
    std::cout << "replica router: ok\n";
    return 0;
}
