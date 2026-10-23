// The host KV tier's pool and keys (family/impl/kv_tier/), without a GPU: chained SHA-256 keys,
// shared chunks, least-recently-used eviction of whole entries across owners, leases, readiness,
// abandoned writes, and the C back-end table (api/kv_tier.h) driven through CBackend.
#include "family/impl/kv_tier/host_pool.h"
#include "family/impl/kv_tier/keys.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using sinfer::family::detail::CBackend;
using sinfer::family::detail::HostKvPool;
namespace kvkeys = sinfer::family::detail::kvkeys;
namespace kvtier = sinfer::kvtier;

namespace {

kvtier::ObjectRef chunk(std::uint8_t tag, std::size_t bytes) {
    kvtier::ObjectRef ref;
    ref.key.bytes.fill(tag);
    ref.kind  = kvtier::Kind::Chunk;
    ref.bytes = bytes;
    return ref;
}

void keys() {
    std::vector<std::int32_t> tokens(3000);
    std::iota(tokens.begin(), tokens.end(), 7);
    const std::vector<std::uint8_t> model{1, 2, 3};
    const kvkeys::AdapterDigest base{};
    kvkeys::AdapterDigest adapter{};
    adapter[0] = 9;
    const auto root = kvkeys::namespace_root(model, base);
    const auto chain = kvkeys::chunk_chain(root, tokens, 2);
    assert(chain.size() == 2);
    // Deterministic, and chained: chunk 1's key depends on chunk 0's tokens.
    assert(kvkeys::chunk_chain(root, tokens, 2) == chain);
    auto changed = tokens;
    changed[5] = 1;
    const auto other = kvkeys::chunk_chain(root, changed, 2);
    assert(other[0] != chain[0] && other[1] != chain[1]);
    // A change after chunk 0 leaves its key alone.
    changed = tokens;
    changed[1500] = 1;
    assert(kvkeys::chunk_chain(root, changed, 2)[0] == chain[0]);
    // The namespace separates models, adapters and tenants.
    assert(kvkeys::namespace_root(model, adapter) != root);
    assert(kvkeys::namespace_root(std::vector<std::uint8_t>{1, 2, 4}, base) != root);
    assert(kvkeys::namespace_root(model, base, "tenant") != root);
    // A draft head's chunk also names the next token; a boundary names its frontier and kind.
    assert(!(kvkeys::chunk_key(chain[0], 5) == kvkeys::chunk_key(chain[0], 6)));
    assert(!(kvkeys::chunk_key(chain[0], std::nullopt) == kvkeys::chunk_key(chain[0], 5)));
    const auto tail = std::span<const std::int32_t>(tokens).subspan(2048, 100);
    assert(!(kvkeys::boundary_key(chain[1], tail, kvkeys::BoundaryKind::AppendFrontier, 2148) ==
             kvkeys::boundary_key(chain[1], tail, kvkeys::BoundaryKind::RewriteCheckpoint, 2148)));
    std::cout << "keys: OK\n";
}

void pool() {
    auto pool = HostKvPool::create_unpinned(1024, 8);
    assert(pool->capacity_bytes() == 8 * 1024);
    const auto a = pool->add_owner();
    const auto b = pool->add_owner();

    // Owner a writes an entry of a shared chunk and a 2-block boundary.
    auto shared = pool->acquire(chunk(1, 1024), a);
    auto own    = pool->acquire(chunk(2, 1500), a);
    assert(shared && shared->created && own && own->created);
    assert(pool->block_count(own->object) == 2 && pool->used_bytes() == 3 * 1024);
    std::memset(pool->block(shared->object, 0).data(), 0x11, 1024);
    const auto ea = pool->add_entry(a, {shared->object, own->object});
    // Not ready yet: the writer may lease it (its copy stream orders the read), another owner not.
    assert(pool->lease(ea, a));
    assert(!pool->lease(ea, b));
    pool->mark_ready(std::vector<HostKvPool::ObjectId>{shared->object, own->object});
    assert(pool->lease(ea, b));

    // Owner b finds the shared chunk stored: no new blocks.
    auto again = pool->acquire(chunk(1, 1024), b);
    assert(again && !again->created && again->object == shared->object);
    auto b_own = pool->acquire(chunk(3, 1024), b);
    assert(b_own && b_own->created);
    const auto eb = pool->add_entry(b, {again->object, b_own->object});
    pool->mark_ready(std::vector<HostKvPool::ObjectId>{b_own->object});
    assert(pool->used_bytes() == 4 * 1024 && pool->shared_objects() == 1);

    // Room for 5 more blocks evicts the least recently used entry no lease holds: a's (b's is
    // newer). Its own blocks go; the shared chunk stays with b. a learns of it.
    pool->touch(eb);
    {
        auto lease = pool->lease(eb, b);
        auto big   = pool->acquire(chunk(4, 5 * 1024), b);
        assert(big && big->created);
        assert(!pool->alive(ea) && pool->alive(eb));
        const auto dropped = pool->take_dropped(a);
        assert(dropped.size() == 1 && dropped[0] == ea);
        assert(pool->take_dropped(a).empty());
        // One block is free, and nothing is left to evict while b's entry is leased.
        assert(pool->used_bytes() == 7 * 1024);
        assert(!pool->acquire(chunk(5, 2 * 1024), b));
        pool->release_objects(std::vector<HostKvPool::Placement>{*big});
    }
    // A writer that gives up an unfinished object poisons it: an entry that found it cannot lease.
    auto unfinished = pool->acquire(chunk(6, 1024), a);
    auto finder     = pool->acquire(chunk(6, 1024), b);
    assert(unfinished && unfinished->created && finder && !finder->created);
    const auto eb2 = pool->add_entry(b, {finder->object});
    pool->release_objects(std::vector<HostKvPool::Placement>{*unfinished});
    assert(!pool->lease(eb2, b));
    // ... and the next writer of that key starts afresh.
    auto fresh = pool->acquire(chunk(6, 1024), a);
    assert(fresh && fresh->created && fresh->object != finder->object);
    pool->release_objects(std::vector<HostKvPool::Placement>{*fresh});
    pool->drop_entry(eb2);

    // Dropping an owner's entries returns their blocks.
    pool->drop_owner(b);
    assert(pool->entries() == 0 && pool->used_bytes() == 0);
    std::cout << "pool: OK\n";
}

void backend() {
    auto pool  = HostKvPool::create_unpinned(64, 16);
    auto table = sinfer::family::detail::host_pool_c_table(pool.get());
    CBackend backend(&table);
    assert(backend.caps().chunk_tokens == kvtier::kChunkTokens);
    std::size_t stored = 0, removed = 0;
    backend.set_listener([&](std::span<const kvtier::ObjectRef> refs, bool is_stored) {
        (is_stored ? stored : removed) += refs.size();
    });
    std::vector<std::uint8_t> data(100);
    std::iota(data.begin(), data.end(), 0);
    const auto ref = chunk(7, data.size());
    const kvtier::Span in{data.data(), data.size()};
    const auto put = backend.put(std::span<const kvtier::ObjectRef>(&ref, 1), std::span<const kvtier::Span>(&in, 1));
    assert(backend.poll(put) == SINFER_KVTIER_DONE && stored == 1);
    std::uint8_t present = 0;
    backend.contains(std::span<const kvtier::ObjectRef>(&ref, 1), std::span<std::uint8_t>(&present, 1));
    assert(present == 1);
    std::vector<std::uint8_t> out(100, 0);
    const kvtier::Span target{out.data(), out.size()};
    const auto get = backend.get(std::span<const kvtier::ObjectRef>(&ref, 1), std::span<const kvtier::Span>(&target, 1));
    assert(backend.poll(get) == SINFER_KVTIER_DONE && out == data);
    backend.erase(std::span<const kvtier::Key>(&ref.key, 1));
    backend.contains(std::span<const kvtier::ObjectRef>(&ref, 1), std::span<std::uint8_t>(&present, 1));
    assert(present == 0 && removed == 1);
    const auto missing = backend.get(std::span<const kvtier::ObjectRef>(&ref, 1), std::span<const kvtier::Span>(&target, 1));
    assert(backend.poll(missing) == SINFER_KVTIER_FAILED);
    std::cout << "C back end: OK\n";
}

} // namespace

int main() {
    keys();
    pool();
    backend();
    std::cout << "host KV pool: OK\n";
    return 0;
}
