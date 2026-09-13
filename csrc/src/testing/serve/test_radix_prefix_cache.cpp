// Scenarios adapted from SGLang's radix/hybrid-cache operations: branches,
// checkpoint-only matches, independent namespaces, pins, and LRU eviction.
#include "family/impl/radix_prefix_cache.h"
#include <array>
#include <cassert>

struct Value { unsigned frontier; int identity; };
using Cache = sinfer::family::detail::RadixPrefixCache<Value>;

int main() {
    Cache cache(30, 8);
    const std::array<int, 6> a{1,2,3,4,5,6}, b{1,2,3,7,8,9};
    const std::array<int, 4> branch{1,2,3,99};
    const std::array<unsigned, 1> six{6}, three{3};
    const auto match = [&](auto key, int identity = 0) {
        return cache.match(key, [&](const Value& v) { return v.identity == identity ? v.frontier : 0; });
    };
    assert(cache.insert(a, six, std::make_shared<Value>(6, 0), 10));
    assert(cache.insert(b, six, std::make_shared<Value>(6, 0), 10));
    assert(match(a)->frontier == 6 && match(b)->frontier == 6);
    // Splitting the token edge does not manufacture a recurrent-state image.
    assert(!match(branch));
    assert(cache.insert(a, three, std::make_shared<Value>(3, 0), 10));
    assert(match(branch)->frontier == 3);
    // A matched value stays alive and cannot be evicted while a request pins it.
    auto pinned = match(a);
    const std::array<int, 2> unrelated{80,81};
    const std::array<unsigned, 1> two{2};
    assert(cache.insert(unrelated, two, std::make_shared<Value>(2, 0), 10));
    assert(match(a)->frontier == 6);
    assert(cache.bytes() == 30);
    assert(!cache.reserve(31));
    assert(!match(a, 1));
    assert(cache.insert(a, six, std::make_shared<Value>(6, 1), 10));
    assert(match(a, 1)->identity == 1 && match(a, 0)->identity == 0);
    cache.clear();
    assert(cache.empty() && cache.bytes() == 0 && pinned->frontier == 6);

    Cache locked(10);
    assert(locked.insert(a, six, std::make_shared<Value>(6, 0), 10));
    auto lease = locked.match(a, [](const Value& v) { return v.frontier; });
    assert(!locked.insert(b, six, std::make_shared<Value>(6, 0), 10));
    lease.reset();
    assert(locked.insert(b, six, std::make_shared<Value>(6, 0), 10));
    assert(!locked.match(a, [](const Value& v) { return v.frontier; }));
    // SGLang test_advanced_prefix_match_with_node_splits: dense attention
    // can resume inside an edge while complete-state matching cannot.
    Cache dense(100);
    const std::array<int, 8> long_key{1,2,3,4,5,6,7,8};
    const std::array<int, 4> short_key{1,2,9,10};
    const std::array<unsigned, 1> eight{8}, four{4};
    assert(dense.insert(long_key, eight, std::make_shared<Value>(8, 0), 10));
    assert(dense.insert(short_key, four, std::make_shared<Value>(4, 1), 10));
    const std::array<int, 6> query{1,2,3,4,999,1000};
    auto dense_match = [&](const Value& value) { return value.identity == 0 ? 4U : 2U; };
    assert(!dense.match(query, dense_match));
    assert(dense.match(query, dense_match, true)->identity == 0);
    const auto before = dense.bytes();
    assert(dense.match(long_key, [](const Value& value) { return value.identity == 0 ? 8U : 2U; }, true)->frontier == 8);
    assert(dense.bytes() == before);

    Cache continuations(20, 2);
    std::vector<int> growing{1};
    for (unsigned end = 2; end < 1024; ++end) {
        growing.push_back(end);
        const std::array<unsigned, 1> boundary{end};
        assert(continuations.insert(growing, boundary, std::make_shared<Value>(end, 0), 10));
        assert(continuations.match(growing, [](const Value& v) { return v.frontier; })->frontier == end);
        assert(continuations.bytes() <= 20);
    }

}
