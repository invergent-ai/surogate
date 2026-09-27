#include "runtime/engine/transient_slots.h"
#include <cassert>
#include <random>
#include <vector>

using sinfer::runtime::TransientSlots;

int main() {
    TransientSlots unaligned_capacity(900);
    unaligned_capacity.activate(0, 768, 256);
    assert(unaligned_capacity.fit(1, 100, 256) == 768);
    TransientSlots pool(1024);
    pool.activate(4, 300, 256);
    pool.activate(9, 256, 256);
    assert(pool.region(4)->offset == 0 && pool.region(9)->offset == 512);
    assert(!pool.fit(7, 512, 256));
    pool.release(4);
    pool.activate(7, 512, 256);
    assert(pool.region(9)->offset == 512);
    assert(pool.used() == 768 && pool.peak() == 768);
    try { pool.activate(7, 2048, 256); assert(false); }
    catch (const std::invalid_argument&) {}
    assert(pool.region(7)->bytes == 512 && pool.used() == 768);
    pool.clear();
    assert(pool.used() == 0 && pool.peak() == 768);
    pool.reset_peak();
    assert(pool.peak() == 0);
    assert(!pool.fit(0, 1, 3) && !pool.fit(0, 1, 512));

    // Independently mark every byte after fragmented allocations and cancellations.
    std::mt19937 rng(42);
    for (int step = 0; step < 10000; ++step) {
        const unsigned lane = rng() % 32;
        const unsigned bytes = 1 + rng() % 257;
        const unsigned alignment = 1U << (rng() % 9);
        if (rng() % 3 == 0) { pool.release(lane); }
        else if (pool.fit(lane, bytes, alignment)) { pool.activate(lane, bytes, alignment); }
        std::vector<bool> used(1024);
        std::size_t total = 0;
        for (unsigned l = 0; l < 32; ++l) {
            const auto r = pool.region(l);
            if (!r) { continue; }
            assert(r->offset % r->alignment == 0 && r->offset + r->bytes <= 1024);
            total += r->bytes;
            for (std::size_t i = r->offset; i < r->offset + r->bytes; ++i) {
                assert(!used[i]);
                used[i] = true;
            }
        }
        assert(total == pool.used() && total <= pool.peak());
    }
}
