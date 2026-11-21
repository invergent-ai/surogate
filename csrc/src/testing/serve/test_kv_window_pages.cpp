// Window-bounded KV pages (lane f1, rl/engine-inkling-2026-11-21/lanes/f1/DESIGN-window-kv.md):
// the host-only page manager KvWindowPages, driven the way the runtime drives it, against a
// simulated page store. No device: every check runs on the CPU.
//
// What is checked, for every simulated forward:
//  - every key a query reads (q - j < window) sits in a live page that holds exactly the value the
//    model last wrote at that position (no page lost, none recycled under a reader);
//  - a page holding committed keys is never freshly mapped (prepare throws instead);
//  - the class sized by plan_kv_window_class never runs dry with many lanes mixing prefill
//    chunks (graph-bucket pads included), decode, speculative verify with rollback, rewrite
//    checkpoints (pins, restore), shared-prefix images + forks, archive restores;
//  - covers() is true exactly where a plan may reuse (frontier, frontier - 1, pinned checkpoint);
//  - an inactive manager (a model without windowed layers) does nothing.

#include "core/kv_window_pages.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using sinfer::KvWindowChange;
using sinfer::KvWindowPages;
using sinfer::KvWindowSlot;

constexpr std::uint32_t kPage = sinfer::kKvWindowPageTokens;
int g_failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        ++g_failures;
        if (g_failures < 20) { std::fprintf(stderr, "FAIL: %s\n", what.c_str()); }
    }
}

/// Page store with owner tracking and a value per token slot.
struct Store final : sinfer::KvWindowPageSource {
    explicit Store(std::uint32_t pages) : owner(pages, -1), cells(std::size_t(pages) * kPage, -1) {
        for (std::uint32_t p = pages; p > 0; --p) { free.push_back(static_cast<std::int32_t>(p - 1)); }
        capacity = pages;
    }
    std::int32_t take_window_page() override {
        if (free.empty()) { throw std::logic_error("store exhausted"); }
        const std::int32_t page = free.back();
        free.pop_back();
        if (owner[page] != -1) { throw std::logic_error("page taken twice"); }
        owner[page] = 1;
        peak = std::max<std::uint32_t>(peak, capacity - static_cast<std::uint32_t>(free.size()));
        // A recycled page holds whatever its last owner wrote: poison it so a stale read shows.
        std::fill(cells.begin() + std::ptrdiff_t(page) * kPage, cells.begin() + std::ptrdiff_t(page + 1) * kPage, -7);
        return page;
    }
    void give_back_window_page(std::int32_t page) noexcept override {
        if (owner[page] == -1) { ++g_failures; std::fprintf(stderr, "FAIL: page %d given back twice\n", page); }
        owner[page] = -1;
        if (sorted) {
            // Elastic class (M3-C): the free list stays sorted descending, so takes are lowest-first.
            free.insert(std::upper_bound(free.begin(), free.end(), page, std::greater<>()), page);
        } else {
            free.push_back(page);
        }
    }
    std::uint32_t in_use() const { return capacity - static_cast<std::uint32_t>(free.size()); }
    /// Granules holding at least one page (what a demand-mapped class keeps mapped, reserve aside).
    std::uint32_t granules_in_use(std::uint32_t granule) const {
        std::uint32_t count = 0;
        for (std::uint32_t g = 0; g * granule < capacity; ++g) {
            bool any = false;
            for (std::uint32_t p = g * granule; p < std::min(capacity, (g + 1) * granule) && !any; ++p) { any = owner[p] != -1; }
            count += any ? 1U : 0U;
        }
        return count;
    }
    bool sorted = false;
    std::vector<std::int32_t> free;
    std::vector<int> owner;
    std::vector<std::int64_t> cells;
    std::uint32_t capacity = 0;
    std::uint32_t peak     = 0;
};

/// One sequence: its page manager and the value the model last wrote at each position.
struct Seq {
    KvWindowPages pages;
    std::vector<std::int64_t> truth; // value per position (index = position)
    std::uint32_t committed = 0;     // text_kv_valid
    std::int64_t epoch      = 1;
};

void write(Store& store, Seq& seq, std::uint32_t position, std::int64_t value) {
    const std::int32_t physical = seq.pages.physical(position / kPage);
    if (physical < 0) {
        expect(false, "write to unmapped page at " + std::to_string(position));
        return;
    }
    store.cells[std::size_t(physical) * kPage + position % kPage] = value;
    if (seq.truth.size() <= position) { seq.truth.resize(position + 1, -1); }
    seq.truth[position] = value;
}

/// A forward over queries [first, first + tokens) that writes [first, first + reach) (pads past
/// `tokens`), reading each query's window after writing its own key (causal, as the kernels do).
void forward(Store& store, Seq& seq, std::uint32_t first, std::uint32_t tokens, std::uint32_t reach,
             std::uint32_t cap, const std::vector<std::uint32_t>& pins) {
    const KvWindowChange change = seq.pages.prepare(first, first + reach, cap, pins, store);
    (void)change;
    const std::uint32_t window = seq.pages.window_tokens();
    for (std::uint32_t q = first; q < first + reach; ++q) {
        write(store, seq, q, q < first + tokens ? seq.epoch * 1000000 + q : -3 /* pad */);
    }
    // The keys the forward's queries read are the union of their windows: [lo(first), last query].
    // Nothing changes pages during a forward, so checking that union once checks every query.
    if (tokens == 0) { return; }
    const std::uint32_t lo = first + 1 > window ? first + 1 - window : 0;
    for (std::uint32_t k = lo; k < first + tokens; ++k) {
        const std::int32_t physical = seq.pages.physical(k / kPage);
        if (physical < 0) {
            expect(false, "read of released page " + std::to_string(k / kPage) + " by the forward at " +
                              std::to_string(first));
            return;
        }
        const std::int64_t value = store.cells[std::size_t(physical) * kPage + k % kPage];
        if (value != seq.truth[k]) {
            expect(false, "forward at " + std::to_string(first) + " read key " + std::to_string(k) + " = " +
                              std::to_string(value) + ", expected " + std::to_string(seq.truth[k]));
            return;
        }
    }
}

void rewind(Store& store, Seq& seq, std::uint32_t tokens) {
    (void)seq.pages.rewind(tokens, store);
    seq.committed = tokens;
    if (seq.truth.size() > tokens) { seq.truth.resize(tokens); }
    ++seq.epoch;
}

/// Copy of the window behind `tokens` into image pages (shared prefix capture), and back (fork).
std::vector<KvWindowSlot> capture_image(Store& lane_store, const Seq& seq, Store& image_store, std::uint32_t tokens) {
    std::vector<KvWindowSlot> image;
    const std::uint32_t window = seq.pages.window_tokens();
    for (std::uint32_t page = KvWindowPages::keep_first_page(tokens, window); page < KvWindowPages::pages_for(tokens); ++page) {
        const std::int32_t source = seq.pages.physical(page);
        if (source < 0) { return {}; }
        const std::int32_t dest = image_store.take_window_page();
        std::copy_n(lane_store.cells.begin() + std::ptrdiff_t(source) * kPage, kPage,
                    image_store.cells.begin() + std::ptrdiff_t(dest) * kPage);
        image.push_back({page, dest});
    }
    return image;
}

void restore_image(Store& lane_store, Seq& seq, const Store& image_store, const std::vector<KvWindowSlot>& image,
                   const std::vector<std::int64_t>& truth, std::uint32_t tokens) {
    std::vector<std::uint32_t> logical;
    for (const auto& slot : image) { logical.push_back(slot.logical); }
    const KvWindowChange change = seq.pages.map_restored(logical, lane_store);
    for (std::size_t i = 0; i < image.size(); ++i) {
        std::copy_n(image_store.cells.begin() + std::ptrdiff_t(image[i].physical) * kPage, kPage,
                    lane_store.cells.begin() + std::ptrdiff_t(change.mapped[i].physical) * kPage);
    }
    seq.truth.assign(truth.begin(), truth.begin() + tokens);
    seq.committed = tokens;
}

void test_inactive() {
    Store store(4);
    KvWindowPages pages; // a model without windowed layers
    const std::vector<std::uint32_t> pins{100};
    expect(pages.prepare(0, 5000, 8192, pins, store).empty(), "inactive prepare is empty");
    expect(pages.settle(5000, pins, store).empty(), "inactive settle is empty");
    expect(pages.rewind(10, store).empty(), "inactive rewind is empty");
    expect(pages.covers(12345), "inactive covers everything");
    expect(store.in_use() == 0, "inactive takes no page");
}

void test_single_lane(std::uint32_t window, std::uint32_t chunk) {
    const auto size = sinfer::plan_kv_window_class(window, 1, chunk, 0);
    Store store(size.lane_pages);
    Seq seq{KvWindowPages(window)};
    const std::uint32_t wp = KvWindowPages::pages_for(window) + 1;
    const std::uint32_t prompt = 9000 + window;
    // Admission maps no window page ahead (prepare with end == first).
    (void)seq.pages.prepare(0, 0, size.forward_tokens, {}, store);
    expect(store.in_use() == 0, "admission maps no window page");
    // Chunked prefill, graph buckets (pads to a multiple of 128) on every other chunk; the
    // rewrite checkpoint is planned at C (pinned from the start of the prefill).
    const std::uint32_t checkpoint = prompt - 37;
    std::vector<std::uint32_t> pins{checkpoint};
    int index = 0;
    while (seq.committed < prompt) {
        std::uint32_t n = std::min(chunk, prompt - seq.committed);
        if (seq.committed < checkpoint) { n = std::min(n, checkpoint - seq.committed); }
        const std::uint32_t reach = (index++ % 2) ? (n + 127) / 128 * 128 : n;
        forward(store, seq, seq.committed, n, reach, size.forward_tokens, pins);
        seq.committed += n;
        (void)seq.pages.settle(seq.committed, pins, store);
        expect(seq.pages.page_count() <= 2 * wp, "settled prefill lane holds <= window + pin");
    }
    expect(seq.pages.covers(prompt) && seq.pages.covers(prompt - 1), "frontier and frontier - 1 covered");
    expect(seq.pages.covers(checkpoint), "pinned checkpoint covered");
    // Decode bursts and speculative verify with random acceptance; the commit trim rewinds.
    std::mt19937 rng(7 + window);
    for (int round = 0; round < 3000; ++round) {
        const std::uint32_t width = (round % 3 == 0) ? 1 : 1 + rng() % 6;
        forward(store, seq, seq.committed, width, width, size.forward_tokens, pins);
        const std::uint32_t accepted = 1 + rng() % width;
        rewind(store, seq, seq.committed + accepted);
        expect(seq.pages.page_count() <= 2 * wp + 1, "decoding lane holds <= window + pin + reach");
    }
    const std::uint32_t frontier = seq.committed;
    expect(seq.pages.covers(frontier) && seq.pages.covers(frontier - 1), "decode frontier covered");
    expect(!seq.pages.covers(frontier - window - 2 * kPage), "a point far behind the window is not covered");
    // Restore at the rewrite checkpoint: its window survived the whole answer.
    expect(seq.pages.covers(checkpoint), "checkpoint still covered after decode");
    rewind(store, seq, checkpoint);
    forward(store, seq, checkpoint, 200, 200, size.forward_tokens, {});
    seq.committed += 200;
    // Dense reuse far behind the frontier must be refused by prepare, never refilled.
    bool threw = false;
    try {
        rewind(store, seq, checkpoint - window - 3 * kPage);
        forward(store, seq, seq.committed, 1, 1, size.forward_tokens, {});
    } catch (const std::logic_error&) { threw = true; }
    expect(threw, "prepare refuses a reuse point whose window was released");
    seq.pages.release_all(store);
    expect(store.in_use() == 0, "release_all gives every page back");
    expect(store.peak <= size.lane_pages, "single lane stays inside the planned class");
}

void test_images(std::uint32_t window) {
    const std::uint32_t chunk = 1024;
    const auto size = sinfer::plan_kv_window_class(window, 2, chunk, 2);
    Store lanes(size.lane_pages);
    Store images(size.image_pages);
    Seq a{KvWindowPages(window)};
    const std::uint32_t shared = 64 * 70; // page-aligned shared-prefix boundary
    while (a.committed < shared) {
        const std::uint32_t n = std::min(chunk, shared - a.committed);
        forward(lanes, a, a.committed, n, n, size.forward_tokens, {});
        a.committed += n;
        (void)a.pages.settle(a.committed, {}, lanes);
    }
    const auto image = capture_image(lanes, a, images, shared);
    expect(!image.empty(), "shared-prefix window image captured");
    const std::vector<std::int64_t> truth = a.truth;
    // The capturing lane runs on and drops the prefix's window.
    for (int i = 0; i < 40; ++i) {
        forward(lanes, a, a.committed, 64, 64, size.forward_tokens, {});
        a.committed += 64;
        (void)a.pages.settle(a.committed, {}, lanes);
    }
    expect(!a.pages.covers(shared), "the lane itself no longer holds the prefix window");
    // A fork copies the image back and prefills its own suffix.
    Seq b{KvWindowPages(window)};
    restore_image(lanes, b, images, image, truth, shared);
    expect(b.pages.covers(shared), "fork covers the shared boundary");
    b.epoch = 9;
    forward(lanes, b, shared, 777, 777, size.forward_tokens, {});
    b.committed += 777;
    (void)b.pages.settle(b.committed, {}, lanes);
    // An archive restore at the frontier is the same operation (map_restored + upload).
    Seq c{KvWindowPages(window)};
    restore_image(lanes, c, images, image, truth, shared);
    c.epoch = 11;
    forward(lanes, c, shared, 1, 1, size.forward_tokens, {});
    for (const auto& slot : image) { images.give_back_window_page(slot.physical); }
    a.pages.release_all(lanes);
    b.pages.release_all(lanes);
    c.pages.release_all(lanes);
    expect(lanes.in_use() == 0 && images.in_use() == 0, "images and forks give every page back");
}

/// Many lanes in engine-like rounds: one round's prefill budget is shared by the prefilling lanes
/// (mixed rounds), the others decode or verify; lanes finish, stay retained, and take new
/// requests as appends at their frontier, checkpoint restores or full resets.
void test_many_lanes(std::uint32_t window, std::uint32_t rows, std::uint32_t chunk, std::uint32_t seed,
                     bool image_checkpoints = false, std::uint32_t elastic_granule = 0) {
    // elastic_granule != 0: M3-C -- the store hands out the lowest free page, and at every round boundary the
    // compaction planner empties the highest granules into holes below; reads must stay exact across moves.
    // image_checkpoints: M3 lever A -- a checkpoint's window is an image copy made when the prefill passes it, the
    // lane pins nothing, and the class is sized without pins; the image region is small and may refuse a capture.
    const std::uint32_t image_windows = image_checkpoints ? rows / 2 + 2 : 0;
    const auto size = sinfer::plan_kv_window_class(window, rows, chunk, image_windows, 1, !image_checkpoints);
    Store store(size.lane_pages);
    store.sorted = elastic_granule != 0;
    Store images(std::max(size.image_pages, 1U));
    std::uint32_t captured = 0, refused = 0, restored = 0;
    std::uint32_t moved = 0, worst_waste = 0, over_bound_rounds = 0, boundaries = 0;
    std::mt19937 rng(seed);
    struct Lane {
        Seq seq;
        std::uint32_t prompt = 0, remaining = 0;
        std::uint32_t checkpoint = 0; // pinned frontier, 0 = none
        bool active = false;
        std::vector<KvWindowSlot> image; // image_checkpoints: the window behind `checkpoint`, empty = none
        std::vector<std::int64_t> image_truth;
    };
    std::vector<Lane> lanes(rows);
    for (auto& lane : lanes) { lane.seq = Seq{KvWindowPages(window)}; }
    const auto pins_of = [&](const Lane& lane) {
        return lane.checkpoint && !image_checkpoints ? std::vector<std::uint32_t>{lane.checkpoint}
                                                     : std::vector<std::uint32_t>{};
    };
    const auto drop_image = [&](Lane& lane) {
        for (const auto& slot : lane.image) { images.give_back_window_page(slot.physical); }
        lane.image.clear();
    };
    const auto start = [&](Lane& lane) {
        const std::uint32_t choice = rng() % 3;
        if (choice == 0 || lane.seq.committed == 0) {
            lane.seq.pages.release_all(store);
            lane.seq.truth.clear();
            lane.seq.committed = 0;
        } else if (choice == 1 && lane.checkpoint && !image_checkpoints && lane.seq.pages.covers(lane.checkpoint)) {
            rewind(store, lane.seq, lane.checkpoint);
        } else if (choice == 1 && lane.checkpoint && image_checkpoints && !lane.image.empty()) {
            // Restore at the checkpoint from its image: the lane's own window pages go, the image's come back.
            lane.seq.pages.release_all(store);
            restore_image(store, lane.seq, images, lane.image, lane.image_truth, lane.checkpoint);
            ++restored;
        } else {
            rewind(store, lane.seq, lane.seq.committed); // append at the frontier
        }
        ++lane.seq.epoch;
        (void)lane.seq.pages.prepare(lane.seq.committed, lane.seq.committed, size.forward_tokens, pins_of(lane), store);
        lane.prompt    = lane.seq.committed + 100 + rng() % (4 * window + 3000);
        drop_image(lane);
        lane.checkpoint = (rng() % 2) ? lane.prompt - rng() % 50 : 0;
        lane.remaining = 50 + rng() % 400;
        lane.active    = true;
    };
    for (int round = 0; round < 2500 && g_failures == 0; ++round) {
        for (auto& lane : lanes) { if (!lane.active && rng() % 4 == 0) { start(lane); } }
        std::uint32_t budget = chunk;
        for (auto& lane : lanes) {
            if (!lane.active) { continue; }
            Seq& seq = lane.seq;
            if (seq.committed < lane.prompt) {
                if (budget == 0) { continue; }
                std::uint32_t n = std::min(budget, lane.prompt - seq.committed);
                if (lane.checkpoint > seq.committed) { n = std::min(n, lane.checkpoint - seq.committed); }
                budget -= n;
                const std::uint32_t reach = (rng() % 2) ? (n + 127) / 128 * 128 : n;
                forward(store, seq, seq.committed, n, reach, size.forward_tokens, pins_of(lane));
                seq.committed += n;
                (void)seq.pages.settle(seq.committed, pins_of(lane), store);
                if (image_checkpoints && lane.checkpoint && seq.committed == lane.checkpoint) {
                    // The prefill stops at the checkpoint: copy its window if the image region has room.
                    const std::uint32_t need = KvWindowPages::pages_for(lane.checkpoint) -
                                               KvWindowPages::keep_first_page(lane.checkpoint, window);
                    if (images.free.size() >= need) {
                        lane.image       = capture_image(store, seq, images, lane.checkpoint);
                        lane.image_truth = seq.truth;
                        ++captured;
                    } else {
                        ++refused; // the checkpoint keeps its state but not its window: no restore there
                    }
                }
            } else {
                const std::uint32_t width = 1 + rng() % 6;
                forward(store, seq, seq.committed, width, width, size.forward_tokens, pins_of(lane));
                const std::uint32_t accepted = std::min<std::uint32_t>(1 + rng() % width, lane.remaining);
                rewind(store, seq, seq.committed + accepted);
                lane.remaining -= accepted;
                if (lane.remaining == 0) { lane.active = false; } // retained
            }
            // Between forwards every lane holds at most its charge (window + pin + reach).
            if (!image_checkpoints && seq.pages.page_count() > sinfer::kv_window_lane_charge(window)) {
                expect(false, "a lane holds more than its window charge between forwards");
            }
        }
        if (elastic_granule != 0) {
            // Round boundary: compaction (bitwise page copies; the owner's slot follows).
            std::vector<std::int32_t> used, free_sorted(store.free.rbegin(), store.free.rend());
            for (std::uint32_t p = 0; p < store.capacity; ++p) { if (store.owner[p] != -1) { used.push_back(std::int32_t(p)); } }
            const auto moves = sinfer::plan_kv_window_compaction(std::span<const std::int32_t>(used),
                std::span<const std::int32_t>(free_sorted), elastic_granule, 64, [](std::int32_t) { return true; });
            for (const auto& move : moves) {
                bool found = false;
                for (auto& lane : lanes) {
                    for (const auto& slot : lane.seq.pages.slots()) {
                        if (slot.physical != move.from) { continue; }
                        // Take exactly `move.to` out of the free list, copy, relocate, give the old page back.
                        const auto it = std::find(store.free.begin(), store.free.end(), move.to);
                        if (it == store.free.end()) { expect(false, "compaction target is not free"); break; }
                        store.free.erase(it);
                        store.owner[move.to] = 1;
                        std::copy_n(store.cells.begin() + std::ptrdiff_t(move.from) * kPage, kPage,
                                    store.cells.begin() + std::ptrdiff_t(move.to) * kPage);
                        const std::uint32_t logical = slot.logical;
                        const std::int32_t old = lane.seq.pages.relocate(logical, move.to);
                        expect(old == move.from, "relocate returns the old page");
                        store.give_back_window_page(old);
                        found = true;
                        break;
                    }
                    if (found) { break; }
                }
                expect(found, "compaction moved a page no lane holds");
                ++moved;
            }
            const std::uint32_t in_use = store.in_use();
            const std::uint32_t granules = store.granules_in_use(elastic_granule);
            const std::uint32_t waste = granules * elastic_granule - in_use;
            worst_waste = std::max(worst_waste, waste);
            ++boundaries;
            if (granules > (in_use + elastic_granule - 1) / elastic_granule + 1) { ++over_bound_rounds; }
        }
    }
    for (auto& lane : lanes) { lane.seq.pages.release_all(store); drop_image(lane); }
    expect(store.in_use() == 0 && images.in_use() == 0, "many lanes give every page back");
    std::printf("  window %u rows %u chunk %u%s%s: class %u lane pages, peak %u", window, rows, chunk,
                image_checkpoints ? " (checkpoints as images)" : "", elastic_granule ? " (elastic)" : "",
                size.lane_pages, store.peak);
    if (elastic_granule != 0) {
        std::printf(", granule %u: %u pages moved, worst waste after compaction %u pages, %u of %u boundaries over "
                    "ceil(in use / granule) + 1 granules", elastic_granule, moved, worst_waste, over_bound_rounds,
                    boundaries);
        // Compaction keeps the mapped granules within one of the minimum at nearly every boundary
        // (a granule whose pages have no lower holes yet waits for the next round).
        expect(over_bound_rounds * 20 <= boundaries, "elastic class: compaction keeps granules near the minimum");
    }
    if (image_checkpoints) {
        std::printf(", images %u pages: %u captured, %u refused, %u restores", size.image_pages, captured, refused,
                    restored);
        expect(restored > 0, "checkpoint restores from images happened");
    }
    std::printf("\n");
}

/// M3-C: relocation keeps reads exact; the planner empties whole top granules into holes below,
/// never into the granule it empties, never past its budget, never moving an immovable page.
void test_compaction_planner() {
    const std::uint32_t g = 4;
    // Granule 0: pages 0, 2 used (holes 1, 3); granule 1: 5 used; granule 2: 8, 9 used.
    const std::vector<std::int32_t> used{0, 2, 5, 8, 9};
    const std::vector<std::int32_t> free{1, 3, 4, 6, 7, 10, 11, 12, 13, 14, 15};
    auto moves = sinfer::plan_kv_window_compaction(std::span<const std::int32_t>(used), std::span<const std::int32_t>(free),
                                                   g, 64, [](std::int32_t) { return true; });
    // Granule 2 (2 pages) -> holes 1, 3; then granule 1 (page 5) has holes 4? no: 4, 6, 7 are in granule 1 itself,
    // and granule 0 is full now -> stop.
    expect(moves.size() == 2 && moves[0].from == 8 && moves[0].to == 1 && moves[1].from == 9 && moves[1].to == 3,
           "planner empties the top granule into the lowest holes");
    moves = sinfer::plan_kv_window_compaction(std::span<const std::int32_t>(used), std::span<const std::int32_t>(free),
                                              g, 64, [](std::int32_t page) { return page != 9; });
    expect(moves.empty(), "planner leaves a granule with an immovable page");
    moves = sinfer::plan_kv_window_compaction(std::span<const std::int32_t>(used), std::span<const std::int32_t>(free),
                                              g, 1, [](std::int32_t) { return true; });
    expect(moves.empty(), "planner never splits a granule over its budget");
    // Relocation: the slot follows, reads stay exact.
    Store store(8);
    Seq seq{KvWindowPages(256)};
    forward(store, seq, 0, 200, 200, 1024, {});
    seq.committed = 200;
    const std::int32_t from = seq.pages.physical(1);
    const std::int32_t to   = store.take_window_page();
    std::copy_n(store.cells.begin() + std::ptrdiff_t(from) * kPage, kPage, store.cells.begin() + std::ptrdiff_t(to) * kPage);
    expect(seq.pages.relocate(1, to) == from && seq.pages.physical(1) == to, "relocate moves the slot");
    store.give_back_window_page(from);
    forward(store, seq, 200, 10, 10, 1024, {});
    bool threw = false;
    try { (void)seq.pages.relocate(40, to); } catch (const std::logic_error&) { threw = true; }
    expect(threw, "relocating a page that is not live throws");
    // Sizing: Inkling (window 1,024, 32-page granules): charge 35 window pages per lane.
    expect(sinfer::kv_window_lane_charge(1024) == 35, "lane charge = 2 x (window pages + 1) + 1");
    const auto size = sinfer::plan_kv_window_class(1024, 8, 4096, 0);
    expect(sinfer::kv_window_floor_pages(size, 8, 1024, 32) == 96 + 32, "floor = one round's pages in granules + 1");
    expect(sinfer::kv_window_main_equivalent(35, 5u << 20, 1u << 20) == 175, "Inkling: a lane charge is 175 main pages");
}

} // namespace

int main() try {
    test_inactive();
    for (const std::uint32_t window : {512U, 1024U, 4096U}) {
        test_single_lane(window, 1024);
        test_single_lane(window, 8192);
        test_images(window);
    }
    test_many_lanes(1024, 8, 2048, 1);
    test_many_lanes(1024, 16, 8192, 2);
    test_many_lanes(512, 4, 1024, 3);
    test_many_lanes(1000, 6, 4096, 4); // a window that is no multiple of the page
    // M3 lever A: checkpoint windows as image copies, the class sized without pins.
    test_many_lanes(1024, 8, 2048, 5, true);
    test_many_lanes(1024, 16, 8192, 6, true);
    test_many_lanes(512, 4, 1024, 7, true);
    // M3-C: the elastic class (Inkling's 32-page granules), lowest-first takes, compaction at round boundaries.
    test_compaction_planner();
    test_many_lanes(1024, 8, 4096, 8, false, 32);
    test_many_lanes(1024, 8, 8192, 9, false, 32);
    test_many_lanes(1024, 2, 4096, 10, false, 32);
    test_many_lanes(512, 4, 1024, 11, false, 16);
    if (g_failures != 0) {
        std::fprintf(stderr, "test_kv_window_pages: %d failures\n", g_failures);
        return 1;
    }
    std::printf("test_kv_window_pages: PASS\n");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "test_kv_window_pages: FAIL (exception: %s)\n", error.what());
    return 1;
}
