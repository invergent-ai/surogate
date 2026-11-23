#pragma once

// Window-bounded KV pages (ENGINE-INKLING lane f1; design: rl/engine-inkling-2026-11-21/lanes/f1/
// DESIGN-window-kv.md). Host-only bookkeeping of one sequence's pages in a paged pool's WINDOW
// class -- the planes of layers that attend through a sliding window. Such a plane only ever
// reads the last `window` keys before a query, so the pages behind the window go back to the
// class's free list and their block-table entries point at a zeroed sentinel page.
//
// Nothing here touches a device: the pool (core/paged_kv_cache.*) turns each returned change into
// block-table writes and copies. Keeping the decisions host-only lets the rules be unit tested
// without a card (testing/serve/test_kv_window_pages.cpp).

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sinfer {

/// Tokens per page; the paged pool asserts that it equals kPagedKVPageSize.
inline constexpr std::uint32_t kKvWindowPageTokens = 64;

struct KvWindowSlot {
    std::uint32_t logical = 0; ///< page index in the sequence (position / 64)
    std::int32_t physical = -1;
};

/// Where a sequence's window pages come from and go back to: the pool's lane region, a fake in
/// the unit test. A take that finds nothing is a sizing bug (the class is sized by construction,
/// DESIGN-window-kv.md section 5), so it throws rather than reporting.
class KvWindowPageSource {
public:
    virtual ~KvWindowPageSource() = default;
    [[nodiscard]] virtual std::int32_t take_window_page() = 0;
    virtual void give_back_window_page(std::int32_t physical) noexcept = 0;
};

/// What one operation changed: the pool publishes `released` entries as the sentinel and
/// `mapped` entries as their new physical page (when the allocation has a table row).
struct KvWindowChange {
    std::vector<std::uint32_t> released;
    std::vector<KvWindowSlot> mapped;
    [[nodiscard]] bool empty() const noexcept { return released.empty() && mapped.empty(); }
};

class KvWindowPages {
public:
    KvWindowPages() noexcept = default;
    explicit KvWindowPages(std::uint32_t window_tokens) noexcept : window_(window_tokens) {}

    [[nodiscard]] std::uint32_t window_tokens() const noexcept { return window_; }
    [[nodiscard]] bool active() const noexcept { return window_ != 0; }
    [[nodiscard]] std::span<const KvWindowSlot> slots() const noexcept { return slots_; }
    [[nodiscard]] std::uint32_t page_count() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }

    [[nodiscard]] static constexpr std::uint32_t pages_for(std::uint64_t tokens) noexcept {
        return static_cast<std::uint32_t>((tokens + kKvWindowPageTokens - 1) / kKvWindowPageTokens);
    }
    /// First page a sequence with `tokens` committed keeps: one token of margin beyond the
    /// window, so the frontier also serves a reuse at `tokens - 1` (dense "replay the last token").
    [[nodiscard]] static constexpr std::uint32_t keep_first_page(std::uint32_t tokens,
                                                                 std::uint32_t window) noexcept {
        return tokens > window ? (tokens - window) / kKvWindowPageTokens : 0U;
    }
    /// First page a query at position `first` can read: keys j with first - j < window.
    [[nodiscard]] static constexpr std::uint32_t needed_first_page(std::uint32_t first,
                                                                   std::uint32_t window) noexcept {
        return first + 1 > window ? (first + 1 - window) / kKvWindowPageTokens : 0U;
    }
    /// The page range [first, last) a pin at frontier `pin` keeps.
    [[nodiscard]] static constexpr std::pair<std::uint32_t, std::uint32_t>
    pin_pages(std::uint32_t pin, std::uint32_t window) noexcept {
        return {keep_first_page(pin, window), pages_for(pin)};
    }

    [[nodiscard]] bool live(std::uint32_t logical) const noexcept {
        const auto it = find(logical);
        return it != slots_.end() && it->logical == logical;
    }
    [[nodiscard]] std::int32_t physical(std::uint32_t logical) const noexcept {
        const auto it = find(logical);
        return it != slots_.end() && it->logical == logical ? it->physical : -1;
    }

    /// Can a forward whose first query is `tokens` run on these pages: every page holding a key
    /// that query reads is live. The test a prefix-reuse plan must pass.
    [[nodiscard]] bool covers(std::uint32_t tokens) const noexcept {
        if (window_ == 0 || tokens == 0) { return true; }
        for (std::uint32_t page = needed_first_page(tokens, window_); page < pages_for(tokens); ++page) {
            if (!live(page)) { return false; }
        }
        return true;
    }

    /// Before a forward. `first` = the sequence's committed KV length (the forward's first query),
    /// `end` = where its writes end, `forward_cap` = the most tokens any one forward writes (only
    /// that much is mapped ahead: admission asks for the whole prompt, each chunk maps its own).
    /// Throws if a page holding committed keys the forward reads is missing: a plan reused a
    /// point whose window is gone, which must never be refilled silently. Releases only pages
    /// below the keep range (pages past the frontier may belong to a round still being staged;
    /// the commit trim gives them back), then maps the missing pages that hold no committed token.
    [[nodiscard]] KvWindowChange prepare(std::uint32_t first, std::uint32_t end, std::uint32_t forward_cap,
                                         std::span<const std::uint32_t> pins, KvWindowPageSource& source) {
        KvWindowChange change;
        if (window_ == 0) { return change; }
        const std::uint32_t committed_last = pages_for(first);
        for (std::uint32_t page = needed_first_page(first, window_); page < committed_last; ++page) {
            if (!live(page)) {
                throw std::logic_error("window KV page " + std::to_string(page) +
                                       " holding committed keys is missing for a forward at " +
                                       std::to_string(first) + " (window " + std::to_string(window_) +
                                       "): the plan reused a point whose window was released");
            }
        }
        const std::uint32_t keep_first = keep_first_page(first, window_);
        const std::uint64_t reach = std::min<std::uint64_t>(std::max(end, first),
                                                            std::uint64_t{first} + forward_cap);
        const std::uint32_t map_last = std::max(pages_for(reach), committed_last);
        release_if(change, source, [&](std::uint32_t logical) {
            return logical < keep_first && !pinned(logical, pins);
        });
        for (std::uint32_t page = std::max(keep_first, committed_last); page < map_last; ++page) {
            if (!live(page)) { map(change, source, page); }
        }
        return change;
    }

    /// After a forward's tokens are committed (a staged prefill chunk; a commit trim): keep the
    /// window behind `committed` and the pins, give back everything else.
    [[nodiscard]] KvWindowChange settle(std::uint32_t committed, std::span<const std::uint32_t> pins,
                                        KvWindowPageSource& source) {
        KvWindowChange change;
        if (window_ == 0) { return change; }
        const std::uint32_t keep_first = keep_first_page(committed, window_);
        const std::uint32_t keep_last  = pages_for(committed);
        release_if(change, source, [&](std::uint32_t logical) {
            return (logical < keep_first || logical >= keep_last) && !pinned(logical, pins);
        });
        return change;
    }

    /// Trim to `tokens`: pages that hold no token below it go back (append reuse, checkpoint
    /// restore, rejected verify columns).
    [[nodiscard]] KvWindowChange rewind(std::uint32_t tokens, KvWindowPageSource& source) {
        KvWindowChange change;
        if (window_ == 0) { return change; }
        const std::uint32_t keep_last = pages_for(tokens);
        release_if(change, source, [&](std::uint32_t logical) { return logical >= keep_last; });
        return change;
    }

    /// Maps fresh pages for `logical` (ascending, none live), whose bytes the caller then fills:
    /// an archive restore's upload, a fork's copy of a shared prefix's window image.
    [[nodiscard]] KvWindowChange map_restored(std::span<const std::uint32_t> logical,
                                              KvWindowPageSource& source) {
        KvWindowChange change;
        if (window_ == 0) { return change; }
        for (std::size_t i = 0; i < logical.size(); ++i) {
            if ((i > 0 && logical[i] <= logical[i - 1]) || live(logical[i])) {
                throw std::invalid_argument("window KV restore pages must be ascending and unmapped");
            }
        }
        for (const std::uint32_t page : logical) { map(change, source, page); }
        return change;
    }

    void release_all(KvWindowPageSource& source) noexcept {
        for (const KvWindowSlot& slot : slots_) { source.give_back_window_page(slot.physical); }
        slots_.clear();
    }

    /// M3-C compaction: the live page `logical` now lives at `physical` (the caller copied its
    /// bytes, bitwise, and owns the page it gave). Returns the page it lived at.
    std::int32_t relocate(std::uint32_t logical, std::int32_t physical) {
        const auto it = std::lower_bound(slots_.begin(), slots_.end(), logical,
                                         [](const KvWindowSlot& slot, std::uint32_t value) {
                                             return slot.logical < value;
                                         });
        if (it == slots_.end() || it->logical != logical) {
            throw std::logic_error("window KV relocation of a page that is not live");
        }
        const std::int32_t old = it->physical;
        it->physical           = physical;
        return old;
    }

private:
    [[nodiscard]] std::vector<KvWindowSlot>::const_iterator find(std::uint32_t logical) const noexcept {
        return std::lower_bound(slots_.begin(), slots_.end(), logical,
                                [](const KvWindowSlot& slot, std::uint32_t value) {
                                    return slot.logical < value;
                                });
    }
    [[nodiscard]] bool pinned(std::uint32_t logical, std::span<const std::uint32_t> pins) const noexcept {
        for (const std::uint32_t pin : pins) {
            const auto [first, last] = pin_pages(pin, window_);
            if (logical >= first && logical < last) { return true; }
        }
        return false;
    }
    template <class Predicate>
    void release_if(KvWindowChange& change, KvWindowPageSource& source, Predicate&& release) {
        std::vector<KvWindowSlot> kept;
        kept.reserve(slots_.size());
        for (const KvWindowSlot& slot : slots_) {
            if (release(slot.logical)) {
                source.give_back_window_page(slot.physical);
                change.released.push_back(slot.logical);
            } else {
                kept.push_back(slot);
            }
        }
        slots_ = std::move(kept);
    }
    void map(KvWindowChange& change, KvWindowPageSource& source, std::uint32_t logical) {
        const std::int32_t physical = source.take_window_page();
        const auto it = std::lower_bound(slots_.begin(), slots_.end(), logical,
                                         [](const KvWindowSlot& slot, std::uint32_t value) {
                                             return slot.logical < value;
                                         });
        slots_.insert(it, KvWindowSlot{logical, physical});
        change.mapped.push_back(KvWindowSlot{logical, physical});
    }

    std::uint32_t window_ = 0;
    std::vector<KvWindowSlot> slots_; ///< sorted by logical page, unique
};

/// Pages of a pool's window class: steady per lane (window + pin + decode reach), one round's new
/// prefill pages, and the image region (DESIGN-window-kv.md section 5). Pure arithmetic, shared by
/// the planner and the unit test.
struct KvWindowClassSize {
    std::uint32_t lane_pages  = 0;
    std::uint32_t image_pages = 0;
    std::uint32_t forward_tokens = 0;
    // M3-C (set by the planner, not by plan_kv_window_class): demand-mapped class whose round
    // transient (`transient_pages`, rounded to granules + one granule by the pool) is the floor,
    // and whose holders are charged `lane_charge` window pages each in the main ledger.
    bool elastic                  = false;
    std::uint32_t lane_charge     = 0;
    std::uint32_t transient_pages = 0;
};

[[nodiscard]] inline KvWindowClassSize plan_kv_window_class(std::uint32_t window_tokens, std::uint32_t rows,
                                                            std::uint32_t prefill_chunk,
                                                            std::uint32_t image_windows,
                                                            std::uint32_t rounds_in_flight = 1,
                                                            bool checkpoint_pins = true) noexcept {
    KvWindowClassSize out;
    if (window_tokens == 0 || rows == 0) { return out; }
    const std::uint32_t window_pages = KvWindowPages::pages_for(window_tokens) + 1U; // unaligned
    // A forward writes at most one prefill chunk rounded to its 128-token graph bucket; decode
    // bursts and verify rounds are far shorter. 64 more tokens cover an unaligned start.
    out.forward_tokens = prefill_chunk + 128U + 64U;
    const std::uint32_t round_pages =
        KvWindowPages::pages_for(std::uint64_t{prefill_chunk} + 128U) + 2U * std::min(rows, 64U);
    // A pipeline stage can hold one round per micro-batch group in flight, each with its own
    // prefill budget mapped until the round is consumed.
    // checkpoint_pins = false is M3 lever A (DESIGN-M3-kv-lever.md): a rewrite checkpoint's window is an image copy
    // in the image region, not pinned lane pages, so a lane's steady holding is its window + the decode reach.
    out.lane_pages  = rows * ((checkpoint_pins ? 2U : 1U) * window_pages + 1U) +
                      std::max(rounds_in_flight, 1U) * round_pages;
    out.image_pages = image_windows * window_pages;
    return out;
}

// ---- M3-C: the elastic window class (lanes/f1/DESIGN-M3-kv-lever.md section C) ---------------
// The class's planes are demand-mapped in granules of `granule_pages` pages (2 MiB per plane:
// Inkling's 80 window planes of 64 KiB per page give 32 pages, 160 MiB, per granule). The pool's
// static bound above stays the VIRTUAL span; physical memory is (a) a floor the engine always
// holds -- one round's prefill pages and a granule of compaction slack -- and (b) per holder
// charges booked in the main pool's page ledger, so admission, the watermark and retained-lane
// eviction see window pages as ordinary KV.

/// Window pages one sequence may hold between forwards: its window, the rewrite checkpoint's
/// pinned window, and the decode reach. The bound test_kv_window_pages checks for every lane.
[[nodiscard]] inline std::uint32_t kv_window_lane_charge(std::uint32_t window_tokens) noexcept {
    if (window_tokens == 0) { return 0; }
    return 2U * (KvWindowPages::pages_for(window_tokens) + 1U) + 1U;
}

/// The guaranteed window pages of an elastic class: one round's prefill pages per round in
/// flight, rounded to whole granules, and one granule of slack (compaction leaves at most one
/// granule's worth of holes; the sentinel shares granule 0 with the lowest lane pages).
[[nodiscard]] inline std::uint32_t kv_window_floor_pages(const KvWindowClassSize& size, std::uint32_t rows,
                                                         std::uint32_t window_tokens,
                                                         std::uint32_t granule_pages) noexcept {
    if (window_tokens == 0 || rows == 0 || granule_pages == 0) { return 0; }
    const std::uint32_t steady = rows * kv_window_lane_charge(window_tokens);
    const std::uint32_t transient = size.lane_pages > steady ? size.lane_pages - steady : 0U;
    const std::uint32_t rounded = (transient + granule_pages - 1U) / granule_pages * granule_pages;
    return rounded + granule_pages;
}

/// Main-pool pages that stand for `window_pages` window pages in the shared ledger (bytes, rounded up).
[[nodiscard]] inline std::uint32_t kv_window_main_equivalent(std::uint64_t window_pages, std::size_t window_page_bytes,
                                                             std::size_t main_page_bytes) noexcept {
    if (window_pages == 0 || main_page_bytes == 0) { return 0; }
    const std::uint64_t bytes = window_pages * window_page_bytes;
    return static_cast<std::uint32_t>((bytes + main_page_bytes - 1U) / main_page_bytes);
}

struct KvWindowMove {
    std::int32_t from = -1;
    std::int32_t to   = -1;
};

/// Compaction plan for a demand-mapped class: empty the highest granules that hold pages into
/// free pages of LOWER granules that already hold pages (so no granule is mapped for it), as
/// long as a whole granule empties and every page in it may move. `used` and `free_pages` are
/// sorted ascending page ids inside the managed range; `movable(page)` says whether a used page
/// may be copied now (an image, the sentinel, a page a host copy still reads may not). Pure:
/// the pool copies the bytes and rewrites the owner's slot and block-table entry.
template <class Movable>
[[nodiscard]] std::vector<KvWindowMove> plan_kv_window_compaction(std::span<const std::int32_t> used,
                                                                  std::span<const std::int32_t> free_pages,
                                                                  std::uint32_t granule_pages,
                                                                  std::uint32_t max_moves, Movable&& movable) {
    std::vector<KvWindowMove> moves;
    if (granule_pages == 0 || used.empty() || free_pages.empty() || max_moves == 0) { return moves; }
    // Pages per granule in use.
    std::vector<std::uint32_t> count;
    for (const std::int32_t page : used) {
        const auto g = static_cast<std::size_t>(page) / granule_pages;
        if (g >= count.size()) { count.resize(g + 1, 0U); }
        ++count[g];
    }
    // Holes: free pages in granules that hold pages, lowest first.
    std::vector<std::int32_t> holes;
    for (const std::int32_t page : free_pages) {
        const auto g = static_cast<std::size_t>(page) / granule_pages;
        if (g < count.size() && count[g] != 0) { holes.push_back(page); }
    }
    std::size_t next_hole = 0;
    std::size_t top       = used.size(); // used[top - 1] is the highest page not yet moved
    while (top > 0) {
        const auto g = static_cast<std::size_t>(used[top - 1]) / granule_pages;
        std::size_t first = top;
        while (first > 0 && static_cast<std::size_t>(used[first - 1]) / granule_pages == g) { --first; }
        const std::size_t pages = top - first;
        // A granule that already received moved pages stays (holes are consumed lowest first).
        if (next_hole > 0 && static_cast<std::size_t>(holes[next_hole - 1]) / granule_pages >= g) { break; }
        // Holes strictly below this granule that are still free.
        std::size_t available = 0;
        for (std::size_t h = next_hole; h < holes.size() && static_cast<std::size_t>(holes[h]) / granule_pages < g; ++h) {
            ++available;
        }
        if (pages > available || moves.size() + pages > max_moves) { break; }
        bool all_movable = true;
        for (std::size_t i = first; i < top && all_movable; ++i) { all_movable = movable(used[i]); }
        if (!all_movable) { break; }
        for (std::size_t i = first; i < top; ++i) {
            moves.push_back(KvWindowMove{used[i], holes[next_hole++]});
        }
        count[g] = 0;
        top      = first;
    }
    return moves;
}

} // namespace sinfer
