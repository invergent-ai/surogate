// The request entitlement must cover every KV page a prefill graph chunk can
// write, from ANY cursor. A captured body writes its whole 128-rounded bucket,
// pad columns included, and chunk starts are not aligned: prefix reuse and
// rewrite-checkpoint restores begin at arbitrary frontiers. Reserving to the
// rounded prompt alone assumed 128-aligned starts and let a multi-turn follow-up
// with a short output budget map past its entitlement, which is engine-fatal.
//
// This pins the arithmetic that keeps the mapping inside the entitlement:
// for every capacity, chunk, prompt and cursor, the window a chunk maps is
// entitled, and the lemma behind it (cursor + roundup128(prompt - cursor)
// <= prompt + 127) holds exactly.
#include "core/paged_kv_cache.h"
#include "family/impl/runtime/prefill_graph.h"

#include <algorithm>
#include <cstdint>
#include <iostream>

namespace {

using sinfer::family::detail::PrefillGraphFamily;

// The pool's page arithmetic (paged_kv_cache.cpp), restated from the page size
// it is derived from; the plan keeps its own copy in request_plan_impl.h.
std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    if (tokens == 0) { return 0; }
    return 1U + (tokens - 1U) / static_cast<std::uint32_t>(sinfer::kPagedKVPageSize);
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;

    // The lemma, exhaustively, for every prompt a 4096 context can hold.
    for (std::uint32_t prompt = 1; prompt <= 4096; ++prompt) {
        const std::uint32_t reach = PrefillGraphFamily::graph_prefill_reach(prompt);
        std::uint32_t worst       = 0;
        for (std::uint32_t cursor = 0; cursor < prompt; ++cursor) {
            const auto bucket = static_cast<std::uint32_t>(
                PrefillGraphFamily::chunk_bucket_for(prompt - cursor));
            worst = std::max(worst, cursor + bucket);
        }
        if (worst > reach) {
            std::cerr << "prompt " << prompt << ": a chunk reaches " << worst
                      << " past graph_prefill_reach " << reach << '\n';
            ++failures;
            break;
        }
        // And the bound is tight somewhere, or the reservation is over-sized.
        if (prompt % 128 == 1 && worst != reach) {
            std::cerr << "prompt " << prompt << ": reach " << reach << " is loose, worst is "
                      << worst << '\n';
            ++failures;
            break;
        }
    }

    // The entitlement as request_plan sizes it, against the window a chunk maps,
    // over unaligned capacities too (an explicit --max-model-len need not be a
    // multiple of 128). A window past capacity is not mapped -- the graph layer
    // refuses that chunk -- so it is skipped here as it is there.
    const std::uint32_t capacities[] = {64U, 100U, 127U, 128U, 129U, 1930U, 2048U, 4032U, 4096U};
    const std::uint32_t chunks[]     = {128U, 512U, 2048U};
    for (const std::uint32_t capacity : capacities) {
        for (const std::uint32_t chunk : chunks) {
            const std::uint32_t effective_chunk = std::min(chunk, capacity);
            for (std::uint32_t prompt = 1; prompt <= capacity; ++prompt) {
                // Output budget 1: the smallest extent, so the graph reach term decides.
                const std::uint32_t reserved = std::min(
                    capacity, std::max(prompt, PrefillGraphFamily::graph_prefill_reach(prompt)));
                const std::uint32_t entitled = pages_for_tokens(reserved);
                for (std::uint32_t cursor = 0; cursor < prompt; ++cursor) {
                    const std::uint32_t nominal = std::min(effective_chunk, prompt - cursor);
                    const std::uint32_t window =
                        cursor + static_cast<std::uint32_t>(
                                     PrefillGraphFamily::chunk_bucket_for(nominal));
                    if (window > capacity) { continue; }
                    if (pages_for_tokens(window) > entitled) {
                        std::cerr << "capacity " << capacity << " chunk " << chunk << " prompt "
                                  << prompt << " cursor " << cursor << ": maps "
                                  << pages_for_tokens(window) << " pages, entitled "
                                  << entitled << '\n';
                        ++failures;
                        goto done;
                    }
                }
            }
        }
    }
done:
    // The two recipes that took the engine down before the reach was sized this
    // way, stated as page counts: base 560 / prompt 600 / out 16 needed 11 pages
    // and was entitled 10; base 700 / prompt 1000 / out 16 needed 17, entitled 16.
    {
        const auto entitled = [](std::uint32_t prompt, std::uint32_t out) {
            return pages_for_tokens(std::min(
                4096U, std::max(prompt + out - 1U, PrefillGraphFamily::graph_prefill_reach(prompt))));
        };
        const auto needed = [](std::uint32_t base, std::uint32_t prompt) {
            return pages_for_tokens(
                base + static_cast<std::uint32_t>(PrefillGraphFamily::chunk_bucket_for(prompt - base)));
        };
        failures += check(needed(560, 600) == 11 && entitled(600, 16) >= 11,
                          "base 560 / prompt 600 / out 16 is not entitled to its window");
        failures += check(needed(700, 1000) == 17 && entitled(1000, 16) >= 17,
                          "base 700 / prompt 1000 / out 16 is not entitled to its window");
    }

    std::cout << (failures == 0 ? "PASS" : "FAIL") << " prefill graph reach\n";
    return failures == 0 ? 0 : 1;
}
