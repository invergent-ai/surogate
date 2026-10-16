// Rollout Routing Replay row bookkeeping (family/impl/runtime/routing_rows.h): the column ->
// position spans every round kind stores. Simulated generations check that ordinary rounds,
// chained bursts and speculative verify rounds with random acceptance (alone and behind a mixed
// round's prompt segments) all build the same rows, that a rejected draft's column never reaches
// a request, and that the last generated token has no row.

#include "family/impl/runtime/routing_rows.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>

namespace rr = sinfer::family::routing_replay;

namespace {

constexpr std::size_t kLayers = 3;
constexpr std::size_t kTopK   = 4;
constexpr std::int32_t kRejected = 30000; // ids no true position uses

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++failures;
    }
}

// The expert ids position `p` routes to in layer `layer`: distinct per (p, layer, r).
std::int32_t routed(std::size_t p, std::size_t layer, std::size_t r) {
    return static_cast<std::int32_t>((p * 97 + layer * 13 + r) % 20000);
}

// A capture `columns` wide; `position_of(c)` is the position column c ran, or -1 for a column
// that ran a token the sequence does not keep (a rejected draft, padding).
template <class PositionOf>
std::vector<std::int32_t> capture(std::size_t columns, PositionOf position_of) {
    std::vector<std::int32_t> ids(kLayers * columns * kTopK);
    for (std::size_t layer = 0; layer < kLayers; ++layer) {
        for (std::size_t c = 0; c < columns; ++c) {
            const long p = position_of(c);
            for (std::size_t r = 0; r < kTopK; ++r) {
                ids[(layer * columns + c) * kTopK + r] =
                    p < 0 ? kRejected + static_cast<std::int32_t>(r) : routed(static_cast<std::size_t>(p), layer, r);
            }
        }
    }
    return ids;
}

std::vector<std::int16_t> expected_rows(std::size_t positions) {
    std::vector<std::int16_t> rows(positions * kLayers * kTopK);
    for (std::size_t p = 0; p < positions; ++p) {
        for (std::size_t layer = 0; layer < kLayers; ++layer) {
            for (std::size_t r = 0; r < kTopK; ++r) {
                rows[(p * kLayers + layer) * kTopK + r] = static_cast<std::int16_t>(routed(p, layer, r));
            }
        }
    }
    return rows;
}

bool holds_rejected(const std::vector<std::int16_t>& rows) {
    for (const auto id : rows) {
        if (id >= kRejected) { return true; }
    }
    return false;
}

// The prompt in chunks of `chunk` columns (the lone prefill path).
void prefill(std::vector<std::int16_t>& rows, std::size_t prompt, std::size_t chunk) {
    for (std::size_t cursor = 0; cursor < prompt; cursor += chunk) {
        const std::size_t n = std::min(chunk, prompt - cursor);
        const auto ids      = capture(n, [&](std::size_t c) { return static_cast<long>(cursor + c); });
        rr::store_routing_rows(rows, ids.data(), kLayers, n, kTopK, {.column = 0, .position = cursor, .count = n});
    }
}

// Ordinary rounds in bursts of up to `burst`: each round one column per lane (here lane 0 of
// `lanes`), its input at frontier + round; the generation ends with `generated` tokens.
std::vector<std::int16_t> ordinary(std::size_t prompt, std::size_t generated, std::size_t burst, std::size_t lanes) {
    std::vector<std::int16_t> rows;
    prefill(rows, prompt, 7);
    std::size_t frontier = prompt, produced = 1; // the prefill samples the first token
    while (produced < generated) {
        const std::size_t rounds = std::min(burst, generated - produced);
        for (std::size_t round = 0; round < rounds; ++round) {
            // The other lanes' columns hold other sequences: mark them so a wrong column shows.
            const auto ids = capture(lanes, [&](std::size_t c) { return c == 0 ? static_cast<long>(frontier + round) : -1L; });
            rr::store_routing_rows(rows, ids.data(), kLayers, lanes, kTopK, {.column = 0, .position = frontier + round, .count = 1});
        }
        frontier += rounds;
        produced += rounds;
    }
    return rows;
}

// Speculative rounds: `draft_window` drafts, random acceptance; the lane is row `row` of `lanes`,
// behind `offset` prompt columns of a mixed round when `offset` > 0.
std::vector<std::int16_t> speculative(std::size_t prompt, std::size_t generated, std::size_t draft_window,
                                      std::size_t lanes, std::size_t row, std::size_t offset, std::mt19937& rng,
                                      std::size_t* rejected_columns) {
    std::vector<std::int16_t> rows;
    prefill(rows, prompt, 7);
    const std::size_t stride = draft_window + 1;
    std::size_t frontier = prompt, produced = 1;
    while (produced < generated) {
        const std::size_t remaining = generated - produced;
        const std::size_t extent    = std::min(draft_window, remaining - 1); // launch_mtp_round's budget cap
        const std::size_t accepted  = std::uniform_int_distribution<std::size_t>(0, extent)(rng);
        const std::size_t columns   = offset + lanes * stride;
        const auto ids = capture(columns, [&](std::size_t c) {
            if (c < offset || (c - offset) / stride != row) { return -1L; } // prompts, other lanes
            const std::size_t j = (c - offset) % stride;
            return j <= accepted ? static_cast<long>(frontier + j) : -1L;    // rejected, padding
        });
        *rejected_columns += stride - (accepted + 1);
        const auto span = rr::verify_routing_span(offset, row, stride, frontier, accepted + 1);
        rr::store_routing_rows(rows, ids.data(), kLayers, columns, kTopK, span);
        frontier += accepted + 1;
        produced += accepted + 1;
    }
    return rows;
}

} // namespace

int main() {
    constexpr std::size_t prompt = 23, generated = 61;
    // Positions run: the prompt and every generated token but the last.
    const auto expected = expected_rows(prompt + generated - 1);

    check(ordinary(prompt, generated, 1, 1) == expected, "ordinary rounds, one lane");
    check(ordinary(prompt, generated, 1, 5) == expected, "ordinary rounds, five lanes");
    check(ordinary(prompt, generated, 8, 3) == expected, "chained bursts of 8");
    check(ordinary(prompt, generated, 3, 2) == expected, "chained bursts of 3");

    std::mt19937 rng(7);
    for (const std::size_t window : {1U, 2U, 3U, 5U}) {
        for (const std::size_t offset : {0U, 9U}) {
            std::size_t rejected = 0;
            const auto rows = speculative(prompt, generated, window, 4, 2, offset, rng, &rejected);
            char what[96];
            std::snprintf(what, sizeof(what), "speculative rows equal ordinary (window %zu, offset %zu)", window, offset);
            check(rows == expected, what);
            std::snprintf(what, sizeof(what), "rejected drafts ran (%zu columns) but never stored", rejected);
            check(rejected > 0 && !holds_rejected(rows), what);
        }
    }

    // A mixed round: two prompts' segments, then two decode rows, in one capture.
    {
        const std::size_t a = 5, b = 3, columns = a + b + 2;
        const auto ids = capture(columns, [](std::size_t c) {
            if (c < 5) { return static_cast<long>(10 + c); }    // prompt A, positions 10..14
            if (c < 8) { return static_cast<long>(c - 5); }     // prompt B, positions 0..2
            return c == 8 ? 40L : 77L;                          // decode rows at 40 and 77
        });
        std::vector<std::int16_t> lane_a(10 * kLayers * kTopK, std::int16_t{-1}), lane_b, row0, row1;
        rr::store_routing_rows(lane_a, ids.data(), kLayers, columns, kTopK, {.column = 0, .position = 10, .count = a});
        rr::store_routing_rows(lane_b, ids.data(), kLayers, columns, kTopK, {.column = a, .position = 0, .count = b});
        rr::store_routing_rows(row0, ids.data(), kLayers, columns, kTopK, {.column = a + b, .position = 40, .count = 1});
        rr::store_routing_rows(row1, ids.data(), kLayers, columns, kTopK, {.column = a + b + 1, .position = 77, .count = 1});
        const auto full = expected_rows(78);
        const auto row_length = kLayers * kTopK;
        const auto same = [&](const std::vector<std::int16_t>& rows, std::size_t p) {
            return std::equal(rows.begin() + p * row_length, rows.begin() + (p + 1) * row_length, full.begin() + p * row_length);
        };
        bool ok = lane_a.size() == 15 * row_length && lane_b.size() == 3 * row_length;
        for (std::size_t p = 0; p < 10; ++p) { ok = ok && lane_a[p * row_length] == -1; }
        for (std::size_t p = 10; p < 15; ++p) { ok = ok && same(lane_a, p); }
        for (std::size_t p = 0; p < 3; ++p) { ok = ok && same(lane_b, p); }
        ok = ok && row0.size() == 41 * row_length && same(row0, 40) && row0[39 * row_length] == -1;
        ok = ok && row1.size() == 78 * row_length && same(row1, 77);
        check(ok, "mixed round: prompt segments and decode rows land at their positions");
    }

    // Spans that run past what the round copied are refused, not read out of bounds.
    {
        std::vector<std::int32_t> ids(kLayers * 4 * kTopK, 1);
        std::vector<std::int16_t> rows;
        bool threw = false;
        try {
            rr::store_routing_rows(rows, ids.data(), kLayers, 4, kTopK, {.column = 2, .position = 0, .count = 3});
        } catch (const std::logic_error&) { threw = true; }
        check(threw && rows.empty(), "span past the copied columns refused");
        threw = false;
        try { (void)rr::verify_routing_span(0, 0, 3, 0, 4); } catch (const std::logic_error&) { threw = true; }
        check(threw, "verify span licensing more columns than it ran refused");
    }

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("ok routing rows\n");
    return 0;
}
