#pragma once

// Rollout Routing Replay bookkeeping, kept free of CUDA so it can be tested on its own
// (testing/serve/test_routing_rows.cpp).
//
// A forward's capture is [layers][columns][top_k] expert ids, one column per token it ran. A
// request keeps [positions][layers][top_k] int16 rows, -1 where nothing was captured. Every round
// kind reduces to "these consecutive columns ran these consecutive positions": a prompt chunk or
// one prompt's segment of a mixed round, a decode row, one round of an ordinary burst, or the
// accepted columns of a speculative lane.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace sinfer::family::routing_replay {

struct RoutingSpan {
    std::size_t column   = 0;
    std::size_t position = 0;
    std::size_t count    = 0;
};

/// Columns [column, column + count) of a capture `capture_columns` wide become the rows of
/// positions [position, position + count); `rows` grows (with -1 rows) to hold them.
inline void store_routing_rows(std::vector<std::int16_t>& rows, const std::int32_t* capture,
                               std::size_t layers, std::size_t capture_columns, std::size_t top_k,
                               RoutingSpan span) {
    if (span.count == 0) { return; }
    if (span.column + span.count > capture_columns) {
        throw std::logic_error("routing capture span runs past the columns the round copied");
    }
    const std::size_t row_length = layers * top_k;
    const std::size_t end        = (span.position + span.count) * row_length;
    if (rows.size() < end) { rows.resize(end, std::int16_t{-1}); }
    for (std::size_t c = 0; c < span.count; ++c) {
        std::int16_t* row = rows.data() + (span.position + c) * row_length;
        for (std::size_t layer = 0; layer < layers; ++layer) {
            const std::int32_t* ids = capture + (layer * capture_columns + span.column + c) * top_k;
            for (std::size_t r = 0; r < top_k; ++r) {
                row[layer * top_k + r] = static_cast<std::int16_t>(ids[r]);
            }
        }
    }
}

/// A speculative verify gives lane row `row` the columns [offset + row * stride, + stride): the
/// anchor at position `base`, then the drafts after it, then padding. After the accept the first
/// `licensed` columns (the anchor and the drafts it accepted) ran tokens the sequence now holds;
/// the rejected drafts and the padding are never stored, and the token sampled after the last
/// accepted column gets its row when the next round runs it as the anchor.
inline RoutingSpan verify_routing_span(std::size_t offset, std::size_t row, std::size_t stride,
                                       std::size_t base, std::size_t licensed) {
    if (licensed > stride) { throw std::logic_error("a verify lane licensed more columns than it ran"); }
    return RoutingSpan{.column = offset + row * stride, .position = base, .count = licensed};
}

} // namespace sinfer::family::routing_replay
