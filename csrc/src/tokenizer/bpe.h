// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
// Core BPE merge algorithm, ported from tiktoken (MIT License).
// Two-path approach: O(mn) linear scan for small pieces (< 128 bytes),
// O(m log n) heap-based for large pieces. The small-piece path wins on
// cache locality for the common case.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace tokenizer {

using Rank = uint32_t;
static constexpr Rank RANK_MAX = std::numeric_limits<Rank>::max();

// ---- Fast hash for byte vectors (FNV-1a) ----
struct ByteVecHash {
    size_t operator()(const std::vector<uint8_t>& v) const noexcept {
        size_t h = 14695981039346656037ULL; // FNV offset basis
        for (uint8_t b : v) {
            h ^= b;
            h *= 1099511628211ULL; // FNV prime
        }
        return h;
    }
};

// String-view-like key for hash lookups without allocation.
struct ByteSpan {
    const uint8_t* data;
    size_t len;
};

struct ByteSpanHash {
    size_t operator()(const ByteSpan& s) const noexcept {
        size_t h = 14695981039346656037ULL;
        for (size_t i = 0; i < s.len; i++) {
            h ^= s.data[i];
            h *= 1099511628211ULL;
        }
        return h;
    }
};

struct ByteSpanEqual {
    bool operator()(const ByteSpan& a, const ByteSpan& b) const noexcept {
        return a.len == b.len && std::memcmp(a.data, b.data, a.len) == 0;
    }
};

// The encoder map: byte sequence -> rank (token ID).
using Encoder = std::unordered_map<std::vector<uint8_t>, Rank, ByteVecHash>;

// Temporary lookup that accepts ByteSpan keys (avoids allocations in hot loop).
// Built once from the Encoder for the duration of a merge operation.
class EncoderLookup {
public:
    explicit EncoderLookup(const Encoder& enc) {
        map_.reserve(enc.size());
        for (const auto& [k, v] : enc) {
            map_[ByteSpan{k.data(), k.size()}] = v;
        }
    }

    Rank get(const uint8_t* data, size_t len) const {
        auto it = map_.find(ByteSpan{data, len});
        return it != map_.end() ? it->second : RANK_MAX;
    }

private:
    std::unordered_map<ByteSpan, Rank, ByteSpanHash, ByteSpanEqual> map_;
};

// ---- Small-piece BPE: O(mn) linear scan ----
// For pieces < SMALL_PIECE_THRESHOLD bytes. Cache-locality beats algorithmic
// complexity for the overwhelmingly common case of short regex pieces.
static constexpr size_t SMALL_PIECE_THRESHOLD = 128;

inline std::vector<Rank> bpe_merge_small(const uint8_t* piece, size_t piece_len, const EncoderLookup& enc) {
    // parts[i] = (start_offset, rank_of_pair_starting_at_i)
    // Two sentinel entries at the end.
    struct Part { size_t start; Rank rank; };

    std::vector<Part> parts;
    parts.reserve(piece_len + 1);

    Rank min_rank = RANK_MAX;
    size_t min_idx = 0;

    for (size_t i = 0; i < piece_len - 1; i++) {
        Rank r = enc.get(piece + i, 2);
        if (r < min_rank) {
            min_rank = r;
            min_idx = i;
        }
        parts.push_back({i, r});
    }
    parts.push_back({piece_len - 1, RANK_MAX});
    parts.push_back({piece_len, RANK_MAX});

    // Recompute the rank for the pair at index i in parts.
    // After merging parts[i+1] is removed, so parts[i]..parts[i+3] spans 3 original tokens.
    auto get_rank = [&](size_t i) -> Rank {
        if (i + 3 < parts.size()) {
            return enc.get(piece + parts[i].start, parts[i + 3].start - parts[i].start);
        }
        return RANK_MAX;
    };

    while (min_rank != RANK_MAX) {
        size_t i = min_idx;

        // Update neighbors before removing parts[i+1].
        if (i > 0) {
            parts[i - 1].rank = get_rank(i - 1);
        }
        parts[i].rank = get_rank(i);
        parts.erase(parts.begin() + static_cast<std::ptrdiff_t>(i + 1));

        // Rescan for new minimum.
        min_rank = RANK_MAX;
        for (size_t j = 0; j + 1 < parts.size(); j++) {
            if (parts[j].rank < min_rank) {
                min_rank = parts[j].rank;
                min_idx = j;
            }
        }
    }

    // Extract token ranks from parts.
    std::vector<Rank> result;
    result.reserve(parts.size() - 1);
    for (size_t i = 0; i + 1 < parts.size(); i++) {
        Rank r = enc.get(piece + parts[i].start, parts[i + 1].start - parts[i].start);
        assert(r != RANK_MAX);
        result.push_back(r);
    }
    return result;
}

// ---- Large-piece BPE: O(m log n) heap-based ----
// For pieces >= SMALL_PIECE_THRESHOLD bytes. Uses a min-heap with lazy
// invalidation (tiktoken's approach).

inline std::vector<Rank> bpe_merge_large(const uint8_t* piece, size_t piece_len, const EncoderLookup& enc) {
    struct State {
        size_t prev;       // Previous token start
        size_t end;        // Current token end
        size_t next_end;   // Next token's end (for lookahead)
        Rank next_rank;    // Rank of potential merge at this position
        Rank cur_rank;     // Rank of the already-merged token
    };

    struct Merge {
        size_t start;
        Rank rank;
        bool operator>(const Merge& o) const {
            return rank != o.rank ? rank > o.rank : start > o.start;
        }
    };

    std::vector<State> state;
    state.reserve(piece_len);
    state.push_back({SIZE_MAX, 1, 2, RANK_MAX, RANK_MAX}); // sentinel at index 0

    std::priority_queue<Merge, std::vector<Merge>, std::greater<Merge>> heap;

    for (size_t i = 0; i < piece_len - 1; i++) {
        Rank r = enc.get(piece + i, 2);
        if (r != RANK_MAX) {
            heap.push({i, r});
        }
        state.push_back({
            i,                                         // prev
            i + 2,                                     // end
            std::min(i + 3, piece_len),                // next_end
            r,                                         // next_rank (valid merge rank or MAX)
            RANK_MAX                                   // cur_rank (not yet merged)
        });
    }

    auto potential_merge = [&](size_t start, size_t next_end_item) {
        state[start].next_end = next_end_item;
        state[start].next_rank = RANK_MAX;
        if (next_end_item <= piece_len) {
            Rank r = enc.get(piece + start, next_end_item - start);
            if (r != RANK_MAX) {
                heap.push({start, r});
                state[start].next_rank = r;
            }
        }
    };

    while (!heap.empty()) {
        auto [left_start, rank] = heap.top();
        heap.pop();

        if (rank == RANK_MAX) break;
        if (rank != state[left_start].next_rank) continue; // invalidated

        size_t right_start = state[left_start].end;
        size_t right_end = state[left_start].next_end;
        size_t right_next_end = state[right_start].next_end;

        // Merge left and right.
        state[left_start].cur_rank = rank;
        state[left_start].end = right_end;
        potential_merge(left_start, right_next_end);

        // Update backward link.
        if (right_end < state.size()) {
            state[right_end].prev = left_start;
        }

        // Update merge that ends at left_start.
        if (left_start > 0) {
            size_t prev_start = state[left_start].prev;
            potential_merge(prev_start, right_end);
        }

        // Invalidate merge starting at right_start.
        state[right_start].next_rank = RANK_MAX;
    }

    // Extract final token ranks.
    std::vector<Rank> result;
    size_t i = 0;
    while (i < state.size()) {
        if (state[i].cur_rank != RANK_MAX) {
            result.push_back(state[i].cur_rank);
        } else {
            Rank r = enc.get(piece + i, state[i].end - i);
            assert(r != RANK_MAX);
            result.push_back(r);
        }
        i = state[i].end;
    }
    return result;
}

// ---- Public dispatch ----

inline std::vector<Rank> byte_pair_encode(const uint8_t* piece, size_t piece_len, const EncoderLookup& enc) {
    if (piece_len == 1) {
        Rank r = enc.get(piece, 1);
        assert(r != RANK_MAX);
        return {r};
    }
    if (piece_len < SMALL_PIECE_THRESHOLD) {
        return bpe_merge_small(piece, piece_len, enc);
    }
    return bpe_merge_large(piece, piece_len, enc);
}

inline std::vector<Rank> byte_pair_encode(const std::string& piece, const EncoderLookup& enc) {
    return byte_pair_encode(reinterpret_cast<const uint8_t*>(piece.data()), piece.size(), enc);
}

} // namespace tokenizer
