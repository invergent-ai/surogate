#pragma once

// Whether a thought goes round in circles: StopPolicy::thinking_guard.loop (--thinking-loop-guard).
//
// A thought position repeats when the 32 tokens ending there have already appeared at least twice
// earlier in the same thought, so its third showing on. The detector fires once at least 90% of
// the last 2,048 positions repeat. Calibrated offline on the release model's own replies
// (rl/answer-guard-2026-10-28/loopstat.py in the 35B-A3B work tree): it ends each of four looping
// thoughts 5,500 to 11,000 tokens in, and fires on none of 346 chat replies or 6,900 long
// SWE-agent turns, whose highest window shares are 0.31 and 0.55.
//
// Model-free. The output session feeds it the thought's tokens as they are previewed: a round
// that does not finish the request is always accepted whole, and one that finishes it ends the
// session, so the detector never has to take a token back.

#include "api/types.h"

#include <array>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace sinfer::family {

class ThinkingLoopDetector {
public:
    static constexpr std::uint32_t kGram    = 32;
    static constexpr std::uint8_t kRepeats  = 2;
    static constexpr std::uint32_t kWindow  = 2048;
    static constexpr std::uint32_t kLooping = kWindow * 9 / 10;

    /// Feeds the thought's next token; true once the thought loops.
    bool push(TokenId token) {
        ring_[pushed_ % kGram] = token;
        ++pushed_;
        bool repeats = false;
        if (pushed_ >= kGram) {
            std::uint64_t hash = 0x9e3779b97f4a7c15ULL;
            for (std::uint32_t i = 0; i < kGram; ++i) { // oldest first
                hash = mix(hash ^ static_cast<std::uint32_t>(ring_[(pushed_ + i) % kGram]));
            }
            std::uint8_t& seen = seen_[hash];
            repeats            = seen >= kRepeats;
            if (!repeats) { ++seen; }
        }
        if (window_.empty()) { window_.assign(kWindow, 0); }
        std::uint8_t& slot = window_[(pushed_ - 1) % kWindow];
        repeating_ -= slot;
        slot = repeats ? 1 : 0;
        repeating_ += slot;
        return looping();
    }

    [[nodiscard]] bool looping() const noexcept { return pushed_ >= kWindow && repeating_ >= kLooping; }
    [[nodiscard]] std::uint64_t tokens() const noexcept { return pushed_; }

private:
    static std::uint64_t mix(std::uint64_t x) noexcept { // splitmix64's finalizer
        x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
        return x ^ (x >> 31U);
    }

    std::array<TokenId, kGram> ring_{};
    std::uint64_t pushed_ = 0;
    /// Each 32-token gram's showings so far, saturating at kRepeats.
    std::unordered_map<std::uint64_t, std::uint8_t> seen_;
    /// Whether each of the last kWindow positions repeats, and how many do.
    std::vector<std::uint8_t> window_;
    std::uint32_t repeating_ = 0;
};

} // namespace sinfer::family
