#pragma once

#include "core/limits.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace sinfer::family {

inline std::vector<std::uint32_t> dflash_draft_windows(std::uint32_t maximum, bool adaptive) {
    if (!adaptive) { return {maximum}; }
    std::vector<std::uint32_t> windows{0};
    for (auto width : {1U, 3U, 7U, 15U}) {
        if (width < maximum) { windows.push_back(width); }
    }
    windows.push_back(maximum);
    return windows;
}

// Bounded online calibration. Each exact batch size and context band learns its own
// elapsed time per committed token. Zero drafts participates as a measured candidate;
// periodic probes let speculation recover after a difficult stretch of output.
class AdaptiveDFlash {
    struct Arm {
        double seconds = 0, tokens = 0;
        std::uint64_t samples = 0, last = 0;
        double cost() const { return seconds / tokens; }
    };

    struct History {
        std::array<Arm, 16> arms{};
        std::uint64_t rounds    = 0;
        std::uint64_t decisions = 0;
        std::uint32_t incumbent = 3;
    };

    std::vector<std::uint32_t> windows_;
    std::array<std::array<History, 5>, kMaximumBatchColumns> histories_{};

    History& history(std::uint32_t batch, std::uint32_t context) {
        if (batch == 0 || batch > histories_.size()) {
            throw std::invalid_argument("adaptive DFlash batch size is out of bounds");
        }
        const auto band = context < 512     ? 0
                          : context < 2048  ? 1
                          : context < 8192  ? 2
                          : context < 32768 ? 3
                                            : 4;
        return histories_[batch - 1][band];
    }
public:
    explicit AdaptiveDFlash(std::uint32_t maximum) : windows_(dflash_draft_windows(maximum, true)) {
        if (maximum == 0 || maximum > 15) {
            throw std::invalid_argument("adaptive DFlash maximum must be in [1,15]");
        }
        for (auto& batch : histories_) {
            for (auto& h : batch) { h.incumbent = std::min(3U, maximum); }
        }
    }

    std::uint32_t choose(std::uint32_t batch, std::uint32_t context) {
        auto& h = history(batch, context);
        ++h.decisions;
        if (h.arms[h.incumbent].samples < 2) { return h.incumbent; }
        for (auto width : windows_) {
            if (h.arms[width].samples < 2) { return width; }
        }
        auto best = h.incumbent;
        for (auto width : windows_) {
            if (h.arms[width].cost() < h.arms[best].cost()) { best = width; }
        }
        if (h.arms[best].cost() < h.arms[h.incumbent].cost() * 0.9) { h.incumbent = best; }
        if (h.decisions % 32 == 0) {
            return *std::min_element(windows_.begin(), windows_.end(), [&](auto a, auto b) {
                return h.arms[a].last < h.arms[b].last;
            });
        }
        return h.incumbent;
    }

    void observe(std::uint32_t batch, std::uint32_t context, std::uint32_t drafts, double seconds,
                 std::uint32_t committed) {
        if (!std::isfinite(seconds) || seconds <= 0 || committed == 0) { return; }
        if (std::find(windows_.begin(), windows_.end(), drafts) == windows_.end()) {
            throw std::invalid_argument("unconfigured adaptive DFlash window");
        }
        auto& h           = history(batch, context);
        auto& arm         = h.arms[drafts];
        // Ratio of accumulated time and tokens measures throughput. Averaging
        // time/token ratios instead penalizes windows with variable acceptance.
        arm.seconds = arm.samples == 0 ? seconds : 0.25 * seconds + 0.75 * arm.seconds;
        arm.tokens = arm.samples == 0 ? committed : 0.25 * committed + 0.75 * arm.tokens;
        ++arm.samples;
        arm.last = ++h.rounds;
    }
};

} // namespace sinfer::family
