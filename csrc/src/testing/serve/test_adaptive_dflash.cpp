#include "family/impl/adaptive_dflash.h"

#include <cassert>
#include <limits>

using sinfer::family::AdaptiveDFlash;
using sinfer::family::dflash_draft_windows;

int main() {
    assert((dflash_draft_windows(15, true) == std::vector<std::uint32_t>{0, 1, 3, 7, 15}));
    assert((dflash_draft_windows(5, true) == std::vector<std::uint32_t>{0, 1, 3, 5}));
    assert((dflash_draft_windows(15, false) == std::vector<std::uint32_t>{15}));
    AdaptiveDFlash policy(15);
    std::array<unsigned, 16> chosen{};
    for (int round = 0; round < 300; ++round) {
        auto width = policy.choose(1, 100);
        if (round >= 172) {
            ++chosen[width];
        }
        // High acceptance alone is insufficient: the longer windows cost more per token.
        policy.observe(1, 100, width, (width == 0 ? 1.0 : 2.0) * (width + 1), width + 1);
    }
    assert(chosen[0] > 115);
    unsigned probes_without_feedback = 0;
    for (int i = 0; i < 128; ++i) { probes_without_feedback += policy.choose(1, 100) != 0; }
    assert(probes_without_feedback <= 4); // cancellations cannot stick on a probe
    assert(chosen[3] > 0);  // probes continue while ordinary decoding wins
    assert(policy.choose(sinfer::kMaximumBatchColumns, 100) == 3);
    assert(policy.choose(4, 100) == 3);   // no cross-batch contamination
    assert(policy.choose(1, 3000) == 3);  // no cross-context contamination
    chosen = {};
    for (int round = 0; round < 1400; ++round) {
        auto width = policy.choose(1, 100);
        if (round >= 1272) {
            ++chosen[width];
        }
        policy.observe(1, 100, width, width == 3 ? 0.2 : 2.0, 1);
    }
    assert(chosen[3] > 115);  // speculation recovers after the workload changes
    AdaptiveDFlash variable(15);
    for (int i = 0; i < 32; ++i) {
        for (auto width : dflash_draft_windows(15, true)) {
            variable.observe(1, 100, width, width == 0 ? 1 : width == 3 ? 3 : 10,
                width == 3 ? (i % 2 ? 7 : 1) : 1);
        }
    }
    assert(variable.choose(1, 100) == 3); // 4 tokens/3 seconds beats 1 token/second
    AdaptiveDFlash small(2);
    for (int round = 0; round < 300; ++round) {
        auto width = small.choose(2, 10000);
        assert(width <= 2);
        small.observe(2, 10000, width, 1, 2 * (width + 1));
    }
    assert(small.choose(2, 10000) == 2);
    const auto before = policy.choose(1, 100);
    policy.observe(1, 100, before, std::numeric_limits<double>::quiet_NaN(), 1);
    policy.observe(1, 100, before, 0, 1);
    policy.observe(1, 100, before, 1, 0);
    assert(policy.choose(1, 100) == before);
}
