// Which W8 route a shape takes, as a rule rather than as a measurement.
//
// `select_w8_launch` is the only thing standing between a shape and a kernel whose alignment
// assumptions it does not meet, and nothing tested it. Two such assumptions have now been found
// the same way -- by a model producing wrong numbers -- and both are invisible until a real
// checkpoint lands on them:
//
//   * scale rows are read 16 bytes at a time, which needs `k % 256 == 0`  (EmbeddingGemma);
//   * rows are tiled in groups of 128, which needs `n % 128 == 0`         (Gemma 4 26B-A4B).
//
// The second was measured against `transformers` on the mixture's feed-forward, n = 2,112 and
// k = 2,816, one token either side of the untuned fallback's SIMT/MMA boundary:
//
//     t = 16 (SIMT)  cosine 0.999806
//     t = 17 (MMA)   cosine 0.000693     <- orthogonal, not merely inaccurate
//
// A shape that misses either constraint must take a SIMT route at every width. These cases are
// the shapes the engine actually serves, so a future tuning entry that reintroduces the fault
// fails here rather than in a checkpoint.

#include "ops/linear/w8/w8_dispatch.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdio>
#include <cstdint>

using sinfer::ops::LinearPolicy;
using sinfer::ops::detail::W8Launch;
using sinfer::ops::detail::kW8MmaRowAlignmentN;
using sinfer::ops::detail::kW8MmaScaleRowAlignmentK;
using sinfer::ops::detail::launch_w8_simt_r8_c4;
using sinfer::ops::detail::select_w8_launch;

namespace {

int failures = 0;

void expect_simt(std::int32_t n, std::int32_t k, std::int32_t t, const char* why) {
    const W8Launch chosen = select_w8_launch(n, k, t, LinearPolicy::A16Only);
    if (chosen != launch_w8_simt_r8_c4) {
        std::printf("FAIL n=%d k=%d t=%d: expected the SIMT route (%s)\n", n, k, t, why);
        ++failures;
    }
}

} // namespace

int main() {
    static_assert(kW8MmaRowAlignmentN == 128);
    static_assert(kW8MmaScaleRowAlignmentK == 256);

    // Gemma 4 26B-A4B, the shape that found the row constraint. 2,112 is 16.5 x 128, so every
    // width must stay off the MMA routes -- including the ones a tuned entry might claim.
    for (const std::int32_t t : {17, 27, 64, 129, 512, 4096}) {
        expect_simt(2112, 2816, t, "n = 2112 is not a whole number of 128-row groups");
    }
    // Its down projection misses the K constraint instead, and must be refused for that.
    for (const std::int32_t t : {17, 27, 512}) {
        expect_simt(2816, 2112, t, "k = 2112 is not a multiple of 256");
    }
    // A shape that meets both must NOT be pushed to SIMT above the small-T band -- the guards
    // are a floor on correctness, not an excuse to abandon the tuned routes. The 12B's
    // feed-forward and a 1,024 x 512 fixture both measured clean on MMA at t = 27.
    for (const auto [n, k] : {std::pair{15360, 3840}, std::pair{1024, 512}}) {
        if (select_w8_launch(n, k, 27, LinearPolicy::A16Only) == launch_w8_simt_r8_c4) {
            std::printf("FAIL n=%d k=%d t=27: a conforming shape was forced onto SIMT\n", n, k);
            ++failures;
        }
    }

    if (failures != 0) {
        std::printf("w8 route selection: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("OK w8 route selection\n");
    return 0;
}
