#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

// ops::linear_swiglu_preactivation_delta (an adapter round's shared expert, sparse_moe_nvfp4.cu): a zero delta is
// bit-identical to linear_swiglu, a random delta matches the fp64 oracle, and only W8 parents outside the route tables
// take the delta form.
int main() {
    using namespace sinfer;
    using namespace sinfer::test::linear_swiglu;
    try {
        // Qwen3.6-35B-A3B's shared expert (2048 -> 2 x 512, stored W8): the per-pair kernel at T=1, the runtime-MMA
        // r32 tile through T=1024 (Full at 1024) and the r64 tile past it (Full at 2048).
        constexpr std::array<std::int32_t, 8> kSharedTokens{1, 2, 17, 64, 1024, 1025, 1500, 2048};
        int failures = run_preactivation_delta_profile(
            "W8 shared expert", {QType::W8G32_F16S, 1024, 2048, 512, 1709U, ActivationCompute::A16}, kSharedTokens,
            true);
        // A shape whose k misses the MMA alignment takes the per-pair kernel at every width.
        constexpr std::array<std::int32_t, 3> kNarrowTokens{1, 17, 128};
        failures += run_preactivation_delta_profile(
            "W8 narrow scale loads", {QType::W8G32_F16S, 256, 1152, 128, 1711U, ActivationCompute::A16},
            kNarrowTokens, true);
        // A registered route table's shape keeps linear_swiglu's own plan: the delta form is refused.
        constexpr std::array<std::int32_t, 2> kTableTokens{1, 33};
        failures += run_preactivation_delta_profile(
            "W8 35B dense mlp (refused)", {QType::W8G32_F16S, 12288, 2048, 6144, 1713U, ActivationCompute::A16},
            kTableTokens, false);
        if (failures == 77) { return 77; }
        if (failures != 0) {
            std::cerr << "linear_swiglu preactivation delta: " << failures << " failure(s)\n";
            return 1;
        }
        std::cout << "linear_swiglu preactivation delta: zero delta bit-identical, delta within the oracle bound\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "linear_swiglu preactivation delta: " << error.what() << '\n';
        return 1;
    }
}
