#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int q5_a16_conformance() {
    int failures = 0;

    constexpr std::array kN1024K5120{
        convenience(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {1024, 5120, 151U, Comparison::Full, true, kN1024K5120});

    constexpr std::array kN6144K5120{
        a16(1), a16(2), a16(6), a16(7), a16(24), a16(25), a16(64), a16(65), a16(128),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {6144, 5120, 157U, Comparison::Sampled, false, kN6144K5120});

    constexpr std::array kN7168K5120{
        a16(1), a16(2), a16(6), a16(7), a16(16), a16(17), a16(128),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {7168, 5120, 163U, Comparison::Sampled, false, kN7168K5120});

    constexpr std::array kN5120K6144{
        a16(1), a16(2), a16(6), a16(7), a16(24), a16(25), a16(128),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {5120, 6144, 167U, Comparison::Sampled, false, kN5120K6144});

    constexpr std::array kN5120K17408{
        a16(1), a16(2), a16(6), a16(7), a16(24), a16(25), a16(128),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {5120, 17408, 173U, Comparison::Sampled, false, kN5120K17408});

    constexpr std::array kN1152K1152{
        a16(4),   a16(76),   a16(80),   a16(636),  a16(640),  a16(700),    a16(704),
        a16(708), a16(828),  a16(832),  a16(836),  a16(896),  a16(900),    a16(960),
        a16(964), a16(1024), a16(1028), a16(1088), a16(1092), a16(131072),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {1152, 1152, 179U, Comparison::Sampled, false, kN1152K1152});

    constexpr std::array kN1152K4304{
        a16(4), a16(120), a16(124), a16(1148), a16(1152), a16(131072),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {1152, 4304, 181U, Comparison::Sampled, false, kN1152K4304});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
