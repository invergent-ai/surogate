#include "api/ops/logit_softcap.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr PointwiseCriterion softcap_bf16_criterion() {
    return {/*absolute*/ 2.0e-6, /*relative*/ 4.05e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        bits[index] = f32_to_bf16(values[index]);
    }
    return bits;
}

/// `tanh(x / cap) * cap`, stated as the reference states it -- with a divide, where the
/// kernel multiplies by a precomputed reciprocal. Writing the oracle the kernel's way would
/// hide exactly the difference it is here to measure.
double softcap_oracle(double input, double cap) { return std::tanh(input / cap) * cap; }

int run_case(const char* label, float cap, std::int32_t rows, std::int32_t columns,
             std::uint32_t seed, float span = 80.0F) {
    const std::size_t count = static_cast<std::size_t>(rows) * columns;
    std::vector<float> input(count);
    // Logits reach well past the cap in both directions, which is the whole point of it: the
    // interesting region is where tanh saturates, not where it is nearly linear.
    fill_uniform(input, seed, -span, span);
    round_to_bf16(input);

    std::vector<double> expected(count);
    for (std::size_t index = 0; index < count; ++index) {
        expected[index] = softcap_oracle(input[index], cap);
    }
    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());

    Tensor input_tensor(device_input.data(), DType::BF16, {rows, columns});
    ops::logit_softcap(input_tensor, cap, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_input.data(), count), expected,
                                    softcap_bf16_criterion());
    failures += device_input.verify_guards("logit_softcap input");
    return failures;
}

/// The values a vocabulary's logits actually reach, including the saturating tails and an odd
/// element count -- the kernel walks BF16 *pairs* and handles a trailing odd element apart, so
/// a run of even sizes would never exercise that branch.
int run_edge_case() {
    std::vector<float> input{-1.0e4F, -300.0F, -90.0F, -30.0F, -1.0F, -0.0F,
                             0.0F,    1.0F,    30.0F,  90.0F,  300.0F};
    round_to_bf16(input);
    std::vector<double> expected(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        expected[index] = softcap_oracle(input[index], 30.0);
    }

    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    Tensor input_tensor(device_input.data(), DType::BF16,
                        {static_cast<std::int32_t>(input.size())});
    ops::logit_softcap(input_tensor, 30.0F, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise("logit_softcap edge values",
                                    from_device_bf16(device_input.data(), input.size()), expected,
                                    softcap_bf16_criterion());
    failures += device_input.verify_guards("logit_softcap edge input");
    return failures;
}

/// Nothing leaves the bound, which is the property the model depends on rather than a
/// numerical tolerance: the cap is what keeps a trained-with-it head's ratios where the
/// training put them.
int run_bound_case() {
    constexpr float kCap = 30.0F;
    std::vector<float> input(4096);
    fill_uniform(input, 909U, -5.0e3F, 5.0e3F);
    round_to_bf16(input);
    const auto input_bits = encode_bf16(input);
    GuardedDeviceBuffer device_input(input_bits.size() * sizeof(std::uint16_t));
    device_input.copy_from_host(input_bits.data(), device_input.bytes());
    Tensor input_tensor(device_input.data(), DType::BF16,
                        {static_cast<std::int32_t>(input.size())});
    ops::logit_softcap(input_tensor, kCap, nullptr);
    cuda_synchronize();

    const auto got = from_device_bf16(device_input.data(), input.size());
    int failures   = 0;
    for (std::size_t index = 0; index < got.size(); ++index) {
        if (!(std::abs(got[index]) <= static_cast<double>(kCap))) {
            std::cout << "FAIL logit_softcap bound: element " << index << " is " << got[index]
                      << ", outside +-" << kCap << "\n";
            ++failures;
            break;
        }
    }
    failures += device_input.verify_guards("logit_softcap bound input");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    // Gemma 4's cap, over a vocabulary-shaped plane.
    failures += run_case("logit_softcap 30 [262144,1]", 30.0F, 262144, 1, 601U);
    failures += run_case("logit_softcap 30 [262144,4]", 30.0F, 262144, 4, 602U);
    // A second cap, so nothing can pass by treating 30 as a constant.
    failures += run_case("logit_softcap 50 [4096,7]", 50.0F, 4096, 7, 603U);
    // Small values, where tanh is nearly linear and the relative tolerance is tightest.
    failures += run_case("logit_softcap 30 near-zero", 30.0F, 2048, 3, 604U, 0.5F);
    failures += run_edge_case();
    failures += run_bound_case();
    std::cout << (failures ? "FAIL" : "OK") << " logit_softcap\n";
    return failures ? 1 : 0;
}
