// head_linear -- the per-head projection a latent attention folds its query through on the way
// in and unfolds the attended latent through on the way out. Every head has its own matrix,
// which is what `linear` cannot say.
#include "api/ops/head_linear.h"
#include "ops/input_projection_test_common.h"

#include <cmath>
#include <functional>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::input_projection;

namespace {

// W8's dequantisation -- an int8 code times an FP16 scale -- is exact in FP32, so what is left
// to the kernel is the FP32 accumulation order over at most 512 terms and the BF16 output
// rounding. The rounding is the bound: a BF16 value just above a power of two is 2^-8 of
// itself from its neighbour, so a correctly rounded output can sit 2^-8 (3.9e-3) of the
// largest reference away from it, and a seven-element sample dominated by one such element
// reads the same in relative L2. 4e-3 on each admits exactly that and nothing structural.
constexpr ReductionCriterion kHeadLinearTolerance{4.5e-3, 4.0e-3, 4.0e-3};

int run_case(std::int32_t heads, std::int32_t n, std::int32_t k, std::int32_t tokens,
             float out_scale, std::uint32_t seed) {
    const std::string label = "head_linear heads=" + std::to_string(heads) + " n=" +
                              std::to_string(n) + " k=" + std::to_string(k) + " T=" +
                              std::to_string(tokens) + " scale=" + std::to_string(out_scale);
    DevicePackedWeight weight(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, heads * n, k, seed));
    const std::vector<float> activation = make_bf16_activation(heads * k, tokens, seed + 1U);
    const std::vector<std::uint16_t> bits = bf16_bits(activation);
    DeviceBuffer device_x                 = to_device(bits);
    GuardedBf16Tensor output(heads * n, tokens);
    Tensor x(device_x.p, DType::BF16, {heads * k, tokens});
    Tensor out = output.tensor();

    ops::head_linear(x, weight.view(), heads, out_scale, out, nullptr);
    cuda_synchronize();

    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<double> got = output.values();
    for (std::int32_t head = 0; head < heads; ++head) {
        // Head h reads its own k rows of every column and writes its own n rows; the oracle
        // sees one head's problem as an ordinary projection of that slice.
        std::vector<float> slice(static_cast<std::size_t>(k) * tokens);
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t j = 0; j < k; ++j) {
                slice[static_cast<std::size_t>(t) * k + j] =
                    activation[static_cast<std::size_t>(t) * (heads * k) +
                               static_cast<std::size_t>(head) * k + j];
            }
        }
        std::vector<double> expected =
            projection_oracle(weight.host, head * n, n, slice, k, tokens);
        for (double& value : expected) { value *= static_cast<double>(out_scale); }
        failures += compare(label + " head " + std::to_string(head),
                            gather_rows(got, heads * n, head * n, n, tokens), expected,
                            kHeadLinearTolerance);
    }
    failures += verify_preserved(label + " x", device_x, bits);
    failures += weight.verify_preserved(label + " w");
    return failures;
}

int expect_rejection(const char* what, const std::function<void()>& call) {
    try {
        call();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << "head_linear: " << what << " was not rejected\n";
    return 1;
}

int run_rejections() {
    DevicePackedWeight weight(quantized_weight::make_patterned_weight(QType::W8G32_F16S, 48, 64, 3U));
    const std::vector<float> activation = make_bf16_activation(4 * 64, 2, 4U);
    DeviceBuffer device_x               = to_device(bf16_bits(activation));
    GuardedBf16Tensor output(48, 2);
    Tensor out = output.tensor();
    int failures = 0;
    failures += expect_rejection("rows that are not a whole number of heads", [&] {
        Tensor x(device_x.p, DType::BF16, {5 * 64, 2});
        ops::head_linear(x, weight.view(), 5, 1.0F, out, nullptr);
    });
    failures += expect_rejection("an activation of the wrong width", [&] {
        Tensor x(device_x.p, DType::BF16, {4 * 32, 2});
        ops::head_linear(x, weight.view(), 4, 1.0F, out, nullptr);
    });
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    // A row count that is not a whole number of warps, two scale groups, every token tile.
    failures += run_case(4, 48, 64, 1, 1.0F, 11U);
    failures += run_case(4, 48, 64, 3, 1.0F, 12U);
    failures += run_case(4, 48, 64, 8, 1.0F, 13U);
    failures += run_case(4, 48, 64, 17, 1.0F, 14U);
    // GLM-5.3-Flash's two sites: the absorb, 64 heads of 256 into the 512-wide latent with the
    // sqrt(2) that turns the kernels' 1/sqrt(512) into the model's 1/sqrt(256); and the
    // unabsorb, 512 back to 256.
    failures += run_case(64, 512, 256, 1, std::sqrt(2.0F), 21U);
    failures += run_case(64, 512, 256, 9, std::sqrt(2.0F), 22U);
    failures += run_case(64, 256, 512, 1, 1.0F, 23U);
    failures += run_case(64, 256, 512, 5, 1.0F, 24U);
    failures += run_rejections();
    if (failures != 0) {
        std::cerr << "head_linear: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "head_linear: ok\n";
    return 0;
}
