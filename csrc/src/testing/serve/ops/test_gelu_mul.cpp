// gelu_mul: out = gelu(gate) * up, elementwise, in both GELU formulations.
//
// The two modes are tested separately and their outputs compared, because the
// tanh formulation is not an approximation the caller may substitute freely:
// Gemma's MLP specifies it, and the exact-erf form differs from it by far more
// than BF16 rounding.

#include "api/ops/gelu_mul.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr PointwiseCriterion gelu_mul_bf16_criterion() {
    return {/*absolute*/ 2.0e-2, /*relative*/ 8.0e-3};
}

double gelu_one(double z, bool tanh_approx) {
    constexpr double kSqrt2Pi = 0.79788456080286535588;
    if (tanh_approx) {
        return 0.5 * z * (1.0 + std::tanh(kSqrt2Pi * (z + 0.044715 * z * z * z)));
    }
    return 0.5 * z * (1.0 + std::erf(z * 0.70710678118654752440));
}

int run(std::size_t n, bool tanh_approx, std::uint32_t seed) {
    std::vector<float> gate(n);
    std::vector<float> up(n);
    fill_uniform(gate, seed, -6.0F, 6.0F);
    fill_uniform(up, seed + 1, -4.0F, 4.0F);
    round_to_bf16(gate);
    round_to_bf16(up);

    std::vector<double> reference(n);
    for (std::size_t i = 0; i < n; ++i) {
        reference[i] = gelu_one(gate[i], tanh_approx) * static_cast<double>(up[i]);
    }

    DeviceBuffer dg = to_device_bf16(gate);
    DeviceBuffer du = to_device_bf16(up);
    DeviceBuffer dout(n * sizeof(std::uint16_t));
    Tensor tg(dg.p, DType::BF16, {static_cast<std::int64_t>(n)});
    Tensor tu(du.p, DType::BF16, {static_cast<std::int64_t>(n)});
    Tensor to(dout.p, DType::BF16, {static_cast<std::int64_t>(n)});

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ops::gelu_mul(tg, tu, tanh_approx ? ops::GeluMode::Tanh : ops::GeluMode::Exact, to, stream);
    cuda_synchronize(stream);
    cuda_check(cudaStreamDestroy(stream), "stream destroy");

    const std::string label =
        "gelu_mul n=" + std::to_string(n) + (tanh_approx ? " tanh" : " exact");
    return verify_pointwise(label, from_device_bf16(dout, n), reference,
                            gelu_mul_bf16_criterion());
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "gelu_mul: no CUDA device, skipping\n";
        return 0;
    }
    int failures = 0;
    for (const bool tanh_approx : {true, false}) {
        failures += run(4096, tanh_approx, 21);
        failures += run(1152 * 7, tanh_approx, 22); // Gemma's intermediate width
        failures += run(1, tanh_approx, 23);        // odd tail, no pair covers it
        failures += run(1023, tanh_approx, 24);     // odd length
    }
    if (failures != 0) {
        std::cerr << "gelu_mul: " << failures << " case(s) failed\n";
        return 1;
    }
    std::cout << "gelu_mul: all cases passed\n";
    return 0;
}
