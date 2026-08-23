#include "ninfer/ops/gdn_gating.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHeads = 48;

constexpr PointwiseCriterion kGdnGatingFp32{/*absolute=*/1.0e-7, /*relative=*/2.2e-7};

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_fp32(const void* device, std::size_t elements) {
    const std::vector<float> values = from_device<float>(device, elements);
    return {values.begin(), values.end()};
}

void gating_oracle(const std::vector<float>& a, const std::vector<float>& b,
                   const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                   std::vector<double>& g, std::vector<double>& beta) {
    g.resize(a.size());
    beta.resize(b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        const std::size_t head = i % kHeads;
        const double av        = static_cast<double>(a[i]);
        const double bv        = static_cast<double>(b[i]);
        const double bias      = static_cast<double>(dt_bias[head]);
        const double scale     = std::exp(static_cast<double>(a_log[head]));
        g[i]                   = -scale * softplus(av + bias);
        beta[i]                = sigmoid(bv);
    }
}

int run_case(std::int32_t tokens, std::uint32_t seed, bool stress_transcendentals) {
    const std::size_t elements = static_cast<std::size_t>(kHeads) * tokens;
    std::vector<float> a(elements), b(elements), a_log(kHeads), dt_bias(kHeads);
    fill_uniform(a, seed, -8.0F, 8.0F);
    fill_uniform(b, seed + 1u, -8.0F, 8.0F);
    fill_uniform(a_log, seed + 2u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 3u, -1.0F, 1.0F);
    if (stress_transcendentals) {
        constexpr float values[] = {-30.0F, -15.0F, 0.0F, 15.0F, 30.0F};
        for (std::size_t i = 0; i < elements; ++i) {
            a[i] = values[i % 5];
            b[i] = values[(i / 5) % 5];
        }
    }
    round_to_bf16(a);
    round_to_bf16(b);

    std::vector<double> reference_g, reference_beta;
    gating_oracle(a, b, a_log, dt_bias, reference_g, reference_beta);

    const std::vector<std::uint16_t> a_bits = bf16_bits(a);
    const std::vector<std::uint16_t> b_bits = bf16_bits(b);
    DeviceBuffer device_a                   = to_device(a_bits);
    DeviceBuffer device_b                   = to_device(b_bits);
    DeviceBuffer device_a_log               = to_device(a_log);
    DeviceBuffer device_dt_bias             = to_device(dt_bias);
    GuardedDeviceBuffer device_g(elements * sizeof(float));
    GuardedDeviceBuffer device_beta(elements * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_a(device_a.p, DType::BF16, {kHeads, tokens});
    Tensor tensor_b(device_b.p, DType::BF16, {kHeads, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {kHeads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {kHeads});
    Tensor tensor_g(device_g.data(), DType::FP32, {kHeads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {kHeads, tokens});

    ops::gdn_gating(tensor_a, tensor_b, tensor_a_log, tensor_dt_bias, tensor_g, tensor_beta,
                    nullptr);
    cuda_synchronize();

    const std::string label = std::string("gdn_gating T=") + std::to_string(tokens) +
                              (stress_transcendentals ? " transcendental-range" : "");
    int failures = 0;
    failures += verify_pointwise((label + " g").c_str(), read_fp32(device_g.data(), elements),
                                 reference_g, kGdnGatingFp32);
    failures += verify_pointwise((label + " beta").c_str(), read_fp32(device_beta.data(), elements),
                                 reference_beta, kGdnGatingFp32);
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_exact((label + " a immutable").c_str(),
                             from_device<std::uint16_t>(device_a, elements), a_bits);
    failures += verify_exact((label + " b immutable").c_str(),
                             from_device<std::uint16_t>(device_b, elements), b_bits);
    failures += verify_exact((label + " A_log immutable").c_str(),
                             from_device<float>(device_a_log, a_log.size()), a_log);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(1, 0x101u, false);
    failures += run_case(7, 0x202u, false);
    failures += run_case(128, 0x303u, false);
    failures += run_case(4096, 0x404u, false);
    failures += run_case(17, 0x505u, true);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating correctness\n";
    return failures == 0 ? 0 : 1;
}
