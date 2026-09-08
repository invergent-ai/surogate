#include "api/ops/gdn_gating.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr std::int32_t kHeads = 48;

constexpr PointwiseCriterion kGdnGatingFp32{/*absolute=*/1.0e-7, /*relative=*/2.2e-7};

/// The bounded logistic decay costs more than the softplus one, and the arithmetic says how
/// much: the sigmoid's argument is `exp(A_log) * (a + dt_bias)`, so a relative error of about
/// one ulp on each of the exponential, the product and the sigmoid's own exponential is
/// amplified by that argument's magnitude -- around 21 for the ranges a checkpoint occupies.
/// Twenty-one ulps of FP32 is 1.3e-6, and this is the next round number above it.
constexpr PointwiseCriterion kKdaGatingFp32{/*absolute=*/1.0e-6, /*relative=*/5.0e-6};

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
        const std::size_t head = i % a_log.size();
        const double av        = static_cast<double>(a[i]);
        const double bv        = static_cast<double>(b[i]);
        const double bias      = static_cast<double>(dt_bias[head]);
        const double scale     = std::exp(static_cast<double>(a_log[head]));
        g[i]                   = -scale * softplus(av + bias);
        beta[i]                = sigmoid(bv);
    }
}

int run_case(std::int32_t tokens, std::uint32_t seed, bool stress_transcendentals, std::int32_t heads = kHeads) {
    const std::size_t elements = static_cast<std::size_t>(heads) * tokens;
    std::vector<float> a(elements), b(elements), a_log(heads), dt_bias(heads);
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

    Tensor tensor_a(device_a.p, DType::BF16, {heads, tokens});
    Tensor tensor_b(device_b.p, DType::BF16, {heads, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {heads, tokens});

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

/// Kimi Delta Attention's gates: a decay per key channel, bounded by a logistic rather than
/// unbounded by a softplus, and an update gate per head.
int run_kda_case(std::int32_t heads, std::int32_t head_dim, std::int32_t tokens, float lower_bound,
                 std::uint32_t seed) {
    const std::size_t width  = static_cast<std::size_t>(heads) * head_dim;
    const std::size_t decays = width * tokens;
    const std::size_t betas  = static_cast<std::size_t>(heads) * tokens;
    std::vector<float> a(decays), b(betas), a_log(heads), dt_bias(width);
    fill_uniform(a, seed, -6.0f, 6.0f);
    fill_uniform(b, seed + 1, -6.0f, 6.0f);
    // A_log is exponentiated, so keep it in the range a trained checkpoint's occupies rather
    // than a range where exp() alone decides the answer.
    fill_uniform(a_log, seed + 2, -2.0f, 1.0f);
    fill_uniform(dt_bias, seed + 3, -2.0f, 2.0f);
    round_to_bf16(a);
    round_to_bf16(b);

    std::vector<double> reference_g(decays), reference_beta(betas);
    for (std::size_t i = 0; i < decays; ++i) {
        const std::size_t within = i % width;
        const std::size_t head   = within / static_cast<std::size_t>(head_dim);
        const double value = static_cast<double>(a[i]) + static_cast<double>(dt_bias[within]);
        reference_g[i] =
            static_cast<double>(lower_bound) * sigmoid(std::exp(static_cast<double>(a_log[head])) * value);
    }
    for (std::size_t i = 0; i < betas; ++i) { reference_beta[i] = sigmoid(b[i]); }

    const auto a_bits = bf16_bits(a), b_bits = bf16_bits(b);
    GuardedDeviceBuffer device_a(a_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_b(b_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_a_log(a_log.size() * sizeof(float));
    GuardedDeviceBuffer device_dt_bias(dt_bias.size() * sizeof(float));
    GuardedDeviceBuffer device_g(decays * sizeof(float));
    GuardedDeviceBuffer device_beta(betas * sizeof(float));
    device_a.copy_from_host(a_bits.data(), device_a.bytes());
    device_b.copy_from_host(b_bits.data(), device_b.bytes());
    device_a_log.copy_from_host(a_log.data(), device_a_log.bytes());
    device_dt_bias.copy_from_host(dt_bias.data(), device_dt_bias.bytes());

    Tensor at(device_a.data(), DType::BF16, {static_cast<std::int32_t>(width), tokens});
    Tensor bt(device_b.data(), DType::BF16, {heads, tokens});
    Tensor a_log_t(device_a_log.data(), DType::FP32, {heads});
    Tensor dt_bias_t(device_dt_bias.data(), DType::FP32, {static_cast<std::int32_t>(width)});
    Tensor gt(device_g.data(), DType::FP32, {static_cast<std::int32_t>(width), tokens});
    Tensor beta_t(device_beta.data(), DType::FP32, {heads, tokens});
    ops::kda_gating(at, bt, a_log_t, dt_bias_t, lower_bound, gt, beta_t, nullptr);
    cuda_synchronize();

    const std::string label = "kda_gating H=" + std::to_string(heads) + " D=" +
                              std::to_string(head_dim) + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_pointwise((label + " g").c_str(), read_fp32(device_g.data(), decays),
                                 reference_g, kKdaGatingFp32);
    failures += verify_pointwise((label + " beta").c_str(), read_fp32(device_beta.data(), betas),
                                 reference_beta, kKdaGatingFp32);
    // The bound is what makes this gate safe: the decay cannot leave [lower_bound, 0), so the
    // state neither grows nor survives forever. Closed at the lower end and open at the upper
    // because that is what FP32 does with a logistic -- sigmoid saturates to exactly 1 for an
    // argument past about 17, and never reaches 0.
    const std::vector<double> got = read_fp32(device_g.data(), decays);
    for (double value : got) {
        if (!(value >= static_cast<double>(lower_bound) && value < 0.0)) {
            std::cerr << label << ": a decay of " << value << " is outside ["
                      << lower_bound << ", 0)\n";
            failures += 1;
            break;
        }
    }
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_exact((label + " a immutable").c_str(),
                             from_device<std::uint16_t>(device_a.data(), a_bits.size()), a_bits);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias.data(), dt_bias.size()), dt_bias);
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

    for (int heads : {1, 7, 16, 32, 64}) {
        failures += run_case(1, 0xA01u, false, heads);
        failures += run_case(129, 0xA02u, true, heads);
    }

    // GLM-5.3's shape, and two others so the head/channel split is exercised rather than
    // assumed: the decay is indexed by channel and the update gate by head.
    failures += run_kda_case(64, 128, 1, -5.0f, 0x606u);
    failures += run_kda_case(64, 128, 37, -5.0f, 0x707u);
    failures += run_kda_case(4, 8, 5, -2.5f, 0x808u);
    failures += run_kda_case(8, 64, 129, -5.0f, 0x909u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating correctness\n";
    return failures == 0 ? 0 : 1;
}
