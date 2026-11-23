#include "ops/parallel_rows.h"
#include "api/ops/linear.h"
#include "api/ops/linear_bias.h"

#include "core/device.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits(std::int32_t hidden, std::int32_t tokens) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            const int centered = ((column * 29 + token * 71 + 17) & 0xff) - 128;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    sinfer::test::parallel_rows(weight.n, [&](std::int32_t row) {
        result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
    });
    return result;
}

std::vector<std::int32_t> sampled_rows(std::int32_t rows) {
    std::vector<std::int32_t> result{0, 1, rows / 4, rows / 2, (3 * rows) / 4, rows - 2, rows - 1};
    if (rows == 14336) {
        result.insert(result.end(), {1023, 6143, 6144, 7167, 7168, 13311, 13312});
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int run_bf16_linear_case(DeviceWeight& weight, std::int32_t tokens) {
    const std::int32_t rows                          = weight.host.n;
    const std::int32_t hidden                        = weight.host.k;
    const std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    const std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(rows) * tokens *
                                       sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    DeviceArena workspace(256);
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " T=" + std::to_string(tokens);
    int failures             = guarded_output.verify_guards("BF16_A16 Linear output" + suffix);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < output_bits.size(); ++index) {
        const std::uint16_t bits = output_bits[index];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output" << suffix << " element " << index
                      << " is not finite\n";
            ++failures;
            break;
        }
    }

    std::vector<double> actual;
    std::vector<double> expected;
    if (tokens == 1) {
        const std::vector<double> complete =
            oracle_all_rows(weight.host, std::span<const float>(activation));
        actual.reserve(rows);
        for (const std::uint16_t bits : output_bits) { actual.push_back(bf16_to_f32(bits)); }
        expected = complete;
    } else {
        const std::vector<std::int32_t> sampled       = sampled_rows(rows);
        const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
        actual.reserve(sampled.size() * token_samples.size());
        expected.reserve(actual.capacity());
        for (const std::int32_t row : sampled) {
            for (const std::int32_t token : token_samples) {
                actual.push_back(
                    bf16_to_f32(output_bits[static_cast<std::size_t>(token) * rows + row]));
                expected.push_back(dot_fp64(
                    weight.host, row,
                    std::span<const float>(
                        activation.data() + static_cast<std::size_t>(token) * hidden, hidden)));
            }
        }
    }
    failures += verify_reduction("BF16_A16 Linear [" + std::to_string(rows) + "," +
                                     std::to_string(hidden) + "]" + suffix,
                                 actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear" << suffix << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight" + suffix);
    return failures;
}

int run_bf16_linear() {
    int failures = 0;
    DeviceWeight attention_weight(make_patterned(14336, 5120, 401U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 17, 27, 28, 32, 33, 128, 129, 1024}) {
        failures += run_bf16_linear_case(attention_weight, tokens);
    }
    DeviceWeight output_weight(make_patterned(5120, 6144, 409U));
    for (const std::int32_t tokens : {1, 2, 4, 8, 16, 27, 28, 32, 33, 127, 128, 129, 1024, 1536}) {
        failures += run_bf16_linear_case(output_weight, tokens);
    }
    return failures;
}

// The fused routes a vision tower's blocks take (ops::try_linear_epilogue,
// ops::try_linear_bias_accumulate), at its four shapes, against an FP64 reference.
enum class FusedKind { Bias, GeluTanhBias, BiasAccumulate };

double gelu_tanh(double value) {
    constexpr double kSqrt2OverPi = 0.79788456080286535588;
    return 0.5 * value * (1.0 + std::tanh(kSqrt2OverPi * (value + 0.044715 * value * value * value)));
}

int run_bf16_fused_case(DeviceWeight& weight, std::int32_t tokens, FusedKind kind) {
    const std::int32_t rows                          = weight.host.n;
    const std::int32_t hidden                        = weight.host.k;
    const std::vector<std::uint16_t> activation_bits = make_activation_bits(hidden, tokens);
    const std::vector<float> activation              = materialize(activation_bits);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    std::vector<std::uint16_t> bias_bits(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        bias_bits[static_cast<std::size_t>(row)] =
            f32_to_bf16(static_cast<float>(((row * 37 + 11) % 97) - 48) * (1.0F / 64.0F));
    }
    DeviceBuffer device_bias = to_device(bias_bits);
    // The accumulating form starts from a residual; the others must overwrite whatever is there.
    std::vector<std::uint16_t> initial_bits(static_cast<std::size_t>(rows) * tokens);
    for (std::size_t index = 0; index < initial_bits.size(); ++index) {
        initial_bits[index] =
            kind == FusedKind::BiasAccumulate
                ? f32_to_bf16(static_cast<float>(static_cast<int>((index * 13 + 5) % 61) - 30) *
                              (1.0F / 16.0F))
                : static_cast<std::uint16_t>(0x7fc0); // NaN: must not leak through
    }
    GuardedDeviceBuffer guarded_output(initial_bits.size() * sizeof(std::uint16_t));
    guarded_output.fill(0xff);
    CUDA_CHECK(cudaMemcpy(guarded_output.data(), initial_bits.data(),
                          initial_bits.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice));

    Tensor x(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor bias(device_bias.p, DType::BF16, {rows});
    Tensor output(guarded_output.data(), DType::BF16, {rows, tokens});
    const char* name = kind == FusedKind::Bias           ? "bias"
                       : kind == FusedKind::GeluTanhBias ? "gelu_tanh_bias"
                                                         : "bias_accumulate";
    const std::string label = std::string("BF16 fused ") + name + " [" + std::to_string(rows) +
                              "," + std::to_string(hidden) + "] T=" + std::to_string(tokens);
    const bool served =
        kind == FusedKind::BiasAccumulate
            ? ops::try_linear_bias_accumulate(x, weight.view(), bias, output, nullptr)
            : ops::try_linear_epilogue(x, weight.view(), bias,
                                       kind == FusedKind::Bias
                                           ? ops::LinearEpilogue::Bias
                                           : ops::LinearEpilogue::GeluTanhBias,
                                       output, nullptr);
    cuda_synchronize();
    if (!served) {
        // A plain contiguous BF16 weight with no adapter is exactly what the route exists for.
        std::cerr << label << ": the fused route declined\n";
        return 1;
    }

    int failures = guarded_output.verify_guards(label);
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), initial_bits.size());
    std::vector<double> actual;
    std::vector<double> expected;
    for (const std::int32_t row : sampled_rows(rows)) {
        for (const std::int32_t token : sampled_tokens(tokens)) {
            const std::size_t at = static_cast<std::size_t>(token) * rows + row;
            double value = dot_fp64(weight.host, row,
                                    std::span<const float>(activation.data() +
                                                               static_cast<std::size_t>(token) *
                                                                   hidden,
                                                           hidden)) +
                           bf16_to_f32(bias_bits[static_cast<std::size_t>(row)]);
            if (kind == FusedKind::GeluTanhBias) { value = gelu_tanh(value); }
            if (kind == FusedKind::BiasAccumulate) { value += bf16_to_f32(initial_bits[at]); }
            actual.push_back(bf16_to_f32(output_bits[at]));
            expected.push_back(value);
        }
    }
    failures += verify_reduction(label, actual, expected, kA16Tolerance);
    if (from_device<std::uint16_t>(device_activation, activation_bits.size()) != activation_bits) {
        std::cerr << label << " modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved(label + " weight");
    return failures;
}

int run_bf16_fused() {
    int failures = 0;
    // The 1152-wide vision tower: qkv, attention output, fc1, fc2.
    struct Shape {
        std::int32_t n, k;
        FusedKind kinds[2];
        int kind_count;
    };
    const Shape shapes[] = {
        {3456, 1152, {FusedKind::Bias}, 1},
        {1152, 1152, {FusedKind::BiasAccumulate, FusedKind::Bias}, 2},
        {4304, 1152, {FusedKind::GeluTanhBias}, 1},
        {1152, 4304, {FusedKind::BiasAccumulate}, 1},
    };
    std::uint32_t seed = 421U;
    for (const Shape& shape : shapes) {
        DeviceWeight weight(make_patterned(shape.n, shape.k, seed++));
        for (int index = 0; index < shape.kind_count; ++index) {
            for (const std::int32_t tokens : {1, 17, 784, 3136}) {
                failures += run_bf16_fused_case(weight, tokens, shape.kinds[index]);
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (sinfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int linear_failures = run_bf16_linear();
        std::cout << (linear_failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        const int fused_failures = run_bf16_fused();
        std::cout << (fused_failures == 0 ? "OK" : "FAIL") << " BF16 fused epilogue/accumulate\n";
        return linear_failures + fused_failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
