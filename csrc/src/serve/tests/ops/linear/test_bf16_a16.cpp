#include "ninfer/ops/linear.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

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

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

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
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
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

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = run_bf16_linear();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
