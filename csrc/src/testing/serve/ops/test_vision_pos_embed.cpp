#include "api/ops/vision_pos_embed.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

// Full-matrix measurements against the direct FP64 ideal peak at 3.891e-3 relative error. One
// strict pointwise criterion covers every output element in both registered cases; there is no
// tail or normwise escape path.
constexpr PointwiseCriterion kVisionPositionEmbeddingTolerance{
    /*absolute=*/1.0e-5,
    /*relative=*/3.95e-3,
};

std::vector<std::uint16_t> as_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

int run_case(std::int32_t patches, std::uint32_t seed) {
    constexpr std::int32_t kDimensions = 1152;
    constexpr std::int32_t kTableRows  = 2304;

    std::vector<float> table_values(static_cast<std::size_t>(kDimensions) * kTableRows);
    std::vector<float> x_values(static_cast<std::size_t>(kDimensions) * patches);
    std::vector<float> weights(static_cast<std::size_t>(4) * patches);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * patches);
    fill_uniform(table_values, seed, -2.0f, 2.0f);
    fill_uniform(x_values, seed + 1u, -8.0f, 8.0f);
    fill_uniform(weights, seed + 2u, 0.05f, 1.0f);
    round_to_bf16(table_values);
    round_to_bf16(x_values);

    for (std::int32_t patch = 0; patch < patches; ++patch) {
        double sum = 0.0;
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            indices[control] =
                static_cast<std::int32_t>((patch * 37 + corner * 101 + 11) % kTableRows);
            sum += static_cast<double>(weights[control]);
        }
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            weights[control] = static_cast<float>(static_cast<double>(weights[control]) / sum);
        }
    }

    std::vector<double> reference(x_values.size());
    for (std::int32_t patch = 0; patch < patches; ++patch) {
        for (std::int32_t dimension = 0; dimension < kDimensions; ++dimension) {
            double position = 0.0;
            for (std::int32_t corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                position +=
                    static_cast<double>(
                        table_values[static_cast<std::size_t>(indices[control]) * kDimensions +
                                     dimension]) *
                    static_cast<double>(weights[control]);
            }
            const std::size_t output = static_cast<std::size_t>(patch) * kDimensions + dimension;
            reference[output]        = static_cast<double>(x_values[output]) + position;
        }
    }

    const auto table_bits = as_bf16_bits(table_values);
    const auto x_bits     = as_bf16_bits(x_values);
    GuardedDeviceBuffer device_table(table_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(weights.size() * sizeof(float));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_table.copy_from_host(table_bits.data(), table_bits.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_weights.copy_from_host(weights.data(), weights.size() * sizeof(float));
    device_x.copy_from_host(x_bits.data(), x_bits.size() * sizeof(std::uint16_t));

    Tensor table_tensor(device_table.data(), DType::BF16, {kDimensions, kTableRows});
    Tensor indices_tensor(device_indices.data(), DType::I32, {4, patches});
    Tensor weights_tensor(device_weights.data(), DType::FP32, {4, patches});
    Tensor x_tensor(device_x.data(), DType::BF16, {kDimensions, patches});
    ops::vision_pos_embed_add(table_tensor, indices_tensor, weights_tensor, x_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "vision_pos_embed D=1152 R=2304 P=" + std::to_string(patches);
    int failures = verify_pointwise(label.c_str(), from_device_bf16(device_x.data(), x_bits.size()),
                                    reference, kVisionPositionEmbeddingTolerance);
    failures += verify_exact((label + " preserves table").c_str(),
                             from_device<std::uint16_t>(device_table.data(), table_bits.size()),
                             table_bits);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += verify_exact((label + " preserves weights").c_str(),
                             from_device<float>(device_weights.data(), weights.size()), weights);
    failures += device_table.verify_guards((label + " table").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_weights.verify_guards((label + " weights").c_str());
    failures += device_x.verify_guards((label + " x").c_str());
    return failures;
}

// A separable FP64 interpolation oracle, including the BF16 storage between passes.
// Covers downsampling antialiasing, border normalization, rectangular grids and block order.
int run_siglip_case(int height, int width) {
    constexpr int side = 16, channels = 64, merge = 2;
    std::vector<float> table(side * side * channels), input(height * width * channels);
    fill_uniform(table, 81u, -2.0f, 2.0f);
    fill_uniform(input, 82u, -1.0f, 1.0f);
    round_to_bf16(table);
    round_to_bf16(input);
    const auto weights = [](int target) {
        std::vector<double> result(target * side);
        const double scale = static_cast<double>(side) / target;
        for (int out = 0; out < target; ++out) {
            double total = 0;
            for (int in = 0; in < side; ++in) {
                const double distance = std::abs((in + 0.5 - (out + 0.5) * scale) / std::max(1.0, scale));
                total += result[out * side + in] = std::max(0.0, 1.0 - distance);
            }
            for (int in = 0; in < side; ++in) {
                const float raw = bf16_to_f32(f32_to_bf16(result[out * side + in]));
                result[out * side + in] = bf16_to_f32(f32_to_bf16(raw / total));
            }
        }
        return result;
    };
    const auto wx = weights(width), wy = weights(height);
    const auto rounded = [](double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); };
    std::vector<float> horizontal(side * width * channels);
    for (int y = 0; y < side; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < channels; ++c) {
        double sum = 0;
        for (int source = 0; source < side; ++source) {
            sum += wx[x * side + source] * table[(y * side + source) * channels + c];
        }
        horizontal[(y * width + x) * channels + c] = rounded(sum);
    }
    std::vector<double> reference(input.size());
    for (int y = 0; y < height; ++y) for (int x = 0; x < width; ++x) for (int c = 0; c < channels; ++c) {
        double sum = 0;
        for (int source = 0; source < side; ++source) {
            sum += wy[y * side + source] * horizontal[(source * width + x) * channels + c];
        }
        const int patch = ((y / merge) * (width / merge) + x / merge) * merge * merge +
                           (y % merge) * merge + x % merge;
        reference[patch * channels + c] = rounded(input[patch * channels + c] + rounded(sum));
    }
    const auto table_bits = as_bf16_bits(table), input_bits = as_bf16_bits(input);
    GuardedDeviceBuffer dt(table_bits.size() * 2), dx(input_bits.size() * 2);
    dt.copy_from_host(table_bits.data(), table_bits.size() * 2);
    dx.copy_from_host(input_bits.data(), input_bits.size() * 2);
    Tensor t(dt.data(), DType::BF16, {channels, side * side});
    Tensor x(dx.data(), DType::BF16, {channels, height * width});
    ops::siglip2_pos_embed_add(t, height, width, merge, x, nullptr);
    cuda_synchronize();
    const std::string label = "siglip2 position " + std::to_string(height) + "x" + std::to_string(width);
    // FP32 GPU sums can resolve BF16 halfway cases differently from the FP64 oracle.
    int failures = verify_pointwise(label.c_str(), from_device_bf16(dx.data(), input.size()),
                                    reference, PointwiseCriterion{0.015625, 0.0});
    failures += verify_exact((label + " preserves table").c_str(),
        from_device<std::uint16_t>(dt.data(), table_bits.size()), table_bits);
    failures += dt.verify_guards((label + " table").c_str());
    failures += dx.verify_guards((label + " output").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    for (auto [h, w] : {std::pair{8, 8}, {8, 12}, {16, 16}, {24, 32}, {32, 8}}) {
        failures += run_siglip_case(h, w);
    }
    failures += run_case(17, 1u);
    failures += run_case(1024, 11u);
    std::cout << (failures ? "FAIL" : "OK") << " vision_pos_embed\n";
    return failures ? 1 : 0;
}
