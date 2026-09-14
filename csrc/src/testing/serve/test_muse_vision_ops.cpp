#include "api/ops/linear_bias.h"
#include "family/muse_vision.h"
#include "ops/op_tester.h"
#include <iostream>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

int main() {
    if (cuda_unavailable()) {
        return 77;
    }
    constexpr int h = 96, n = 192, t = 12;
    std::vector<float> weights(h * n), inputs(h * t), biases(n);
    fill_uniform(weights, 211, -.5F, .5F);
    fill_uniform(inputs, 212, -2.F, 2.F);
    fill_uniform(biases, 213, -1.F, 1.F);
    round_to_bf16(weights);
    round_to_bf16(inputs);
    round_to_bf16(biases);
    const auto upload = [](const std::vector<float>& values, GuardedDeviceBuffer& buffer) {
        std::vector<std::uint16_t> bits(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            bits[i] = f32_to_bf16(values[i]);
        }
        buffer.copy_from_host(bits.data(), buffer.bytes());
    };
    GuardedDeviceBuffer dw(weights.size() * 2), dx(inputs.size() * 2), db(biases.size() * 2), dy(n * t * 2),
        ds(h * t * 2);
    upload(weights, dw);
    upload(inputs, dx);
    upload(biases, db);
    Weight weight;
    weight.qtype = QType::BF16_CTRL;
    weight.layout = QuantLayout::Contiguous;
    weight.ndim = 2;
    weight.n = n;
    weight.k = h;
    weight.qdata = dw.data();
    Tensor x(dx.data(), DType::BF16, {h, t}), bias(db.data(), DType::BF16, {n}), y(dy.data(), DType::BF16, {n, t});
    ops::linear_bias(x, weight, bias, y, nullptr);
    cuda_synchronize();
    std::vector<double> expected(n * t);
    for (int token = 0; token < t; ++token)
        for (int row = 0; row < n; ++row) {
            double sum = biases[row];
            for (int k = 0; k < h; ++k) {
                sum += double(weights[row * h + k]) * inputs[token * h + k];
            }
            expected[token * n + row] = sum;
        }
    int failures = verify_pointwise("Muse bias before BF16 rounding",
                                    from_device_bf16(dy.data(), n * t),
                                    expected,
                                    PointwiseCriterion{1.e-5, 3.95e-3});
    Tensor shuffled(ds.data(), DType::BF16, {h * 4, t / 4});
    family::muse_pixel_shuffle(x, shuffled, nullptr);
    cuda_synchronize();
    const auto got = from_device_bf16(ds.data(), h * t);
    for (int token = 0; token < t / 4; ++token)
        for (int channel = 0; channel < h; ++channel)
            for (int neighbor = 0; neighbor < 4; ++neighbor) {
                if (got[token * h * 4 + channel * 4 + neighbor] != inputs[(token * 4 + neighbor) * h + channel]) {
                    ++failures;
                }
            }
    std::vector<int> positions(t * 2);
    for (int i = 0; i < t; ++i) {
        positions[i] = i + 1;
        positions[t + i] = 33 - i;
    }
    GuardedDeviceBuffer dp(positions.size() * sizeof(int));
    dp.copy_from_host(positions.data(), dp.bytes());
    upload(inputs, ds);
    Tensor q = x.view({h, 1, t}), k(ds.data(), DType::BF16, {h, 1, t}), pos(dp.data(), DType::I32, {t, 2});
    std::vector<double> rotated(h * t);
    for (int token = 0; token < t; ++token)
        for (int pair = 0; pair < h / 2; ++pair) {
            const double angle =
                positions[(pair / (h / 4)) * t + token] * std::pow(10000., -2. * (pair % (h / 4)) / (h / 2));
            const float cosine = bf16_to_f32(f32_to_bf16(float(std::cos(angle))));
            const float sine = bf16_to_f32(f32_to_bf16(float(std::sin(angle))));
            const int at = token * h + pair;
            rotated[at] = double(inputs[at]) * cosine - double(inputs[at + h / 2]) * sine;
            rotated[at + h / 2] = double(inputs[at + h / 2]) * cosine + double(inputs[at]) * sine;
        }
    family::muse_vision_rope(pos, 10000.F, q, k, nullptr);
    cuda_synchronize();
    // As in the shared RoPE tests, measure cancellation against the input-pair
    // norm. BF16 tables and BF16 output each contribute rounding.
    for (auto* data : {dx.data(), ds.data()}) {
        const auto actual = from_device_bf16(data, h * t);
        for (int token = 0; token < t; ++token)
            for (int pair = 0; pair < h / 2; ++pair) {
                const int at = token * h + pair;
                const double bound = 6.9e-3 * std::hypot(inputs[at], inputs[at + h / 2]);
                if (std::abs(actual[at] - rotated[at]) > bound ||
                    std::abs(actual[at + h / 2] - rotated[at + h / 2]) > bound) {
                    ++failures;
                }
            }
    }
    bool rejected = false;
    Weight square = weight;
    square.n = h;
    Tensor small_bias(db.data(), DType::BF16, {h});
    try {
        ops::linear_bias(x, square, small_bias, x, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        ++failures;
    }
    failures += dy.verify_guards("biased projection guards");
    failures += ds.verify_guards("pixel shuffle guards");
    std::cout << (failures ? "FAIL" : "OK") << " Muse vision ops\n";
    return failures ? 1 : 0;
}
