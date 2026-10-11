#include "api/ops/dflash2.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

std::vector<float> bf16_values(std::size_t count, std::uint32_t seed, float lo, float hi) {
    std::vector<float> values(count);
    fill_uniform(values, seed, lo, hi);
    round_to_bf16(values);
    return values;
}

// One BF16 rounding of an fp32 result: the kernel may contract its products into FMAs, so the
// two sides can round the fp32 value differently by an ulp before that.
int verify_bf16(const std::string& label, const std::vector<double>& got, const std::vector<double>& expected) {
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double limit = std::ldexp(std::abs(expected[i]), -7) + 1e-6;
        if (!(std::abs(got[i] - expected[i]) <= limit)) {
            std::cerr << label << ": mismatch at " << i << " got " << got[i] << " expected " << expected[i] << '\n';
            return 1;
        }
    }
    return 0;
}

int run_conv(int stage, bool residual) {
    constexpr int hidden = 128, groups = 8, width = 8, batch = 3, columns = width * batch;
    const auto input   = bf16_values(static_cast<std::size_t>(hidden) * columns, 11 + stage, -2.f, 2.f);
    const auto dynamic = bf16_values(static_cast<std::size_t>(4 * groups) * columns, 23 + stage, -.5f, .5f);
    const auto base    = bf16_values(static_cast<std::size_t>(4) * hidden, 31, -1.f, 1.f);
    const auto stream  = bf16_values(static_cast<std::size_t>(hidden) * columns, 47, -3.f, 3.f);

    std::vector<double> expected(static_cast<std::size_t>(hidden) * columns);
    for (int column = 0; column < columns; ++column) {
        const int row = column % width;
        for (int c = 0; c < hidden; ++c) {
            const int g = c / (hidden / groups);
            const float w0 = base[(stage * 2) * hidden + c] + dynamic[column * 4 * groups + (stage * 2) * groups + g];
            const float w1 = base[(stage * 2 + 1) * hidden + c] + dynamic[column * 4 * groups + (stage * 2 + 1) * groups + g];
            float value = input[column * hidden + c] * w0;
            if (row > 0) value += input[(column - 1) * hidden + c] * w1;
            if (residual) value += stream[column * hidden + c];
            expected[static_cast<std::size_t>(column) * hidden + c] = bf16_to_f32(f32_to_bf16(value));
        }
    }

    DeviceBuffer input_d = to_device_bf16(input), dynamic_d = to_device_bf16(dynamic), base_d = to_device_bf16(base);
    DeviceBuffer out_d   = to_device_bf16(stream);
    Tensor input_t(input_d.p, DType::BF16, {hidden, columns});
    Tensor dynamic_t(dynamic_d.p, DType::BF16, {4 * groups, columns});
    Tensor base_t(base_d.p, DType::BF16, {hidden, 2, 2});
    Tensor out_t(out_d.p, DType::BF16, {hidden, columns});
    // With a residual the stage adds into the stream in place, as the draft forward runs it.
    ops::dflash_conv(input_t, dynamic_t, base_t, stage, width, residual ? &out_t : nullptr, out_t, nullptr);
    cuda_synchronize();
    return verify_bf16("dflash_conv stage=" + std::to_string(stage) + (residual ? " residual" : ""),
                       from_device_bf16(out_d, expected.size()), expected);
}

int run_top_k(int rows, int domain, int k, std::uint32_t seed) {
    constexpr int columns = 5;
    // A narrow range on the BF16 grid gives plenty of exact ties for the id order to settle.
    auto logits = bf16_values(static_cast<std::size_t>(rows) * columns, seed, -4.f, 4.f);
    for (int column = 0; column < columns; ++column) {
        logits[static_cast<std::size_t>(column) * rows + (7 * column + 3) % domain] = std::numeric_limits<float>::quiet_NaN();
    }
    std::vector<int> expected_ids;
    std::vector<float> expected_values;
    for (int column = 0; column < columns; ++column) {
        std::vector<int> order(static_cast<std::size_t>(domain));
        std::iota(order.begin(), order.end(), 0);
        const auto value = [&](int row) {
            const float v = logits[static_cast<std::size_t>(column) * rows + row];
            return std::isnan(v) ? -std::numeric_limits<float>::infinity() : v;
        };
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return value(a) > value(b); });
        for (int i = 0; i < k; ++i) {
            const bool real = i < domain;
            expected_ids.push_back(real ? order[static_cast<std::size_t>(i)] : 0);
            expected_values.push_back(real ? value(order[static_cast<std::size_t>(i)])
                                           : -std::numeric_limits<float>::infinity());
        }
    }
    DeviceBuffer logits_d = to_device_bf16(logits);
    GuardedDeviceBuffer ids(static_cast<std::size_t>(k) * columns * sizeof(std::int32_t));
    GuardedDeviceBuffer values(static_cast<std::size_t>(k) * columns * sizeof(float));
    Tensor logits_t(logits_d.p, DType::BF16, {rows, columns});
    Tensor ids_t(ids.data(), DType::I32, {k, columns});
    Tensor values_t(values.data(), DType::FP32, {k, columns});
    ops::dflash_select_top_k(logits_t, domain, ids_t, values_t, nullptr);
    cuda_synchronize();
    const std::string label = "dflash_select_top_k R=" + std::to_string(rows) + " domain=" +
                              std::to_string(domain) + " K=" + std::to_string(k);
    int failures = verify_exact((label + " ids").c_str(),
                                from_device<std::int32_t>(ids.data(), expected_ids.size()), expected_ids);
    failures += verify_exact((label + " values").c_str(),
                             from_device<float>(values.data(), expected_values.size()), expected_values);
    failures += ids.verify_guards((label + " ids guards").c_str());
    failures += values.verify_guards((label + " values guards").c_str());
    return failures;
}

int run_walk(int k, int dims) {
    constexpr int vocab = 300, positions = 4, batch = 3, columns = positions * batch;
    const auto predecessor = bf16_values(static_cast<std::size_t>(vocab) * dims, 5, -1.f, 1.f);
    const auto successor   = bf16_values(static_cast<std::size_t>(vocab) * dims, 6, -1.f, 1.f);
    const auto hidden      = bf16_values(static_cast<std::size_t>(dims) * columns, 7, -1.f, 1.f);
    std::vector<float> values(static_cast<std::size_t>(k) * columns);
    fill_uniform(values, 8, -1.f, 1.f);
    std::vector<int> ids(static_cast<std::size_t>(k) * columns);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int>((i * 37 + 11) % vocab);
    const std::vector<int> anchors{3, 150, 299};

    std::vector<int> expected(static_cast<std::size_t>(columns));
    for (int b = 0; b < batch; ++b) {
        int previous = anchors[static_cast<std::size_t>(b)];
        for (int p = 0; p < positions; ++p) {
            const int column = b * positions + p;
            int best = 0;
            double best_score = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < k; ++j) {
                const int candidate = ids[static_cast<std::size_t>(column) * k + j];
                double edge = 0.0;
                for (int d = 0; d < dims; ++d) {
                    edge += double(predecessor[static_cast<std::size_t>(previous) * dims + d]) *
                            hidden[static_cast<std::size_t>(column) * dims + d] *
                            successor[static_cast<std::size_t>(candidate) * dims + d];
                }
                const double score = values[static_cast<std::size_t>(column) * k + j] + edge;
                if (score > best_score) {
                    best_score = score;
                    best = j;
                }
            }
            previous = ids[static_cast<std::size_t>(column) * k + best];
            expected[static_cast<std::size_t>(column)] = previous;
        }
    }
    DeviceBuffer anchors_d = to_device(anchors), ids_d = to_device(ids), values_d = to_device(values);
    DeviceBuffer hidden_d = to_device_bf16(hidden), predecessor_d = to_device_bf16(predecessor);
    DeviceBuffer successor_d = to_device_bf16(successor);
    GuardedDeviceBuffer drafts(static_cast<std::size_t>(columns) * sizeof(std::int32_t));
    Tensor anchors_t(anchors_d.p, DType::I32, {batch});
    Tensor ids_t(ids_d.p, DType::I32, {k, columns});
    Tensor values_t(values_d.p, DType::FP32, {k, columns});
    Tensor hidden_t(hidden_d.p, DType::BF16, {dims, columns});
    Tensor predecessor_t(predecessor_d.p, DType::BF16, {dims, vocab});
    Tensor successor_t(successor_d.p, DType::BF16, {dims, vocab});
    Tensor drafts_t(drafts.data(), DType::I32, {columns});
    ops::dflash_select_walk(anchors_t, ids_t, values_t, hidden_t, predecessor_t, successor_t, positions,
                            drafts_t, nullptr);
    cuda_synchronize();
    const std::string label = "dflash_select_walk K=" + std::to_string(k) + " D=" + std::to_string(dims);
    int failures = verify_exact(label.c_str(), from_device<std::int32_t>(drafts.data(), expected.size()), expected);
    failures += drafts.verify_guards((label + " guards").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "dflash2: SKIP (CUDA unavailable)\n";
        return 77;
    }
    int failures = 0;
    failures += run_conv(0, false);
    failures += run_conv(1, true);
    failures += run_top_k(1024, 1024, 16, 1);
    failures += run_top_k(1024, 997, 16, 2);
    failures += run_top_k(1003, 1001, 5, 3);
    failures += run_top_k(4096, 4000, 32, 4);
    failures += run_top_k(64, 10, 16, 5);  // fewer rows than candidates: padded with row 0 at -inf
    failures += run_walk(16, 64);
    failures += run_walk(32, 256);
    failures += run_walk(1, 32);
    if (failures != 0) {
        std::cerr << "dflash2 failures=" << failures << '\n';
        return 1;
    }
    std::cout << "dflash2: PASS\n";
    return 0;
}
