// Public-contract qualification for the low-rank hyper-connection mixer (Qwen3.8-Flash-Next).
//
// The oracle evaluates the documented math in FP64 from the represented BF16 inputs. It rounds
// where the contract stores BF16 -- the normalised projection operand, the low-rank activations
// and the gate logits -- because those roundings are part of what the op computes; everything
// else (the mixed input's n, the inject dot) is FP64, since the op keeps them in FP32.
//
// The mixer spreads a token over several blocks and sums the inject dot from per-stream shares,
// so beside the pointwise comparison it is asked to be bit-reproducible, to give the same mixed
// input whether or not the inject gates are requested, and to leave its inputs untouched.
//
// The low-rank projections may also be W8 row-split. Their oracle weights are then the BF16
// rounding of each dequantised value, the matrix every A16 W8 GEMM multiplies; the cases cover
// the narrow rounds the mixer's own W8 kernels serve and a wide one it hands to cuBLASLt.
#include "api/ops/hyper_connection.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr PointwiseCriterion mixed_criterion() {
    return {/*absolute*/ 3.0e-3, /*relative*/ 1.0e-2};
}

constexpr PointwiseCriterion inject_criterion() {
    return {/*absolute*/ 2.0e-5, /*relative*/ 2.0e-5};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

double round_bf16(double value) { return bf16_to_f32(f32_to_bf16(static_cast<float>(value))); }

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

Weight bf16_weight(void* data, std::size_t bytes, int n, int k) {
    Weight weight;
    weight.qtype           = QType::BF16_CTRL;
    weight.payload         = data;
    weight.qdata           = data;
    weight.payload_bytes   = bytes;
    weight.n               = n;
    weight.k               = k;
    weight.ndim            = 2;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.layout          = QuantLayout::Contiguous;
    return weight;
}

struct Case {
    int streams;
    int hidden;
    int low_rank;
    int tokens;
    std::uint32_t seed;
    const char* label;
    bool w8 = false;
};

// A projection's device weight in the case's format, and the matrix the oracle multiplies.
struct Projection {
    std::vector<float> oracle;
    std::vector<std::uint8_t> bytes;
    quantized_weight::PackedWeight packed;
    bool w8 = false;
};

Projection make_projection(const std::vector<float>& source, int n, int k, bool w8) {
    Projection out;
    out.w8 = w8;
    if (!w8) {
        out.oracle = source;
        const auto bits = encode_bf16(source);
        out.bytes.resize(bits.size() * sizeof(std::uint16_t));
        std::memcpy(out.bytes.data(), bits.data(), out.bytes.size());
        return out;
    }
    out.packed = quantized_weight::pack_w8g32_row_split(source, n, k);
    out.bytes  = out.packed.payload;
    out.oracle = out.packed.dequant;
    round_to_bf16(out.oracle);
    return out;
}

Weight device_projection(const Projection& projection, GuardedDeviceBuffer& buffer, int n, int k) {
    return projection.w8 ? projection.packed.device_weight(buffer.data())
                         : bf16_weight(buffer.data(), buffer.bytes(), n, k);
}

int run_case(const Case& item) {
    const int streams = item.streams, hidden = item.hidden, rank = item.low_rank;
    const int width = streams * hidden, tokens = item.tokens;
    constexpr double kEps = 1.0e-6;

    std::vector<float> residual(static_cast<std::size_t>(width) * tokens), gamma(width);
    std::vector<float> down(static_cast<std::size_t>(rank) * width), up(static_cast<std::size_t>(width) * rank);
    std::vector<float> inject_w(static_cast<std::size_t>(streams) * width);
    fill_uniform(residual, item.seed, -3.0f, 3.0f);
    fill_uniform(gamma, item.seed + 1, 0.5f, 1.5f);
    // Small enough that the projections land where a checkpoint's do, not in saturated sigmoids.
    fill_uniform(down, item.seed + 2, -0.02f, 0.02f);
    fill_uniform(up, item.seed + 3, -0.05f, 0.05f);
    fill_uniform(inject_w, item.seed + 4, -0.02f, 0.02f);
    round_to_bf16(residual);
    round_to_bf16(down);
    round_to_bf16(up);
    round_to_bf16(inject_w);
    const Projection down_weight = make_projection(down, rank, width, item.w8);
    const Projection up_weight   = make_projection(up, width, rank, item.w8);
    down = down_weight.oracle;
    up   = up_weight.oracle;

    std::vector<double> expected_mixed(static_cast<std::size_t>(hidden) * tokens);
    std::vector<double> expected_inject(static_cast<std::size_t>(streams) * tokens);
    for (int t = 0; t < tokens; ++t) {
        const float* x = residual.data() + static_cast<std::size_t>(t) * width;
        std::vector<double> n(width), operand(width);
        for (int s = 0; s < streams; ++s) {
            double sum_sq = 0.0;
            for (int d = 0; d < hidden; ++d) { sum_sq += static_cast<double>(x[s * hidden + d]) * x[s * hidden + d]; }
            const double inv = 1.0 / std::sqrt(sum_sq / hidden + kEps);
            for (int d = 0; d < hidden; ++d) {
                const int i = s * hidden + d;
                n[i]        = x[i] * inv * gamma[i];
                operand[i]  = round_bf16(n[i]);
            }
        }
        std::vector<double> low(rank);
        for (int r = 0; r < rank; ++r) {
            double dot = 0.0;
            for (int i = 0; i < width; ++i) { dot += down[static_cast<std::size_t>(r) * width + i] * operand[i]; }
            const double v = round_bf16(dot) / streams;
            low[r]         = round_bf16(v / (1.0 + std::exp(-v)));
        }
        for (int d = 0; d < hidden; ++d) {
            double acc = 0.0;
            for (int s = 0; s < streams; ++s) {
                const int i  = s * hidden + d;
                double logit = 0.0;
                for (int r = 0; r < rank; ++r) { logit += up[static_cast<std::size_t>(i) * rank + r] * low[r]; }
                acc += sigmoid(round_bf16(logit)) * n[i];
            }
            expected_mixed[static_cast<std::size_t>(t) * hidden + d] = acc / streams;
        }
        for (int row = 0; row < streams; ++row) {
            double dot = 0.0;
            for (int i = 0; i < width; ++i) { dot += inject_w[static_cast<std::size_t>(row) * width + i] * n[i]; }
            expected_inject[static_cast<std::size_t>(t) * streams + row] = 2.0 * sigmoid(dot / streams);
        }
    }

    const auto residual_bits = encode_bf16(residual), inject_bits = encode_bf16(inject_w);
    GuardedDeviceBuffer device_residual(residual_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gamma(gamma.size() * sizeof(float));
    GuardedDeviceBuffer device_down(down_weight.bytes.size());
    GuardedDeviceBuffer device_up(up_weight.bytes.size());
    GuardedDeviceBuffer device_inject_w(inject_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_mixed(expected_mixed.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_inject(expected_inject.size() * sizeof(float));
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    device_gamma.copy_from_host(gamma.data(), device_gamma.bytes());
    device_down.copy_from_host(down_weight.bytes.data(), device_down.bytes());
    device_up.copy_from_host(up_weight.bytes.data(), device_up.bytes());
    device_inject_w.copy_from_host(inject_bits.data(), device_inject_w.bytes());

    ops::HyperConnectionWeights weights{
        Tensor(device_gamma.data(), DType::FP32, {width}),
        device_projection(down_weight, device_down, rank, width),
        device_projection(up_weight, device_up, width, rank),
        bf16_weight(device_inject_w.data(), device_inject_w.bytes(), streams, width)};
    Tensor residual_tensor(device_residual.data(), DType::BF16, {width, tokens});
    Tensor mixed(device_mixed.data(), DType::BF16, {hidden, tokens});
    Tensor inject(device_inject.data(), DType::FP32, {streams, tokens});
    WorkspaceArena arena(ops::hyper_connection_mix_workspace_capacity_bytes(streams, hidden, rank, 1, tokens));
    ops::hyper_connection_mix(residual_tensor, weights, streams, static_cast<float>(kEps), mixed, &inject,
                              arena, nullptr);
    cuda_synchronize();

    const std::string label(item.label);
    int failures = verify_pointwise((label + " mixed").c_str(),
                                    from_device_bf16(device_mixed.data(), expected_mixed.size()),
                                    expected_mixed, mixed_criterion());
    const std::vector<float> got_inject = from_device<float>(device_inject.data(), expected_inject.size());
    failures += verify_pointwise((label + " inject").c_str(),
                                 std::vector<double>(got_inject.begin(), got_inject.end()),
                                 expected_inject, inject_criterion());

    // The same input gives the same bits, and the mixed input does not depend on whether the
    // inject gates were asked for. Shorter rounds reuse the arena sized for the longest.
    const auto mixed_bits  = from_device<std::uint16_t>(device_mixed.data(), expected_mixed.size());
    const auto inject_bits_out = from_device<std::uint32_t>(device_inject.data(), expected_inject.size());
    for (const int prefix : {1, std::max(1, tokens / 2), tokens}) {
        Tensor prefix_residual(device_residual.data(), DType::BF16, {width, prefix});
        Tensor prefix_mixed(device_mixed.data(), DType::BF16, {hidden, prefix});
        Tensor prefix_inject(device_inject.data(), DType::FP32, {streams, prefix});
        ops::hyper_connection_mix(prefix_residual, weights, streams, static_cast<float>(kEps),
                                  prefix_mixed, &prefix_inject, arena, nullptr);
    }
    for (int repeat = 0; repeat < 8; ++repeat) {
        ops::hyper_connection_mix(residual_tensor, weights, streams, static_cast<float>(kEps), mixed,
                                  &inject, arena, nullptr);
        cuda_synchronize();
        if (mixed_bits != from_device<std::uint16_t>(device_mixed.data(), mixed_bits.size()) ||
            inject_bits_out != from_device<std::uint32_t>(device_inject.data(), inject_bits_out.size())) {
            std::cerr << label << ": the mixer was not bit-reproducible\n";
            ++failures;
            break;
        }
    }
    ops::hyper_connection_mix(residual_tensor, weights, streams, static_cast<float>(kEps), mixed, nullptr,
                              arena, nullptr);
    cuda_synchronize();
    failures += verify_exact((label + " mixed without inject").c_str(),
                             from_device<std::uint16_t>(device_mixed.data(), mixed_bits.size()), mixed_bits);
    failures += verify_exact((label + " residual unchanged").c_str(),
                             from_device<std::uint16_t>(device_residual.data(), residual_bits.size()),
                             residual_bits);
    failures += device_down.verify_guards("hc down");
    failures += device_up.verify_guards("hc up");
    failures += device_residual.verify_guards("hc residual");
    failures += device_mixed.verify_guards("hc mixed");
    failures += device_inject.verify_guards("hc inject");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    const Case cases[] = {
        // Qwen3.8-Flash-Next: four streams over 2560 with a 320-wide low rank, at decode and in
        // a prefill round.
        {4, 2560, 320, 1, 31u, "hc decode, 4 streams x 2560"},
        {4, 2560, 320, 37, 37u, "hc prefill 37 tokens"},
        // A hidden width that is not a multiple of the block, and the geometry's other ends.
        {2, 1032, 64, 5, 41u, "hc two streams, ragged hidden"},
        {3, 768, 32, 3, 43u, "hc three streams"},
        {8, 512, 64, 2, 47u, "hc maximum streams"},
        {1, 256, 8, 4, 53u, "hc one stream"},
        // W8 projections: decode, a speculative verify round, a ragged multi-tile round on the
        // mixer's kernels, and a prefill-width round past them.
        {4, 2560, 320, 1, 59u, "hc W8 decode, 4 streams x 2560", true},
        {4, 2560, 320, 4, 61u, "hc W8 verify 4 tokens", true},
        {4, 2560, 320, 37, 67u, "hc W8 37 tokens", true},
        {4, 2560, 320, 100, 71u, "hc W8 prefill 100 tokens", true},
        {2, 1032, 64, 5, 73u, "hc W8 two streams, ragged hidden", true},
        {8, 512, 64, 2, 79u, "hc W8 maximum streams", true},
        {1, 256, 8, 3, 83u, "hc W8 one stream, low rank 8", true},
    };
    for (const Case& item : cases) { failures += run_case(item); }
    if (failures != 0) {
        std::cerr << failures << " hyper_connection check(s) failed\n";
        return 1;
    }
    std::cout << "hyper_connection: PASS\n";
    return 0;
}
