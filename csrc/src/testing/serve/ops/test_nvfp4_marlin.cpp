// Weight-only NVFP4 (NVFP4_A16) on Marlin, against the values the stored words stand for.
//
// The load-time rewrite and both GEMMs must reproduce e2m1(code) * e4m3(block) * second level
// with BF16 activations, under the one rule the reference (vLLM's Marlin FP4 path) imposes on top:
// a block scale that, times the weight's power-of-two rescale and 2^7, falls below 2 has no
// S0E5M3 encoding and reads as zero. The dense cases cover every Marlin row-block class (1..8 is
// the 8-row kernel, then 16-row multiples to 64, then the 64-row slices) for a matrix whose
// largest scale is 448 (no rescale, subnormal scales flushed) and one whose largest is small
// (rescaled, so the same subnormals survive). The mixture case runs the public sparse_moe op on
// a weight-only NVFP4 bank at the 35B geometry -- decode, narrow, tiled-router and a round past
// the route's 1,024-token slice -- against an FP64 oracle that takes the round's own top-8 (the
// captured ids, which must be a top-8 of the FP64 logits up to an FP32 router's rounding).

#include "api/ops/linear.h"
#include "api/ops/sparse_moe.h"
#include "core/device.h"
#include "ops/linear/marlin/marlin_nvfp4.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <span>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

constexpr double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

double e2m1(std::uint8_t nibble) {
    const double magnitude = kE2M1[nibble & 7];
    return (nibble & 8) != 0 ? -magnitude : magnitude;
}

double e4m3(std::uint8_t word) {
    const int exponent = (word >> 3) & 0xF;
    const int fraction = word & 7;
    if (exponent == 0) { return fraction * std::ldexp(1.0, -9); }
    return (1.0 + fraction / 8.0) * std::ldexp(1.0, exponent - 7);
}

/// One stored NVFP4_A16 matrix (or bank of `blocks` stacked matrices), as the converter writes
/// it: codes [n, k/2], natural block scales [n, k/16], and a second level per block.
struct StoredNvfp4 {
    std::int32_t n = 0, k = 0, blocks = 1;
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> second;
    double sf = 1.0; // vLLM's power-of-two rescale, over the whole weight

    double scale(std::int32_t row, std::int32_t group) const {
        const double value = e4m3(scales[std::size_t(row) * (k / 16) + group]);
        return value * sf * 128.0 < 2.0 ? 0.0 : value;
    }
    double weight(std::int32_t row, std::int32_t column) const {
        const std::uint8_t byte = codes[std::size_t(row) * (k / 2) + column / 2];
        const std::uint8_t nibble = (column & 1) != 0 ? byte >> 4 : byte & 0xF;
        return e2m1(nibble) * scale(row, column / 16) * second[row / (n / blocks)];
    }
};

StoredNvfp4 make_stored(std::int32_t n, std::int32_t k, std::int32_t blocks, std::uint8_t top_word,
                        std::uint32_t seed, bool shared_pairs = false) {
    StoredNvfp4 w{n, k, blocks, {}, {}, {}, 1.0};
    std::mt19937 generator(seed);
    w.codes.resize(std::size_t(n) * k / 2);
    for (auto& byte : w.codes) { byte = static_cast<std::uint8_t>(generator()); }
    w.scales.resize(std::size_t(n) * k / 16);
    std::uniform_int_distribution<int> normal(0x08, top_word), subnormal(0x01, 0x07);
    for (auto& word : w.scales) {
        word = static_cast<std::uint8_t>((generator() % 10) == 0 ? subnormal(generator) : normal(generator));
    }
    w.scales[generator() % w.scales.size()] = top_word;
    std::uniform_real_distribution<float> level(5.0e-5F, 2.0e-4F);
    w.second.resize(blocks);
    for (auto& value : w.second) { value = level(generator); }
    if (shared_pairs) {
        for (std::int32_t b = 1; b < blocks; b += 2) { w.second[b] = w.second[b - 1]; }
    }
    const double largest = e4m3(*std::max_element(w.scales.begin(), w.scales.end())) * 128.0;
    w.sf = largest < 448.0 * 128.0 ? std::exp2(std::floor(std::log2(448.0 * 128.0 / largest))) : 1.0;
    return w;
}

/// The artifact payload: codes, the 256-aligned swizzled scale plane, the trailing word.
std::vector<std::uint8_t> payload(const StoredNvfp4& w, float trailing) {
    const std::size_t code_bytes  = w.codes.size();
    const std::size_t scale_begin = (code_bytes + 255) / 256 * 256;
    std::vector<std::uint8_t> bytes(scale_begin + w.scales.size() + 4, 0);
    std::memcpy(bytes.data(), w.codes.data(), code_bytes);
    const std::int32_t groups = w.k / 16, k_tiles = w.k / 64;
    for (std::int32_t row = 0; row < w.n; ++row) {
        for (std::int32_t group = 0; group < groups; ++group) {
            const std::size_t offset = (std::size_t(row / 128) * k_tiles + group / 4) * 512 +
                                       (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4;
            bytes[scale_begin + offset] = w.scales[std::size_t(row) * groups + group];
        }
    }
    std::memcpy(bytes.data() + scale_begin + w.scales.size(), &trailing, 4);
    return bytes;
}

Weight stored_weight(const DeviceBuffer& device, const StoredNvfp4& w) {
    Weight out{};
    out.payload         = device.p;
    out.payload_bytes   = device.bytes;
    out.qtype           = QType::NVFP4_A16;
    out.group_size      = 16;
    out.ndim            = 2;
    out.qdata           = device.p;
    out.scales          = static_cast<const std::byte*>(device.p) + (w.codes.size() + 255) / 256 * 256;
    out.n               = w.n;
    out.k               = w.k;
    out.group           = 16;
    out.layout          = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype     = DType::FP8_E4M3FN;
    out.shape[0]        = w.n;
    out.shape[1]        = w.k;
    out.padded_shape[0] = w.n;
    out.padded_shape[1] = w.k;
    return out;
}

const ReductionCriterion kGemmCriterion{4.0e-3, 0.0, 1.0 / 64};

int run_dense(const char* label, std::int32_t n, std::int32_t k, std::uint8_t top_word,
              std::uint32_t seed, DeviceContext& device) {
    const StoredNvfp4 w = make_stored(n, k, 1, top_word, seed);
    DeviceBuffer stored = to_device(payload(w, w.second[0]));
    Weight weight       = stored_weight(stored, w);
    ops::detail::nvfp4_marlin_adopt(weight, 1, nullptr, 1, device.stream);
    if (weight.layout != QuantLayout::MarlinTiles) { throw std::runtime_error("not adopted"); }
    std::vector<double> dequantized(std::size_t(n) * k);
    for (std::int32_t row = 0; row < n; ++row) {
        for (std::int32_t column = 0; column < k; ++column) {
            dequantized[std::size_t(row) * k + column] = w.weight(row, column);
        }
    }
    int failures = 0;
    for (const std::int32_t tokens : {1, 3, 8, 9, 16, 17, 33, 48, 64, 65, 300}) {
        std::vector<float> x(std::size_t(k) * tokens);
        fill_uniform(x, seed + tokens, -1.0F, 1.0F);
        round_to_bf16(x);
        DeviceBuffer dx = to_device_bf16(x), dout(std::size_t(n) * tokens * 2);
        Tensor input(dx.p, DType::BF16, {k, tokens});
        Tensor output(dout.p, DType::BF16, {n, tokens});
        ops::linear(input, weight, output, device.stream);
        device.synchronize();
        std::vector<double> reference(std::size_t(n) * tokens, 0.0);
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t row = 0; row < n; ++row) {
                double sum = 0.0;
                const double* weights = dequantized.data() + std::size_t(row) * k;
                const float* column   = x.data() + std::size_t(t) * k;
                for (std::int32_t i = 0; i < k; ++i) { sum += weights[i] * column[i]; }
                reference[std::size_t(t) * n + row] = sum;
            }
        }
        const std::string name = std::string(label) + " T=" + std::to_string(tokens);
        failures += verify_reduction(name, from_device_bf16(dout, reference.size()), reference,
                                     kGemmCriterion);
    }
    return failures;
}

// -- the mixture -------------------------------------------------------------------------

struct CapturedIds {
    std::int32_t* ids;
    std::int32_t capacity;
};

void capture_ids(void* context, const Tensor& ids, std::int32_t first, std::int32_t tokens,
                 cudaStream_t stream) {
    auto* target = static_cast<CapturedIds*>(context);
    CUDA_CHECK(cudaMemcpyAsync(target->ids + std::size_t(first) * 8, ids.data,
                               std::size_t(tokens) * 8 * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

double silu(double v) { return v / (1.0 + std::exp(-v)); }

int run_mixture(DeviceContext& device) {
    constexpr ops::SparseMoeGeometry g = ops::kSparseMoeQwen36Geometry;
    const std::int32_t hidden = g.hidden, experts = g.experts, inter = g.intermediate;
    const std::int32_t shared = g.shared_intermediate;
    const StoredNvfp4 gate_up = make_stored(experts * 2 * inter, hidden, experts * 2, 0x60, 11, true);
    const StoredNvfp4 down    = make_stored(experts * hidden, inter, experts, 0x60, 12);
    const StoredNvfp4 sgu     = make_stored(2 * shared, hidden, 1, 0x7E, 13);
    const StoredNvfp4 sdown   = make_stored(hidden, shared, 1, 0x7E, 14);
    // The routed banks' trailing word is the identity; their second level is per expert.
    DeviceBuffer gate_up_device = to_device(payload(gate_up, 1.0F));
    DeviceBuffer down_device    = to_device(payload(down, 1.0F));
    DeviceBuffer sgu_device     = to_device(payload(sgu, sgu.second[0]));
    DeviceBuffer sdown_device   = to_device(payload(sdown, sdown.second[0]));
    DeviceBuffer gate_up_scale  = to_device(gate_up.second);
    DeviceBuffer down_scale     = to_device(down.second);
    std::vector<float> router(std::size_t(g.router_rows()) * hidden);
    fill_uniform(router, 21, -0.05F, 0.05F);
    round_to_bf16(router);
    DeviceBuffer router_device = to_device_bf16(router);

    ops::SparseMoeWeights weights{};
    weights.router_shared_gate.qtype = QType::BF16_CTRL;
    weights.router_shared_gate.layout = QuantLayout::Contiguous;
    weights.router_shared_gate.qdata = router_device.p;
    weights.router_shared_gate.payload = router_device.p;
    weights.router_shared_gate.payload_bytes = router_device.bytes;
    weights.router_shared_gate.n = g.router_rows();
    weights.router_shared_gate.k = hidden;
    weights.router_shared_gate.ndim = 2;
    weights.router_shared_gate.shape[0] = weights.router_shared_gate.padded_shape[0] = g.router_rows();
    weights.router_shared_gate.shape[1] = weights.router_shared_gate.padded_shape[1] = hidden;
    weights.routed_gate_up = stored_weight(gate_up_device, gate_up);
    weights.routed_down    = stored_weight(down_device, down);
    weights.shared_gate_up = stored_weight(sgu_device, sgu);
    weights.shared_down    = stored_weight(sdown_device, sdown);
    weights.routed_gate_up_scale = static_cast<const float*>(gate_up_scale.p);
    weights.routed_down_scale    = static_cast<const float*>(down_scale.p);
    weights.experts_per_token    = g.experts_per_token;
    ops::detail::nvfp4_marlin_adopt(weights.routed_gate_up, experts, weights.routed_gate_up_scale, 2,
                                    device.stream);
    ops::detail::nvfp4_marlin_adopt(weights.routed_down, experts, weights.routed_down_scale, 1,
                                    device.stream);
    ops::detail::nvfp4_marlin_adopt(weights.shared_gate_up, 1, nullptr, 1, device.stream);
    ops::detail::nvfp4_marlin_adopt(weights.shared_down, 1, nullptr, 1, device.stream);

    int failures = 0;
    for (const std::int32_t tokens : {1, 5, 40, 1100}) {
        std::vector<float> x(std::size_t(hidden) * tokens), residual(x.size());
        fill_uniform(x, 31 + tokens, -1.0F, 1.0F);
        fill_uniform(residual, 41 + tokens, -1.0F, 1.0F);
        round_to_bf16(x);
        round_to_bf16(residual);
        DeviceBuffer dx = to_device_bf16(x), dout = to_device_bf16(residual);
        DeviceBuffer ids(std::size_t(tokens) * 8 * sizeof(std::int32_t));
        CapturedIds captured{static_cast<std::int32_t*>(ids.p), tokens};
        ops::SparseMoeRoundHook hook{};
        hook.capture         = &capture_ids;
        hook.capture_context = &captured;
        DeviceArena workspace(ops::sparse_moe_workspace_capacity_bytes(
            g, QType::NVFP4_A16, QType::NVFP4_A16, tokens, tokens));
        Tensor input(dx.p, DType::BF16, {hidden, tokens});
        Tensor output(dout.p, DType::BF16, {hidden, tokens});
        ops::sparse_moe(input, weights, ops::SparseMoeEpilogue::AddResidual, output, workspace,
                        device.stream, hook);
        device.synchronize();
        const auto chosen = from_device<std::int32_t>(ids, std::size_t(tokens) * 8);
        const auto actual = from_device_bf16(dout, x.size());
        std::vector<std::int32_t> sample{0, tokens - 1, tokens / 2};
        if (tokens > 1024) { sample.insert(sample.end(), {1023, 1024}); }
        std::sort(sample.begin(), sample.end());
        sample.erase(std::unique(sample.begin(), sample.end()), sample.end());
        std::vector<double> got, reference;
        for (const std::int32_t t : sample) {
            const float* column = x.data() + std::size_t(t) * hidden;
            std::vector<double> logits(g.router_rows());
            for (std::int32_t r = 0; r < g.router_rows(); ++r) {
                double sum = 0.0;
                for (std::int32_t i = 0; i < hidden; ++i) { sum += double(router[std::size_t(r) * hidden + i]) * column[i]; }
                logits[r] = sum;
            }
            std::vector<std::int32_t> mine(chosen.begin() + std::size_t(t) * 8, chosen.begin() + std::size_t(t) * 8 + 8);
            std::vector<double> sorted(logits.begin(), logits.begin() + experts);
            std::nth_element(sorted.begin(), sorted.begin() + 7, sorted.end(), std::greater<>());
            const double eighth = sorted[7];
            for (const std::int32_t e : mine) {
                if (e < 0 || e >= experts || logits[e] < eighth - 1.0e-4) {
                    std::cerr << "token " << t << " selected expert " << e << " outside the top-8\n";
                    ++failures;
                }
            }
            const double top = *std::max_element(logits.begin(), logits.begin() + experts);
            double denominator = 0.0;
            for (const std::int32_t e : mine) { denominator += std::exp(logits[e] - top); }
            std::vector<double> out(hidden, 0.0);
            for (const std::int32_t e : mine) {
                const double alpha = std::exp(logits[e] - top) / denominator;
                std::vector<double> act(inter);
                for (std::int32_t j = 0; j < inter; ++j) {
                    double gv = 0.0, uv = 0.0;
                    const std::int32_t gate_row = e * 2 * inter + j, up_row = gate_row + inter;
                    for (std::int32_t i = 0; i < hidden; ++i) {
                        gv += gate_up.weight(gate_row, i) * column[i];
                        uv += gate_up.weight(up_row, i) * column[i];
                    }
                    act[j] = silu(gv) * uv;
                }
                for (std::int32_t h = 0; h < hidden; ++h) {
                    double sum = 0.0;
                    for (std::int32_t j = 0; j < inter; ++j) { sum += down.weight(e * hidden + h, j) * act[j]; }
                    out[h] += alpha * sum;
                }
            }
            const double gate = 1.0 / (1.0 + std::exp(-logits[experts]));
            std::vector<double> act(shared);
            for (std::int32_t j = 0; j < shared; ++j) {
                double gv = 0.0, uv = 0.0;
                for (std::int32_t i = 0; i < hidden; ++i) {
                    gv += sgu.weight(j, i) * column[i];
                    uv += sgu.weight(shared + j, i) * column[i];
                }
                act[j] = silu(gv) * uv;
            }
            for (std::int32_t h = 0; h < hidden; ++h) {
                double sum = 0.0;
                for (std::int32_t j = 0; j < shared; ++j) { sum += sdown.weight(h, j) * act[j]; }
                // the residual is part of the result; the paths are what this checks
                reference.push_back(out[h] + gate * sum);
                got.push_back(actual[std::size_t(t) * hidden + h] - residual[std::size_t(t) * hidden + h]);
            }
        }
        failures += verify_reduction("sparse_moe nvfp4_a16 T=" + std::to_string(tokens), got, reference,
                                     ReductionCriterion{2.0e-2, 0.0, 1.0 / 16});
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        DeviceContext device;
        int failures = 0;
        failures += run_dense("nvfp4_a16 2048x3072 top 448", 2048, 3072, 0x7E, 1, device);
        failures += run_dense("nvfp4_a16 3072x1024 top 15", 3072, 1024, 0x57, 2, device);
        failures += run_dense("nvfp4_a16 4096x3072 top 448", 4096, 3072, 0x7E, 3, device);
        failures += run_mixture(device);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " nvfp4_a16 marlin\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
