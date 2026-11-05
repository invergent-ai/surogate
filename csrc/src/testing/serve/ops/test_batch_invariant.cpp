// Batch invariance of the Ops `--batch-invariant` reroutes (api/ops/batch_invariant.h).
//
// A token's output must be the same bits whatever else shares the launch: alone, as one of many
// columns, and at any column offset. Without the switch both routes below fail this: cuBLASLt
// picks its BF16 algorithm (tile, split-K) per width, and the registered GDN gating routes split
// k by the token count. Each case compares every token against that token computed alone -- the
// property the server relies on -- and against an independent oracle, so an invariant but wrong
// kernel cannot pass.
//
//   - BF16 projections through every public entry a projection takes: Linear, Linear-add, the
//     registered 27B shape whose own kernels change with T, and the raw cuBLASLt-route call.
//   - The fused GDN norm + gating projection at the Qwen3.5-0.8B shape (16 heads, 1024 rows).
//   - The low-rank hyper-connection mixer at the Qwen3.8-Flash-Next shape (four streams over
//     2560, low rank 320), with BF16 projections (whose up projection changes kernels at 128
//     tokens, both in the fixed-order chain) and with W8 row-split ones, which run the mixer's own
//     kernels up to 64 tokens unless the switch sends them to the fixed-order GEMM.

#include "api/ops/batch_invariant.h"
#include "api/ops/gdn_gating_proj.h"
#include "api/ops/hyper_connection.h"
#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::direct_bf16_weight;

constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr std::int32_t kTokens = 300;
// Widths straddle every schedule boundary of the invariant GEMM (8, 32, and the 64-token tile);
// offsets move a token to a different column, and tile position, of the same launch.
constexpr std::int32_t kWidths[]  = {2,  3,  7,  8,  9,   15,  16,  17, 31,
                                     32, 33, 64, 65, 129, 256, 257, 300};
constexpr std::int32_t kOffsets[] = {0, 1, 5};

std::vector<std::uint16_t> make_tokens(std::int32_t hidden, std::int32_t tokens,
                                       std::uint32_t seed) {
    std::vector<std::uint16_t> result(static_cast<std::size_t>(hidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t column = 0; column < hidden; ++column) {
            std::uint32_t hash = static_cast<std::uint32_t>(token) * 0x27d4eb2fU ^
                                 static_cast<std::uint32_t>(column) * 0x165667b1U ^
                                 seed * 0x9e3779b9U;
            hash ^= hash >> 15;
            hash *= 0x85ebca6bU;
            hash ^= hash >> 13;
            const int centered = static_cast<int>((hash >> 8) & 0x3ffU) - 512;
            result[static_cast<std::size_t>(token) * hidden + column] =
                f32_to_bf16(static_cast<float>(centered) * (1.0F / 256.0F));
        }
    }
    return result;
}

std::vector<float> column_values(const std::vector<std::uint16_t>& bits, std::int32_t rows,
                                 std::int32_t column) {
    std::vector<float> result(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        result[static_cast<std::size_t>(row)] =
            bf16_to_f32(bits[static_cast<std::size_t>(column) * rows + row]);
    }
    return result;
}

// Compares every column of a wide launch with the same token run alone; `run(first, count)`
// returns `count` columns of `bytes_per_column` bytes each.
template <class Run>
int check_widths(const std::string& label, std::size_t bytes_per_column, Run&& run) {
    std::vector<std::vector<std::uint8_t>> alone(kTokens);
    for (std::int32_t token = 0; token < kTokens; ++token) {
        alone[static_cast<std::size_t>(token)] = run(token, 1);
    }
    int failures = 0;
    for (const std::int32_t width : kWidths) {
        for (const std::int32_t first : kOffsets) {
            if (first + width > kTokens) { continue; }
            const std::vector<std::uint8_t> wide = run(first, width);
            for (std::int32_t column = 0; column < width; ++column) {
                if (std::memcmp(wide.data() + static_cast<std::size_t>(column) * bytes_per_column,
                                alone[static_cast<std::size_t>(first + column)].data(),
                                bytes_per_column) != 0) {
                    std::cerr << label << ": token " << first + column << " differs at width "
                              << width << " column " << column << " from the same token alone\n";
                    ++failures;
                    break;
                }
            }
        }
    }
    return failures;
}

enum class Route { Linear, LinearAdd, Raw };

const char* route_name(Route route) {
    switch (route) {
    case Route::Linear:
        return "linear";
    case Route::LinearAdd:
        return "linear_add";
    case Route::Raw:
        return "cublaslt_raw";
    }
    return "?";
}

// `count` tokens from `first` through one route. A linear_add reads a per-token residual, so an
// accumulating column is still a function of its token alone.
std::vector<std::uint8_t> run_projection(Route route, DeviceWeight& weight,
                                         const DeviceBuffer& tokens,
                                         const std::vector<std::uint16_t>& residual_bits,
                                         std::int32_t first, std::int32_t count) {
    const std::int32_t rows   = weight.host.n;
    const std::int32_t hidden = weight.host.k;
    auto* base                = static_cast<std::uint16_t*>(tokens.p);
    Tensor x(base + static_cast<std::size_t>(first) * hidden, DType::BF16, {hidden, count});
    const std::size_t out_elements = static_cast<std::size_t>(rows) * count;
    std::vector<std::uint16_t> initial(out_elements, 0);
    if (route == Route::LinearAdd) {
        std::memcpy(initial.data(), residual_bits.data() + static_cast<std::size_t>(first) * rows,
                    out_elements * sizeof(std::uint16_t));
    }
    DeviceBuffer out = to_device(initial);
    Tensor output(out.p, DType::BF16, {rows, count});
    DeviceArena workspace(256);
    switch (route) {
    case Route::Linear:
        ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
        break;
    case Route::LinearAdd:
        ops::linear_add(x, weight.view(), output, ops::LinearPolicy::A16Only, workspace, nullptr);
        break;
    case Route::Raw:
        ops::detail::bf16_cublaslt_gemm_raw(weight.view().qdata, rows, hidden, x.data, count,
                                            output.data, rows, 0.0F, nullptr);
        break;
    }
    cuda_synchronize();
    return from_device<std::uint8_t>(out.p, out_elements * sizeof(std::uint16_t));
}

int check_projection(std::int32_t rows, std::int32_t hidden, std::uint32_t seed) {
    int failures = 0;
    DeviceWeight weight(make_patterned(rows, hidden, seed));
    const std::vector<std::uint16_t> token_bits    = make_tokens(hidden, kTokens, seed + 1);
    const std::vector<std::uint16_t> residual_bits = make_tokens(rows, kTokens, seed + 2);
    DeviceBuffer tokens                            = to_device(token_bits);
    const std::string shape = "[" + std::to_string(rows) + "," + std::to_string(hidden) + "]";

    for (const Route route : {Route::Linear, Route::LinearAdd, Route::Raw}) {
        const std::string label = std::string("batch-invariant ") + route_name(route) + " " + shape;
        failures += check_widths(label, static_cast<std::size_t>(rows) * sizeof(std::uint16_t),
                                 [&](std::int32_t first, std::int32_t count) {
                                     return run_projection(route, weight, tokens, residual_bits,
                                                           first, count);
                                 });
        // Still a correct GEMM: sampled rows of tokens run alone against FP64.
        std::vector<double> actual;
        std::vector<double> expected;
        for (const std::int32_t token : {0, 1, 150, kTokens - 1}) {
            const std::vector<std::uint8_t> bytes =
                run_projection(route, weight, tokens, residual_bits, token, 1);
            std::vector<std::uint16_t> out(static_cast<std::size_t>(rows));
            std::memcpy(out.data(), bytes.data(), bytes.size());
            const std::vector<float> activation = column_values(token_bits, hidden, token);
            for (const std::int32_t row : {0, 1, rows / 3, rows / 2, rows - 1}) {
                double reference = dot_fp64(weight.host, row, activation);
                if (route == Route::LinearAdd) {
                    reference +=
                        bf16_to_f32(residual_bits[static_cast<std::size_t>(token) * rows + row]);
                }
                actual.push_back(bf16_to_f32(out[static_cast<std::size_t>(row)]));
                expected.push_back(reference);
            }
        }
        failures += verify_reduction(label, actual, expected, kA16Tolerance);
    }
    failures += weight.verify_preserved("batch-invariant weight " + shape);
    return failures;
}

// The GDN norm + gating projection: h = RMSNorm(x) with a unit-offset weight, then per head
// g = -exp(A_log) * softplus(a.h + dt_bias) and beta = sigmoid(b.h).
int check_gdn_gating() {
    constexpr std::int32_t kHeads = 16;
    constexpr std::int32_t kRows  = 1024;
    constexpr float kEps          = 1.0e-6F;
    DeviceWeight a_weight(make_patterned(kHeads, kRows, 31U));
    DeviceWeight b_weight(make_patterned(kHeads, kRows, 37U));
    std::vector<std::uint16_t> norm_bits(kRows);
    for (std::int32_t i = 0; i < kRows; ++i) {
        norm_bits[static_cast<std::size_t>(i)] =
            f32_to_bf16(0.25F * static_cast<float>((i * 7) % 13 - 6) / 6.0F);
    }
    std::vector<float> a_log(kHeads);
    std::vector<float> dt_bias(kHeads);
    for (std::int32_t h = 0; h < kHeads; ++h) {
        a_log[static_cast<std::size_t>(h)]   = -1.0F + 0.125F * static_cast<float>(h);
        dt_bias[static_cast<std::size_t>(h)] = 0.5F - 0.0625F * static_cast<float>(h);
    }
    const std::vector<std::uint16_t> token_bits = make_tokens(kRows, kTokens, 41U);
    DeviceBuffer tokens                         = to_device(token_bits);
    DeviceBuffer norm                           = to_device(norm_bits);
    DeviceBuffer a_log_d                        = to_device(a_log);
    DeviceBuffer dt_d                           = to_device(dt_bias);
    DeviceArena workspace(64u << 20);

    // One token's output: its gates and betas (FP32, one per head), then its normalized hidden.
    constexpr std::size_t kColumnBytes = 2 * kHeads * sizeof(float) + kRows * sizeof(std::uint16_t);
    const auto run                     = [&](std::int32_t first, std::int32_t count) {
        auto* base = static_cast<std::uint16_t*>(tokens.p);
        Tensor x(base + static_cast<std::size_t>(first) * kRows, DType::BF16, {kRows, count});
        Tensor norm_weight(norm.p, DType::BF16, {kRows});
        Tensor a_log_t(a_log_d.p, DType::FP32, {kHeads});
        Tensor dt_t(dt_d.p, DType::FP32, {kHeads});
        DeviceBuffer h(static_cast<std::size_t>(kRows) * count * sizeof(std::uint16_t));
        DeviceBuffer g(static_cast<std::size_t>(kHeads) * count * sizeof(float));
        DeviceBuffer beta(static_cast<std::size_t>(kHeads) * count * sizeof(float));
        Tensor h_t(h.p, DType::BF16, {kRows, count});
        Tensor g_t(g.p, DType::FP32, {kHeads, count});
        Tensor beta_t(beta.p, DType::FP32, {kHeads, count});
        workspace.reset();
        ops::gdn_norm_gating_proj(x, norm_weight, kEps, a_weight.view(), b_weight.view(), a_log_t,
                                                      dt_t, workspace, h_t, g_t, beta_t, nullptr);
        cuda_synchronize();
        const auto g_host    = from_device<float>(g, static_cast<std::size_t>(kHeads) * count);
        const auto beta_host = from_device<float>(beta, static_cast<std::size_t>(kHeads) * count);
        const auto h_host = from_device<std::uint16_t>(h, static_cast<std::size_t>(kRows) * count);
        std::vector<std::uint8_t> packed(kColumnBytes * static_cast<std::size_t>(count));
        for (std::int32_t c = 0; c < count; ++c) {
            std::uint8_t* at = packed.data() + kColumnBytes * static_cast<std::size_t>(c);
            std::memcpy(at, g_host.data() + static_cast<std::size_t>(c) * kHeads,
                                            kHeads * sizeof(float));
            std::memcpy(at + kHeads * sizeof(float),
                                            beta_host.data() + static_cast<std::size_t>(c) * kHeads,
                                            kHeads * sizeof(float));
            std::memcpy(at + 2 * kHeads * sizeof(float),
                                            h_host.data() + static_cast<std::size_t>(c) * kRows,
                                            kRows * sizeof(std::uint16_t));
        }
        return packed;
    };
    int failures =
        check_widths("batch-invariant gdn_norm_gating_proj [16,1024]", kColumnBytes, run);

    // Oracle: FP64 gates from the kernel's own BF16 normalized hidden (the RMSNorm Op has its own
    // qualification), for a few tokens run alone.
    std::vector<double> actual;
    std::vector<double> expected;
    for (const std::int32_t token : {0, 77, kTokens - 1}) {
        const std::vector<std::uint8_t> column = run(token, 1);
        std::vector<float> g(kHeads);
        std::vector<float> beta(kHeads);
        std::vector<std::uint16_t> h(kRows);
        std::memcpy(g.data(), column.data(), kHeads * sizeof(float));
        std::memcpy(beta.data(), column.data() + kHeads * sizeof(float), kHeads * sizeof(float));
        std::memcpy(h.data(), column.data() + 2 * kHeads * sizeof(float),
                    kRows * sizeof(std::uint16_t));
        std::vector<float> hidden(kRows);
        for (std::int32_t i = 0; i < kRows; ++i) {
            hidden[static_cast<std::size_t>(i)] = bf16_to_f32(h[static_cast<std::size_t>(i)]);
        }
        for (std::int32_t head = 0; head < kHeads; ++head) {
            const double a = dot_fp64(a_weight.host, head, hidden) +
                             static_cast<double>(dt_bias[static_cast<std::size_t>(head)]);
            const double b        = dot_fp64(b_weight.host, head, hidden);
            const double softplus = a > 20.0 ? a : std::log1p(std::exp(a));
            actual.push_back(g[static_cast<std::size_t>(head)]);
            expected.push_back(
                -std::exp(static_cast<double>(a_log[static_cast<std::size_t>(head)])) * softplus);
            actual.push_back(beta[static_cast<std::size_t>(head)]);
            expected.push_back(1.0 / (1.0 + std::exp(-b)));
        }
    }
    failures += verify_reduction("batch-invariant gdn_norm_gating_proj oracle", actual, expected,
                                 ReductionCriterion{1.0e-4, 1.0e-4, 1.0e-3});
    return failures;
}

// A BF16 [n,k] matrix in device memory as the mixer binds it.
Weight bf16_matrix(void* data, std::int32_t n, std::int32_t k) {
    Weight weight;
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.qdata           = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(n) * k * sizeof(std::uint16_t);
    weight.ndim            = 2;
    weight.n               = n;
    weight.k               = k;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    return weight;
}

std::vector<std::uint8_t> bf16_bytes(const std::vector<float>& values) {
    std::vector<std::uint8_t> bytes(values.size() * sizeof(std::uint16_t));
    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint16_t bits = f32_to_bf16(values[i]);
        std::memcpy(bytes.data() + i * sizeof(std::uint16_t), &bits, sizeof(bits));
    }
    return bytes;
}

// The hyper-connection mixer: one token's output is its mixed block input (BF16 [hidden]) and its
// inject gates (FP32 [streams]). The oracle is the op's documented math in FP64 with the BF16
// roundings its contract stores (api/ops/hyper_connection.h; test_hyper_connection.cpp).
int check_hyper_connection(bool w8) {
    constexpr std::int32_t kStreams = 4;
    constexpr std::int32_t kHidden  = 2560;
    constexpr std::int32_t kRank    = 320;
    constexpr std::int32_t kWidth   = kStreams * kHidden;
    constexpr double kEps           = 1.0e-6;
    const std::uint32_t seed        = w8 ? 151U : 131U;
    std::vector<float> residual(static_cast<std::size_t>(kWidth) * kTokens);
    std::vector<float> gamma(kWidth);
    std::vector<float> down(static_cast<std::size_t>(kRank) * kWidth);
    std::vector<float> up(static_cast<std::size_t>(kWidth) * kRank);
    std::vector<float> inject_w(static_cast<std::size_t>(kStreams) * kWidth);
    fill_uniform(residual, seed, -3.0F, 3.0F);
    fill_uniform(gamma, seed + 1, 0.5F, 1.5F);
    fill_uniform(down, seed + 2, -0.02F, 0.02F);
    fill_uniform(up, seed + 3, -0.05F, 0.05F);
    fill_uniform(inject_w, seed + 4, -0.02F, 0.02F);
    round_to_bf16(residual);
    round_to_bf16(down);
    round_to_bf16(up);
    round_to_bf16(inject_w);
    quantized_weight::PackedWeight down_packed;
    quantized_weight::PackedWeight up_packed;
    std::vector<std::uint8_t> down_bytes;
    std::vector<std::uint8_t> up_bytes;
    if (w8) {
        down_packed = quantized_weight::pack_w8g32_row_split(down, kRank, kWidth);
        up_packed   = quantized_weight::pack_w8g32_row_split(up, kWidth, kRank);
        down_bytes  = down_packed.payload;
        up_bytes    = up_packed.payload;
        // The oracle multiplies the BF16 rounding of each dequantised value, as every A16 W8 GEMM.
        down = down_packed.dequant;
        up   = up_packed.dequant;
        round_to_bf16(down);
        round_to_bf16(up);
    } else {
        down_bytes = bf16_bytes(down);
        up_bytes   = bf16_bytes(up);
    }
    DeviceBuffer d_residual = to_device_bf16(residual);
    DeviceBuffer d_gamma    = to_device(gamma);
    DeviceBuffer d_down     = to_device(down_bytes);
    DeviceBuffer d_up       = to_device(up_bytes);
    DeviceBuffer d_inject_w = to_device_bf16(inject_w);
    const ops::HyperConnectionWeights weights{
        Tensor(d_gamma.p, DType::FP32, {kWidth}),
        w8 ? down_packed.device_weight(d_down.p) : bf16_matrix(d_down.p, kRank, kWidth),
        w8 ? up_packed.device_weight(d_up.p) : bf16_matrix(d_up.p, kWidth, kRank),
        bf16_matrix(d_inject_w.p, kStreams, kWidth)};
    WorkspaceArena arena(
        ops::hyper_connection_mix_workspace_capacity_bytes(kStreams, kHidden, kRank, 1, kTokens));

    constexpr std::size_t kMixedBytes  = kHidden * sizeof(std::uint16_t);
    constexpr std::size_t kColumnBytes = kMixedBytes + kStreams * sizeof(float);
    const auto run = [&](std::int32_t first, std::int32_t count) {
        auto* base = static_cast<std::uint16_t*>(d_residual.p);
        Tensor x(base + static_cast<std::size_t>(first) * kWidth, DType::BF16, {kWidth, count});
        DeviceBuffer mixed(static_cast<std::size_t>(kHidden) * count * sizeof(std::uint16_t));
        DeviceBuffer inject(static_cast<std::size_t>(kStreams) * count * sizeof(float));
        Tensor mixed_t(mixed.p, DType::BF16, {kHidden, count});
        Tensor inject_t(inject.p, DType::FP32, {kStreams, count});
        ops::hyper_connection_mix(x, weights, kStreams, static_cast<float>(kEps), mixed_t,
                                  &inject_t, arena, nullptr);
        cuda_synchronize();
        const auto m = from_device<std::uint16_t>(mixed, static_cast<std::size_t>(kHidden) * count);
        const auto g = from_device<float>(inject, static_cast<std::size_t>(kStreams) * count);
        std::vector<std::uint8_t> packed(kColumnBytes * static_cast<std::size_t>(count));
        for (std::int32_t c = 0; c < count; ++c) {
            std::uint8_t* at = packed.data() + kColumnBytes * static_cast<std::size_t>(c);
            std::memcpy(at, m.data() + static_cast<std::size_t>(c) * kHidden, kMixedBytes);
            std::memcpy(at + kMixedBytes, g.data() + static_cast<std::size_t>(c) * kStreams,
                        kStreams * sizeof(float));
        }
        return packed;
    };
    const std::string label = std::string("batch-invariant hyper_connection_mix ") +
                              (w8 ? "W8" : "BF16") + " [4 x 2560, rank 320]";
    int failures = check_widths(label, kColumnBytes, run);

    // Still the mixer's math: tokens run alone against the FP64 oracle.
    std::vector<double> got_mixed;
    std::vector<double> want_mixed;
    std::vector<double> got_inject;
    std::vector<double> want_inject;
    for (const std::int32_t token : {0, 151, kTokens - 1}) {
        const std::vector<std::uint8_t> column = run(token, 1);
        const float* x = residual.data() + static_cast<std::size_t>(token) * kWidth;
        std::vector<double> n(kWidth);
        std::vector<double> operand(kWidth);
        for (std::int32_t s = 0; s < kStreams; ++s) {
            double sum_sq = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                sum_sq += static_cast<double>(x[s * kHidden + d]) * x[s * kHidden + d];
            }
            const double inv = 1.0 / std::sqrt(sum_sq / kHidden + kEps);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const std::int32_t i = s * kHidden + d;
                n[static_cast<std::size_t>(i)]       = x[i] * inv * gamma[static_cast<std::size_t>(i)];
                operand[static_cast<std::size_t>(i)] = bf16_to_f32(f32_to_bf16(static_cast<float>(n[static_cast<std::size_t>(i)])));
            }
        }
        std::vector<double> low(kRank);
        for (std::int32_t r = 0; r < kRank; ++r) {
            double dot = 0.0;
            for (std::int32_t i = 0; i < kWidth; ++i) {
                dot += down[static_cast<std::size_t>(r) * kWidth + i] * operand[static_cast<std::size_t>(i)];
            }
            const double v = bf16_to_f32(f32_to_bf16(static_cast<float>(dot))) / kStreams;
            low[static_cast<std::size_t>(r)] = bf16_to_f32(f32_to_bf16(static_cast<float>(v / (1.0 + std::exp(-v)))));
        }
        std::vector<std::uint16_t> mixed_bits(kHidden);
        std::memcpy(mixed_bits.data(), column.data(), kMixedBytes);
        std::vector<float> gates(kStreams);
        std::memcpy(gates.data(), column.data() + kMixedBytes, kStreams * sizeof(float));
        for (std::int32_t d = 0; d < kHidden; ++d) {
            double acc = 0.0;
            for (std::int32_t s = 0; s < kStreams; ++s) {
                const std::int32_t i = s * kHidden + d;
                double logit = 0.0;
                for (std::int32_t r = 0; r < kRank; ++r) {
                    logit += up[static_cast<std::size_t>(i) * kRank + r] * low[static_cast<std::size_t>(r)];
                }
                const double rounded = bf16_to_f32(f32_to_bf16(static_cast<float>(logit)));
                acc += n[static_cast<std::size_t>(i)] / (1.0 + std::exp(-rounded));
            }
            got_mixed.push_back(bf16_to_f32(mixed_bits[static_cast<std::size_t>(d)]));
            want_mixed.push_back(acc / kStreams);
        }
        for (std::int32_t row = 0; row < kStreams; ++row) {
            double dot = 0.0;
            for (std::int32_t i = 0; i < kWidth; ++i) {
                dot += inject_w[static_cast<std::size_t>(row) * kWidth + i] * n[static_cast<std::size_t>(i)];
            }
            got_inject.push_back(gates[static_cast<std::size_t>(row)]);
            want_inject.push_back(2.0 / (1.0 + std::exp(-dot / kStreams)));
        }
    }
    failures += verify_pointwise((label + " oracle mixed").c_str(), got_mixed, want_mixed,
                                 PointwiseCriterion{3.0e-3, 1.0e-2});
    failures += verify_pointwise((label + " oracle inject").c_str(), got_inject, want_inject,
                                 PointwiseCriterion{2.0e-5, 2.0e-5});
    return failures;
}

} // namespace

int main() {
    if (sinfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        sinfer::ops::set_batch_invariant(true);
        int failures = 0;
        // Qwen3.5-0.8B projections (GDN q/k/v/z parent, output, MLP, attention), a k tail that
        // is not a multiple of the 64-wide k tile, and the registered 27B output shape.
        failures += check_projection(8192, 1024, 11U);
        failures += check_projection(1024, 2048, 13U);
        failures += check_projection(3584, 1024, 17U);
        failures += check_projection(1024, 3584, 19U);
        failures += check_projection(1000, 72, 23U);
        failures += check_projection(5120, 6144, 29U);
        failures += check_gdn_gating();
        failures += check_hyper_connection(false);
        failures += check_hyper_connection(true);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " batch-invariant Ops\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "batch-invariant Ops: " << error.what() << '\n';
        return 1;
    }
}
