// MXFP8 operands and GEMM (ops/linear/mxfp8/mxfp8.h) at the vision tower's four projection
// shapes. The quantizer is checked element by element against FP8's rounding bound; the GEMM
// against an FP64 product of the operands decoded on the host -- codes from e4m3, scales from
// the 128x4 tiled layout -- so a wrong layout or epilogue fails on its own, apart from FP8's
// own rounding.

#include "core/device.h"
#include "ops/linear/mxfp8/mxfp8.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;
using ops::detail::Mxfp8Matrix;

constexpr ReductionCriterion kOutputTolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

double e4m3_value(std::uint8_t bits) {
    const int sign     = bits >> 7;
    const int exponent = (bits >> 3) & 15;
    const int mantissa = bits & 7;
    double value       = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                       : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -value : value;
}

double gelu_tanh(double x) {
    constexpr double kSqrt2OverPi = 0.79788456080286535588;
    return 0.5 * x * (1.0 + std::tanh(kSqrt2OverPi * (x + 0.044715 * x * x * x)));
}

std::vector<std::uint16_t> patterned(std::int64_t count, std::uint32_t seed, float magnitude) {
    std::vector<std::uint16_t> out(static_cast<std::size_t>(count));
    std::uint32_t state = seed;
    for (auto& value : out) {
        state = state * 1664525U + 1013904223U;
        // A spread of magnitudes inside each block, as real activations have.
        const float unit  = static_cast<float>(state >> 8) / 16777216.0F - 0.5F;
        const float scale = (state & 3U) == 0 ? 8.0F : 1.0F;
        value             = f32_to_bf16(unit * magnitude * scale);
    }
    return out;
}

struct Quantized {
    DeviceBuffer codes, scales;
    Mxfp8Matrix matrix;
};

Quantized quantize(const DeviceBuffer& source, std::int32_t rows, std::int32_t k,
                   ops::detail::Mxfp8Transform transform) {
    Quantized q{DeviceBuffer(ops::detail::mxfp8_codes_bytes(rows, k)),
                DeviceBuffer(ops::detail::mxfp8_scales_bytes(rows, k)), {}};
    q.codes.fill(0xff); // the padding must be written, not inherited
    q.matrix = Mxfp8Matrix{q.codes.p, q.scales.p, rows, k};
    ops::detail::mxfp8_quantize(source.p, k, transform, q.matrix, nullptr);
    return q;
}

/// Host decode of a quantized matrix: rows x padded_k doubles.
std::vector<double> decode(const Quantized& q, int& failures, const std::string& label) {
    const std::int32_t rows = q.matrix.rows, k = q.matrix.k;
    const std::int32_t kp   = ops::detail::mxfp8_padded_k(k);
    const std::int32_t groups = kp / 32, tiles = (groups + 3) / 4;
    const auto codes  = from_device<std::uint8_t>(q.codes.p, ops::detail::mxfp8_codes_bytes(rows, k));
    const auto scales = from_device<std::uint8_t>(q.scales.p, ops::detail::mxfp8_scales_bytes(rows, k));
    std::vector<double> out(static_cast<std::size_t>(rows) * kp);
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t g = 0; g < groups; ++g) {
            const std::uint8_t code = scales[ops::detail::nvfp4_tiled_scale_offset(r, g, tiles)];
            if (code == 0xff) {
                std::cerr << label << ": NaN scale at row " << r << " group " << g << '\n';
                ++failures;
                return out;
            }
            const double scale = std::ldexp(1.0, static_cast<int>(code) - 127);
            for (std::int32_t i = g * 32; i < g * 32 + 32; ++i) {
                const std::uint8_t c = codes[static_cast<std::size_t>(r) * kp + i];
                if (i >= k && c != 0) {
                    std::cerr << label << ": K padding not zero at row " << r << '\n';
                    ++failures;
                    return out;
                }
                out[static_cast<std::size_t>(r) * kp + i] = e4m3_value(c) * scale;
            }
        }
    }
    return out;
}

/// Every decoded value within e4m3's rounding of the (transformed) source: half an ulp, 2^-4
/// relative in the normal range, 2^-10 of the block scale among the subnormals.
int check_quantization(const std::vector<std::uint16_t>& source, const std::vector<double>& decoded,
                       const Quantized& q, bool gelu, const std::string& label) {
    const std::int32_t rows = q.matrix.rows, k = q.matrix.k, kp = ops::detail::mxfp8_padded_k(k);
    const auto scales = from_device<std::uint8_t>(q.scales.p, ops::detail::mxfp8_scales_bytes(rows, k));
    const std::int32_t tiles = (kp / 32 + 3) / 4;
    for (std::int32_t r = 0; r < rows; ++r) {
        for (std::int32_t i = 0; i < k; ++i) {
            double x = bf16_to_f32(source[static_cast<std::size_t>(r) * k + i]);
            if (gelu) { x = gelu_tanh(x); }
            const double scale = std::ldexp(
                1.0, static_cast<int>(scales[ops::detail::nvfp4_tiled_scale_offset(r, i / 32, tiles)]) - 127);
            // tanh.approx (~2^-11 relative) feeds the GELU input's rounding.
            const double slack = gelu ? std::ldexp(std::max(1.0, std::fabs(x)), -10) : 1e-6 * std::max(1.0, std::fabs(x));
            const double bound = std::ldexp(std::fabs(x), -4) + std::ldexp(scale, -10) + slack;
            const double got = decoded[static_cast<std::size_t>(r) * kp + i];
            if (!(std::fabs(got - x) <= bound)) {
                std::cerr << label << ": row " << r << " column " << i << " decodes to " << got
                          << ", source " << x << " (bound " << bound << ")\n";
                return 1;
            }
        }
    }
    return 0;
}

int run_case(std::int32_t n, std::int32_t k, std::int32_t tokens, bool bias, bool accumulate,
             bool gelu_input) {
    const std::string label = "MXFP8 [" + std::to_string(n) + "," + std::to_string(k) + "] T=" +
                              std::to_string(tokens) + (bias ? " +bias" : "") +
                              (accumulate ? " +=" : "") + (gelu_input ? " gelu(x)" : "");
    if (!ops::detail::mxfp8_gemm_available(n, k, tokens, bias, accumulate)) {
        std::cerr << label << ": no cuBLASLt algorithm\n";
        return 1;
    }
    int failures = 0;
    const auto weight_bits     = patterned(static_cast<std::int64_t>(n) * k, 977U + n + k, 0.05F);
    const auto activation_bits = patterned(static_cast<std::int64_t>(tokens) * k, 131U + tokens, 2.0F);
    DeviceBuffer weight_device     = to_device(weight_bits);
    DeviceBuffer activation_device = to_device(activation_bits);
    const Quantized w = quantize(weight_device, n, k, ops::detail::Mxfp8Transform::None);
    const Quantized x = quantize(activation_device, tokens, k,
                                 gelu_input ? ops::detail::Mxfp8Transform::GeluTanh
                                            : ops::detail::Mxfp8Transform::None);
    cuda_synchronize();
    const auto wd = decode(w, failures, label + " weight");
    const auto xd = decode(x, failures, label + " activation");
    if (failures != 0) { return failures; }
    failures += check_quantization(weight_bits, wd, w, false, label + " weight");
    failures += check_quantization(activation_bits, xd, x, gelu_input, label + " activation");

    std::vector<std::uint16_t> bias_bits(static_cast<std::size_t>(n));
    for (std::int32_t r = 0; r < n; ++r) {
        bias_bits[static_cast<std::size_t>(r)] = f32_to_bf16(static_cast<float>(((r * 37 + 11) % 97) - 48) / 64.0F);
    }
    DeviceBuffer bias_device = to_device(bias_bits);
    std::vector<std::uint16_t> initial(static_cast<std::size_t>(n) * tokens);
    for (std::size_t i = 0; i < initial.size(); ++i) {
        initial[i] = accumulate ? f32_to_bf16(static_cast<float>(static_cast<int>((i * 13 + 5) % 61) - 30) / 16.0F)
                                : static_cast<std::uint16_t>(0x7fc0); // NaN: must be overwritten
    }
    DeviceBuffer out_device = to_device(initial);
    if (!ops::detail::mxfp8_gemm(w.matrix, x.matrix, bias ? bias_device.p : nullptr, accumulate,
                                 out_device.p, nullptr)) {
        std::cerr << label << ": the GEMM declined\n";
        return failures + 1;
    }
    cuda_synchronize();
    const auto out = from_device<std::uint16_t>(out_device.p, initial.size());
    const std::int32_t kp = ops::detail::mxfp8_padded_k(k);
    std::vector<double> actual, expected;
    for (const std::int32_t r : {0, 1, n / 4, n / 2, (3 * n) / 4, n - 2, n - 1}) {
        for (const std::int32_t t : {0, 1, tokens / 2, tokens - 2, tokens - 1}) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < kp; ++i) {
                sum += wd[static_cast<std::size_t>(r) * kp + i] * xd[static_cast<std::size_t>(t) * kp + i];
            }
            if (bias) { sum += bf16_to_f32(bias_bits[static_cast<std::size_t>(r)]); }
            const std::size_t at = static_cast<std::size_t>(t) * n + r;
            if (accumulate) { sum += bf16_to_f32(initial[at]); }
            actual.push_back(bf16_to_f32(out[at]));
            expected.push_back(sum);
        }
    }
    failures += verify_reduction(label, actual, expected, kOutputTolerance);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        if (!ops::detail::mxfp8_device_supported()) {
            std::cout << "SKIP: MXFP8 needs compute capability 10.0 or newer\n";
            return 77;
        }
        int failures = 0;
        // Patch counts are multiples of the 2x2 merge (4), not of a tile: include widths that
        // leave partial MMA and scale tiles.
        for (const std::int32_t tokens : {4, 36, 64, 100, 784, 1852, 3136}) {
            failures += run_case(3456, 1152, tokens, /*bias=*/true, false, false);  // qkv
            failures += run_case(1152, 1152, tokens, true, /*accumulate=*/true, false);  // attn out
            failures += run_case(4304, 1152, tokens, true, false, false);           // fc1
            failures += run_case(1152, 4304, tokens, true, true, /*gelu=*/true);    // fc2
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " MXFP8 quantize and GEMM\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "MXFP8: " << error.what() << '\n';
        return 1;
    }
}
