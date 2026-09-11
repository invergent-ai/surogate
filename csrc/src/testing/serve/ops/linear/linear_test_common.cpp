#include "ops/parallel_rows.h"
#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace sinfer::test::linear {
namespace {

constexpr std::size_t kOutputGuardBytes   = 256;
constexpr std::uint8_t kOutputGuardByte   = 0xa5;
constexpr std::uint8_t kOutputPoisonByte  = 0xff;
constexpr std::size_t kOutputScanWords    = 1U << 20;
constexpr int kOracleTBlock               = 8;
constexpr double kBf16UnitRoundoff        = 1.0 / 256.0;
constexpr double kA8QuantizationAllowance = 0.04;

// The criterion belongs to the activation compute path, not to a private kernel, schedule, or
// launcher selected inside that path. The relative-L2 allowance is one BF16 unit roundoff; the
// gross allowance covers final BF16 storage plus accumulation/reduction rounding.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {kBf16UnitRoundoff, kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff};
    case ActivationCompute::A8:
        return {kA8QuantizationAllowance, kBf16UnitRoundoff, 1.5 * kA8QuantizationAllowance};
    case ActivationCompute::A4:
        // The A4 oracle reproduces the kernel's quantiser exactly (materialize_activation), so
        // the criterion is the A16 one: what is left is accumulation order and BF16 storage.
        return tolerance_for(ActivationCompute::A16);
    }
    throw std::invalid_argument("linear test: unknown activation compute path");
}

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear test: invalid ") + label + " extent");
    }
    const auto a = static_cast<std::size_t>(first);
    const auto b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear test: ") + label + " size overflow");
    }
    return a * b;
}

class GuardedOutput {
public:
    explicit GuardedOutput(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte) {
        poison();
    }

    void* data() { return storage_.data(); }

    const void* data() const { return storage_.data(); }

    void poison() { storage_.fill(kOutputPoisonByte); }

    int verify_guards(std::string_view label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
};

std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> indices(static_cast<std::size_t>(extent));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    constexpr std::int32_t kSamples = 32;
    for (std::int32_t sample = 0; sample < kSamples; ++sample) {
        const std::int32_t index = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(extent - 1) * sample) / (kSamples - 1));
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed,
                                           ActivationCompute activation_compute) {
    const std::size_t elements = checked_elements(k, t, "activation");
    std::vector<std::uint16_t> result(elements);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t column = 0; column < k; ++column) {
            std::uint32_t coordinate = seed ^ (static_cast<std::uint32_t>(column) * 0x9e3779b9U) ^
                                       (static_cast<std::uint32_t>(token) * 0x85ebca6bU);
            if (activation_compute == ActivationCompute::A16) {
                const std::uint64_t token_block = static_cast<std::uint64_t>(token / 256);
                coordinate                      = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(column) * 17U +
                    static_cast<std::uint64_t>(token) * 31U +
                    static_cast<std::uint64_t>(seed) * 13U +
                    token_block * (static_cast<std::uint64_t>(column / 256) * 13U + 47U));
            } else {
                coordinate ^= coordinate >> 16;
                coordinate *= 0x7feb352dU;
                coordinate ^= coordinate >> 15;
                coordinate *= 0x846ca68bU;
                coordinate ^= coordinate >> 16;
            }
            const int raw     = static_cast<int>(coordinate & 0xffU);
            const float value = static_cast<float>(raw - 128) * (1.0F / 256.0F);
            result[static_cast<std::size_t>(token) * k + column] = test::f32_to_bf16(value);
        }
    }
    return result;
}

namespace {

// E4M3FN: code -> value for positive codes 0x00..0x7e (0x7f is NaN; a saturating encode never
// produces it).
float e4m3_value(std::uint8_t code) {
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    if (exponent == 0) { return static_cast<float>(mantissa) * (1.0F / 512.0F); }
    return std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 7);
}

// __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) for x >= 0: nearest E4M3FN value, ties
// to the even code, saturating to the largest finite (448).
std::uint8_t encode_e4m3_satfinite(float value) {
    if (!(value > 0.0F)) { return 0; }
    if (value >= e4m3_value(0x7e)) { return 0x7e; }
    std::uint8_t below = 0;
    while (below < 0x7e && e4m3_value(static_cast<std::uint8_t>(below + 1)) <= value) { ++below; }
    const std::uint8_t above = static_cast<std::uint8_t>(below + 1);
    const float low = e4m3_value(below), high = e4m3_value(above);
    if (value == low) { return below; }
    const float to_low = value - low, to_high = high - value;
    if (to_low < to_high) { return below; }
    if (to_high < to_low) { return above; }
    return (below & 1) == 0 ? below : above;
}

// cvt.rn.satfinite.e2m1x2: nearest of {0, .5, 1, 1.5, 2, 3, 4, 6}, ties to the even code,
// saturating at 6, sign kept.
float round_e2m1_satfinite(float value) {
    constexpr float grid[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
    const float magnitude   = std::fabs(value);
    int code                = 7;
    if (magnitude < 6.0F) {
        int below = 0;
        while (below < 7 && grid[below + 1] <= magnitude) { ++below; }
        const float to_low = magnitude - grid[below], to_high = grid[below + 1] - magnitude;
        code = to_low < to_high ? below : to_high < to_low ? below + 1 : ((below & 1) == 0 ? below : below + 1);
    }
    return std::copysign(grid[code], value);
}

} // namespace

std::vector<float> materialize_activation(const std::vector<std::uint16_t>& bits, std::int32_t k,
                                          std::span<const std::int32_t> columns,
                                          ActivationCompute activation_compute,
                                          float input_scale_divisor) {
    std::vector<float> result(
        checked_elements(k, static_cast<std::int32_t>(columns.size()), "oracle activation"));
    for (std::size_t oracle_column = 0; oracle_column < columns.size(); ++oracle_column) {
        const std::uint16_t* source =
            bits.data() + static_cast<std::size_t>(columns[oracle_column]) * k;
        float* destination = result.data() + oracle_column * static_cast<std::size_t>(k);
        for (std::int32_t column = 0; column < k; ++column) {
            destination[column] = test::bf16_to_f32(source[column]);
        }
        if (activation_compute != ActivationCompute::A4) { continue; }
        // quantize_nvfp4_k16, step for step, in the kernel's fp32 operation order.
        for (std::int32_t block = 0; block < k; block += 16) {
            float* values = destination + block;
            float max_abs = 0.0F;
            for (int i = 0; i < 16; ++i) { max_abs = std::fmax(max_abs, std::fabs(values[i])); }
            const float scaled          = input_scale_divisor * max_abs;
            const float scale_unencoded = scaled / 6.0F;
            const std::uint8_t scale    = encode_e4m3_satfinite(scale_unencoded);
            if (scale == 0) {
                for (int i = 0; i < 16; ++i) { values[i] = 0.0F; }
                continue;
            }
            const float decoded_scale = e4m3_value(scale);
            for (int i = 0; i < 16; ++i) {
                const float lifted = values[i] * input_scale_divisor;
                const float code   = round_e2m1_satfinite(lifted / decoded_scale);
                // what the GEMM sees, undone by the divisor the epilogue's alpha applies
                values[i] = static_cast<float>(static_cast<double>(code) * decoded_scale /
                                               input_scale_divisor);
            }
        }
    }
    return result;
}

struct OutputRead {
    int failures = 0;
    std::vector<double> selected;
};

OutputRead read_output(const void* device, std::int32_t n, std::int32_t t,
                       std::span<const std::int32_t> rows, std::span<const std::int32_t> columns,
                       std::string_view label) {
    const std::size_t total_words = checked_elements(n, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * n + static_cast<std::size_t>(row));
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t poison_count    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear output");
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint16_t bits = chunk[index];
            if (bits == 0xffffU) { ++poison_count; }
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(test::bf16_to_f32(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (poison_count != 0) {
        std::cerr << label << ": output retains " << poison_count << " poison values\n";
        ++result.failures;
    }
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++result.failures;
    }
    return result;
}

int compare_output(std::string_view label, std::span<const double> actual,
                   std::span<const double> reference, ActivationCompute activation_compute) {
    return verify_reduction(label, actual, reference, tolerance_for(activation_compute));
}

} // namespace

quantized_weight::PackedWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q4G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q5G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::Q6G64_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k,
                                                      std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::W8G32_F16S, n, k, seed,
                                                   {quantized_weight::RowSplitScalePattern::Small,
                                                    quantized_weight::RowSplitCodePattern::Hashed});
}

quantized_weight::PackedWeight make_nvfp4_weight(std::int32_t n, std::int32_t k,
                                                 std::uint32_t seed) {
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    return quantized_weight::make_patterned_weight(QType::NVFP4, n, k, seed, options);
}

quantized_weight::PackedWeight make_fp8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed) {
    return quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, n, k, seed);
}

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t) {
    if (weight == nullptr || activation == nullptr || output == nullptr || n <= 0 || k <= 0 ||
        t <= 0) {
        throw std::invalid_argument("linear test: invalid FP64 GEMM argument");
    }

    sinfer::test::parallel_row_ranges(n, [&](std::int32_t row_begin, std::int32_t row_end) {
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                const float* weight_row = weight + static_cast<std::size_t>(row) * k;
                for (std::int32_t token_begin = 0; token_begin < t; token_begin += kOracleTBlock) {
                    const std::int32_t active = std::min(kOracleTBlock, t - token_begin);
                    std::array<double, kOracleTBlock> accumulators{};
                    for (std::int32_t column = 0; column < k; ++column) {
                        const double weight_value = static_cast<double>(weight_row[column]);
                        for (std::int32_t token = 0; token < active; ++token) {
                            accumulators[static_cast<std::size_t>(token)] +=
                                weight_value *
                                static_cast<double>(
                                    activation[static_cast<std::size_t>(token_begin + token) * k +
                                               column]);
                        }
                    }
                    for (std::int32_t token = 0; token < active; ++token) {
                        output[static_cast<std::size_t>(token_begin + token) * n + row] =
                            accumulators[static_cast<std::size_t>(token)];
                    }
                }
            }
    });
}

bool cuda_available() { return !test::cuda_unavailable(); }

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape) {
    if (shape.invocations.empty()) {
        throw std::invalid_argument("linear test: shape has no invocations");
    }
    const auto maximum = std::max_element(
        shape.invocations.begin(), shape.invocations.end(),
        [](const Invocation& left, const Invocation& right) { return left.t < right.t; });
    if (maximum->t <= 0) {
        throw std::invalid_argument("linear test: token extent must be positive");
    }

    const std::vector<std::int32_t> oracle_rows =
        shape.comparison == Comparison::Full ? all_indices(shape.n) : sampled_indices(shape.n);
    quantized_weight::PackedWeight host_weight = generator(shape.n, shape.k, shape.seed);
    std::vector<float> oracle_weight =
        quantized_weight::materialize_rows_fp32(host_weight, oracle_rows);
    if (host_weight.weight.qtype == QType::W8G32_F16S && activation_compute == ActivationCompute::A16) {
        test::round_to_bf16(oracle_weight);
    }
    const std::vector<std::uint16_t> activation_bits =
        make_activation(shape.k, maximum->t, shape.seed + 1U, activation_compute);

    DeviceBuffer device_activation(activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    DeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);

    std::vector<double> full_reference;
    if (shape.comparison == Comparison::Full) {
        const std::vector<std::int32_t> columns = all_indices(maximum->t);
        const std::vector<float> activation =
            materialize_activation(activation_bits, shape.k, columns, activation_compute,
                                   host_weight.weight.input_scale_divisor);
        full_reference.resize(checked_elements(shape.n, maximum->t, "full reference"));
        cpu_linear_gemm_fp64(oracle_weight.data(), activation.data(), full_reference.data(),
                             shape.n, shape.k, maximum->t);
    }

    int failures = 0;
    for (const Invocation& invocation : shape.invocations) {
        const std::string case_label = std::string(label) + " [" + std::to_string(shape.n) + "," +
                                       std::to_string(shape.k) +
                                       "] T=" + std::to_string(invocation.t);
        GuardedOutput output(checked_elements(shape.n, invocation.t, "guarded output"));
        Tensor input(device_activation.p, DType::BF16, {shape.k, invocation.t});
        Tensor destination(output.data(), DType::BF16, {shape.n, invocation.t});
        const std::size_t capacity = ops::linear_workspace_capacity_bytes(
            weight.qtype, shape.n, shape.k, invocation.policy, invocation.t, invocation.t);
        DeviceArena workspace(std::max<std::size_t>(capacity, 256));
        try {
            if (invocation.call_form == CallForm::A16Convenience) {
                ops::linear(input, weight, destination, nullptr);
            } else {
                ops::linear(input, weight, destination, invocation.policy, workspace, nullptr);
            }
            cuda_check(cudaDeviceSynchronize(), "synchronize linear");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        failures += output.verify_guards(case_label);
        const std::vector<std::int32_t> columns = shape.comparison == Comparison::Full
                                                      ? all_indices(invocation.t)
                                                      : sampled_indices(invocation.t);
        OutputRead actual =
            read_output(output.data(), shape.n, invocation.t, oracle_rows, columns, case_label);
        failures += actual.failures;

        if (shape.comparison == Comparison::Full) {
            failures +=
                compare_output(case_label, actual.selected,
                               std::span<const double>(
                                   full_reference.data(),
                                   checked_elements(shape.n, invocation.t, "reference prefix")),
                               activation_compute);
        } else {
            const std::vector<float> activation =
                materialize_activation(activation_bits, shape.k, columns, activation_compute,
                                   host_weight.weight.input_scale_divisor);
            std::vector<double> reference(
                checked_elements(static_cast<std::int32_t>(oracle_rows.size()),
                                 static_cast<std::int32_t>(columns.size()), "sampled reference"));
            cpu_linear_gemm_fp64(oracle_weight.data(), activation.data(), reference.data(),
                                 static_cast<std::int32_t>(oracle_rows.size()), shape.k,
                                 static_cast<std::int32_t>(columns.size()));
            failures += compare_output(case_label, actual.selected, reference, activation_compute);
        }
    }

    if (shape.verify_input_preservation) {
        std::vector<std::uint16_t> activation_after(activation_bits.size());
        device_activation.copy_to_host(activation_after.data(), device_activation.bytes);
        if (activation_after != activation_bits) {
            std::cerr << label << ": linear modified its activation input\n";
            ++failures;
        }
        std::vector<std::uint8_t> weight_after(host_weight.payload.size());
        device_weight.copy_to_host(weight_after.data(), device_weight.bytes);
        if (weight_after != host_weight.payload) {
            std::cerr << label << ": linear modified its persistent weight\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace sinfer::test::linear
