#include "api/ops/gated_rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion gated_rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 4.5e-5,
            /*gross_relative_to_max_reference*/ 2.8e-3};
}

std::vector<double> gated_rmsnorm_oracle(const std::vector<float>& input,
                                         const std::vector<float>& weight,
                                         const std::vector<float>& gate, const Shape& shape,
                                         ops::GatedRmsGate activation = ops::GatedRmsGate::Silu) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gate_value = gate[base + column];
            const double sigmoid    = 1.0 / (1.0 + std::exp(-gate_value));
            const double activated =
                activation == ops::GatedRmsGate::Sigmoid ? sigmoid : gate_value * sigmoid;
            output[base + column] = static_cast<double>(input[base + column]) * inverse *
                                    static_cast<double>(weight[column]) * activated;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float input_scale = 4.0F,
             bool bf16x2_unaligned = false,
             ops::GatedRmsGate activation = ops::GatedRmsGate::Silu) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), gate(count);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, 0.25F, 1.75F);
    fill_uniform(gate, seed + 2U, -5.0F, 5.0F);
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(gate);
    const std::vector<double> reference =
        gated_rmsnorm_oracle(input, weight, gate, shape, activation);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    DeviceInput device_gate   = make_input(gate, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor gate_tensor   = tensor_for(device_gate.data, shape);
    Tensor output_tensor = tensor_for(output_data, shape);
    if (activation == ops::GatedRmsGate::Silu) {
        ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, output_tensor, nullptr);
    } else {
        ops::gated_rmsnorm(input_tensor, weight_tensor, gate_tensor, kEps, activation,
                           output_tensor, nullptr);
    }
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    gated_rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    failures += verify_preserved(std::string(label) + " preserves gate", device_gate);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("gated_rmsnorm [128,48,1]", {128, 48}, 1401U);
    failures += run_case("gated_rmsnorm [128,48,48]", {128, 48, 48}, 1406U);
    failures += run_case("gated_rmsnorm [128,32,7]", {128, 32, 7}, 1402U);
    failures += run_case("gated_rmsnorm [128,32,128]", {128, 32, 128}, 1403U);
    failures += run_case("gated_rmsnorm near-zero [128,32]", {128, 32}, 1404U, 1.0e-5F);
    failures += run_case("gated_rmsnorm unaligned [128,48]", {128, 48}, 1405U, 4.0F, true);
    // surogate vendor patch (PATCHES.md #13): qwen3.5-0.8b (16 GDN value heads).
    failures += run_case("gated_rmsnorm [128,16,1]", {128, 16}, 1407U);
    failures += run_case("gated_rmsnorm [128,16,33]", {128, 16, 33}, 1408U);
    // Qwen3.8-Flash-Next gates the GDN output with the logistic sigmoid; the tiny-input case
    // is the eps-dominated regime the first token of a sequence produces.
    constexpr auto kSigmoid = ops::GatedRmsGate::Sigmoid;
    failures += run_case("gated_rmsnorm sigmoid [128,48,1]", {128, 48}, 1409U, 4.0F, false, kSigmoid);
    failures += run_case("gated_rmsnorm sigmoid [128,48,17]", {128, 48, 17}, 1410U, 4.0F, false, kSigmoid);
    failures += run_case("gated_rmsnorm sigmoid near-zero [128,48]", {128, 48}, 1411U, 1.0e-4F, false, kSigmoid);
    failures += run_case("gated_rmsnorm sigmoid unaligned [128,48]", {128, 48}, 1412U, 4.0F, true, kSigmoid);
    failures += run_case("gated_rmsnorm sigmoid [1024,4,3]", {1024, 4, 3}, 1413U, 4.0F, false, kSigmoid);
    failures += run_case("gated_rmsnorm sigmoid [4096,2,3]", {4096, 2, 3}, 1414U, 4.0F, false, kSigmoid);
    failures += run_case("gated_rmsnorm silu [1024,4,3]", {1024, 4, 3}, 1415U);
    failures += run_case("gated_rmsnorm silu [4096,2,3]", {4096, 2, 3}, 1416U);
    std::cout << (failures ? "FAIL" : "OK") << " gated_rmsnorm\n";
    return failures ? 1 : 0;
}
