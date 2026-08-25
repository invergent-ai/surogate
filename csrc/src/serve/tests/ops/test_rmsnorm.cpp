#include "ninfer/ops/rmsnorm.h"
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

constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.4e-3};
}

std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                   const std::vector<float>& weight, const Shape& shape,
                                   bool unit_offset) {
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
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

int run_case(const char* label, const Shape& shape, bool unit_offset, std::uint32_t seed,
             float input_scale = 4.0F, bool bf16x2_unaligned = false) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d);
    fill_uniform(input, seed, -input_scale, input_scale);
    fill_uniform(weight, seed + 1U, unit_offset ? -0.5F : 0.25F, unit_offset ? 0.5F : 1.75F);
    round_to_bf16(input);
    round_to_bf16(weight);
    const std::vector<double> reference = rmsnorm_oracle(input, weight, shape, unit_offset);

    DeviceInput device_input  = make_input(input, bf16x2_unaligned);
    DeviceInput device_weight = make_input(weight, bf16x2_unaligned);
    const std::size_t leading = bf16x2_unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output(leading + count * sizeof(std::uint16_t));
    output.fill(0xff);
    void* output_data = static_cast<std::uint8_t*>(output.data()) + leading;

    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor output_tensor = tensor_for(output_data, shape);
    ops::rmsnorm(input_tensor, weight_tensor, kEps, unit_offset, output_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(output_data, count), reference,
                                    rmsnorm_bf16_criterion());
    failures += verify_output_storage(std::string(label) + " output", output, bf16x2_unaligned);
    failures += verify_preserved(std::string(label) + " preserves input", device_input);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case("rmsnorm offset [5120,1]", {5120, 1}, true, 1101U);
    failures += run_case("rmsnorm offset [5120,128]", {5120, 128}, true, 1102U);
    failures += run_case("rmsnorm offset [2048,7]", {2048, 7}, true, 1103U);
    failures += run_case("rmsnorm offset [256,24,7]", {256, 24, 7}, true, 1104U);
    failures += run_case("rmsnorm offset [256,2,1]", {256, 2}, true, 1105U);
    failures += run_case("rmsnorm offset [256,4,48]", {256, 4, 48}, true, 1106U);
    failures += run_case("rmsnorm plain [2048,1]", {2048, 1}, false, 1201U);
    failures += run_case("rmsnorm plain [2048,128]", {2048, 128}, false, 1202U);
    failures += run_case("rmsnorm plain [128,32,7]", {128, 32, 7}, false, 1203U);
    failures += run_case("rmsnorm plain [128,8,128]", {128, 8, 128}, false, 1204U);
    failures += run_case("rmsnorm offset unaligned [128,32]", {128, 32}, true, 1301U, 4.0F, true);
    failures += run_case("rmsnorm plain unaligned [128,8]", {128, 8}, false, 1302U, 4.0F, true);
    failures += run_case("rmsnorm plain near-zero [128,32]", {128, 32}, false, 1303U, 1.0e-5F);
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm\n";
    return failures ? 1 : 0;
}
