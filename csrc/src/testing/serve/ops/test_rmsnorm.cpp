#include "api/ops/residual_add.h"
#include "api/ops/rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::norm;

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

// `rmsnorm_add` exists to replace `rmsnorm` into a scratch plane followed by
// `residual_add`, so what it owes is not "close to the oracle" but "the same
// bits as the pair". This runs both and compares the raw BF16 storage: if the
// fused epilogue ever stopped rounding the normalised term before adding, or
// routed to a kernel that reduces in a different order than its base epilogue,
// the two would drift in the last bit and this fails while a tolerance-based
// oracle check would not.
int run_composition_case(const char* label, const Shape& shape, bool unit_offset,
                         std::uint32_t seed) {
    const std::size_t count = shape.elements();
    std::vector<float> input(count), weight(shape.d), prior(count);
    fill_uniform(input, seed, -4.0F, 4.0F);
    fill_uniform(weight, seed + 1U, unit_offset ? -0.5F : 0.25F, unit_offset ? 0.5F : 1.75F);
    fill_uniform(prior, seed + 2U, -8.0F, 8.0F);
    round_to_bf16(input);
    round_to_bf16(weight);
    round_to_bf16(prior);

    DeviceInput device_input  = make_input(input, false);
    DeviceInput device_weight = make_input(weight, false);
    Tensor input_tensor = tensor_for(device_input.data, shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});

    // Fused: the prior is the destination, read and written in place.
    DeviceInput fused_storage = make_input(prior, false);
    Tensor fused = tensor_for(fused_storage.data, shape);
    ops::rmsnorm_add(input_tensor, weight_tensor, kEps, unit_offset, fused, nullptr);

    // The pair it replaces: normalise into a plane, then add that plane on.
    GuardedDeviceBuffer plane(count * sizeof(std::uint16_t));
    plane.fill(0xff);
    Tensor plane_tensor = tensor_for(plane.data(), shape);
    ops::rmsnorm(input_tensor, weight_tensor, kEps, unit_offset, plane_tensor, nullptr);
    DeviceInput staged_storage = make_input(prior, false);
    Tensor staged = tensor_for(staged_storage.data, shape);
    ops::residual_add(plane_tensor, staged, nullptr);
    cuda_synchronize();

    int failures = verify_exact(label, from_device<std::uint16_t>(fused_storage.data, count),
                                from_device<std::uint16_t>(staged_storage.data, count));
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

    // The accumulating form, against the two-kernel sequence it replaces. 640 is
    // Gemma 3's hidden extent and the shape the engine actually runs; the others
    // cover the warp, d128, cta and d2048 routes so a route that diverged from
    // its base epilogue is caught rather than assumed away.
    failures += run_composition_case("rmsnorm_add offset [640,1]", {640, 1}, true, 1401U);
    failures += run_composition_case("rmsnorm_add offset [640,37]", {640, 37}, true, 1402U);
    failures += run_composition_case("rmsnorm_add offset [256,4,9]", {256, 4, 9}, true, 1403U);
    failures += run_composition_case("rmsnorm_add offset [2048,5]", {2048, 5}, true, 1404U);
    failures += run_composition_case("rmsnorm_add plain [128,32]", {128, 32}, false, 1405U);
    failures += run_composition_case("rmsnorm_add plain [2048,9]", {2048, 9}, false, 1406U);
    failures += run_composition_case("rmsnorm_add plain [5120,3]", {5120, 3}, false, 1407U);
    std::cout << (failures ? "FAIL" : "OK") << " rmsnorm\n";
    return failures ? 1 : 0;
}
