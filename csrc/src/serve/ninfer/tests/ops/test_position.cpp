#include "ninfer/ops/position.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

int fill_case(std::int32_t count, std::int32_t start) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) { expected[static_cast<std::size_t>(i)] = start + i; }

    GuardedDeviceBuffer output(static_cast<std::size_t>(count) * sizeof(std::int32_t));
    output.fill(0xcd);
    Tensor output_tensor(output.data(), DType::I32, {count});
    ops::fill_i32_positions(output_tensor, start, nullptr);
    cuda_synchronize();

    const std::string label =
        "fill_i32_positions T=" + std::to_string(count) + " start=" + std::to_string(start);
    int failures = verify_exact(
        label.c_str(), from_device<std::int32_t>(output.data(), expected.size()), expected);
    failures += output.verify_guards(label.c_str());
    return failures;
}

int offset_case(std::int32_t count, std::int32_t delta_value, bool in_place) {
    std::vector<std::int32_t> source(static_cast<std::size_t>(count));
    std::vector<std::int32_t> expected(static_cast<std::size_t>(count));
    for (std::int32_t i = 0; i < count; ++i) {
        source[static_cast<std::size_t>(i)]   = 131072 + 3 * i + (i % 5);
        expected[static_cast<std::size_t>(i)] = source[static_cast<std::size_t>(i)] + delta_value;
    }
    const std::vector<std::int32_t> delta{delta_value};

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_delta(sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(source.size() * sizeof(std::int32_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::int32_t));
    device_delta.copy_from_host(delta.data(), sizeof(std::int32_t));
    device_output.fill(0xcd);

    Tensor source_tensor(device_source.data(), DType::I32, {count});
    Tensor delta_tensor(device_delta.data(), DType::I32, {1});
    Tensor output_tensor(device_output.data(), DType::I32, {count});
    Tensor& destination = in_place ? source_tensor : output_tensor;
    ops::offset_i32_positions(source_tensor, delta_tensor, destination, nullptr);
    cuda_synchronize();

    const std::string label = "offset_i32_positions T=" + std::to_string(count) +
                              (in_place ? " in-place" : " out-of-place");
    int failures =
        verify_exact(label.c_str(),
                     from_device<std::int32_t>(
                         in_place ? device_source.data() : device_output.data(), expected.size()),
                     expected);
    if (!in_place) {
        failures +=
            verify_exact((label + " preserves source").c_str(),
                         from_device<std::int32_t>(device_source.data(), source.size()), source);
    }
    failures += verify_exact((label + " preserves delta").c_str(),
                             from_device<std::int32_t>(device_delta.data(), delta.size()), delta);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_delta.verify_guards((label + " delta").c_str());
    if (!in_place) { failures += device_output.verify_guards((label + " destination").c_str()); }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += fill_case(1, 0);
    failures += fill_case(6, 262144);
    failures += fill_case(1024, 131072);
    failures += offset_case(1, -17, false);
    failures += offset_case(6, 31, true);
    failures += offset_case(1024, -257, false);
    std::cout << (failures ? "FAIL" : "OK") << " position\n";
    return failures ? 1 : 0;
}
