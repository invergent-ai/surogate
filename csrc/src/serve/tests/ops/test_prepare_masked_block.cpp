#include "ninfer/ops/prepare_masked_block.h"
#include "ops/op_tester.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kMaskId = 248077;

struct MaskedBlockExpected {
    std::vector<std::int32_t> ids;
    std::vector<std::int32_t> positions;
};

MaskedBlockExpected masked_block_oracle(std::int32_t anchor, std::int32_t length, int block_size) {
    MaskedBlockExpected expected{
        .ids       = std::vector<std::int32_t>(static_cast<std::size_t>(block_size), kMaskId),
        .positions = std::vector<std::int32_t>(static_cast<std::size_t>(block_size)),
    };
    expected.ids[0] = anchor;
    for (int i = 0; i < block_size; ++i) {
        expected.positions[static_cast<std::size_t>(i)] = length + i;
    }
    return expected;
}

int run_case(int block_size, std::int32_t anchor_value, std::int32_t length_value) {
    const auto expected = masked_block_oracle(anchor_value, length_value, block_size);
    DeviceBuffer anchor = to_device<std::int32_t>({anchor_value});
    DeviceBuffer length = to_device<std::int32_t>({length_value});
    DeviceBuffer valid  = to_device<std::int32_t>({block_size});
    GuardedDeviceBuffer ids(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    GuardedDeviceBuffer positions(static_cast<std::size_t>(block_size) * sizeof(std::int32_t));
    ids.fill(0xcd);
    positions.fill(0xef);

    Tensor anchor_tensor(anchor.p, DType::I32, {1});
    Tensor length_tensor(length.p, DType::I32, {1});
    Tensor valid_tensor(valid.p, DType::I32, {1});
    Tensor ids_tensor(ids.data(), DType::I32, {block_size, 1});
    Tensor positions_tensor(positions.data(), DType::I32, {block_size, 1});
    ops::prepare_masked_block(anchor_tensor, length_tensor, valid_tensor, kMaskId, ids_tensor,
                              positions_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "prepare_masked_block W=" + std::to_string(block_size);
    int failures            = verify_exact(
        (label + " ids").c_str(),
        from_device<std::int32_t>(ids.data(), static_cast<std::size_t>(block_size)), expected.ids);
    failures += verify_exact(
        (label + " positions").c_str(),
        from_device<std::int32_t>(positions.data(), static_cast<std::size_t>(block_size)),
        expected.positions);
    failures += verify_exact((label + " anchor unchanged").c_str(),
                             from_device<std::int32_t>(anchor, 1), {anchor_value});
    failures += verify_exact((label + " length unchanged").c_str(),
                             from_device<std::int32_t>(length, 1), {length_value});
    failures += ids.verify_guards((label + " ids guards").c_str());
    failures += positions.verify_guards((label + " positions guards").c_str());
    return failures;
}

int run_batch_case() {
    constexpr std::int32_t width = 4;
    constexpr std::int32_t batch = 2;
    const std::vector<std::int32_t> anchors{9173, 100007};
    const std::vector<std::int32_t> lengths{37, 900};
    const std::vector<std::int32_t> valid{4, 2};
    const std::vector<std::int32_t> expected_ids{
        anchors[0], kMaskId, kMaskId, kMaskId, anchors[1], kMaskId, kMaskId, kMaskId,
    };
    const std::vector<std::int32_t> expected_positions{
        37, 38, 39, 40, 900, 901, 901, 901,
    };

    DeviceBuffer device_anchors = to_device(anchors);
    DeviceBuffer device_lengths = to_device(lengths);
    DeviceBuffer device_valid   = to_device(valid);
    GuardedDeviceBuffer ids(static_cast<std::size_t>(width * batch) * sizeof(std::int32_t));
    GuardedDeviceBuffer positions(static_cast<std::size_t>(width * batch) * sizeof(std::int32_t));
    ids.fill(0xcd);
    positions.fill(0xef);

    Tensor anchors_tensor(device_anchors.p, DType::I32, {batch});
    Tensor lengths_tensor(device_lengths.p, DType::I32, {batch});
    Tensor valid_tensor(device_valid.p, DType::I32, {batch});
    Tensor ids_tensor(ids.data(), DType::I32, {width, batch});
    Tensor positions_tensor(positions.data(), DType::I32, {width, batch});
    ops::prepare_masked_block(anchors_tensor, lengths_tensor, valid_tensor, kMaskId, ids_tensor,
                              positions_tensor, nullptr);
    cuda_synchronize();

    int failures =
        verify_exact("prepare_masked_block B=2 ids",
                     from_device<std::int32_t>(ids.data(), static_cast<std::size_t>(width * batch)),
                     expected_ids);
    failures += verify_exact(
        "prepare_masked_block B=2 positions",
        from_device<std::int32_t>(positions.data(), static_cast<std::size_t>(width * batch)),
        expected_positions);
    failures += ids.verify_guards("prepare_masked_block B=2 ids guards");
    failures += positions.verify_guards("prepare_masked_block B=2 positions guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "prepare_masked_block: SKIP (CUDA unavailable)\n";
        return 77;
    }

    int failures = 0;
    for (const int block_size : std::array{2, 7, 16}) {
        failures += run_case(block_size, 9173 + block_size, 37);
        failures += run_case(block_size, 100000 + block_size,
                             std::numeric_limits<std::int32_t>::max() - (block_size - 1));
    }
    failures += run_batch_case();

    if (failures != 0) {
        std::cerr << "prepare_masked_block failures=" << failures << '\n';
        return 1;
    }
    std::cout << "prepare_masked_block: PASS\n";
    return 0;
}
