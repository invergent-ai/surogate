#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/sparse_moe/marlin/marlin_moe_gemm.h"

#include <iostream>

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::ops::detail;

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        DeviceContext device;
        constexpr int experts = 256, n = 1024, k = 2048, block = 16;
        int sms = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device.device));
        DeviceBuffer codes(std::size_t(experts) * n * k / 2);
        DeviceBuffer scales(std::size_t(experts) * n * (k / 64) * 2);
        DeviceBuffer packed(marlin_moe_b_bytes(n, k) * experts);
        DeviceBuffer reordered_scales(marlin_moe_scale_bytes(n, k) * experts);
        DeviceBuffer temporary(marlin_moe_b_bytes(n, k));
        CUDA_CHECK(cudaMemset(codes.p, 0x99, codes.bytes)); // each signed Q4 nibble is -7
        const std::vector<std::uint16_t> scale_values(scales.bytes / 2, 0x3c00); // FP16 1
        CUDA_CHECK(cudaMemcpy(scales.p, scale_values.data(), scales.bytes, cudaMemcpyHostToDevice));
        marlin_moe_repack_q4g64(codes.p, scales.p, experts, n, k, temporary.p,
                                packed.p, reordered_scales.p, device.stream);
        for (const int assignments : {376, 8, 1536}) {
            const int padded = marlin_moe_padded_rows(assignments, experts, block);
            DeviceBuffer reduction(marlin_moe_c_tmp_bytes(assignments, n, experts, block, sms));
            DeviceBuffer sorted(std::size_t(padded) * sizeof(int));
            DeviceBuffer expert_ids(std::size_t(padded / block + 1) * sizeof(int));
            DeviceBuffer padded_count(sizeof(int)), locks(marlin_moe_lock_bytes());
            DeviceBuffer output(std::size_t(assignments) * n * 2);
            std::vector<int> offsets(experts + 1);
            for (int expert = 0; expert < experts; ++expert) {
                offsets[expert + 1] = offsets[expert] + assignments / experts + (expert < assignments % experts);
            }
            auto offsets_device = to_device_i32(offsets);
            std::vector<std::uint16_t> values(std::size_t(assignments) * k);
            for (int row = 0; row < assignments; ++row) {
                std::fill_n(values.begin() + std::size_t(row) * k, k, f32_to_bf16(float(1 + row % 7) / 64));
            }
            auto input = to_device(values);
            CUDA_CHECK(cudaMemsetAsync(locks.p, 0, locks.bytes, device.stream));
            marlin_moe_build_routing(static_cast<const int*>(offsets_device.p), experts, assignments,
                block, static_cast<int*>(sorted.p), static_cast<int*>(expert_ids.p),
                static_cast<int*>(padded_count.p), device.stream);
            marlin_moe_gemm_q4g64_bf16(input.p, packed.p, reordered_scales.p, output.p, reduction.p,
                static_cast<const int*>(sorted.p), static_cast<const int*>(expert_ids.p),
                static_cast<const int*>(padded_count.p), assignments, n, k, experts, block,
                static_cast<int*>(locks.p), device.stream);
            device.synchronize();
            const auto actual = from_device<std::uint16_t>(output.p, output.bytes / 2);
            for (int row = 0; row < assignments; ++row) {
                const auto expected = f32_to_bf16(float(-7 * k * (1 + row % 7)) / 64);
                for (int col = 0; col < n; ++col) {
                    if (actual[std::size_t(row) * n + col] != expected) {
                        std::cerr << "assignments=" << assignments << " row=" << row << " col=" << col
                                  << " actual=" << bf16_to_f32(actual[std::size_t(row) * n + col])
                                  << " expected=" << bf16_to_f32(expected) << '\n';
                        throw std::runtime_error("Marlin MoE reduction differs from the exact projection");
                    }
                }
            }
            std::cout << "assignments=" << assignments << " reduction bytes=" << reduction.bytes << " OK\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
