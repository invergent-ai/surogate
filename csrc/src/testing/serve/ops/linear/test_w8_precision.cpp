#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "api/ops/linear_pair.h"
#include "api/ops/gdn_input_proj.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <iostream>
#include <memory>
#include <utility>

using namespace sinfer;
using namespace sinfer::test;

int verify_batch_invariance() {
    constexpr int n = 1024, k = 2560, columns = 129;
    auto packed = quantized_weight::make_patterned_weight(QType::W8G32_F16S, n, k, 123,
        {quantized_weight::RowSplitScalePattern::Small, quantized_weight::RowSplitCodePattern::Hashed});
    DeviceBuffer dw(packed.payload.size()), dx(std::size_t(k) * columns * 2), dy(std::size_t(n) * columns * 2);
    dw.copy_from_host(packed.payload.data(), dw.bytes);
    const Weight weight = packed.device_weight(dw.p);
    std::vector<float> input(std::size_t(k) * columns);
    fill_uniform(input, 987, -3.0F, 3.0F);
    std::vector<std::uint16_t> bits(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) { bits[i] = f32_to_bf16(input[i]); }
    dx.copy_from_host(bits.data(), dx.bytes);
    ops::detail::marlin_plane_set_enabled(true);
    std::vector<std::uint16_t> reference;
    int failures = 0;
    for (const int t : {columns, 1, 4, 8, 16, 17, 32, 33, 64, 65, 128}) {
        Tensor x(dx.p, DType::BF16, {k, t}), y(dy.p, DType::BF16, {n, t});
        ops::linear(x, weight, y, nullptr);
        cuda_synchronize();
        const auto actual = from_device<std::uint16_t>(y.data, y.numel());
        if (reference.empty()) { reference = actual; }
        else if (!std::equal(actual.begin(), actual.end(), reference.begin())) {
            std::cerr << "Generic W8 reduction changed at batch width " << t << '\n';
            ++failures;
        }
    }
    return failures;
}

int verify_gdn_batch_invariance() {
    int failures = 0;
    constexpr int columns = 257;
    for (const auto [n, k] : {std::pair{8192, 1024}, std::pair{8192, 2048},
                              std::pair{12288, 2048}, std::pair{12288, 2560}}) {
        const int qkv_rows = n == 8192 ? 6144 : 8192;
        const int z_rows = n - qkv_rows;
        auto packed = quantized_weight::make_patterned_weight(QType::W8G32_F16S, n, k, 127,
            {quantized_weight::RowSplitScalePattern::Small, quantized_weight::RowSplitCodePattern::Hashed});
        DeviceBuffer dw(packed.payload.size()), dx(std::size_t(k) * columns * 2),
                     dy(std::size_t(n) * columns * 2);
        dw.copy_from_host(packed.payload.data(), dw.bytes);
        const Weight weight = packed.device_weight(dw.p);
        std::vector<float> input(std::size_t(k) * columns);
        fill_uniform(input, 1991, -3.0F, 3.0F);
        std::vector<std::uint16_t> bits(input.size());
        for (std::size_t i = 0; i < input.size(); ++i) { bits[i] = f32_to_bf16(input[i]); }
        dx.copy_from_host(bits.data(), dx.bytes);
        std::vector<std::uint16_t> reference_qkv, reference_z;
        for (const int t : {columns, 2, 4, 8, 16, 17, 32, 33, 49, 64, 65, 128, 129, 256}) {
            Tensor x(dx.p, DType::BF16, {k, t});
            Tensor qkv(dy.p, DType::BF16, {qkv_rows, t});
            Tensor z(static_cast<std::uint16_t*>(dy.p) + qkv_rows * t, DType::BF16, {z_rows, t});
            ops::gdn_input_proj(x, weight, qkv, z, nullptr);
            cuda_synchronize();
            const auto actual_qkv = from_device<std::uint16_t>(qkv.data, qkv.numel());
            const auto actual_z = from_device<std::uint16_t>(z.data, z.numel());
            if (reference_qkv.empty()) {
                reference_qkv = actual_qkv;
                reference_z = actual_z;
            } else if (!std::equal(actual_qkv.begin(), actual_qkv.end(), reference_qkv.begin()) ||
                       !std::equal(actual_z.begin(), actual_z.end(), reference_z.begin())) {
                std::cerr << "GDN W8 batch reduction changed: " << n << 'x' << k << " T=" << t << '\n';
                ++failures;
            }
        }
    }
    return failures;
}

int main() {
    if (cuda_unavailable()) { return 77; }
    // Two nonzero activations remove reduction-order ambiguity. Their weights
    // use FP16 scales outside the BF16 grid, and their difference exposes both
    // early scale rounding and missing per-weight rounding before the GEMM.
    std::vector<std::unique_ptr<DeviceBuffer>> weights;
    int failures = verify_batch_invariance() + verify_gdn_batch_invariance();
    for (const auto [n, k, padding] : {std::array{1024, 2560, 0}, std::array{768, 1152, 0},
                                      std::array{2048, 16384, 0}, std::array{5120, 10240, 0},
                                      std::array{2560, 5120, 0}, std::array{1024, 2560, 256},
                                      std::array{12288, 2048, 0}, std::array{2048, 6144, 0},
                                      std::array{6144, 2048, 0}}) {
        const int padded_k = k + padding;
        auto packed = quantized_weight::make_patterned_weight(QType::W8G32_F16S, n, padded_k, 91);
        const int groups = padded_k / 32;
        std::fill_n(packed.payload.begin(), packed.code_plane_bytes, 0);
        std::vector<std::uint16_t> reference(n);
        std::vector<std::uint16_t> residual_reference(n);
        for (int row = 0; row < n; ++row) {
            const auto scale0 = std::uint16_t(0x2c41 + row % 31);
            const auto scale1 = std::uint16_t(0x2c83 + row % 29);
            const auto code0 = std::int8_t(91 + row % 31);
            const auto code1 = std::int8_t(73 + row % 37);
            packed.payload[std::size_t(row) * padded_k] = std::uint8_t(code0);
            packed.payload[std::size_t(row) * padded_k + k - 32] = std::uint8_t(code1);
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + std::size_t(row) * groups * 2, scale0);
            quantized_weight::detail::store_u16_le(packed.payload,
                packed.scale_plane_offset + (std::size_t(row) * groups + k / 32 - 1) * 2, scale1);
            const float w0 = bf16_to_f32(f32_to_bf16(float(code0) * quantized_weight::detail::f16_to_f32(scale0)));
            const float w1 = bf16_to_f32(f32_to_bf16(float(code1) * quantized_weight::detail::f16_to_f32(scale1)));
            reference[row] = f32_to_bf16(w0 - w1);
            residual_reference[row] = f32_to_bf16(w0 - w1 + 0.25F);
        }
        // The derived-plane registry keys weights by address; retain allocations
        // until every case has finished so a different matrix cannot reuse one.
        weights.push_back(std::make_unique<DeviceBuffer>(packed.payload.size()));
        auto& dw = *weights.back();
        dw.copy_from_host(packed.payload.data(), dw.bytes);
        Weight weight = packed.device_weight(dw.p);
        weight.k = weight.shape[1] = k;
        constexpr int max_t = 129;
        std::vector<std::uint16_t> input(std::size_t(k) * max_t, 0);
        for (int col = 0; col < max_t; ++col) {
            input[std::size_t(col) * k] = f32_to_bf16(1.0F);
            input[std::size_t(col) * k + k - 32] = f32_to_bf16(-1.0F);
        }
        DeviceBuffer dx(input.size() * 2), dy(std::size_t(n) * max_t * 2);
        dx.copy_from_host(input.data(), dx.bytes);
        for (bool marlin : {false, true}) {
            ops::detail::marlin_plane_set_enabled(marlin);
            for (const int t : {1, 4, 16, 17, 32, 33, 49, 64, 128, 129}) {
                Tensor x(dx.p, DType::BF16, {k, t}), y(dy.p, DType::BF16, {n, t});
                ops::linear(x, weight, y, nullptr);
                cuda_synchronize();
                const auto actual = from_device<std::uint16_t>(y.data, y.numel());
                std::size_t mismatches = 0;
                for (std::size_t i = 0; i < actual.size(); ++i) {
                    mismatches += actual[i] != reference[i % n];
                }
                if (n == 12288) {
                    Tensor qkv(dy.p, DType::BF16, {8192, t});
                    Tensor z(static_cast<std::uint16_t*>(dy.p) + 8192 * t, DType::BF16, {4096, t});
                    ops::gdn_input_proj(x, weight, qkv, z, nullptr);
                    cuda_synchronize();
                    const auto projected = from_device<std::uint16_t>(dy.p, std::size_t(n) * t);
                    for (int col = 0; col < t; ++col) {
                        for (int row = 0; row < 8192; ++row) {
                            mismatches += projected[col * 8192 + row] != reference[row];
                        }
                        for (int row = 0; row < 4096; ++row) {
                            mismatches += projected[8192 * t + col * 4096 + row] != reference[8192 + row];
                        }
                    }
                }
                if (n == 2048 && k == 6144) {
                    std::vector<std::uint16_t> residual(std::size_t(n) * t, f32_to_bf16(0.25F));
                    dy.copy_from_host(residual.data(), residual.size() * 2);
                    WorkspaceArena workspace(256);
                    ops::linear_add(x, weight, y, workspace, nullptr);
                    cuda_synchronize();
                    const auto updated = from_device<std::uint16_t>(y.data, y.numel());
                    for (std::size_t i = 0; i < updated.size(); ++i) {
                        mismatches += updated[i] != residual_reference[i % n];
                    }
                }
                if (n == 6144 && k == 2048) {
                    const Weight key = packed.device_row_view(dw.p, 4096, 1024);
                    const Weight value = packed.device_row_view(dw.p, 5120, 1024);
                    Tensor keys(dy.p, DType::BF16, {1024, t});
                    Tensor values(static_cast<std::uint16_t*>(dy.p) + 1024 * t,
                                  DType::BF16, {1024, t});
                    ops::linear_pair(x, key, value, keys, values, nullptr);
                    cuda_synchronize();
                    const auto projected = from_device<std::uint16_t>(dy.p, std::size_t(2048) * t);
                    for (int col = 0; col < t; ++col) {
                        for (int row = 0; row < 1024; ++row) {
                            mismatches += projected[col * 1024 + row] != reference[4096 + row];
                            mismatches += projected[1024 * t + col * 1024 + row] != reference[5120 + row];
                        }
                    }
                }
                if (mismatches) {
                    std::cerr << "W8 precision " << n << 'x' << k << " T=" << t
                              << " padding=" << padding << " Marlin=" << marlin << ": " << mismatches << " mismatches\n";
                    ++failures;
                }
            }
        }
    }
    if (!failures) { std::cout << "W8 weights agree across scalar, tensor-core and Marlin batches\n"; }
    return failures ? 1 : 0;
}
