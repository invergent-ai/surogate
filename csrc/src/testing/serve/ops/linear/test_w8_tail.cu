#include "ops/linear/w8/w8_launch.h"
#include "ops/linear/w8/w8_rowsplit_gemm_simt.cuh"
#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <array>
#include <iostream>

using namespace sinfer;
using namespace sinfer::test;

namespace {

template <int Columns, bool AddResidual = false>
int check_case(int rows, int hidden, int padding, int tokens, int input_offset,
               cudaStream_t stream) {
    const int padded_k = ((hidden + 31) / 32) * 32 + padding;
    auto packed = quantized_weight::make_patterned_weight(
        QType::W8G32_F16S, rows, padded_k, 179,
        {quantized_weight::RowSplitScalePattern::Small,
         quantized_weight::RowSplitCodePattern::Hashed});
    DeviceBuffer weights = to_device(packed.payload);
    Weight weight = packed.device_weight(weights.p);
    weight.k = hidden;
    weight.shape[1] = hidden;

    std::vector<float> activation(static_cast<std::size_t>(hidden) * tokens + input_offset);
    fill_uniform(activation, 193, -3.0F, 3.0F);
    DeviceBuffer input = to_device_bf16(activation);
    auto* x = static_cast<__nv_bfloat16*>(input.p) + input_offset;
    const std::size_t elements = static_cast<std::size_t>(rows) * tokens;
    GuardedDeviceBuffer expected(elements * 2), actual(elements * 2);
    expected.fill(0xff);
    actual.fill(0xff);
    if constexpr (AddResidual) {
        std::vector<float> residual(elements);
        fill_uniform(residual, 197, -1.0F, 1.0F);
        std::vector<std::uint16_t> bits(elements);
        for (std::size_t i = 0; i < elements; ++i) { bits[i] = f32_to_bf16(residual[i]); }
        expected.copy_from_host(bits.data(), bits.size() * 2);
        actual.copy_from_host(bits.data(), bits.size() * 2);
    }

    // The global-read fallback is the numerical reference. Staging may change
    // memory access, but it must preserve every output bit, including at the
    // final incomplete weight group and across padded rows and token tiles.
    const bool aligned = hidden % 8 == 0 && (reinterpret_cast<std::uintptr_t>(x) & 15) == 0;
    const int full_slabs = aligned ? hidden / 1024 : 0;
    using namespace ops::detail;
    constexpr auto epilogue = AddResidual ? W8Epilogue::Residual : W8Epilogue::Store;
    w8_rowsplit_gemm_simt_kernel<W8RowSplitSimtSchedule, Columns, 8, 2, false, epilogue>
        <<<dim3((rows + 7) / 8, (tokens + Columns - 1) / Columns), 256, 0, stream>>>(
            x, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales),
            W8ContiguousOutput{static_cast<__nv_bfloat16*>(expected.data()), rows},
            rows, hidden, tokens, weight.padded_shape[1], full_slabs);
    cuda_check_last_launch("W8 tail reference");
    Tensor tx(x, DType::BF16, {hidden, tokens});
    Tensor ty(actual.data(), DType::BF16, {rows, tokens});
    if constexpr (AddResidual) {
        const bool full = rows % 8 == 0 && tokens % Columns == 0;
        if constexpr (Columns == 4) { w8_linear_add_simt_r8_c4_launch(full, tx, weight, ty, stream); }
        else { w8_linear_add_simt_r8_c8_launch(full, tx, weight, ty, stream); }
    } else {
        if constexpr (Columns == 4) { launch_w8_simt_r8_c4(tx, weight, ty, stream); }
        else { launch_w8_simt_r8_c8(tx, weight, ty, stream); }
    }
    cuda_synchronize(stream);

    int failures = expected.verify_guards("W8 tail reference") + actual.verify_guards("W8 tail");
    const auto reference = from_device<std::uint16_t>(expected.data(), elements);
    const auto result = from_device<std::uint16_t>(actual.data(), elements);
    if (reference != result) {
        std::cerr << "W8 tail changed values: rows=" << rows << " K=" << hidden
                  << " padding=" << padding << " tokens=" << tokens
                  << " offset=" << input_offset << " tile=" << Columns
                  << " residual=" << AddResidual << '\n';
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreate(&stream), "create W8 tail stream");
    int failures = 0;
    for (int hidden : {512, 1023, 1024, 1032, 1056, 1088, 1280, 1536, 2016, 2040, 2048, 2560, 2568, 3073}) {
        for (int rows : {17, 32}) {
            for (int tokens : {1, 4, 5, 8, 9, 17}) {
                failures += check_case<4>(rows, hidden, 0, tokens, 0, stream);
                failures += check_case<8>(rows, hidden, 256, tokens, 0, stream);
                failures += check_case<4, true>(rows, hidden, 256, tokens, 0, stream);
                failures += check_case<8, true>(rows, hidden, 0, tokens, 0, stream);
            }
        }
    }
    for (int tokens : {1, 4, 9}) {
        failures += check_case<4>(17, 2560, 256, tokens, 1, stream);
        failures += check_case<8>(32, 1032, 0, tokens, 1, stream);
        failures += check_case<4, true>(17, 1032, 0, tokens, 1, stream);
        failures += check_case<8, true>(32, 2560, 256, tokens, 1, stream);
    }
    cuda_check(cudaStreamDestroy(stream), "destroy W8 tail stream");
    if (failures != 0) { return 1; }
    std::cout << "W8 staged tails match global reads exactly\n";
    return 0;
}
