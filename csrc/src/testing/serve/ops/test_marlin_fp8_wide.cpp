#include "core/device.h"
#include "ops/linear/marlin/marlin_plane.h"
#include "ops/op_tester.h"

#include <iostream>

using namespace sinfer;
using namespace sinfer::test;

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        DeviceContext device;
        // This shape selects 64-row, 256-column reduction tiles on a wide
        // prefill even though the deployment's decode band is only 32 rows.
        constexpr int n = 5120, k = 17408, maximum = 512;
        ops::detail::marlin_plane_set_enabled(true);
        ops::detail::marlin_set_fixed_m(1);
        DeviceBuffer codes(static_cast<std::size_t>(n) * k);
        auto scales = to_device(std::vector<std::uint16_t>(n, f32_to_bf16(1.0F)));
        CUDA_CHECK(cudaMemset(codes.p, 0x38, codes.bytes)); // E4M3 value 1
        Weight weight{};
        weight.qtype = QType::FP8_E4M3FN_ROW_BF16S;
        weight.layout = QuantLayout::RowScale;
        weight.scale_dtype = DType::BF16;
        weight.qdata = weight.payload = codes.p;
        weight.scales = scales.p;
        weight.n = n;
        weight.k = k;
        if (!ops::detail::marlin_fp8_adopt_residency(weight, device.stream)) {
            throw std::runtime_error("FP8 Marlin adoption did not engage");
        }
        DeviceBuffer input(static_cast<std::size_t>(maximum) * k * 2);
        DeviceBuffer output(static_cast<std::size_t>(maximum) * n * 2);
        for (const int tokens : {1, 32, 64, 129, maximum}) {
            std::vector<std::uint16_t> values(static_cast<std::size_t>(tokens) * k);
            for (int t = 0; t < tokens; ++t) {
                std::fill_n(values.begin() + static_cast<std::size_t>(t) * k, k,
                            f32_to_bf16(float(t % 7 + 1) / 64.0F));
            }
            CUDA_CHECK(cudaMemcpyAsync(input.p, values.data(), values.size() * 2,
                                       cudaMemcpyHostToDevice, device.stream));
            Tensor x(input.p, DType::BF16, {k, tokens});
            Tensor out(output.p, DType::BF16, {n, tokens});
            for (const bool captured : {false, true}) {
                cudaGraph_t graph{};
                cudaGraphExec_t executable{};
                if (captured) { CUDA_CHECK(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeThreadLocal)); }
                if (!ops::detail::marlin_fp8_run(x, weight, out, device.stream)) {
                    throw std::runtime_error("FP8 Marlin projection declined");
                }
                if (captured) {
                    CUDA_CHECK(cudaStreamEndCapture(device.stream, &graph));
                    CUDA_CHECK(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
                    CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
                    CUDA_CHECK(cudaGraphLaunch(executable, device.stream));
                }
                device.synchronize();
                const auto actual = from_device<std::uint16_t>(output.p, out.numel());
                for (int t = 0; t < tokens; ++t) {
                    const auto expected = f32_to_bf16(float(k * (t % 7 + 1)) / 64.0F);
                    for (int row = 0; row < n; ++row) {
                        if (actual[static_cast<std::size_t>(t) * n + row] != expected) {
                            throw std::runtime_error("FP8 Marlin output differs at width " + std::to_string(tokens));
                        }
                    }
                }
                if (captured) {
                    CUDA_CHECK(cudaGraphExecDestroy(executable));
                    CUDA_CHECK(cudaGraphDestroy(graph));
                }
            }
        }
        std::cout << "FP8 Marlin narrow/wide eager and graph projections: OK\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
