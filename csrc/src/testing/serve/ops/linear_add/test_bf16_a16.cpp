#include "ops/linear_add/linear_add_test_common.h"

#include "api/ops/linear_add.h"
#include "ops/op_tester.h"

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {

using sinfer::test::linear_add::ShapeCase;
using sinfer::test::linear_add::WeightFormat;

template <class Callable>
int expect_invalid(Callable&& callable, const char* label) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int bf16_a16_rejections() {
    int failures = 0;
    failures += expect_invalid(
        [] {
            (void)sinfer::ops::linear_add_workspace_capacity_bytes(sinfer::QType::BF16_CTRL, 5120,
                                                                   6143, 1, 32);
        },
        "BF16_A16 LinearAdd workspace shape");

    sinfer::DeviceBuffer input(static_cast<std::size_t>(6144) * sizeof(std::uint16_t));
    sinfer::DeviceBuffer residual(static_cast<std::size_t>(5120) * sizeof(std::uint16_t));
    sinfer::DeviceBuffer weight_storage(256);
    sinfer::Weight weight{};
    weight.qtype  = sinfer::QType::BF16_CTRL;
    weight.layout = sinfer::QuantLayout::RowSplit;
    weight.qdata  = weight_storage.p;
    weight.n      = 5120;
    weight.k      = 6144;
    sinfer::Tensor x(input.p, sinfer::DType::BF16, {6144, 1});
    sinfer::Tensor out(residual.p, sinfer::DType::BF16, {5120, 1});
    sinfer::WorkspaceArena workspace(1);
    failures += expect_invalid([&] { sinfer::ops::linear_add(x, weight, out, workspace, nullptr); },
                               "BF16_A16 LinearAdd weight layout");
    return failures;
}

int bf16_a16_conformance() {
    constexpr std::array<std::int32_t, 3> kRouteStarts{2, 5, 49};
    constexpr std::array<std::int32_t, 10> kRouteInteriors{
        4, 8, 16, 32, 48, 127, 128, 129, 1024, 1536,
    };
    return sinfer::test::linear_add::run_shape(
               "BF16_A16 LinearAdd", WeightFormat::BF16,
               ShapeCase{5120, 6144, 431U, kRouteStarts, kRouteInteriors}) +
           bf16_a16_rejections();
}

} // namespace

int main() {
    if (!sinfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = bf16_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
