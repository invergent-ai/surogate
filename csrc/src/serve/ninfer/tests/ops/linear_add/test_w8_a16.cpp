#include "ops/linear_add/linear_add_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

int w8_a16_conformance() {
    int failures = 0;

    constexpr std::array<std::int32_t, 4> kK4096RouteStarts{2, 49, 129, 641};
    constexpr std::array<std::int32_t, 5> kK4096RouteInteriors{1, 24, 96, 256, 1024};
    failures += ninfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{2048, 4096, 419U, kK4096RouteStarts, kK4096RouteInteriors});

    constexpr std::array<std::int32_t, 32> kK6144RouteStarts{
        2,    49,   129,  192,  193,  257,  385,  400,  401,  448,  449,
        481,  641,  673,  705,  785,  897,  961,  1024, 1025, 1121, 1281,
        1345, 1409, 1681, 1792, 1793, 1920, 1921, 2017, 2048, 2049,
    };
    constexpr std::array<std::int32_t, 33> kK6144RouteInteriors{
        1,    24,   96,   160,  192,  224,  320,  392,  400,  424,  448,
        464,  560,  656,  688,  744,  840,  928,  992,  1024, 1072, 1200,
        1312, 1376, 1544, 1736, 1792, 1856, 1920, 1968, 2032, 2048, 4096,
    };
    failures += ninfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{2048, 6144, 421U, kK6144RouteStarts, kK6144RouteInteriors});
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
