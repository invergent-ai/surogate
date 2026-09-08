#include "ops/linear_add/linear_add_test_common.h"

#include "ops/linear_add/w8/w8_linear_add_kernels.h"
#include "ops/linear_add/w8/w8_linear_add_plan.h"
#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using sinfer::test::linear_add::ShapeCase;
using sinfer::test::linear_add::WeightFormat;

// Every case lists the first column of each band of its route table, so the
// harness samples both sides of every band edge (boundary-1, boundary,
// boundary+1) plus the interiors. Widening a band without moving its start here
// leaves the new edge unsampled; the plan/launcher agreement below catches a
// band that outgrew its bake, not a numerical fault inside it.
int w8_a16_conformance() {
    int failures = 0;

    constexpr std::array<std::int32_t, 0> kGenericStarts{};
    constexpr std::array<std::int32_t, 3> kGenericTokens{1, 17, 128};
    for (const auto& [n, k] : std::array<std::pair<int, int>, 3>{
             {{1024, 1024}, {128, 256}, {768, 1152}}}) {
        failures += sinfer::test::linear_add::run_shape(
            "W8_A16 LinearAdd fallback", WeightFormat::W8G32F16S,
            ShapeCase{n, k, 469U, kGenericStarts, kGenericTokens});
    }

    constexpr std::array<std::int32_t, 4> kK4096RouteStarts{2, 49, 129, 641};
    constexpr std::array<std::int32_t, 5> kK4096RouteInteriors{1, 24, 96, 256, 1024};
    failures += sinfer::test::linear_add::run_shape(
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
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{2048, 6144, 421U, kK6144RouteStarts, kK6144RouteInteriors});

    // qwen3.5-0.8b residual projections (attn/gdn output 1024x2048, mlp down
    // 1024x3584): exact-T 2..32, r32c128 33..1024, r48c128 above (1024 % 48 ==
    // 16, so the r48 tile runs partial at every T there).
    constexpr std::array<std::int32_t, 4> kQ08RouteStarts{2, 33, 129, 1025};
    constexpr std::array<std::int32_t, 8> kQ08RouteInteriors{1, 4, 8, 16, 24, 64, 256, 2048};
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{1024, 2048, 431U, kQ08RouteStarts, kQ08RouteInteriors});
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{1024, 3584, 433U, kQ08RouteStarts, kQ08RouteInteriors});

    // qwen3.5-2b output projections: exact-T 2..32, r32c96, r48c128, r32c128,
    // r48c128.
    constexpr std::array<std::int32_t, 5> kQ2BRouteStarts{2, 33, 513, 901, 1025};
    constexpr std::array<std::int32_t, 8> kQ2BRouteInteriors{1, 4, 8, 16, 24, 64, 256, 2048};
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{2048, 2048, 439U, kQ2BRouteStarts, kQ2BRouteInteriors});

    // Shapes on the extent-agnostic table: SIMT at T=1, r32c128 2..1024,
    // r48c128 above. 640 % 48 == 16 and 1024 % 48 == 16, so the r48 tile runs
    // partial rows at every T past 1024 for all three of these.
    constexpr std::array<std::int32_t, 2> kExtentAgnosticRouteStarts{2, 1025};
    constexpr std::array<std::int32_t, 6> kExtentAgnosticRouteInteriors{1, 8, 16, 512, 1024,
                                                                        2048};
    // qwen3-0.6b mlp down.
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{1024, 3072, 449U, kExtentAgnosticRouteStarts, kExtentAgnosticRouteInteriors});
    // gemma-3-270m attention output and mlp down. These are the first exercise of
    // the fused op at 640 rows: the wrapper used to refuse them at every T while
    // the plan admitted them.
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{640, 1024, 457U, kExtentAgnosticRouteStarts, kExtentAgnosticRouteInteriors});
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{640, 2048, 461U, kExtentAgnosticRouteStarts, kExtentAgnosticRouteInteriors});

    // tinyllama-1.1b mlp down {2048, 5632}. Starts cover the extent-agnostic
    // table today and the exact-T / r32c128 / r48c128 bands if the shape gets a
    // bake of its own; both sides of T=32/33 and 128/129 are sampled either way.
    constexpr std::array<std::int32_t, 4> kR2048K5632RouteStarts{2, 33, 129, 1025};
    constexpr std::array<std::int32_t, 8> kR2048K5632RouteInteriors{1,  8,   16,  24,
                                                                    32, 64, 512, 2048};
    failures += sinfer::test::linear_add::run_shape(
        "W8_A16 LinearAdd", WeightFormat::W8G32F16S,
        ShapeCase{2048, 5632, 443U, kR2048K5632RouteStarts, kR2048K5632RouteInteriors});
    return failures;
}

int expect_throw_containing(std::string_view label, std::string_view needle, auto&& body) {
    try {
        body();
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find(needle) != std::string_view::npos) { return 0; }
        std::cerr << label << ": threw without naming the constraint: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << label << ": wrong exception type: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": did not throw\n";
    return 1;
}

// The scale-row alignment rule, at both levels that enforce it. k = 1152 is
// EmbeddingGemma's hidden: k % 32 == 0 (whole scale groups), k % 128 == 0 (no
// artifact padding), k % 256 != 0 (a scale row is 72 bytes, so every odd row
// starts 8 bytes off the 16-byte cp.async).
int w8_alignment_refusals() {
    using sinfer::ops::detail::W8LinearAddProblem;
    int failures = 0;
    constexpr std::int32_t kMisalignedK = 1152;

    // Plan level: resolve names the constraint, and the shape is not admitted.
    failures += expect_throw_containing("w8 plan k%256 refusal", "k % 256", [&] {
        (void)sinfer::ops::detail::w8_linear_add_resolve_plan({2048, kMisalignedK, kMisalignedK, 8});
    });
    if (sinfer::ops::detail::w8_linear_add_admits({2048, kMisalignedK, kMisalignedK, 8})) {
        std::cerr << "w8 admits: k " << kMisalignedK << " must not be admitted\n";
        ++failures;
    }

    // Launcher level: a direct MMA launch over a misaligned k throws before it
    // launches (the bench's candidate rows reach the launcher without the plan).
    constexpr std::int32_t kRows = 2048;
    constexpr std::int32_t kT    = 8;
    const sinfer::test::quantized_weight::PackedWeight host_weight =
        sinfer::test::quantized_weight::make_patterned_weight(sinfer::QType::W8G32_F16S, kRows,
                                                              kMisalignedK, 467U);
    if (host_weight.weight.padded_shape[1] != kMisalignedK) {
        std::cerr << "w8 launcher refusal: fixture padded k to " << host_weight.weight.padded_shape[1]
                  << ", the case needs padded_k == k\n";
        return failures + 1;
    }
    sinfer::test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const sinfer::Weight weight = host_weight.device_weight(device_weight.data());
    sinfer::test::GuardedDeviceBuffer device_x(static_cast<std::size_t>(kMisalignedK) * kT * 2);
    sinfer::test::GuardedDeviceBuffer device_out(static_cast<std::size_t>(kRows) * kT * 2);
    device_x.fill(0);
    device_out.fill(0);
    const sinfer::Tensor x(device_x.data(), sinfer::DType::BF16, {kMisalignedK, kT});
    sinfer::Tensor out(device_out.data(), sinfer::DType::BF16, {kRows, kT});
    failures += expect_throw_containing("w8 mma_r32_c128 launcher k%256 refusal", "k % 256", [&] {
        sinfer::ops::detail::w8_linear_add_mma_r32_c128_launch(false, x, weight, out, nullptr);
    });
    failures += expect_throw_containing("w8 mma_r48_c128 launcher k%256 refusal", "k % 256", [&] {
        sinfer::ops::detail::w8_linear_add_mma_r48_c128_launch(false, x, weight, out, nullptr);
    });
    sinfer::test::cuda_check(cudaDeviceSynchronize(), "synchronize after refused launches");
    failures += device_out.verify_guards("w8 launcher refusal output");
    return failures;
}

// Plan and launcher agree on every registered shape: every T resolves; a band
// that names the exact-T bake has a table covering it (the fall-through
// PATCHES.md #89 describes ran a neighbour's bake); every baked T is routed to
// its bake (no dead bakes); the medium split-K and decode bakes are the 35B's.
int w8_plan_launcher_agreement() {
    using sinfer::ops::detail::W8LinearAddScheduleId;
    int failures = 0;
    std::vector<std::int32_t> tokens;
    for (std::int32_t t = 1; t <= 1100; ++t) { tokens.push_back(t); }
    tokens.push_back(2048);
    tokens.push_back(4096);

    const auto shapes = sinfer::ops::detail::w8_linear_add_registered_shapes();
    if (shapes.empty()) {
        std::cerr << "w8 registered shapes: empty\n";
        return 1;
    }
    for (const auto& shape : shapes) {
        const std::string label = "w8 plan/launcher [" + std::to_string(shape.rows) + "," +
                                  std::to_string(shape.k) + "]";
        if (shape.rows <= 0 || shape.k <= 0) {
            std::cerr << label << ": phantom registration\n";
            ++failures;
            continue;
        }
        const std::int32_t baked_last =
            sinfer::ops::detail::w8_linear_add_exact_t_last_cols(shape.rows, shape.k);
        for (const std::int32_t t : tokens) {
            W8LinearAddScheduleId schedule;
            try {
                schedule = sinfer::ops::detail::w8_linear_add_resolve_plan(
                               {shape.rows, shape.k, shape.k, t})
                               .schedule;
            } catch (const std::exception& error) {
                std::cerr << label << " T=" << t << ": resolve threw: " << error.what() << '\n';
                ++failures;
                continue;
            }
            const bool exact = schedule == W8LinearAddScheduleId::SplitKMmaExactT;
            const bool baked =
                sinfer::ops::detail::w8_linear_add_exact_t_covers(shape.rows, shape.k, t);
            if (exact && !baked) {
                std::cerr << label << " T=" << t << ": routed to the exact-T bake, which has "
                          << (baked_last == 0 ? "no table for this shape"
                                              : "a table ending at T=" + std::to_string(baked_last))
                          << '\n';
                ++failures;
            }
            if (baked && !exact) {
                std::cerr << label << " T=" << t << ": baked (table to T=" << baked_last
                          << ") but routed to "
                          << sinfer::ops::detail::w8_linear_add_schedule_name(schedule) << '\n';
                ++failures;
            }
            if ((schedule == W8LinearAddScheduleId::MediumSplitK ||
                 schedule == W8LinearAddScheduleId::DecodeR16) &&
                !(shape.rows == 2048 && (shape.k == 4096 || shape.k == 6144))) {
                std::cerr << label << " T=" << t << ": "
                          << sinfer::ops::detail::w8_linear_add_schedule_name(schedule)
                          << " is baked for 2048 rows over k 4096/6144 only\n";
                ++failures;
            }
        }
    }
    return failures;
}

} // namespace

int main() {
    if (!sinfer::test::linear_add::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        int failures = 0;
        failures += w8_plan_launcher_agreement();
        failures += w8_alignment_refusals();
        failures += w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
