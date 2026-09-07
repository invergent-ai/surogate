// The row-scaled FP8 route on shapes its table does not hold.
//
// Every registered geometry is a Qwen3.8-27B projection; a weight of any other shape in this
// format used to have no launch at all. These cases are the generic route: the runtime-shaped
// BF16 kernel at narrow widths, the cuBLASLt GEMM over per-token E4M3 activations at prefill
// widths, and the same two through `linear_add`'s residual form. The shapes are deliberately not
// in the table -- one of them is a fused MLP parent's half, which is what an adapter projects.

#include "ops/linear/linear_test_common.h"

#include "api/ops/linear.h"
#include "api/ops/linear_add.h"
#include "core/arena.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/quantized_weight.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <vector>

namespace {

using namespace sinfer;
using namespace sinfer::test;
using namespace sinfer::test::linear;

/// Round a float to BF16 the way the device does, and keep the bits: the reference then reads
/// exactly the activation the kernel reads.
std::uint16_t to_bf16_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t rounding = 0x7FFFU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

float from_bf16_bits(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value              = 0.0F;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

bool cuda_ok(cudaError_t status, const char* what) {
    if (status == cudaSuccess) { return true; }
    std::cerr << "FP8 generic: " << what << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

/// `linear_add` on a shape outside the table. What is new here is the accumulate path, so the
/// check is differential: the same route's `linear` into a fresh output, and `linear_add` into a
/// residual, must agree to the rounding of one BF16 store. The GEMM's own accuracy is what the
/// `linear` cases above measure against an FP64 reference.
int run_linear_add(std::int32_t n, std::int32_t k, std::int32_t tokens, ops::LinearPolicy policy) {
    const std::string label = "FP8_generic linear_add [" + std::to_string(n) + "," +
                              std::to_string(k) + "] T=" + std::to_string(tokens);
    quantized_weight::PackedWeight host_weight =
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, n, k, 4231U);

    std::vector<std::uint16_t> activation(static_cast<std::size_t>(k) * tokens);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(n) * tokens);
    for (std::size_t index = 0; index < activation.size(); ++index) {
        activation[index] = to_bf16_bits(0.05F * static_cast<float>((index * 37U) % 41U) - 1.0F);
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
        residual[index] = to_bf16_bits(0.25F * static_cast<float>((index * 11U) % 7U) - 0.5F);
    }

    void* device_weight_bytes = nullptr;
    void* device_activation   = nullptr;
    void* device_residual     = nullptr;
    void* device_store        = nullptr;
    const std::size_t output_bytes = residual.size() * sizeof(std::uint16_t);
    if (!cuda_ok(cudaMalloc(&device_weight_bytes, host_weight.payload.size()), "malloc weight") ||
        !cuda_ok(cudaMalloc(&device_activation, activation.size() * sizeof(std::uint16_t)),
                 "malloc activation") ||
        !cuda_ok(cudaMalloc(&device_residual, output_bytes), "malloc residual") ||
        !cuda_ok(cudaMalloc(&device_store, output_bytes), "malloc store")) {
        return 1;
    }
    int failures = 0;
    if (cuda_ok(cudaMemcpy(device_weight_bytes, host_weight.payload.data(),
                           host_weight.payload.size(), cudaMemcpyHostToDevice),
                "copy weight") &&
        cuda_ok(cudaMemcpy(device_activation, activation.data(),
                           activation.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
                "copy activation") &&
        cuda_ok(cudaMemcpy(device_residual, residual.data(), output_bytes, cudaMemcpyHostToDevice),
                "copy residual")) {
        const Weight weight = host_weight.device_weight(device_weight_bytes);
        Tensor input(device_activation, DType::BF16, {k, tokens});
        Tensor accumulated(device_residual, DType::BF16, {n, tokens});
        Tensor stored(device_store, DType::BF16, {n, tokens});
        try {
            const std::size_t add_capacity =
                ops::linear_add_workspace_capacity_bytes(weight.qtype, n, k, policy, tokens, tokens);
            const std::size_t store_capacity =
                ops::linear_workspace_capacity_bytes(weight.qtype, n, k, policy, tokens, tokens);
            DeviceArena add_workspace(std::max<std::size_t>(add_capacity, 256));
            DeviceArena store_workspace(std::max<std::size_t>(store_capacity, 256));
            ops::linear_add(input, weight, accumulated, policy, add_workspace, nullptr);
            ops::linear(input, weight, stored, policy, store_workspace, nullptr);
            if (cuda_ok(cudaDeviceSynchronize(), "synchronize linear_add")) {
                std::vector<std::uint16_t> got(residual.size());
                std::vector<std::uint16_t> store(residual.size());
                cuda_ok(cudaMemcpy(got.data(), device_residual, output_bytes, cudaMemcpyDeviceToHost),
                        "read back residual");
                cuda_ok(cudaMemcpy(store.data(), device_store, output_bytes, cudaMemcpyDeviceToHost),
                        "read back store");
                double worst   = 0.0;
                double largest = 0.0;
                for (std::size_t index = 0; index < got.size(); ++index) {
                    const double projection = from_bf16_bits(store[index]);
                    const double expected   = projection + from_bf16_bits(residual[index]);
                    const double actual     = from_bf16_bits(got[index]);
                    largest                 = std::max(largest, std::abs(projection));
                    worst = std::max(worst, std::abs(actual - expected) /
                                                std::max(1.0, std::abs(projection)));
                }
                if (largest == 0.0) {
                    std::cerr << label << ": the projection is all zeros, nothing was measured\n";
                    ++failures;
                } else if (!(worst <= 0.02)) {
                    std::cerr << label << ": residual and projection disagree by " << worst << '\n';
                    ++failures;
                } else {
                    std::cout << label << ": accumulate agrees to " << worst << '\n';
                }
            } else {
                ++failures;
            }
        } catch (const std::exception& error) {
            std::cerr << label << ": " << error.what() << '\n';
            ++failures;
        }
    } else {
        ++failures;
    }
    cudaFree(device_weight_bytes);
    cudaFree(device_activation);
    cudaFree(device_residual);
    cudaFree(device_store);
    return failures;
}

int run_fp8_generic() {
    // Not one of the six registered geometries, and the route says so.
    static_assert(!ops::detail::is_fp8_registered_problem(4096, 2048));
    static_assert(ops::detail::is_fp8_generic_problem(4096, 2048));
    static_assert(ops::detail::is_fp8_generic_problem(17408, 5120)); // a fused MLP parent's half
    static_assert(!ops::detail::is_fp8_generic_problem(34816, 5120)); // the parent itself
    static_assert(!ops::detail::is_fp8_linear_problem(4096, 2044));   // K is not whole groups

    constexpr std::array narrow_invocations{
        Invocation{1, CallForm::A16Convenience, ops::LinearPolicy::A16Only},
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{3, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{64, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    int failures = run_shape("FP8_generic", ActivationCompute::A16, make_fp8_weight,
                             {4096, 2048, 4211U, Comparison::Sampled, true, narrow_invocations});
    failures += run_shape("FP8_generic", ActivationCompute::A16, make_fp8_weight,
                          {17408, 5120, 4217U, Comparison::Sampled, true, narrow_invocations});
    // A tail that is not a whole number of 1024-value chunks, and a single row.
    failures += run_shape("FP8_generic", ActivationCompute::A16, make_fp8_weight,
                          {512, 1568, 4219U, Comparison::Full, true, narrow_invocations});

    constexpr std::array wide_invocations{
        Invocation{65, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{128, CallForm::Policy, ops::LinearPolicy::AllowA8},
        Invocation{257, CallForm::Policy, ops::LinearPolicy::AllowA8},
    };
    failures += run_shape("FP8_generic", ActivationCompute::A8, make_fp8_weight,
                          {4096, 2048, 4223U, Comparison::Sampled, true, wide_invocations});
    failures += run_shape("FP8_generic", ActivationCompute::A8, make_fp8_weight,
                          {17408, 5120, 4229U, Comparison::Sampled, true, wide_invocations});

    failures += run_linear_add(1024, 1024, 3, ops::LinearPolicy::A16Only);
    failures += run_linear_add(1024, 1024, 96, ops::LinearPolicy::AllowA8);
    return failures;
}

} // namespace

int main() {
    if (!cuda_available()) {
        std::cout << "FP8 generic linear test skipped: no CUDA device\n";
        return 0;
    }
    try {
        const int failures = run_fp8_generic();
        if (failures != 0) {
            std::cerr << "FP8 generic linear test: " << failures << " failure(s)\n";
            return 1;
        }
    } catch (const std::exception& error) {
        std::cerr << "FP8 generic linear test: " << error.what() << '\n';
        return 1;
    }
    std::cout << "FP8 generic linear test passed\n";
    return 0;
}
