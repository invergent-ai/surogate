#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace sinfer;
using namespace sinfer::test::linear;

int run_nvfp4_a16() {
    constexpr std::array attn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{8, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{20, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{32, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{33, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    constexpr std::array new_problem_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::A16Only},
        Invocation{16, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {14336, 5120, 701U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {16384, 5120, 703U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {34816, 5120, 704U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 6144, 705U, Comparison::Sampled, true, new_problem_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {5120, 17408, 707U, Comparison::Sampled, true, new_problem_invocations});
    // The hidden-2560 family (Qwen3.5-4B) has the decode GEMV at one token and nothing wider.
    constexpr std::array decode_only_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::A16Only},
    };
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {10240, 2560, 711U, Comparison::Sampled, true, decode_only_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {12288, 2560, 712U, Comparison::Sampled, true, decode_only_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {18432, 2560, 713U, Comparison::Sampled, true, decode_only_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2560, 4096, 714U, Comparison::Sampled, true, decode_only_invocations});
    failures += run_shape("NVFP4_A16", ActivationCompute::A16, make_nvfp4_weight,
                          {2560, 9216, 715U, Comparison::Sampled, true, decode_only_invocations});
    return failures;
}

} // namespace

int main() {
    if (!sinfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a16();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
