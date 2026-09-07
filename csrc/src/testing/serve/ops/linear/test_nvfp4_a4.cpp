#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace sinfer;
using namespace sinfer::test::linear;

int run_nvfp4_a4() {
    constexpr std::array attn_invocations{
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1077, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1300, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gdn_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1077, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1300, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array gate_up_invocations{
        Invocation{5, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1077, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1300, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    constexpr std::array residual_invocations{
        Invocation{8, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1024, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1077, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{1300, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    int failures = 0;
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {14336, 5120, 719U, Comparison::Sampled, true, attn_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {16384, 5120, 721U, Comparison::Sampled, true, gdn_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {34816, 5120, 722U, Comparison::Sampled, true, gate_up_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 6144, 723U, Comparison::Sampled, true, residual_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {5120, 17408, 725U, Comparison::Sampled, true, residual_invocations});
    // The Qwen3.6-35B-A3B geometry as a compressed-tensors export binds it: one object per HF
    // Linear, so attention q/gate [4096,2048], k/v [512,2048], output [2048,4096], shared-expert
    // gate and up [512,2048], shared down [2048,512]. None is a registered shape, so every token
    // count runs the cuBLASLt W4A4 route, which is what admitted K = 2048 and K = 512 to the
    // activation quantizer.
    constexpr std::array generic_invocations{
        Invocation{1, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{2, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{4, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{17, CallForm::Policy, ops::LinearPolicy::AllowA4},
        Invocation{300, CallForm::Policy, ops::LinearPolicy::AllowA4},
    };
    // The oracle quantises the activation exactly as the kernel does (linear_test_common.cpp,
    // materialize_activation), so these compare under the A16 criterion -- BF16 storage plus
    // accumulation order -- at every width, one token included. Before that the A4 oracle used
    // the raw activation with a 16 % allowance, and this shape's one-token case sat 4 % over
    // the allowance's per-element bound on random data; the allowance was hiding the noise,
    // not testing the route.
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {4096, 2048, 731U, Comparison::Sampled, true, generic_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {512, 2048, 732U, Comparison::Sampled, true, generic_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {2048, 4096, 733U, Comparison::Sampled, true, generic_invocations});
    failures += run_shape("NVFP4_A4", ActivationCompute::A4, make_nvfp4_weight,
                          {2048, 512, 734U, Comparison::Sampled, true, generic_invocations});
    return failures;
}

} // namespace

int main() {
    if (!sinfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        const int failures = run_nvfp4_a4();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " NVFP4_A4 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "NVFP4_A4 Linear: " << error.what() << '\n';
        return 1;
    }
}
