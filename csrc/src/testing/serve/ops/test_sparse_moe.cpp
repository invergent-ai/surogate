#include "ops/parallel_rows.h"

#include "api/ops/sparse_moe.h"

#include "ops/op_tester.h"
#include "ops/quantized_weight.h"

#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <future>
#include <thread>
#include <utility>
#include <vector>

using namespace sinfer;
using namespace sinfer::test;

namespace {

// SparseMoe is the exact qwen3_6_35b_a3b post-mixer Op. qwen3_6_27b has a dense SwiGLU
// post-mixer and therefore contributes no second SparseMoe geometry.
/// Which codecs a run exercises. Independent of the mixture, so it is declared once and both
/// instantiations below take it.
struct CodecProfile {
    const char* name;
    QType routed_gate_up;
    QType routed_down;
    std::span<const std::int32_t> token_cases;
    bool verify_graph_replay;
};

/// One registered mixture's constants, as the kernels' own instantiation blocks spell them.
/// `kSharedGateRows` is zero where the mixture has no always-on expert, and the fixture reads
/// that as "bind no shared weights" rather than "bind empty ones".
#define SINFER_MOE_TEST_GEOMETRY(Registered)                                                       \
    constexpr ops::SparseMoeGeometry kGeometry = (Registered);                                     \
    constexpr std::int32_t kHidden         = kGeometry.hidden;                                     \
    constexpr std::int32_t kExperts        = kGeometry.experts;                                    \
    constexpr std::int32_t kTopK           = kGeometry.experts_per_token;                          \
    constexpr std::int32_t kIntermediate   = kGeometry.intermediate;                               \
    constexpr std::int32_t kExpertGateRows = kGeometry.expert_rows();                              \
    constexpr std::int32_t kRoutedGateRows = kGeometry.routed_gate_rows();                         \
    constexpr std::int32_t kRoutedDownRows = kGeometry.routed_down_rows();                          \
    constexpr std::int32_t kSharedGateRows = kGeometry.shared_rows();                              \
    constexpr bool kHasShared              = kGeometry.has_shared();

namespace qwen36 {
SINFER_MOE_TEST_GEOMETRY(ops::kSparseMoeQwen36Geometry)
#include "ops/test_sparse_moe_body.inc"
} // namespace qwen36

namespace qwen3_moe {
SINFER_MOE_TEST_GEOMETRY(ops::kSparseMoeQwen3MoeGeometry)
#include "ops/test_sparse_moe_body.inc"
} // namespace qwen3_moe

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    // These are public-behavior cases, not route assertions. They exercise decode (T=1), the
    // Small-T supported-domain edges, each profile's first prefill T, the wide-prefill boundary,
    // and one call crossing the 4096-token internal slice without observing any private plan.
    constexpr std::array<std::int32_t, 6> kQ4Q5Tokens{{1, 2, 46, 47, 768, 4097}};
    constexpr std::array<std::int32_t, 5> kQ4Q6Tokens{{1, 2, 46, 47, 768}};
    constexpr std::array<std::int32_t, 5> kW8W8Tokens{{1, 2, 19, 20, 768}};
    // The NVFP4 routed profile runs on the vendored TRT-LLM runner at every width, so the cases
    // walk that runner's own tuning ladder: 1 and 2 are buckets of their own, 19/20 and 47 land
    // inside the power-of-two rungs, 139 and 768 inside the linear ones, and 4,097 crosses the
    // 4,096-row slice bound into a second call.
    constexpr std::array<std::int32_t, 8> kNvfp4Tokens{{1, 2, 19, 20, 47, 139, 768, 4097}};
    const std::array<CodecProfile, 4> profiles{{
        {"sparse_moe q4+q5 a16", QType::Q4G64_F16S, QType::Q5G64_F16S, kQ4Q5Tokens, true},
        {"sparse_moe q4+q6 a16", QType::Q4G64_F16S, QType::Q6G64_F16S, kQ4Q6Tokens, false},
        {"sparse_moe w8+w8 a16", QType::W8G32_F16S, QType::W8G32_F16S, kW8W8Tokens, false},
            // NVFP4 routed experts, served by the vendored TRT-LLM runner: e2m1 codes with an
            // e4m3 scale every 16 values in the dense BlockScaleK16M128x4 layout, an expert's
            // gate/up rows stored [up; gate], and the activations rounded to the same format by
            // the runner. Shapes line up — gate/up is 1,024 x 2,048 and down 2,048 x 512, so N
            // is a multiple of 128 and K of 64 for both.
        {"sparse_moe nvfp4 w4a4", QType::NVFP4, QType::NVFP4, kNvfp4Tokens, false},
    }};

    int failures = 0;
    for (const CodecProfile& profile : profiles) { failures += qwen36::run_profile(profile); }

    // The second registered mixture routes every token and has no always-on expert, which is
    // the whole reason it is here: its router is one row narrower, a token sums one fewer path,
    // and the oracle says so rather than subtracting a shared contribution that was never
    // added. It runs the profiles its own width admits -- the Q5 and Q6 routed-down kernels are
    // baked for an intermediate of 512 and this mixture's is 768, which the op refuses by name
    // rather than approximating, so listing them here would test that refusal and nothing else.
    for (const CodecProfile& profile : profiles) {
        // Q5 and Q6 routed-down kernels are baked for an intermediate of 512 and this mixture's
        // is 768, which the op refuses by name rather than approximating.
        const bool baked_for_512 = profile.routed_down == QType::Q5G64_F16S ||
                                   profile.routed_down == QType::Q6G64_F16S;
        if (baked_for_512) { continue; }
        failures += qwen3_moe::run_profile(profile);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " sparse_moe correctness\n";
    return failures == 0 ? 0 : 1;
}

