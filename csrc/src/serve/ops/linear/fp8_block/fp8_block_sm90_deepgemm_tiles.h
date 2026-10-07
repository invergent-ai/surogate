// DeepGEMM's sm90 1D2D block-FP8 kernel (src/third_party/deep_gemm) as ahead-of-time tiles: the
// tile list, each tile's compile-time pipeline and its launcher's declaration. The launchers are
// defined by the fp8_block_sm90_deepgemm_tiles*.cu units (fp8_block_sm90_deepgemm.cuh), and
// fp8_block_sm90_deepgemm.cu picks one per call. Internal to the 90a archive.
#pragma once

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

namespace sinfer::ops::detail::fp8_block::sm90::dg {

// The tiles: (block_m, block_n, cluster_m, cluster_n), in DeepGEMM's enumeration order (cluster m,
// cluster n, block m, block n) so its comparator breaks ties alike. DeepGEMM JIT-compiles the best
// of ~90 candidates per shape; these are the ones its cost model keeps choosing over the q/k/v, o,
// gate/up and down shapes of 0.6B to 70B dense models, 33 to 8192 tokens, on 132, 114 and 78 SMs
// (within 0.1 % of the full set on average under that model), plus the three 2-CTA 64-row tiles it
// picks for 3 % of those shapes at decode widths (33 to 128 tokens), where the next best tile is 7 %
// slower by the model and was 6.5 % slower measured (Qwen3-8B's gate/up at 64 tokens on an H100:
// 39.0 us against vLLM's 36.6). Not compiled: the single-CTA 128 x 144, 128 x 160, 128 x 192,
// 256 x 112 and 256 x 128 tiles, which the model picks when a round fits one wave. Built for a
// run-time N, as here, they spill 80 to 520 bytes of registers in the K loop (DeepGEMM compiles N
// and K in) and ran 1.2 to 3.2x slower than the CUTLASS tiles on an H100. The model still scores
// them; a round that picks one runs on its 2-CTA twin, which doesn't spill (SINFER_DG_STAND_INS):
// 0.67 to 0.99 of the CUTLASS tiles' time on those rounds. Four lists, four translation units.
// clang-format off
#define SINFER_DG_TILES_0(X) X(64, 16, 1, 1) X(64, 32, 1, 1) X(64, 48, 1, 1) X(64, 64, 1, 1) \
                             X(64, 80, 1, 1) X(64, 96, 1, 1) X(64, 112, 1, 1) X(64, 128, 1, 1)
#define SINFER_DG_TILES_1(X) X(64, 144, 1, 1) X(64, 160, 1, 1) X(64, 192, 1, 1) X(64, 80, 2, 1) \
                             X(64, 96, 2, 1) X(64, 112, 2, 1) X(64, 128, 2, 1) X(64, 144, 2, 1)
#define SINFER_DG_TILES_2(X) X(128, 80, 1, 1) X(128, 96, 1, 1) X(128, 112, 1, 1) X(128, 128, 1, 1) \
                             X(128, 80, 1, 2) X(128, 112, 1, 2) X(128, 128, 1, 2) X(64, 160, 2, 1)
#define SINFER_DG_TILES_3(X) X(256, 96, 1, 2) X(256, 112, 1, 2) X(256, 128, 1, 2) X(128, 144, 2, 1) \
                             X(128, 160, 2, 1) X(128, 192, 2, 1)
// (block_m, block_n, cluster_m, cluster_n) of the twin each uncompiled single-CTA tile runs as.
#define SINFER_DG_STAND_INS(X) X(128, 144, 2, 1) X(128, 160, 2, 1) X(128, 192, 2, 1) X(256, 112, 1, 2) \
                               X(256, 128, 1, 2)
// clang-format on
#define SINFER_DG_TILES(X) SINFER_DG_TILES_0(X) SINFER_DG_TILES_1(X) SINFER_DG_TILES_2(X) SINFER_DG_TILES_3(X)
#define SINFER_DG_COUNT(BM, BN, CM, CN) +1
inline constexpr int kNumTiles    = 0 SINFER_DG_TILES(SINFER_DG_COUNT);
inline constexpr int kNumStandIns = 0 SINFER_DG_STAND_INS(SINFER_DG_COUNT);
#undef SINFER_DG_COUNT

inline constexpr int kBlockK        = 128;
inline constexpr int kSmemCapacity  = 232448; // sm_90's opt-in shared memory per block
inline constexpr int kMaxKBlocks    = 256;    // k up to 32768: the B-scale row each CTA stages
inline constexpr int kNumSMsHint    = 132;    // sizes the scheduler's L2 swizzle groups only
inline constexpr int kTmaThreads    = 128;

constexpr int align_up(int x, int a) { return (x + a - 1) / a * a; }

// DeepGEMM's sm90 shared-memory plan (heuristics/sm90.hpp, get_pipeline_config).
constexpr int fixed_smem(int bm, int bn, int k_blocks) {
    return align_up(bm * bn * 2, 1024) + 16 * 8 * 2 + align_up(k_blocks * 4 * (kBlockK % bn == 0 ? 1 : 2), 8);
}
constexpr int stage_smem(int bm, int bn) { return bm * kBlockK + bn * kBlockK + align_up(bm * 4, 128); }
constexpr int stages(int bm, int bn) {
    return std::min((kSmemCapacity - fixed_smem(bm, bn, kMaxKBlocks)) / stage_smem(bm, bn), 16);
}
constexpr int smem_bytes(int bm, int bn, int k_blocks) {
    return fixed_smem(bm, bn, k_blocks) + stages(bm, bn) * stage_smem(bm, bn);
}
constexpr int math_threads(int bm) { return bm <= 64 ? 128 : 256; }
// The output's TMA swizzle: the widest of 128/64/32 bytes that a BF16 row of block_n divides.
constexpr int swizzle_d(int bn) { return (bn * 2) % 128 == 0 ? 128 : (bn * 2) % 64 == 0 ? 64 : 32; }

struct Launch {
    CUtensorMap a, b, d, sfa;
    float* sfb;
    const __nv_bfloat16* residual;
    std::uint32_t m, n, k;
    int sms;
    int smem;
    cudaStream_t stream;
};

using LaunchFn = cudaError_t (*)(const Launch&);

#define SINFER_DG_NAME(BM, BN, CM, CN) launch_##BM##_##BN##_##CM##_##CN
#define SINFER_DG_DECLARE(BM, BN, CM, CN) cudaError_t SINFER_DG_NAME(BM, BN, CM, CN)(const Launch& l);
SINFER_DG_TILES(SINFER_DG_DECLARE)
#undef SINFER_DG_DECLARE

} // namespace sinfer::ops::detail::fp8_block::sm90::dg
