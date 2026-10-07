// The launch of one DeepGEMM sm90 1D2D tile (fp8_block_sm90_deepgemm_tiles.h): the kernel
// instantiation, its shared-memory opt-in and the cluster launch. Included only by the
// fp8_block_sm90_deepgemm_tiles*.cu units, which define the launchers in parallel.
#pragma once

#include "ops/linear/fp8_block/fp8_block_sm90_deepgemm_tiles.h"
#include "ops/kernel/func_attribute.cuh"

#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

namespace sinfer::ops::detail::fp8_block::sm90::dg {

using sinfer::ops::set_func_attribute_per_device;

template <int BM, int BN, int CM, int CN>
cudaError_t launch(const Launch& l) {
    static_assert(stages(BM, BN) >= 3 && (BM * BN >= 128 * 192 || stages(BM, BN) >= 4),
                  "DeepGEMM keeps at least 3 stages, 4 for small tiles");
    constexpr int kCluster = CM * CN;
    auto* kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
        cute::UMMA::Major::K, 0, 0, 0, 1, BM, BN, kBlockK, 128, 128, swizzle_d(BN), stages(BM, BN),
        kTmaThreads, math_threads(BM), kCluster, (CN > 1), kNumSMsHint, deep_gemm::GemmType::Normal,
        cutlass::bfloat16_t>;
    if (cudaError_t e = set_func_attribute_per_device(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemCapacity);
        e != cudaSuccess) {
        return e;
    }
    cudaLaunchConfig_t config{};
    config.gridDim          = dim3(static_cast<unsigned>(l.sms));
    config.blockDim         = dim3(kTmaThreads + math_threads(BM));
    config.dynamicSmemBytes = static_cast<std::size_t>(l.smem);
    config.stream           = l.stream;
    cudaLaunchAttribute cluster{};
    cluster.id               = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim.x = kCluster;
    cluster.val.clusterDim.y = 1;
    cluster.val.clusterDim.z = 1;
    config.attrs             = &cluster;
    config.numAttrs          = 1;
    return cudaLaunchKernelEx(&config, kernel, l.sfb, static_cast<int*>(nullptr), l.m, l.n, l.k, l.a, l.b, l.d,
                              l.sfa, l.residual);
}

#define SINFER_DG_DEFINE(BM, BN, CM, CN) \
    cudaError_t SINFER_DG_NAME(BM, BN, CM, CN)(const Launch& l) { return launch<BM, BN, CM, CN>(l); }

} // namespace sinfer::ops::detail::fp8_block::sm90::dg
