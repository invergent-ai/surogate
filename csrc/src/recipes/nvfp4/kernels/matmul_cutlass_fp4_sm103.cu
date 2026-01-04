// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
/**
 * @file matmul_cutlass_fp4_sm103.cu
 * @brief CUTLASS-based FP4 GEMM kernels for SM103 (Blackwell B300)
 *
 * SM103 uses Sm103BlockScaledConfig with a different scale factor atom layout
 * (8x4x4) compared to SM100/SM120 (32x4). Key constraints:
 * - Tile K dimension must be 768
 * - Scale factor type must be float_ue8m0_t
 * - Uses tuple<float_e2m1_t, float_ue8m0_t> element type format
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cstdio>
#include <string>

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"
#include "cutlass/util/packed_stride.hpp"

// SM103 block-scaled layout
#include "cutlass/detail/sm103_blockscaled_layout.hpp"

#include "cute/tensor.hpp"

#if defined(CUTLASS_ARCH_MMA_SM103_SUPPORTED)

namespace {

namespace sm103_fp4 {

// SM103 requires tuple format for element types with scale factor
using ElementA = cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>;
using ElementB = cute::tuple<cutlass::float_e2m1_t, cutlass::float_ue8m0_t>;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

// Layout configuration
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

// Alignment (32 elements for FP4)
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Architecture and operator class
using ArchTag = cutlass::arch::Sm103;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// SM103 requires K=768 for all tile configurations
constexpr int TileK = 768;

// ============================================================================
// 1SM Tile configuration: 128x128x768, cluster 4x2x1
// Used for smaller problem sizes
// ============================================================================

namespace config_1sm {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_4, cute::_2, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_1sm

// ============================================================================
// 2SM Tile configuration: 256x256x768, cluster 4x2x1
// Used for larger problem sizes with 2-SM cooperative execution
// ============================================================================

namespace config_2sm {
using TileShape = cute::Shape<cute::_256, cute::_256, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_4, cute::_2, cute::_1>;
using EpilogueTileShape = cute::Shape<cute::_128, cute::_64>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileShape,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_2sm

// ============================================================================
// FP32 Output Variants (for alpha scaling before BF16 conversion)
// ============================================================================

namespace fp32_out {

using ElementC_F32 = float;
using ElementD_F32 = float;
constexpr int AlignmentC_F32 = 4;
constexpr int AlignmentD_F32 = 4;

// 1SM FP32 output
namespace config_1sm {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_4, cute::_2, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    void, LayoutCTag, AlignmentC_F32,
    ElementD_F32, LayoutDTag, AlignmentD_F32,
    cutlass::epilogue::NoSmemWarpSpecialized1Sm
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_1sm

// 2SM FP32 output
namespace config_2sm {
using TileShape = cute::Shape<cute::_256, cute::_256, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_4, cute::_2, cute::_1>;
using EpilogueTileShape = cute::Shape<cute::_128, cute::_64>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileShape,
    ElementAccumulator, ElementAccumulator,
    void, LayoutCTag, AlignmentC_F32,
    ElementD_F32, LayoutDTag, AlignmentD_F32,
    cutlass::epilogue::NoSmemWarpSpecialized2Sm
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecialized2SmBlockScaledMxNvf4UltraVs16Sm103
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_2sm

}  // namespace fp32_out

// ============================================================================
// Helper template for running any GEMM variant (BF16 output)
// ============================================================================

template<typename Gemm>
void run_gemm(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const cutlass::float_e2m1_t*>(a),
         stride_A,
         reinterpret_cast<const cutlass::float_e2m1_t*>(b),
         stride_B,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_a),
         layout_SFA,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_b),
         layout_SFB},
        {{1.0f, 0.0f},
         nullptr,
         stride_C,
         reinterpret_cast<ElementD*>(d),
         stride_D}
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM execution failed");
    }
}

// ============================================================================
// Helper template for alpha-pointer GEMM variant (reads alpha from device pointer)
// ============================================================================

template<typename Gemm>
void run_gemm_alpha_ptr(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const cutlass::float_e2m1_t*>(a),
         stride_A,
         reinterpret_cast<const cutlass::float_e2m1_t*>(b),
         stride_B,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_a),
         layout_SFA,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_b),
         layout_SFB},
        {{},  // Default epilogue args, will set alpha_ptr below
         nullptr,
         stride_C,
         reinterpret_cast<ElementD*>(d),
         stride_D}
    };

    // Set alpha_ptr for device-side alpha reading
    args.epilogue.thread.alpha_ptr = alpha_ptr;

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (alpha-ptr) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (alpha-ptr) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (alpha-ptr) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM103 GEMM (alpha-ptr) launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

// ============================================================================
// Helper template for FP32 output GEMMs
// ============================================================================

template<typename Gemm>
void run_gemm_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<const cutlass::float_e2m1_t*>(a),
         stride_A,
         reinterpret_cast<const cutlass::float_e2m1_t*>(b),
         stride_B,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_a),
         layout_SFA,
         reinterpret_cast<const cutlass::float_ue8m0_t*>(scale_b),
         layout_SFB},
        {{1.0f, 0.0f},
         nullptr,
         stride_C,
         d,
         stride_D}
    };

    Gemm gemm_op;
    auto status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (FP32 out) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (FP32 out) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM103 GEMM (FP32 out) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM103 GEMM (FP32 out) launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

}  // namespace sm103_fp4

}  // anonymous namespace

// ============================================================================
// Public API Implementation for SM103
// ============================================================================

void matmul_cutlass_fp4_sm103(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // SM103 tile selection based on problem size:
    // - Small problems (M*N < 256*256): 1SM kernel with 128x128x768 tiles
    // - Large problems: 2SM kernel with 256x256x768 tiles for better throughput
    if (M <= 256 && N <= 256) {
        sm103_fp4::run_gemm<sm103_fp4::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm103_fp4::run_gemm<sm103_fp4::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm103_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // FP32 output variant for alpha scaling before BF16 conversion
    if (M <= 256 && N <= 256) {
        sm103_fp4::run_gemm_f32<sm103_fp4::fp32_out::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm103_fp4::run_gemm_f32<sm103_fp4::fp32_out::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm103_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Alpha-scaled BF16 output via device pointer
    if (M <= 256 && N <= 256) {
        sm103_fp4::run_gemm_alpha_ptr<sm103_fp4::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    } else {
        sm103_fp4::run_gemm_alpha_ptr<sm103_fp4::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    }
}

#endif  // CUTLASS_ARCH_MMA_SM103_SUPPORTED
