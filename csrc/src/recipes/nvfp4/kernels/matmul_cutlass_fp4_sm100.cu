// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
/**
 * @file matmul_cutlass_fp4_sm100.cu
 * @brief CUTLASS-based FP4 GEMM kernels for SM100 (Blackwell B200)
 *
 * SM100 uses Sm100 architecture with NVFP4 kernel schedules. Key constraints:
 * - Uses nv_float4_t<float_e2m1_t> element type (same as SM120)
 * - Scale factor type is float_ue4m3_t (NVFP4 format, 16-element blocks)
 * - Kernel schedules: KernelTmaWarpSpecialized1SmNvf4Sm100, KernelTmaWarpSpecialized2SmNvf4Sm100
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

// SM100 block-scaled layout
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cute/tensor.hpp"

#if defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)

namespace {

namespace sm100_fp4 {

// SM100 uses nv_float4_t wrapper (same as SM120) with ue4m3 scale factors
using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
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
using ArchTag = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// SM100 tile K dimension (NVFP4)
constexpr int TileK = 128;

// ============================================================================
// 1SM Tile configuration: 128x128x128, cluster 1x1x1
// SM100 uses 1x1x1 cluster like SM120 (no multicast TMA)
// ============================================================================

namespace config_1sm {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_1sm

// ============================================================================
// 2SM Tile configuration: 128x256x128, cluster 1x1x1
// SM100 MXF4/NVF4 with 1-CTA cluster requires M=128, so we increase N instead
// Used for larger N dimensions
// ============================================================================

namespace config_2sm {
using TileShape = cute::Shape<cute::_128, cute::_256, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_2sm

// ============================================================================
// FP32 Output Variants
// ============================================================================

namespace fp32_out {

using ElementC_F32 = float;
using ElementD_F32 = float;
constexpr int AlignmentC_F32 = 128 / cutlass::sizeof_bits<ElementC_F32>::value;
constexpr int AlignmentD_F32 = 128 / cutlass::sizeof_bits<ElementD_F32>::value;

// 1SM FP32 output
namespace config_1sm {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC_F32, LayoutCTag, AlignmentC_F32,
    ElementD_F32, LayoutDTag, AlignmentD_F32,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace config_1sm

// 2SM FP32 output (M=128, N=256 due to SM100 constraint)
namespace config_2sm {
using TileShape = cute::Shape<cute::_128, cute::_256, cute::Int<TileK>>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC_F32, LayoutCTag, AlignmentC_F32,
    ElementD_F32, LayoutDTag, AlignmentD_F32,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

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
        {reinterpret_cast<const typename ElementA::DataType*>(a),
         stride_A,
         reinterpret_cast<const typename ElementB::DataType*>(b),
         stride_B,
         reinterpret_cast<const typename ElementA::ScaleFactorType*>(scale_a),
         layout_SFA,
         reinterpret_cast<const typename ElementB::ScaleFactorType*>(scale_b),
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
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM100 GEMM launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

// ============================================================================
// Helper template for alpha-pointer GEMM variant
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
        {reinterpret_cast<const typename ElementA::DataType*>(a),
         stride_A,
         reinterpret_cast<const typename ElementB::DataType*>(b),
         stride_B,
         reinterpret_cast<const typename ElementA::ScaleFactorType*>(scale_a),
         layout_SFA,
         reinterpret_cast<const typename ElementB::ScaleFactorType*>(scale_b),
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
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (alpha-ptr) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (alpha-ptr) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (alpha-ptr) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM100 GEMM (alpha-ptr) launch failed: ") +
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
        {reinterpret_cast<const typename ElementA::DataType*>(a),
         stride_A,
         reinterpret_cast<const typename ElementB::DataType*>(b),
         stride_B,
         reinterpret_cast<const typename ElementA::ScaleFactorType*>(scale_a),
         layout_SFA,
         reinterpret_cast<const typename ElementB::ScaleFactorType*>(scale_b),
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
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (FP32 out) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (FP32 out) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM100 GEMM (FP32 out) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM100 GEMM (FP32 out) launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

}  // namespace sm100_fp4

}  // anonymous namespace

// ============================================================================
// Public API Implementation for SM100
// ============================================================================

void matmul_cutlass_fp4_sm100(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // N-based tile selection for SM100:
    // SM100 NVF4 with 1-CTA cluster requires M=128, so we can only vary N
    // - Small N (<512): 128x128x128
    // - Large N (>=512): 128x256x128 - better throughput for large output dimensions
    if (N < 512) {
        sm100_fp4::run_gemm<sm100_fp4::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm100_fp4::run_gemm<sm100_fp4::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm100_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // FP32 output variant - N-based tile selection (same as BF16 variant)
    if (N < 512) {
        sm100_fp4::run_gemm_f32<sm100_fp4::fp32_out::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm100_fp4::run_gemm_f32<sm100_fp4::fp32_out::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm100_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Alpha-scaled BF16 output - N-based tile selection (same as other variants)
    if (N < 512) {
        sm100_fp4::run_gemm_alpha_ptr<sm100_fp4::config_1sm::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    } else {
        sm100_fp4::run_gemm_alpha_ptr<sm100_fp4::config_2sm::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    }
}

#endif  // CUTLASS_ARCH_MMA_SM100_SUPPORTED
