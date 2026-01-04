// Copyright (c) 2026, Invergent SA, developed by Flavius Burca
// SPDX-License-Identifier: Apache-2.0
//
/**
 * @file matmul_cutlass_fp4_sm120.cu
 * @brief CUTLASS-based FP4 GEMM kernels for SM120/SM121 (Blackwell B200, RTX 50xx)
 *
 * SM120 uses Sm1xxBlockScaledConfig with scale factor atom layout (32x4),
 * which is also compatible with SM100 (B200). Uses nv_float4_t wrapper type.
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

// SM100/SM120 block-scaled layout (Sm1xxBlockScaledConfig)
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cute/tensor.hpp"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

namespace {

namespace sm120_fp4 {

// Common element types for SM120 NVFP4
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

// Alignment
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Architecture and operator class
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// SM120 requires 1x1x1 cluster for all configurations
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

// ============================================================================
// Tile configuration variants based on M dimension
// ============================================================================

// Small M (M < 512): 128x128x128
namespace small {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

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
}  // namespace small

// Large M (M >= 512): 256x128x128
namespace large {
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;

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
}  // namespace large

// Helper template for running any GEMM variant
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
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM120 GEMM launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

// ============================================================================
// FP32 Output Variants (for alpha scaling before BF16 conversion)
// ============================================================================

namespace fp32_out {

using ElementC_F32 = float;
using ElementD_F32 = float;
constexpr int AlignmentC_F32 = 128 / cutlass::sizeof_bits<ElementC_F32>::value;
constexpr int AlignmentD_F32 = 128 / cutlass::sizeof_bits<ElementD_F32>::value;

// Small M FP32 output
namespace small {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

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
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace small

// Large M FP32 output
namespace large {
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;

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
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace large

}  // namespace fp32_out

// ============================================================================
// Alpha-Pointer BF16 Output Variants (alpha from device pointer in epilogue)
// ============================================================================

namespace alpha_ptr {

// Small M (M < 512): 128x128x128 with alpha pointer
namespace small {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;

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
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace small

// Large M (M >= 512): 256x128x128 with alpha pointer
namespace large {
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;

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
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace large

}  // namespace alpha_ptr

// Helper template for alpha-pointer BF16 output GEMMs (reads alpha from device pointer)
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
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (alpha-ptr) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (alpha-ptr) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (alpha-ptr) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM120 GEMM (alpha-ptr) launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

// Helper template for FP32 output GEMMs
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
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (FP32 out) cannot be implemented for given problem size");
    }

    status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (FP32 out) initialization failed");
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS FP4 SM120 GEMM (FP32 out) execution failed");
    }

    cudaError_t cuda_err = cudaPeekAtLastError();
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(std::string("CUTLASS FP4 SM120 GEMM (FP32 out) launch failed: ") +
                                 cudaGetErrorString(cuda_err));
    }
}

}  // namespace sm120_fp4

}  // anonymous namespace

// ============================================================================
// Public API Implementation for SM120
// ============================================================================

void matmul_cutlass_fp4_sm120(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // M-based tile selection for SM120 (GeForce):
    // - Small M (<512): 128x128x128 - better for small batches
    // - Large M (>=512): 256x128x128 - better throughput for training
    if (M < 512) {
        sm120_fp4::run_gemm<sm120_fp4::small::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm120_fp4::run_gemm<sm120_fp4::large::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm120_f32(
    float* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // FP32 output variant for alpha scaling before BF16 conversion
    if (M < 512) {
        sm120_fp4::run_gemm_f32<sm120_fp4::fp32_out::small::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm120_fp4::run_gemm_f32<sm120_fp4::fp32_out::large::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    }
}

void matmul_cutlass_fp4_sm120_alpha(
    nv_bfloat16* d,
    const uint8_t* a, const uint8_t* b,
    const uint8_t* scale_a, const uint8_t* scale_b,
    const float* alpha_ptr,
    std::byte* workspace, std::size_t workspace_size,
    int M, int N, int K,
    cudaStream_t stream)
{
    // Alpha-scaled BF16 output via device pointer (no FP32 intermediate needed)
    if (M < 512) {
        sm120_fp4::run_gemm_alpha_ptr<sm120_fp4::alpha_ptr::small::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    } else {
        sm120_fp4::run_gemm_alpha_ptr<sm120_fp4::alpha_ptr::large::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    }
}

#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED || CUTLASS_ARCH_MMA_SM121_SUPPORTED
