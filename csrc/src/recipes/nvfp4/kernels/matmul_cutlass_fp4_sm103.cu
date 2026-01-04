// SPDX-License-Identifier: Apache-2.0
/**
 * @file matmul_cutlass_fp4_sm103.cu
 * @brief CUTLASS-based FP4 GEMM kernels for SM103 (Blackwell B300)
 *
 * SM103 uses Sm103BlockScaledConfig with a different scale factor atom layout
 * (8x4x4) compared to SM100/SM120 (32x4). This requires separate kernel
 * instantiations.
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

// Common element types for SM103 block-scaled FP4
using ElementA = cutlass::float_e2m1_t;
using ElementSFA = cutlass::float_ue4m3_t;
using ElementB = cutlass::float_e2m1_t;
using ElementSFB = cutlass::float_ue4m3_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;

// Layout configuration
using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;

// Alignment (32 elements = 16 bytes for FP4)
constexpr int AlignmentA = 32;
constexpr int AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Architecture and operator class
using ArchTag = cutlass::arch::Sm103;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// ============================================================================
// Tile configuration variants based on M dimension
// ============================================================================

// Small M (M <= 16): 128x128x256, cluster 1x1x1
namespace small {
using TileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

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
    cute::tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
    cute::tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
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
}  // namespace small

// Medium M (16 < M <= 256): 256x128x256, cluster 2x1x1
namespace medium {
using TileShape = cute::Shape<cute::_256, cute::_128, cute::_256>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using PerSmTileShape = cute::Shape<cute::_128, cute::_128, cute::_256>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    PerSmTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
    cute::tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
}  // namespace medium

// Large M (M > 256): 256x256x256, cluster 2x1x1
namespace large {
using TileShape = cute::Shape<cute::_256, cute::_256, cute::_256>;
using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
using PerSmTileShape = cute::Shape<cute::_128, cute::_256, cute::_256>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    PerSmTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    cute::tuple<ElementA, ElementSFA>, LayoutATag, AlignmentA,
    cute::tuple<ElementB, ElementSFB>, LayoutBTag, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    cute::Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

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
        {reinterpret_cast<const ElementA*>(a),
         stride_A,
         reinterpret_cast<const ElementB*>(b),
         stride_B,
         reinterpret_cast<const ElementSFA*>(scale_a),
         layout_SFA,
         reinterpret_cast<const ElementSFB*>(scale_b),
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

// Helper template for alpha-pointer GEMM variant (reads alpha from device pointer)
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
        {reinterpret_cast<const ElementA*>(a),
         stride_A,
         reinterpret_cast<const ElementB*>(b),
         stride_B,
         reinterpret_cast<const ElementSFA*>(scale_a),
         layout_SFA,
         reinterpret_cast<const ElementSFB*>(scale_b),
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
    // M-based tile selection for optimal performance:
    // - Small M (<=16): 128x128x256, cluster 1x1x1 - single SM, low overhead
    // - Medium M (<=256): 256x128x256, cluster 2x1x1 - 2-SM cooperative
    // - Large M (>256): 256x256x256, cluster 2x1x1 - max throughput
    if (M <= 16) {
        sm103_fp4::run_gemm<sm103_fp4::small::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else if (M <= 256) {
        sm103_fp4::run_gemm<sm103_fp4::medium::Gemm>(
            d, a, b, scale_a, scale_b, workspace, workspace_size, M, N, K, stream);
    } else {
        sm103_fp4::run_gemm<sm103_fp4::large::Gemm>(
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
    // M-based tile selection with alpha pointer for SM103:
    // Uses same tile configs as basic variant, but reads alpha from device pointer
    if (M <= 16) {
        sm103_fp4::run_gemm_alpha_ptr<sm103_fp4::small::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    } else if (M <= 256) {
        sm103_fp4::run_gemm_alpha_ptr<sm103_fp4::medium::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    } else {
        sm103_fp4::run_gemm_alpha_ptr<sm103_fp4::large::Gemm>(
            d, a, b, scale_a, scale_b, alpha_ptr, workspace, workspace_size, M, N, K, stream);
    }
}

#endif  // CUTLASS_ARCH_MMA_SM103_SUPPORTED
