// surogate vendor patch (PATCHES.md #25): cutlass sm_120a blockscaled NVFP4
// GEMM instantiations (see w4fp4_cutlass_gemm.h). Isolated TU: cutlass is
// header-only and heavy; nothing else includes it.

#include "ops/linear/w8a8/w4fp4_cutlass_gemm.h"

#include <cuda_bf16.h>

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"

namespace ninfer::ops::detail {
namespace {

using namespace cute;

// CTA 128x128x128 measured best at every engine shape (679-813 TF/s).
template <bool WithResidual>
struct GemmDef {
    using OutElementType = cutlass::bfloat16_t;
    using CTAShape       = Shape<_128, _128, _128>;
    using Arch           = cutlass::arch::Sm120;
    using ClusterShape   = Shape<_1, _1, _1>;
    using ElementA       = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutA        = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB       = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutB        = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;
    using ElementC       = std::conditional_t<WithResidual, cutlass::bfloat16_t, void>;
    using LayoutC        = cutlass::layout::RowMajor;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits<OutElementType>::value;
    using ElementCompute     = float;
    using ElementAccumulator = float;
    using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;
    using FusionOperation =
        cutlass::epilogue::fusion::LinearCombination<OutElementType, float, ElementC, float>;
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        Arch, cutlass::arch::OpClassTensorOp, CTAShape, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementCompute,
        ElementC, LayoutC, AlignmentC, OutElementType, LayoutC, AlignmentC,
        cutlass::epilogue::TmaWarpSpecialized, FusionOperation>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        Arch, OperatorClass, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB,
        ElementAccumulator, CTAShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                             CollectiveEpilogue,
                                             cutlass::gemm::StaticPersistentScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

int device_sm_count() {
    static const int count = [] {
        int device = 0;
        cudaGetDevice(&device);
        return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    }();
    return count;
}

template <bool WithResidual>
bool launch(const std::uint8_t* act_codes, const std::uint8_t* act_sf_atom,
            const std::uint8_t* w_codes, const std::uint8_t* w_sf_atom, const float* alpha_one,
            void* c_bf16, void* d_bf16, std::int32_t tokens, std::int32_t n, std::int32_t k,
            cudaStream_t stream) {
    using Def  = GemmDef<WithResidual>;
    using Gemm = typename Def::Gemm;
    using Cfg  = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    typename Gemm::Arguments args;
    args.mode          = cutlass::gemm::GemmUniversalMode::kGemm;
    args.problem_shape = make_shape(static_cast<int>(tokens), static_cast<int>(n),
                                    static_cast<int>(k), 1);
    args.mainloop.ptr_A   = reinterpret_cast<cutlass::float_e2m1_t const*>(act_codes);
    args.mainloop.ptr_B   = reinterpret_cast<cutlass::float_e2m1_t const*>(w_codes);
    args.mainloop.ptr_SFA = reinterpret_cast<cutlass::float_ue4m3_t const*>(act_sf_atom);
    args.mainloop.ptr_SFB = reinterpret_cast<cutlass::float_ue4m3_t const*>(w_sf_atom);
    if constexpr (WithResidual) {
        args.epilogue.ptr_C       = static_cast<cutlass::bfloat16_t const*>(c_bf16);
        args.epilogue.thread.beta = 1.0f;
    } else {
        args.epilogue.ptr_C = nullptr;
    }
    args.epilogue.ptr_D            = static_cast<cutlass::bfloat16_t*>(d_bf16);
    args.epilogue.thread.alpha_ptr = alpha_one;
    args.mainloop.dA =
        cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideA{}, {tokens, k, 1});
    args.mainloop.dB =
        cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideB{}, {n, k, 1});
    args.epilogue.dC =
        cutlass::make_cute_packed_stride(typename Gemm::GemmKernel::StrideC{}, {tokens, n, 1});
    args.epilogue.dD         = args.epilogue.dC;
    args.mainloop.layout_SFA = Cfg::tile_atom_to_shape_SFA(args.problem_shape);
    args.mainloop.layout_SFB = Cfg::tile_atom_to_shape_SFB(args.problem_shape);
    args.hw_info.cluster_shape          = dim3(1, 1, 1);
    args.hw_info.cluster_shape_fallback = dim3(1, 1, 1);
    args.hw_info.device_id              = 0;
    args.hw_info.sm_count               = device_sm_count();

    // One adapter per instantiation: initialize validates and sets kernel
    // attributes once; per-call updates go through initialize with a null
    // workspace (StaticPersistentScheduler needs none).
    static thread_local Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { return false; }
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) { return false; }
    return gemm.run(stream) == cutlass::Status::kSuccess;
}

} // namespace

bool w4fp4_cutlass_gemm_store(const std::uint8_t* act_codes, const std::uint8_t* act_sf_atom,
                              const std::uint8_t* w_codes, const std::uint8_t* w_sf_atom,
                              const float* alpha_one, void* out_bf16, std::int32_t tokens,
                              std::int32_t n, std::int32_t k, cudaStream_t stream) {
    return launch<false>(act_codes, act_sf_atom, w_codes, w_sf_atom, alpha_one, nullptr, out_bf16,
                         tokens, n, k, stream);
}

bool w4fp4_cutlass_gemm_residual(const std::uint8_t* act_codes, const std::uint8_t* act_sf_atom,
                                 const std::uint8_t* w_codes, const std::uint8_t* w_sf_atom,
                                 const float* alpha_one, void* residual_bf16, std::int32_t tokens,
                                 std::int32_t n, std::int32_t k, cudaStream_t stream) {
    return launch<true>(act_codes, act_sf_atom, w_codes, w_sf_atom, alpha_one, residual_bf16,
                        residual_bf16, tokens, n, k, stream);
}

} // namespace ninfer::ops::detail
