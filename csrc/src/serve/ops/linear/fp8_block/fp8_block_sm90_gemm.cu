// Hopper block-scaled FP8 GEMM (see fp8_block_sm90_gemm.h). Ported from vLLM's
// csrc/quantization/w8a8/cutlass/c3x/scaled_mm_blockwise_sm90_fp8_dispatch.cuh
// (Copyright contributors to the vLLM project, Apache-2.0): the same CUTLASS 3.x kernel
// definition and the same two configurations, with the torch tensors replaced by the engine's
// pointers, an in-place residual epilogue for linear_add, and the launch held in a per-thread
// adapter like the sm_120 NVFP4 GEMM next door. Isolated translation unit, built for 90a only:
// CUTLASS is header-only and heavy, and nothing else includes it.

#include "ops/linear/fp8_block/fp8_block_sm90_gemm.h"

#include <cuda_bf16.h>

#include <cstdlib>
#include <string_view>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/detail/blockwise_scale_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/packed_stride.hpp"

namespace sinfer::ops::detail::fp8_block {
namespace {

using namespace cute;

// clang-format off
template <bool WithResidual, int ScaleGranularityM, int ScaleGranularityN, int ScaleGranularityK,
          class MmaTileShape, class ClusterShape, class EpilogueScheduler, class MainloopScheduler,
          bool SwapAB>
struct BlockwiseGemm {
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = cutlass::bfloat16_t;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose = typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  // vLLM leaves C void (no bias); the engine's linear_add reads the residual through it.
  using ElementC = std::conditional_t<WithResidual, cutlass::bfloat16_t, void>;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = std::conditional_t<SwapAB,
      cutlass::detail::Sm90BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::GMMA::Major::K, cute::GMMA::Major::MN>,
      cutlass::detail::Sm90BlockwiseScaleConfig<
        ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
        cute::GMMA::Major::MN, cute::GMMA::Major::K>>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm90;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag,
      OperatorClass,
      MmaTileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      std::conditional_t<SwapAB, LayoutC_Transpose, LayoutC>,
      AlignmentC,
      ElementD,
      std::conditional_t<SwapAB, LayoutD_Transpose, LayoutD>,
      AlignmentD,
      EpilogueScheduler,
      DefaultOperation
  >::CollectiveOp;

  using CollectiveMainloop = std::conditional_t<SwapAB,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementB,
          cute::tuple<LayoutB_Transpose, LayoutSFA>,
          AlignmentB,
          ElementA,
          cute::tuple<LayoutA_Transpose, LayoutSFB>,
          AlignmentA,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag,
          OperatorClass,
          ElementA,
          cute::tuple<LayoutA, LayoutSFA>,
          AlignmentA,
          ElementB,
          cute::tuple<LayoutB, LayoutSFB>,
          AlignmentB,
          ElementAccumulator,
          MmaTileShape,
          ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler
      >::CollectiveOp>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};
// clang-format on

// vLLM's two configurations: 128x128x128 cooperative over a 1x2 cluster when the token count
// keeps the activation scales' TMA rows 16-byte aligned, and the swapped 128x16 ping-pong tile
// (weight rows on M, tokens on N) otherwise -- which is also the narrow-round shape.
template <bool WithResidual>
using Wide = BlockwiseGemm<WithResidual, 1, 128, 128, Shape<_128, _128, _128>, Shape<_1, _2, _1>,
                           cutlass::epilogue::TmaWarpSpecializedCooperative,
                           cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8BlockScaledAccum,
                           false>;
template <bool WithResidual>
using Narrow = BlockwiseGemm<WithResidual, 128, 1, 128, Shape<_128, _16, _128>, Shape<_1, _1, _1>,
                             cutlass::epilogue::TmaWarpSpecialized,
                             cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8BlockScaledAccum,
                             true>;

struct Hardware {
    int device   = -1;
    int cc       = 0;
    int sm_count = 0;
};

const Hardware& hardware() {
    static thread_local Hardware cached;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) { return cached; }
    if (cached.device != device) {
        int major = 0, minor = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
        cached.device   = device;
        cached.cc       = major * 10 + minor;
        cached.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(device);
    }
    return cached;
}

template <class Def>
bool launch(const std::uint8_t* act_codes, const float* act_scales, const std::uint8_t* w_codes,
            const float* w_scales, void* out_bf16, std::int32_t tokens, std::int32_t n,
            std::int32_t k, cudaStream_t stream) {
    using Gemm        = typename Def::Gemm;
    using GemmKernel  = typename Def::GemmKernel;
    using ScaleConfig = typename Def::ScaleConfig;
    constexpr bool kSwap     = std::is_same_v<Def, Narrow<false>> || std::is_same_v<Def, Narrow<true>>;
    constexpr bool kResidual = !std::is_void_v<typename Def::ElementC>;

    const int m = tokens;
    // vLLM's convention: A is the [m, k] activation, B the [k, n] column-major weight (the
    // artifact's [n, k] row-major); swapped, the weight is A and the problem is (n, m, k).
    const auto a_stride = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{},
                                                           cute::make_shape(kSwap ? n : m, k, 1));
    const auto b_stride = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{},
                                                           cute::make_shape(kSwap ? m : n, k, 1));
    const auto c_stride = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, kSwap ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));
    const auto problem = kSwap ? cute::make_shape(n, m, static_cast<int>(k), 1)
                               : cute::make_shape(m, n, static_cast<int>(k), 1);

    // The activation scales' rows are sm90_scale_stride(m) long: their layout takes the padded
    // token count, the problem the real one, so the padding is loaded but never stored.
    const int scale_m       = sm90_scale_stride(m);
    const auto scale_shape  = kSwap ? cute::make_shape(n, scale_m, static_cast<int>(k), 1)
                                    : cute::make_shape(scale_m, n, static_cast<int>(k), 1);
    typename GemmKernel::MainloopArguments mainloop{};
    mainloop.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(scale_shape);
    mainloop.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(scale_shape);
    const auto* a = reinterpret_cast<const cutlass::float_e4m3_t*>(act_codes);
    const auto* b = reinterpret_cast<const cutlass::float_e4m3_t*>(w_codes);
    if constexpr (kSwap) {
        mainloop.ptr_A   = b;
        mainloop.dA      = a_stride;
        mainloop.ptr_B   = a;
        mainloop.dB      = b_stride;
        mainloop.ptr_SFA = w_scales;
        mainloop.ptr_SFB = act_scales;
    } else {
        mainloop.ptr_A   = a;
        mainloop.dA      = a_stride;
        mainloop.ptr_B   = b;
        mainloop.dB      = b_stride;
        mainloop.ptr_SFA = act_scales;
        mainloop.ptr_SFB = w_scales;
    }

    auto* d = static_cast<cutlass::bfloat16_t*>(out_bf16);
    typename GemmKernel::EpilogueArguments epilogue{{}, kResidual ? d : nullptr, c_stride, d, c_stride};
    epilogue.thread.alpha = 1.0f;
    epilogue.thread.beta  = kResidual ? 1.0f : 0.0f;

    const Hardware& hw = hardware();
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = hw.device;
    hw_info.sm_count  = hw.sm_count;
    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kGemm, problem, mainloop,
                                  epilogue, hw_info};

    // One adapter per instantiation and thread: initialize validates the arguments and sets the
    // kernel's shared-memory attribute; the persistent scheduler needs no workspace, and a
    // configuration that would ask for one declines rather than allocate under a capture.
    static thread_local Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) { return false; }
    if (Gemm::get_workspace_size(args) != 0) { return false; }
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) { return false; }
    return gemm.run(stream) == cutlass::Status::kSuccess;
}

} // namespace

bool sm90_gemm_available() noexcept {
    // SUROGATE_SERVE_FP8_BLOCK_SM90=0 keeps Hopper on the engine's own tile kernel: an A/B switch
    // and an escape hatch. Read once, so the activation layout and the GEMM always agree.
    static const bool disabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_SM90");
        return raw != nullptr && std::string_view(raw) == "0";
    }();
    return !disabled && hardware().cc == 90;
}

bool sm90_gemm(const std::uint8_t* act_codes, const float* act_scales, const std::uint8_t* w_codes,
               const float* w_scales, void* out_bf16, bool residual, std::int32_t tokens,
               std::int32_t n, std::int32_t k, cudaStream_t stream) {
    // TMA reads and writes every operand: 16-byte aligned bases and rows.
    const auto aligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (!sm90_gemm_available() || tokens <= 0 || n <= 0 || k <= 0 || (k % 128) != 0 ||
        (n % 8) != 0 || !aligned(act_codes) || !aligned(act_scales) || !aligned(w_codes) ||
        !aligned(w_scales) || !aligned(out_bf16)) {
        return false;
    }
    // A narrow round gains from the swapped 16-token tile instead of wasting most of a 128-token
    // one. vLLM also swaps whenever the token count breaks the activation scales' TMA alignment;
    // here their rows are padded to keep it (sm90_scale_stride), so a wide round of any count
    // keeps the 128 x 128 tile. Serving's mixed rounds (a prompt chunk plus the decode rows)
    // rarely hold a multiple of four, and the 16-token tile re-reads the weight for every 16
    // tokens.
    const bool swap = tokens <= 64;
    if (swap) {
        return residual ? launch<Narrow<true>>(act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream)
                        : launch<Narrow<false>>(act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream);
    }
    return residual ? launch<Wide<true>>(act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream)
                    : launch<Wide<false>>(act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream);
}

} // namespace sinfer::ops::detail::fp8_block
