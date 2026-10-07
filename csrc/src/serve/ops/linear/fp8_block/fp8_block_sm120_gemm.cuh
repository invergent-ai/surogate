// The sm_12x block-scaled FP8 GEMM's kernel definitions and launch, shared by the translation
// units that each instantiate a few of its tiles (fp8_block_sm120_gemm*.cu). CUTLASS's sm_120
// blockwise collective, as vLLM's csrc/quantization/w8a8/cutlass/c3x/
// scaled_mm_blockwise_sm120_fp8_dispatch.cuh (Copyright contributors to the vLLM project,
// Apache-2.0) builds it, with an in-place residual epilogue for linear_add, swapped tiles that put
// the weight rows on M, and per-row weights (one scale per row for every k block: a scale layout
// whose k-block stride is zero). The launch follows Hopper's (fp8_block_sm90_gemm.cuh), whose
// operands, hardware query and scale-row padding it shares. Internal: only the sm_12x archive
// includes it, CUTLASS being header-only and heavy.
#pragma once

#include "ops/linear/fp8_block/fp8_block_sm120_gemm.h"
#include "ops/linear/fp8_block/fp8_block_sm90_gemm.cuh"

namespace sinfer::ops::detail::fp8_block::sm120 {

using namespace cute;
using sm90::Operands;

// clang-format off
template <bool WithResidual, bool PerRow, class MmaTileShape, class KernelSchedule, bool SwapAB>
struct BlockwiseGemm {
  static constexpr bool kSwapAB = SwapAB;
  static constexpr bool kPerRow = PerRow;

  // The collective takes TN only: both operands K-major, the activations [tokens, k] and the
  // weight [n, k] row-major, whichever of them is A.
  using ElementA = cutlass::float_e4m3_t;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;
  using ElementB = cutlass::float_e4m3_t;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  // Swapped, D is [n, tokens] column-major: the same bytes as the engine's [tokens, n] row-major.
  using ElementD = cutlass::bfloat16_t;
  using LayoutD = std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  // vLLM leaves C void (no bias); the engine's linear_add reads the residual through it.
  using ElementC = std::conditional_t<WithResidual, cutlass::bfloat16_t, void>;
  using LayoutC = LayoutD;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;

  // The activations: one scale per token per 128 of k, k-block-major (MN-major). The weight: one
  // per 128 x 128 block, [n/128, k/128] row-major (K-major), or one per row (MN-major, its k-block
  // stride set to zero at launch).
  static constexpr int kWeightGranularity = PerRow ? 1 : 128;
  static constexpr auto kWeightMajor = PerRow ? cute::UMMA::Major::MN : cute::UMMA::Major::K;
  using ScaleConfig = cutlass::detail::Sm120BlockwiseScaleConfig<
      SwapAB ? kWeightGranularity : 1, SwapAB ? 1 : kWeightGranularity, 128,
      SwapAB ? kWeightMajor : cute::UMMA::Major::MN, SwapAB ? cute::UMMA::Major::MN : kWeightMajor>;
  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm120;
  using OperatorClass = cutlass::arch::OpClassTensorOp;
  using ClusterShape = Shape<_1, _1, _1>;

  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator, ElementCompute,
      ElementC, LayoutC, AlignmentC,
      ElementD, LayoutD, AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      DefaultOperation
  >::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
      ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
      ElementAccumulator, MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule
  >::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using Cooperative = cutlass::gemm::KernelTmaWarpSpecializedBlockwiseCooperativeSm120;
using Pingpong    = cutlass::gemm::KernelTmaWarpSpecializedBlockwisePingpongSm120;

// The tiles, all over 128 of k. Swapped ones put the weight rows on M and the tokens on N, so a
// round narrower than 128 tokens fills its tile with weight rows instead of padding tokens.
// Narrow: 128 x 32 ping-pong, each consumer warpgroup a tile of its own. (Sixteen tokens is too
// narrow for the collective: each warp's slice of B is one 8-column fragment, under the
// four-matrix ldmatrix it copies with.)
template <bool WithResidual, bool PerRow>
using Narrow = BlockwiseGemm<WithResidual, PerRow, Shape<_128, _32, _128>, Pingpong, true>;
// Mid: swapped cooperative 128 x 32 and 128 x 64, both consumer warpgroups on one tile.
template <bool WithResidual, bool PerRow, int Tokens>
using Mid = BlockwiseGemm<WithResidual, PerRow, Shape<_128, Int<Tokens>, _128>, Cooperative, true>;
// Wide: vLLM's sm_120 tile, 128 tokens x 128 rows, cooperative. A 256-token tile's two operand
// tiles alone leave room for one stage in the 99 KiB a block may use.
template <bool WithResidual, bool PerRow>
using Wide = BlockwiseGemm<WithResidual, PerRow, Shape<_128, _128, _128>, Cooperative, false>;
// clang-format on

/// A per-row weight's scale layout: the layout ScaleConfig tiles for the problem, with its
/// k-block stride zero, so every k block reads the row's one scale.
template <class Layout>
Layout one_scale_per_row(const Layout& tiled) {
    const auto& stride = cute::stride(tiled);
    return Layout(cute::shape(tiled),
                  cute::make_stride(cute::get<0>(stride), cute::make_stride(cute::_0{}, std::int32_t{0}),
                                    cute::get<2>(stride)));
}

// Named apart from Hopper's launch<Def>, which argument-dependent lookup on the shared operands
// would otherwise also find.
template <class Def>
bool launch_blockwise(const Operands& o) {
    using Gemm        = typename Def::Gemm;
    using GemmKernel  = typename Def::GemmKernel;
    using ScaleConfig = typename Def::ScaleConfig;
    constexpr bool kSwap     = Def::kSwapAB;
    constexpr bool kResidual = !std::is_void_v<typename Def::ElementC>;

    const int m = o.tokens;
    const int n = o.n;
    const int k = o.k;
    // A is the [m, k] activation and B the weight, or swapped, the weight is A and the problem
    // is (n, m, k); either way both are K-major.
    const auto a_stride = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{},
                                                           cute::make_shape(kSwap ? n : m, k, 1));
    const auto b_stride = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{},
                                                           cute::make_shape(kSwap ? m : n, k, 1));
    const auto c_stride = cutlass::make_cute_packed_stride(
        typename GemmKernel::StrideC{}, kSwap ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));
    const auto problem = kSwap ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

    // The activation scales' rows are sm90_scale_stride(m) long: their layout takes the padded
    // token count, the problem the real one, so the padding is loaded but never stored.
    const int scale_m      = sm90_scale_stride(m);
    const auto scale_shape = kSwap ? cute::make_shape(n, scale_m, k, 1) : cute::make_shape(scale_m, n, k, 1);
    typename GemmKernel::MainloopArguments mainloop{};
    mainloop.layout_SFA = ScaleConfig::tile_atom_to_shape_SFA(scale_shape);
    mainloop.layout_SFB = ScaleConfig::tile_atom_to_shape_SFB(scale_shape);
    if constexpr (Def::kPerRow) {
        if constexpr (kSwap) {
            mainloop.layout_SFA = one_scale_per_row(mainloop.layout_SFA);
        } else {
            mainloop.layout_SFB = one_scale_per_row(mainloop.layout_SFB);
        }
    }
    const auto* a = reinterpret_cast<const cutlass::float_e4m3_t*>(o.act_codes);
    const auto* b = reinterpret_cast<const cutlass::float_e4m3_t*>(o.w_codes);
    mainloop.ptr_A   = kSwap ? b : a;
    mainloop.dA      = a_stride;
    mainloop.ptr_B   = kSwap ? a : b;
    mainloop.dB      = b_stride;
    mainloop.ptr_SFA = kSwap ? o.w_scales : o.act_scales;
    mainloop.ptr_SFB = kSwap ? o.act_scales : o.w_scales;

    auto* d = static_cast<cutlass::bfloat16_t*>(o.out_bf16);
    typename GemmKernel::EpilogueArguments epilogue{{}, kResidual ? d : nullptr, c_stride, d, c_stride};
    epilogue.thread.alpha = 1.0f;
    epilogue.thread.beta  = kResidual ? 1.0f : 0.0f;

    const sm90::Hardware& hw = sm90::hardware();
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
    if (gemm.initialize(args, nullptr, o.stream) != cutlass::Status::kSuccess) { return false; }
    return gemm.run(o.stream) == cutlass::Status::kSuccess;
}

// The tile families, each instantiated (with and without the residual, per block and per row) in
// translation units of their own so the archive's CUTLASS instantiations compile in parallel.
bool launch_narrow(const Operands& o, bool residual, bool per_row);
bool launch_mid(const Operands& o, bool residual, int tile_tokens);     // 32 or 64, block weights
bool launch_mid_row(const Operands& o, bool residual, int tile_tokens); // 32 or 64, per-row weights
bool launch_wide(const Operands& o, bool residual, bool per_row);

} // namespace sinfer::ops::detail::fp8_block::sm120
