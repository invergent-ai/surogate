#pragma once

#include <cstdint>

namespace ninfer::ops::detail {

enum class Bf16ActivationAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Bf16WeightCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Bf16PhaseOrder : std::uint8_t {
    Sequential,
    RowSwizzled,
};

enum class Bf16SmallTActivationAccess : std::uint8_t {
    DirectStream,
    WarpPacked,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Bf16GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);

    static constexpr std::int32_t kOutputRows = OutputRows;
    static constexpr std::int32_t kInputRows  = InputRows;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, Bf16ActivationAccess ActivationAccess, Bf16WeightCache WeightCache,
          Bf16PhaseOrder PhaseOrder, int PhaseStride, int PrefetchDepth, int PhaseUnroll,
          int MinBlocksPerSm>
struct Bf16GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PhaseStride > 0);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane,
          int AccumulatorChains, int TokenBatch, Bf16SmallTActivationAccess ActivationAccess,
          Bf16WeightCache WeightCache, Bf16PhaseOrder PhaseOrder, int PhaseStride, int PhaseUnroll,
          int PrefetchDepth, int MinBlocksPerSm>
struct Bf16SmallTInnerSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 4 || ValuesPerLane == 8 || ValuesPerLane == 16);
    static_assert(AccumulatorChains > 0 && AccumulatorChains <= ValuesPerLane);
    static_assert((AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(TokenBatch == 1 || TokenBatch == 2 || TokenBatch == 4 || TokenBatch == 8);
    static_assert(PhaseStride > 0);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4 || PhaseUnroll == 8);
    static_assert(PrefetchDepth == 1 || PrefetchDepth == 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr int kTokenBatch        = TokenBatch;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kPhaseOrder       = PhaseOrder;
    static constexpr int kPhaseStride       = PhaseStride;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr int kPrefetchDepth     = PrefetchDepth;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
};

// Measured winner: four CTA warps, one warp per row group, eight rows per warp, eight BF16 values
// per lane, four accumulator chains, direct activation loads, default weight caching, and
// row-swizzled K phases. Geometry remains a template argument so each exact problem can retain or
// replace the schedule independently after measurement.
template <class Geometry>
struct Bf16LinearDecodeScheduleSelector {
    using Type =
        Bf16GemvSchedule<4, 1, 8, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                         Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
};

template <>
struct Bf16LinearDecodeScheduleSelector<Bf16GemvGeometry<5120, 6144>> {
    using Type =
        Bf16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                         Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
};

template <class Geometry>
using Bf16LinearDecodeSchedule = typename Bf16LinearDecodeScheduleSelector<Geometry>::Type;

template <class Geometry, int ActiveTokens>
struct Bf16LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= 2 && ActiveTokens <= 32);
    static constexpr bool kOutputProjectionGeometry =
        Geometry::kOutputRows == 5120 && Geometry::kInputRows == 6144;
    // Per-T winners from the complete production sweep. In particular, two rows per warp wins
    // beyond T=8 despite its higher register count because it removes enough repeated issue work.
    static constexpr int kRowsPerWarp =
        kOutputProjectionGeometry
            ? ((ActiveTokens == 4 || ActiveTokens == 6) ? 2 : (ActiveTokens <= 8 ? 4 : 2))
            : (ActiveTokens <= 4 ? 8 : (ActiveTokens <= 8 ? 4 : 2));
    static constexpr int kValuesPerLane =
        kOutputProjectionGeometry && (ActiveTokens == 3 || ActiveTokens == 8) ? 16 : 8;
    static constexpr Bf16SmallTActivationAccess kActivationAccess =
        ActiveTokens <= 8 ? Bf16SmallTActivationAccess::WarpPacked
                          : Bf16SmallTActivationAccess::DirectStream;
    static constexpr bool kSequential = ActiveTokens <= 9 || ActiveTokens >= 17;
    static constexpr bool kUnroll2 =
        kOutputProjectionGeometry
            ? ((ActiveTokens >= 2 && ActiveTokens <= 8) ||
               (ActiveTokens >= 10 && ActiveTokens <= 19) || ActiveTokens >= 27)
            : (ActiveTokens == 4 || ActiveTokens == 5 || ActiveTokens == 8 || ActiveTokens >= 10);
    static constexpr bool kStreaming = !kOutputProjectionGeometry && ActiveTokens == 7;
    static constexpr Bf16WeightCache kWeightCache =
        kStreaming ? Bf16WeightCache::Streaming : Bf16WeightCache::Default;
    static constexpr Bf16PhaseOrder kPhaseOrder =
        kSequential ? Bf16PhaseOrder::Sequential : Bf16PhaseOrder::RowSwizzled;
    using Type =
        Bf16SmallTInnerSchedule<4, 1, kRowsPerWarp, kValuesPerLane, 1, 4, kActivationAccess,
                                kWeightCache, kPhaseOrder, 1, kUnroll2 ? 2 : 1, 1, 2>;
};

} // namespace ninfer::ops::detail
