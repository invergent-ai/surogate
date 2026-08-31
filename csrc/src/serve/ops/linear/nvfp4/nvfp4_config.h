#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace sinfer::ops::detail {

enum class Nvfp4ScaleAccess : std::uint8_t {
    StagedRaw,
    Direct,
};

enum class Nvfp4CodeCache : std::uint8_t {
    Default,
    Streaming,
};

enum class Nvfp4SmallTActivationAccess : std::uint8_t {
    PairStream,
    TokenPacked,
    SharedPhase,
};

enum class Nvfp4SmallTBlockOrder : std::uint8_t {
    RowsContiguous,
    TokenTilesContiguous,
};

template <std::int32_t OutputRows, std::int32_t InputRows>
struct Nvfp4GemvGeometry {
    static_assert(OutputRows > 0 && InputRows > 0);
    static_assert((OutputRows % 128) == 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kOutputRows       = OutputRows;
    static constexpr std::int32_t kInputRows        = InputRows;
    static constexpr std::int32_t kGroupsPerRow     = InputRows / 16;
    static constexpr std::int32_t kScaleTilesPerRow = InputRows / 64;
    static constexpr std::int32_t kCodeBytesPerRow  = InputRows / 2;
};

template <std::int32_t InputRows>
struct Nvfp4ActivationGeometry {
    static_assert(InputRows > 0);
    static_assert((InputRows % 64) == 0);

    static constexpr std::int32_t kInputRows       = InputRows;
    static constexpr std::int32_t kGroupsPerRow    = InputRows / 16;
    static constexpr std::int32_t kCodeBytesPerRow = InputRows / 2;
};

template <int WarpsPerCta, int RowsPerWarp, int ValuesPerLane, int AccumulatorChains,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int MinBlocksPerSm>
struct Nvfp4GemvSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowsPerCta        = WarpsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

template <int WarpsPerCta, int WarpsPerRow, int RowsPerWarp, int ValuesPerLane, int TokenTile,
          int AccumulatorChains, Nvfp4SmallTActivationAccess ActivationAccess,
          Nvfp4ScaleAccess ScaleAccess, Nvfp4CodeCache CodeCache, int PhaseUnroll,
          Nvfp4SmallTBlockOrder BlockOrder, int MinBlocksPerSm>
struct Nvfp4SmallTSchedule {
    static_assert(WarpsPerCta > 0 && WarpsPerCta <= 32);
    static_assert(WarpsPerRow > 0 && WarpsPerRow <= WarpsPerCta);
    static_assert((WarpsPerCta % WarpsPerRow) == 0);
    static_assert(RowsPerWarp > 0 && RowsPerWarp <= 8);
    static_assert(ValuesPerLane == 8 || ValuesPerLane == 16 || ValuesPerLane == 32);
    static_assert(TokenTile > 0);
    static_assert(AccumulatorChains > 0 && (AccumulatorChains & (AccumulatorChains - 1)) == 0);
    static_assert(AccumulatorChains <= ValuesPerLane / 2);
    static_assert(PhaseUnroll == 1 || PhaseUnroll == 2 || PhaseUnroll == 4);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kWarpsPerCta       = WarpsPerCta;
    static constexpr int kWarpsPerRow       = WarpsPerRow;
    static constexpr int kRowsPerWarp       = RowsPerWarp;
    static constexpr int kValuesPerLane     = ValuesPerLane;
    static constexpr int kTokenTile         = TokenTile;
    static constexpr int kAccumulatorChains = AccumulatorChains;
    static constexpr auto kActivationAccess = ActivationAccess;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kCodeCache        = CodeCache;
    static constexpr int kPhaseUnroll       = PhaseUnroll;
    static constexpr auto kBlockOrder       = BlockOrder;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr int kThreads           = WarpsPerCta * 32;
    static constexpr int kRowGroupsPerCta   = WarpsPerCta / WarpsPerRow;
    static constexpr int kRowsPerCta        = kRowGroupsPerCta * RowsPerWarp;
    static constexpr int kPairsPerLane      = ValuesPerLane / 2;
};

using Nvfp4AttnInputGeometry     = Nvfp4GemvGeometry<14336, 5120>;
using Nvfp4GdnInputGeometry      = Nvfp4GemvGeometry<16384, 5120>;
using Nvfp4MlpGateUpGeometry     = Nvfp4GemvGeometry<34816, 5120>;
using Nvfp4Residual6144Geometry  = Nvfp4GemvGeometry<5120, 6144>;
using Nvfp4Residual17408Geometry = Nvfp4GemvGeometry<5120, 17408>;

// The hidden-2560 family (Qwen3.5-4B). These shapes are *generic* problems — they have no
// in-house small-T or W4A4 ladder and take cuBLASLt at every width — with one exception that
// pays for itself at a single user: the decode GEMV is a plain template over the geometry, and
// at one token a GEMV streams the weight once where the cuBLASLt route runs a 128-row MMA tile
// on a single row (measured 2026-08-30: the 4B decodes at 45 % of memory peak against the
// 27B's 76 %, and its linears are 54 % of the token in that tile). So these geometries exist
// for exactly one route, `launch_nvfp4_decode`, and nothing else keys on them.
using Nvfp4AttnInput2560Geometry   = Nvfp4GemvGeometry<10240, 2560>;
using Nvfp4GdnInput2560Geometry    = Nvfp4GemvGeometry<12288, 2560>;
using Nvfp4MlpGateUp2560Geometry   = Nvfp4GemvGeometry<18432, 2560>;
using Nvfp4Residual4096Geometry    = Nvfp4GemvGeometry<2560, 4096>;
using Nvfp4Residual9216Geometry    = Nvfp4GemvGeometry<2560, 9216>;

enum class Nvfp4GemvOnlyProblem : std::uint8_t {
    AttnInput2560,
    GdnInput2560,
    MlpGateUp2560,
    Residual4096,
    Residual9216,
    None,
};

/// The GEMV-only geometry for a shape, or `None`. Disjoint from the registered problems by
/// construction (no registered shape has K = 2560, 4096 or 9216 with these row counts).
inline constexpr Nvfp4GemvOnlyProblem resolve_nvfp4_gemv_only_problem(std::int32_t output_rows,
                                                                      std::int32_t input_rows) {
    if (output_rows == Nvfp4AttnInput2560Geometry::kOutputRows &&
        input_rows == Nvfp4AttnInput2560Geometry::kInputRows) {
        return Nvfp4GemvOnlyProblem::AttnInput2560;
    }
    if (output_rows == Nvfp4GdnInput2560Geometry::kOutputRows &&
        input_rows == Nvfp4GdnInput2560Geometry::kInputRows) {
        return Nvfp4GemvOnlyProblem::GdnInput2560;
    }
    if (output_rows == Nvfp4MlpGateUp2560Geometry::kOutputRows &&
        input_rows == Nvfp4MlpGateUp2560Geometry::kInputRows) {
        return Nvfp4GemvOnlyProblem::MlpGateUp2560;
    }
    if (output_rows == Nvfp4Residual4096Geometry::kOutputRows &&
        input_rows == Nvfp4Residual4096Geometry::kInputRows) {
        return Nvfp4GemvOnlyProblem::Residual4096;
    }
    if (output_rows == Nvfp4Residual9216Geometry::kOutputRows &&
        input_rows == Nvfp4Residual9216Geometry::kInputRows) {
        return Nvfp4GemvOnlyProblem::Residual9216;
    }
    return Nvfp4GemvOnlyProblem::None;
}

inline constexpr bool is_nvfp4_gemv_only_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return resolve_nvfp4_gemv_only_problem(output_rows, input_rows) != Nvfp4GemvOnlyProblem::None;
}

// Shapes outside the five registered geometries run on the cuBLASLt route alone (#82): it is
// shape-generic, so only the activation quantizer needs an instantiation per K.
using Nvfp4Activation2560Geometry  = Nvfp4ActivationGeometry<2560>;
using Nvfp4Activation4096Geometry  = Nvfp4ActivationGeometry<4096>;
using Nvfp4Activation9216Geometry  = Nvfp4ActivationGeometry<9216>;
using Nvfp4Activation5120Geometry  = Nvfp4ActivationGeometry<5120>;
using Nvfp4Activation6144Geometry  = Nvfp4ActivationGeometry<6144>;
using Nvfp4Activation17408Geometry = Nvfp4ActivationGeometry<17408>;

// Byte offset of the UE4M3 scale for (row, 16-wide group) in the 128x4 tiled layout shared by
// the stored weight scales, the mma/TMA schedules and cuBLASLt's VEC16 block scaling.
__host__ __device__ constexpr std::int64_t nvfp4_tiled_scale_offset(std::int32_t row,
                                                                    std::int32_t group,
                                                                    std::int32_t tiles_per_row) {
    const std::int32_t m_tile    = row / 128;
    const std::int32_t row_inner = row - m_tile * 128;
    return (static_cast<std::int64_t>(m_tile) * tiles_per_row + group / 4) * 512 +
           (row_inner & 31) * 16 + (row_inner >> 5) * 4 + (group & 3);
}

enum class Nvfp4Problem : std::uint8_t {
    AttnInput,
    GdnInput,
    MlpGateUp,
    Residual6144,
    Residual17408,
};

inline constexpr bool is_nvfp4_registered_problem(std::int32_t output_rows,
                                                 std::int32_t input_rows) {
    return (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
            input_rows == Nvfp4AttnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
            input_rows == Nvfp4GdnInputGeometry::kInputRows) ||
           (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
            input_rows == Nvfp4MlpGateUpGeometry::kInputRows) ||
           (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
            input_rows == Nvfp4Residual6144Geometry::kInputRows) ||
           (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
            input_rows == Nvfp4Residual17408Geometry::kInputRows);
}

// The cuBLASLt route needs 128-aligned output rows (its scale tiles) and 64-aligned K; the
// activation quantizer additionally needs a K it was instantiated for. A registered shape is
// never generic: its ladder owns it, and the in-house small-T kernels beat cuBLASLt below the
// route's threshold, so admitting 5120 here must not pull the 27B's own shapes off them (#87).
inline constexpr bool is_nvfp4_generic_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows <= 0 || input_rows <= 0 || (output_rows % 128) != 0 || (input_rows % 64) != 0) {
        return false;
    }
    if (is_nvfp4_registered_problem(output_rows, input_rows)) { return false; }
    return input_rows == 2560 || input_rows == 4096 || input_rows == 5120 || input_rows == 9216;
}

inline constexpr bool is_nvfp4_linear_problem(std::int32_t output_rows, std::int32_t input_rows) {
    return is_nvfp4_registered_problem(output_rows, input_rows) ||
           is_nvfp4_generic_problem(output_rows, input_rows);
}

inline Nvfp4Problem resolve_nvfp4_problem(std::int32_t output_rows, std::int32_t input_rows) {
    if (output_rows == Nvfp4AttnInputGeometry::kOutputRows &&
        input_rows == Nvfp4AttnInputGeometry::kInputRows) {
        return Nvfp4Problem::AttnInput;
    }
    if (output_rows == Nvfp4GdnInputGeometry::kOutputRows &&
        input_rows == Nvfp4GdnInputGeometry::kInputRows) {
        return Nvfp4Problem::GdnInput;
    }
    if (output_rows == Nvfp4MlpGateUpGeometry::kOutputRows &&
        input_rows == Nvfp4MlpGateUpGeometry::kInputRows) {
        return Nvfp4Problem::MlpGateUp;
    }
    if (output_rows == Nvfp4Residual6144Geometry::kOutputRows &&
        input_rows == Nvfp4Residual6144Geometry::kInputRows) {
        return Nvfp4Problem::Residual6144;
    }
    if (output_rows == Nvfp4Residual17408Geometry::kOutputRows &&
        input_rows == Nvfp4Residual17408Geometry::kInputRows) {
        return Nvfp4Problem::Residual17408;
    }
    throw std::invalid_argument("unsupported NVFP4 problem");
}

// RTX 5090 cold-cache winner among the measured decode schedules.
template <class Geometry>
struct Nvfp4LinearDecodeProductionSchedule {
    using Type =
        Nvfp4GemvSchedule<8, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

// The 2,560-row residual shapes cannot fill the card at 16 rows per CTA (160 CTAs on 170 SMs),
// so they take half the rows per CTA and twice the grid. Halving again (4 rows, 640 CTAs)
// measured flat (348.2 against 350.8 tok/s on the 4B, inside noise), so CTA count is not what
// holds these two at 52-59 % of bandwidth; a split-K would be the next thing to try.
template <>
struct Nvfp4LinearDecodeProductionSchedule<Nvfp4Residual4096Geometry> {
    using Type =
        Nvfp4GemvSchedule<4, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};
template <>
struct Nvfp4LinearDecodeProductionSchedule<Nvfp4Residual9216Geometry> {
    using Type =
        Nvfp4GemvSchedule<4, 2, 16, 4, Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 2>;
};

inline constexpr std::int32_t kNvfp4FirstSmallT = 2;
inline constexpr std::int32_t kNvfp4LastSmallT  = 32;

// RTX 5090 cold-cache winners for contiguous Linear output. T=2..4 amortizes activation loads
// through shared staging; T=5..32 keeps one packed activation tile per warp. The warp-count changes
// are measured occupancy/register crossovers, not semantic frontiers.
template <class Geometry, int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens >= 17 ? 4 : (ActiveTokens >= 13 ? 16 : 8);
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens <= 4
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// G1's wider N benefits from keeping four warps per CTA throughout the A16 policy boundary. Only
// T=2 amortizes activation traffic enough for shared staging to win.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4GdnInputGeometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = ActiveTokens == 2
                                                  ? Nvfp4SmallTActivationAccess::SharedPhase
                                                  : Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// At N=5120, R1 needs the larger CTA only for the last three A16 token counts. The unoptimized
// A16-only tail keeps the established generic schedule.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual6144Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 16 ? (ActiveTokens >= 14 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

// R2's longer K moves the stable four-to-sixteen-warp crossover to T=8.
template <int ActiveTokens>
struct Nvfp4LinearSmallTProductionSchedule<Nvfp4Residual17408Geometry, ActiveTokens> {
    static_assert(ActiveTokens >= kNvfp4FirstSmallT);
    static_assert(ActiveTokens <= kNvfp4LastSmallT);
    static constexpr int kWarpsPerCta       = ActiveTokens <= 16 ? (ActiveTokens >= 8 ? 16 : 4) : 4;
    static constexpr int kValuesPerLane     = ActiveTokens >= 17 && ActiveTokens <= 20 ? 8 : 16;
    static constexpr auto kActivationAccess = Nvfp4SmallTActivationAccess::TokenPacked;
    using Type =
        Nvfp4SmallTSchedule<kWarpsPerCta, 1, 2, kValuesPerLane, ActiveTokens, 1, kActivationAccess,
                            Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
};

} // namespace sinfer::ops::detail
