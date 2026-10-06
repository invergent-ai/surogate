// Hopper block-scaled FP8 GEMM (see fp8_block_sm90_gemm.h): vLLM's CUTLASS 3.x kernel
// (fp8_block_sm90_gemm.cuh) and the choice of tile per round. This unit holds the dispatch and
// the narrow tile; the other tiles live in fp8_block_sm90_gemm_{mid,wide}.cu. Built for 90a only.

#include "ops/linear/fp8_block/fp8_block_sm90_gemm.cuh"

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace sinfer::ops::detail::fp8_block {
namespace sm90 {

bool launch_narrow(const Operands& o, bool residual) {
    return residual ? launch<Narrow<true>>(o) : launch<Narrow<false>>(o);
}

namespace {

enum class Family { Narrow, Mid, Wide };

struct TileChoice {
    Family family;
    int tokens;  // the tile's token extent
};

// vLLM's own choice, kept for A/B runs: the swapped 16-token tile up to 64 tokens, else 128 x 128
// (vLLM swaps whenever the token count breaks the scales' TMA alignment; their rows are padded
// here, so a wide round of any count keeps the 128-token tile).
TileChoice vllm_tile(std::int32_t tokens) {
    return tokens <= 64 ? TileChoice{Family::Narrow, 16} : TileChoice{Family::Wide, 128};
}

// Every family splits the weight rows 128 at a time; the round's tokens fill tiles of 16 (up to
// 32 tokens), 32 or 64 (up to 128, swapped so the weight rows fill M), or 128 or 256 rows. A
// taller tile reads the weight fewer times but launches fewer CTAs: between the two heights of a
// family, the persistent grid's wave count decides, a wave of the taller tile costing about 1.8
// of the shorter one's (H100 probe of the q/k/v, o, gate/up and down shapes of 8B to 27B dense
// models, 32 to 8192 tokens, where this picks the faster tile or one within 3% of it).
TileChoice choose_tile(std::int32_t tokens, std::int32_t n, int sms) {
    if (tokens <= 32) { return {Family::Narrow, 16}; }
    const std::int64_t row_tiles = (n + 127) / 128;
    const auto waves = [sms](std::int64_t tiles) { return (tiles + sms - 1) / sms; };
    const auto grid  = [&](int height) { return row_tiles * ((tokens + height - 1) / height); };
    const auto taller_wins = [&](int shorter, int taller) {
        return 9 * waves(grid(taller)) <= 5 * waves(grid(shorter));
    };
    if (tokens <= 128) { return {Family::Mid, taller_wins(32, 64) ? 64 : 32}; }
    return {Family::Wide, taller_wins(128, 256) ? 256 : 128};
}

bool run_tile(const TileChoice& tile, const Operands& o, bool residual) {
    switch (tile.family) {
    case Family::Narrow: return launch_narrow(o, residual);
    case Family::Mid: return launch_mid(o, residual, tile.tokens);
    case Family::Wide: return launch_wide(o, residual, tile.tokens);
    }
    return false;
}

} // namespace
} // namespace sm90

bool sm90_gemm_available() noexcept {
    // SUROGATE_SERVE_FP8_BLOCK_SM90=0 keeps Hopper on the engine's own tile kernel: an A/B switch
    // and an escape hatch. Read once, so the activation layout and the GEMM always agree.
    static const bool disabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_SM90");
        return raw != nullptr && std::string_view(raw) == "0";
    }();
    return !disabled && sm90::hardware().cc == 90;
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
    // SUROGATE_SERVE_FP8_BLOCK_SM90_TILES=0 keeps vLLM's two tiles, for A/B runs.
    static const bool vllm_tiles = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_SM90_TILES");
        return raw != nullptr && std::string_view(raw) == "0";
    }();
    const sm90::Operands o{act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream};
    const sm90::TileChoice fallback = sm90::vllm_tile(tokens);
    const sm90::TileChoice tile =
        vllm_tiles ? fallback : sm90::choose_tile(tokens, n, std::max(1, sm90::hardware().sm_count));
    if (sm90::run_tile(tile, o, residual)) { return true; }
    // A tile CUTLASS declines (can_implement) launches nothing; vLLM's own may still take it.
    return (tile.family != fallback.family || tile.tokens != fallback.tokens) &&
           sm90::run_tile(fallback, o, residual);
}

} // namespace sinfer::ops::detail::fp8_block
