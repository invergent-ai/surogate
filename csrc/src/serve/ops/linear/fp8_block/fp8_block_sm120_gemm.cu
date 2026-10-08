// sm_12x block-scaled FP8 GEMM (see fp8_block_sm120_gemm.h): CUTLASS's sm_120 blockwise kernel
// (fp8_block_sm120_gemm.cuh) and the choice of tile per round. This unit holds the dispatch and
// the narrow tile; the other tiles live in fp8_block_sm120_gemm_{mid,mid_row,wide}.cu. Built for the
// sm_12x targets of SUROGATE_SERVE_CUDA_ARCHS only.

#include "ops/linear/fp8_block/fp8_block_sm120_gemm.cuh"

#include "core/device.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

namespace sinfer::ops::detail::fp8_block {
namespace sm120 {

bool launch_narrow(const Operands& o, bool residual, bool per_row) {
    if (per_row) { return residual ? launch_blockwise<Narrow<true, true>>(o) : launch_blockwise<Narrow<false, true>>(o); }
    return residual ? launch_blockwise<Narrow<true, false>>(o) : launch_blockwise<Narrow<false, false>>(o);
}

namespace {

enum class Family { Narrow, Mid, Wide };

struct TileChoice {
    Family family;
    int tokens; // the tile's token extent
};

// SUROGATE_SERVE_FP8_BLOCK_SM120_TILE runs every round on one tile -- p32 (narrow, ping-pong),
// c32 or c64 (mid, cooperative), c128 (wide) -- for measuring the tiles against each other and
// testing each one on every shape.
const TileChoice* forced_tile() {
    static const TileChoice tiles[] = {{Family::Narrow, 32}, {Family::Mid, 32}, {Family::Mid, 64}, {Family::Wide, 128}};
    static const TileChoice* const forced = []() -> const TileChoice* {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_SM120_TILE");
        const std::string_view name = raw != nullptr ? raw : "";
        if (name == "p32") { return &tiles[0]; }
        if (name == "c32") { return &tiles[1]; }
        if (name == "c64") { return &tiles[2]; }
        if (name == "c128") { return &tiles[3]; }
        return nullptr;
    }();
    return forced;
}

// Every family splits the weight rows 128 at a time; a round's tokens fill tiles of 32 or 64
// (swapped) or 128. Up to 64 tokens the narrowest tile that holds the round runs; past it the
// 64-token tile stays a candidate where the 128-token one would leave SMs idle, each costed as its
// waves times a per-CTA figure (10 : 14, Hopper's measurement of the same two tiles; GB10's
// 48 SMs make the wave count the larger term).
TileChoice choose_tile(std::int32_t tokens, std::int32_t n, int sms) {
    if (const TileChoice* forced = forced_tile(); forced != nullptr) { return *forced; }
    if (tokens <= 32) { return {Family::Narrow, 32}; }
    if (tokens <= 64) { return {Family::Mid, 64}; }
    const std::int64_t row_tiles = (n + 127) / 128;
    const auto waves = [sms](std::int64_t tiles) { return (tiles + sms - 1) / sms; };
    const auto grid  = [&](int height) { return row_tiles * ((tokens + height - 1) / height); };
    return 10 * waves(grid(64)) < 14 * waves(grid(128)) ? TileChoice{Family::Mid, 64}
                                                         : TileChoice{Family::Wide, 128};
}

// Qualified: the operands are Hopper's type, so an unqualified call would also find Hopper's
// launchers by argument-dependent lookup.
bool run_tile(const TileChoice& tile, const Operands& o, bool residual, bool per_row) {
    switch (tile.family) {
    case Family::Narrow: return sm120::launch_narrow(o, residual, per_row);
    case Family::Mid:
        return per_row ? sm120::launch_mid_row(o, residual, tile.tokens) : sm120::launch_mid(o, residual, tile.tokens);
    case Family::Wide: return sm120::launch_wide(o, residual, per_row);
    }
    return false;
}

} // namespace
} // namespace sm120

bool sm120_gemm_available() noexcept {
    // SUROGATE_SERVE_FP8_BLOCK_SM120=0 keeps these GPUs on the engine's own tile kernel: an A/B
    // switch and an escape hatch. Read once, so the activation layout and the GEMM always agree.
    static const bool disabled = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_FP8_BLOCK_SM120");
        return raw != nullptr && std::string_view(raw) == "0";
    }();
    return !disabled && fp4_tensor_cores(sm90::hardware().cc);
}

bool sm120_gemm(const std::uint8_t* act_codes, const float* act_scales, const std::uint8_t* w_codes,
                const float* w_scales, bool per_row, void* out_bf16, bool residual,
                std::int32_t tokens, std::int32_t n, std::int32_t k, cudaStream_t stream) {
    // TMA reads every operand tile and writes the output: 16-byte aligned bases and rows. The
    // weight rows are whole 128-row tiles (and, block-scaled, whole scale blocks).
    const auto aligned = [](const void* p) { return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0; };
    if (!sm120_gemm_available() || tokens <= 0 || n <= 0 || k <= 0 || (k % 128) != 0 ||
        (n % 128) != 0 || !aligned(act_codes) || !aligned(act_scales) || !aligned(w_codes) ||
        !aligned(w_scales) || !aligned(out_bf16)) {
        return false;
    }
    const sm120::Operands o{act_codes, act_scales, w_codes, w_scales, out_bf16, tokens, n, k, stream};
    const sm120::TileChoice tile = sm120::choose_tile(tokens, n, std::max(1, sm90::hardware().sm_count));
    if (sm120::run_tile(tile, o, residual, per_row)) { return true; }
    // A tile CUTLASS declines (can_implement) launches nothing; vLLM's own may still take it.
    return tile.family != sm120::Family::Wide && sm120::run_tile({sm120::Family::Wide, 128}, o, residual, per_row);
}

} // namespace sinfer::ops::detail::fp8_block
