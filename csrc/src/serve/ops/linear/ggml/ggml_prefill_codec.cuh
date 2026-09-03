// Ported from llama.cpp (ggml/src/ggml-cuda), MIT License, Copyright (c) 2023-2026 The ggml
// authors. The block layouts and the dequantisation arithmetic are kept verbatim so that a GGUF's
// K-quant tensors are served as the bytes the file holds; only the surrounding names change.
#pragma once

// Weight decode for the tiled MoE prefill GEMMs, whose shape differs from every other K-quant
// consumer here. Those walk K linearly and let a lane own eight consecutive values; this one
// stages a 64-wide K tile of sixty-four rows into shared memory and hands each lane the *pair*
// the BF16 MMA fragment wants. So each codec answers two questions: which bytes of a superblock
// a 64-value tile needs (the kernel stages exactly those), and how to turn the staged bytes plus
// the superblock's header into that pair.
//
// The tile is 64 values wide and a superblock is 256, so four consecutive tiles share one
// superblock and its header; the header is staged once for the whole row block rather than four
// times over.

#include "ops/linear/ggml/ggml_blocks.h"
#include "ops/linear/ggml/ggml_dequant.cuh"

#include <cuda_bf16.h>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

inline constexpr int kPrefillTileK = 64;
inline constexpr int kTilesPerBlock = QK_K / kPrefillTileK; // four

/// Q4_K. A 64-value tile is exactly the 32 `qs` bytes of quarter `c`: their low nibbles are the
/// tile's first thirty-two values, their high nibbles the last thirty-two. The pair a lane owns
/// is two consecutive values, so it never straddles that halfway point and both of its values
/// share one 32-value sub-block -- one scale lookup per lane, not two.
struct GgmlQ4KPrefill {
    using Block = block_q4_K;
    static constexpr int kTileBytes   = 32;                     // per row, per 64-wide K tile
    static constexpr int kHeaderBytes = 2 * sizeof(__half) + K_SCALE_SIZE; // per row, per block
    static constexpr int kBlockBytes  = sizeof(block_q4_K);     // 144: a multiple of sixteen,
    static constexpr bool kCpAsync    = true;                   // so `cp_async<16>` is in bounds
    /// Its header is one aligned sixteen-byte run, and its tile the narrowest of the three, so
    /// this is the only K-quant whose gate/up fits three blocks to an SM.
    static constexpr int kMinBlocks   = 3;

    /// `blocks` is already advanced to the row's first superblock.
    __device__ static __forceinline__ const std::uint8_t* tile_source(const void* blocks,
                                                                      int tile) {
        const auto* b = static_cast<const block_q4_K*>(blocks) + tile / kTilesPerBlock;
        return b->qs + 32 * (tile % kTilesPerBlock);
    }

    __device__ static __forceinline__ const std::uint8_t* header_source(const void* blocks,
                                                                        int block) {
        return reinterpret_cast<const std::uint8_t*>(
            &(static_cast<const block_q4_K*>(blocks) + block)->dm);
    }

    /// One sixteen-byte run of the tile, for staging.
    __device__ static __forceinline__ const std::uint8_t* chunk_source(const void* blocks, int tile,
                                                                       int chunk) {
        return tile_source(blocks, tile) + 16 * chunk;
    }

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* tile,
                                                            const std::uint8_t* header, int lane,
                                                            int tile_in_block) {
        const __half2 dm = *reinterpret_cast<const __half2*>(header);
        const std::uint8_t* scales = header + 2 * sizeof(__half);
        const int v0 = lane * 2;
        const int hi = v0 >> 5; // low nibbles for the tile's first thirty-two values
        const int r  = v0 & 31;
        std::uint8_t sc = 0;
        std::uint8_t m  = 0;
        get_scale_min_k4(2 * tile_in_block + hi, scales, sc, m);
        const float d  = __low2float(dm) * static_cast<float>(sc);
        const float mo = __high2float(dm) * static_cast<float>(m);
        const std::uint8_t b0 = tile[r];
        const std::uint8_t b1 = tile[r + 1];
        const int q0 = hi ? (b0 >> 4) : (b0 & 0xF);
        const int q1 = hi ? (b1 >> 4) : (b1 & 0xF);
        return __floats2bfloat162_rn(d * static_cast<float>(q0) - mo,
                                     d * static_cast<float>(q1) - mo);
    }
};

/// Q5_K. Same 4-bit body as Q4_K plus a fifth bit per value in `qh`, where one bit *position*
/// serves a whole 32-value sub-block: a 64-value tile therefore needs all thirty-two `qh` bytes,
/// not a slice of them. They are staged with the header, once per superblock, because four tiles
/// read the same thirty-two bytes.
struct GgmlQ5KPrefill {
    using Block = block_q5_K;
    static constexpr int kTileBytes = 32;
    static constexpr int kHeaderBytes =
        2 * sizeof(__half) + K_SCALE_SIZE + QK_K / 8; // dm, scales, qh
    static constexpr int kBlockBytes = sizeof(block_q5_K); // 176, also a multiple of sixteen
    static constexpr bool kCpAsync   = true;
    static constexpr int kMinBlocks  = 2; // its header carries qh as well, so the tile costs more

    __device__ static __forceinline__ const std::uint8_t* tile_source(const void* blocks,
                                                                      int tile) {
        const auto* b = static_cast<const block_q5_K*>(blocks) + tile / kTilesPerBlock;
        return b->qs + 32 * (tile % kTilesPerBlock);
    }

    /// `dm`, `scales` and `qh` are contiguous in that order inside the block, so the header is
    /// one run; `qs` follows them and is staged per tile instead.
    __device__ static __forceinline__ const std::uint8_t* header_source(const void* blocks,
                                                                        int block) {
        return reinterpret_cast<const std::uint8_t*>(
            &(static_cast<const block_q5_K*>(blocks) + block)->dm);
    }

    __device__ static __forceinline__ const std::uint8_t* chunk_source(const void* blocks, int tile,
                                                                       int chunk) {
        return tile_source(blocks, tile) + 16 * chunk;
    }

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* tile,
                                                            const std::uint8_t* header, int lane,
                                                            int tile_in_block) {
        const __half2 dm           = *reinterpret_cast<const __half2*>(header);
        const std::uint8_t* scales = header + 2 * sizeof(__half);
        const std::uint8_t* qh     = scales + K_SCALE_SIZE;
        const int v0 = lane * 2;
        const int hi = v0 >> 5;
        const int r  = v0 & 31;
        std::uint8_t sc = 0;
        std::uint8_t m  = 0;
        get_scale_min_k4(2 * tile_in_block + hi, scales, sc, m);
        const float d   = __low2float(dm) * static_cast<float>(sc);
        const float mo  = __high2float(dm) * static_cast<float>(m);
        const auto bit  = static_cast<std::uint8_t>(1u << (2 * tile_in_block + hi));
        const std::uint8_t b0 = tile[r];
        const std::uint8_t b1 = tile[r + 1];
        const int q0 = (hi ? (b0 >> 4) : (b0 & 0xF)) + ((qh[r] & bit) ? 16 : 0);
        const int q1 = (hi ? (b1 >> 4) : (b1 & 0xF)) + ((qh[r + 1] & bit) ? 16 : 0);
        return __floats2bfloat162_rn(d * static_cast<float>(q0) - mo,
                                     d * static_cast<float>(q1) - mo);
    }
};

/// Q6_K, whose 128-value half interleaves four stripes rather than splitting in two: within a
/// half, value `v` reads `ql[v & 31 (+32 when the stripe is odd)]` and two bits of `qh[v & 31]`.
/// A 64-value tile is one pair of stripes, so it spans all sixty-four `ql` bytes and all
/// thirty-two `qh` bytes of its half -- ninety-six staged bytes rather than thirty-two, and the
/// two tiles of a half stage the same ones. Symmetric, so no min term.
struct GgmlQ6KPrefill {
    using Block = block_q6_K;
    static constexpr int kTileBytes   = 96;                              // ql[64] then qh[32]
    static constexpr int kHeaderBytes = QK_K / 16 + sizeof(__half);      // scales, d
    /// 210 bytes, which is not a multiple of sixteen: consecutive blocks are only two-byte
    /// aligned, so this one stages with scalar loads rather than `cp_async`. llama.cpp reads
    /// Q6_K through its two-byte-aligned accessors for the same reason.
    static constexpr int kBlockBytes = sizeof(block_q6_K);
    static constexpr bool kCpAsync   = false;
    static constexpr int kMinBlocks  = 2;

    __device__ static __forceinline__ const std::uint8_t* tile_source(const void* blocks,
                                                                      int tile) {
        const auto* b = static_cast<const block_q6_K*>(blocks) + tile / kTilesPerBlock;
        return b->ql + 64 * ((tile % kTilesPerBlock) >> 1);
    }
    /// `qh` does not follow `ql` contiguously for the half in question, so it is staged as its
    /// own run and lands directly after the ninety-six-byte tile's first sixty-four.
    __device__ static __forceinline__ const std::uint8_t*
    tile_high_source(const void* blocks, int tile) {
        const auto* b = static_cast<const block_q6_K*>(blocks) + tile / kTilesPerBlock;
        return b->qh + 32 * ((tile % kTilesPerBlock) >> 1);
    }

    __device__ static __forceinline__ const std::uint8_t* header_source(const void* blocks,
                                                                        int block) {
        return reinterpret_cast<const std::uint8_t*>(
            (static_cast<const block_q6_K*>(blocks) + block)->scales);
    }

    /// The tile's first sixty-four bytes come from `ql`, its last thirty-two from `qh`.
    __device__ static __forceinline__ const std::uint8_t* chunk_source(const void* blocks, int tile,
                                                                       int chunk) {
        return chunk < 4 ? tile_source(blocks, tile) + 16 * chunk
                         : tile_high_source(blocks, tile) + 16 * (chunk - 4);
    }

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* tile,
                                                            const std::uint8_t* header, int lane,
                                                            int tile_in_block) {
        const auto* scales = reinterpret_cast<const std::int8_t*>(header);
        const __half d_all = *reinterpret_cast<const __half*>(header + QK_K / 16);
        const std::uint8_t* ql = tile;
        const std::uint8_t* qh = tile + 64;
        // Stripes within the half: the tile's low thirty-two values are stripe `2*odd`, its high
        // thirty-two stripe `2*odd + 1`, where `odd` says which tile of the half this is.
        const int odd = tile_in_block & 1;
        const int v0  = lane * 2;
        const int t   = 2 * odd + (v0 >> 5);
        const int il  = v0 & 31;
        const int ip  = (tile_in_block >> 1); // which 128-value half, for the scale run
        const float d = __half2float(d_all) *
                        static_cast<float>(scales[8 * ip + il / 16 + 2 * t]);
        const std::uint8_t* q = ql + il + 32 * (t & 1);
        const int shift       = (t >> 1) * 4;
        const int q0 = static_cast<int>((q[0] >> shift) & 0xF) |
                       static_cast<int>(((qh[il] >> (2 * t)) & 3) << 4);
        const int q1 = static_cast<int>((q[1] >> shift) & 0xF) |
                       static_cast<int>(((qh[il + 1] >> (2 * t)) & 3) << 4);
        return __floats2bfloat162_rn(d * static_cast<float>(q0 - 32),
                                     d * static_cast<float>(q1 - 32));
    }
};

} // namespace sinfer::ops::detail::ggml
