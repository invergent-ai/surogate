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
#include "ops/linear/ggml/ggml_moe_codec.cuh"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>

namespace sinfer::ops::detail::ggml {

inline constexpr int kPrefillTileK = 64;
inline constexpr int kTilesPerBlock = QK_K / kPrefillTileK; // four

/// Every other GGML format, through the eight-value decoder the gather and the CPU expert path
/// already share. This is the BF16 route only -- the int8 tensor-core route wants an affine
/// code-times-scale structure per 32 that the codebook and float formats do not have -- and it
/// is the fallback, not the fast path: a lane decodes eight values to keep two. What it buys is
/// that a resident-expert MoE in any stored format prefills instead of refusing.
///
/// Staging follows the block size. A block of 64 values or fewer is staged per 64-wide tile
/// (the tile is a whole number of blocks); the 128- and 256-value blocks are staged per
/// superblock as the "header", the way the K-quant codecs stage their scales. Blocks are
/// staged whole and exact, in two-byte units, so nothing is read past a row's last block.
template <GgmlType type>
struct GgmlBlockPrefill {
    static constexpr int kValues      = block_values(type);
    static constexpr int kBytes       = block_bytes(type);
    static constexpr bool kTileStaged = kValues <= kPrefillTileK;
    static constexpr int kTileBytes   = kTileStaged ? (kPrefillTileK / kValues) * kBytes : 2;
    static constexpr int kHeaderBytes = kTileStaged ? 2 : (QK_K / kValues) * kBytes;
    /// Bytes per 256 values of a row: what the kernels stride rows by.
    static constexpr int kBlockBytes  = (QK_K / kValues) * kBytes;
    static constexpr bool kCpAsync    = false;
    static constexpr bool kCover   = false;
    static constexpr int kMinBlocks   = 1;
    static_assert(kPrefillTileK % kValues == 0 || kValues % kPrefillTileK == 0,
                  "a tile is a whole number of blocks, or a block a whole number of tiles");
    static_assert(kTileBytes % 2 == 0 && kHeaderBytes % 2 == 0, "staged in two-byte units");

    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        // the tile's blocks, contiguous: chunk c is bytes 16c.. of that run
        if constexpr (kTileStaged) {
            return (tile / kTilesPerBlock) * kBlockBytes + (tile % kTilesPerBlock) * kTileBytes + 16 * chunk;
        } else {
            (void)chunk;
            return (tile / kTilesPerBlock) * kBlockBytes;
        }
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes;
    }

    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* tile,
                                                            const std::uint8_t* header, int lane,
                                                            int tile_in_block) {
        // the lane's two values, as an index into the staged run of blocks
        const int v0 = kTileStaged ? 2 * lane : tile_in_block * kPrefillTileK + 2 * lane;
        const std::uint8_t* run = kTileStaged ? tile : header;
        float w[8];
        decode_eight<type>(run, v0 / kValues, (v0 % kValues) / 8, w);
        return __floats2bfloat162_rn(w[v0 % 8], w[v0 % 8 + 1]);
    }
};

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
    static constexpr bool kCover     = false;
    /// Its header is one aligned sixteen-byte run, and its tile the narrowest of the three, so
    /// this is the only K-quant whose gate/up fits three blocks to an SM.
    static constexpr int kMinBlocks   = 3;

    // ---- the int8 tensor-core route ----
    /// Padded row strides in shared memory: the unpack reads one word per row across eight
    /// consecutive rows, and twelve words a row visits every bank once before repeating.
    static constexpr int kTileStride   = 48;
    static constexpr int kHeaderStride = 16;
    /// Scales per superblock and the width each covers; an MMA group must not straddle one.
    static constexpr int kScaleGroups = 8;
    static constexpr int kScaleWidth  = 32;
    static constexpr bool kHasMin     = true;

    /// The codes of quad `q` (values 4q..4q+3) of both 32-value halves of a 64-wide tile, each
    /// packed as four int8 in one word with value i in byte i -- the s8 MMA fragment order. The
    /// codes stay unsigned; the affine correction is the epilogue's.
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        const unsigned w = *reinterpret_cast<const unsigned*>(tile + 4 * q);
        lo = w & 0x0F0F0F0Fu;
        hi = (w >> 4) & 0x0F0F0F0Fu;
    }
    /// (d*sc, dmin*m) of sub-block `index` of the superblock whose header is given.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const __half2 dm = *reinterpret_cast<const __half2*>(header);
        std::uint8_t sc  = 0;
        std::uint8_t m   = 0;
        get_scale_min_k4(index, header + 2 * sizeof(__half), sc, m);
        return make_float2(__low2float(dm) * static_cast<float>(sc),
                           __high2float(dm) * static_cast<float>(m));
    }

    /// Byte offset of one staged chunk from the row's first superblock. Row-independent by
    /// construction: the row's base already costs a 64-bit multiply, and folding the block
    /// stride in there too made it two per staged chunk rather than one per row.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return (tile / kTilesPerBlock) * kBlockBytes + offsetof(block_q4_K, qs) +
               32 * (tile % kTilesPerBlock) + 16 * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes; // dm leads the block
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
    static constexpr bool kCover     = false;
    static constexpr int kMinBlocks  = 2; // its header carries qh as well, so the tile costs more

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride   = 48;
    static constexpr int kHeaderStride = 48; // dm, scales, qh: already twelve words
    static constexpr int kScaleGroups  = 8;
    static constexpr int kScaleWidth   = 32;
    static constexpr bool kHasMin      = true;

    /// As Q4_K, plus the fifth bit: one `qh` bit position per 32-value sub-block, so the low
    /// half of the tile takes position `2*tile_in_block` and the high half the one above it.
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        const unsigned w  = *reinterpret_cast<const unsigned*>(tile + 4 * q);
        const unsigned qh = *reinterpret_cast<const unsigned*>(header + 2 * sizeof(__half) +
                                                               K_SCALE_SIZE + 4 * q);
        lo = (w & 0x0F0F0F0Fu) | (((qh >> (2 * tile_in_block)) & 0x01010101u) << 4);
        hi = ((w >> 4) & 0x0F0F0F0Fu) | (((qh >> (2 * tile_in_block + 1)) & 0x01010101u) << 4);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const __half2 dm = *reinterpret_cast<const __half2*>(header);
        std::uint8_t sc  = 0;
        std::uint8_t m   = 0;
        get_scale_min_k4(index, header + 2 * sizeof(__half), sc, m);
        return make_float2(__low2float(dm) * static_cast<float>(sc),
                           __high2float(dm) * static_cast<float>(m));
    }

    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return (tile / kTilesPerBlock) * kBlockBytes + offsetof(block_q5_K, qs) +
               32 * (tile % kTilesPerBlock) + 16 * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes;
    }

    /// `dm`, `scales` and `qh` are contiguous in that order inside the block, so the header is
    /// one run; `qs` follows them and is staged per tile instead.


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
    /// 210 bytes, which is not a multiple of sixteen: consecutive blocks are only two-byte
    /// aligned, so a tile's bytes start at some even offset `off` below sixteen. Staging them
    /// with scalar loads -- forty-eight two-byte copies where every other codec issues six
    /// sixteen-byte `cp_async` -- held the Q6_K down kernel at 22 % of peak bandwidth against
    /// Q4_K's 39 %. Instead the tile is *covered*: eight aligned sixteen-byte copies from the
    /// rounded-down addresses, five for the sixty-four `ql` bytes and three for the thirty-two
    /// `qh`, and the row's `off` is folded into the pointers its readers get (kCover). Within a
    /// block every span sits a multiple of sixteen from the block's start, so one offset serves
    /// `ql`, `qh` and the scales alike.
    static constexpr int kTileBytes   = 128;                             // ql span 80, qh span 48
    static constexpr int kHeaderBytes = 32;                              // scales[16], d: 18 covered
    static constexpr int kHeaderPayloadBytes = 18;
    static constexpr int kBlockBytes = sizeof(block_q6_K);
    static constexpr bool kCpAsync   = true;
    static constexpr bool kCover     = true;
    static constexpr int kMinBlocks  = 2;

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride   = 128;
    static constexpr int kHeaderStride = 32;
    /// Sixteen scales of sixteen values each, so this codec's MMA groups are sixteen deep.
    static constexpr int kScaleGroups = 16;
    static constexpr int kScaleWidth  = 16;
    /// Symmetric: the 32 the format subtracts is folded into the codes here, which makes them
    /// signed int8 and leaves nothing for a min term to do.
    static constexpr bool kHasMin = false;

    __device__ static __forceinline__ unsigned minus_32(unsigned v) {
        // Per byte, v - 32 for v in 0..63: keep the low five bits and let bit five's complement
        // fill the top three (0x20 * 7 == 0xE0, and nothing carries between bytes).
        return (v & 0x1F1F1F1Fu) | (((v & 0x20202020u) ^ 0x20202020u) * 7u);
    }
    /// The low half of the tile is stripe `2*odd`, the high half stripe `2*odd + 1`; a stripe's
    /// nibble sits in `ql[32 * (stripe & 1) + i]` shifted by `4 * (stripe >> 1)`, and its two
    /// high bits in `qh[i]` at `2 * stripe`.
    /// A word at a two-byte-aligned shared address: the two aligned words it straddles,
    /// funnel-shifted. Branchless, since the offset is the row's and differs per thread.
    __device__ static __forceinline__ unsigned load_word(const std::uint8_t* p) {
        const auto addr     = reinterpret_cast<std::uintptr_t>(p);
        const auto* aligned = reinterpret_cast<const unsigned*>(addr & ~std::uintptr_t{3});
        const unsigned sh   = static_cast<unsigned>(addr & 3u) * 8u;
        const unsigned w0   = aligned[0];
        const unsigned w1   = sh ? aligned[1] : 0u;
        return __funnelshift_r(w0, w1, sh);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        const int odd       = tile_in_block & 1;
        const int shift     = 4 * odd;
        const unsigned ql0  = load_word(tile + 4 * q);
        const unsigned ql1  = load_word(tile + 32 + 4 * q);
        const unsigned qh   = load_word(tile + 80 + 4 * q);
        const unsigned lo6  = ((ql0 >> shift) & 0x0F0F0F0Fu) |
                              (((qh >> (4 * odd)) & 0x03030303u) << 4);
        const unsigned hi6  = ((ql1 >> shift) & 0x0F0F0F0Fu) |
                              (((qh >> (4 * odd + 2)) & 0x03030303u) << 4);
        lo = minus_32(lo6);
        hi = minus_32(hi6);
    }
    /// `index` counts sixteen-value groups within the superblock; the format's scale for the
    /// values a tile reads is `scales[4 * tile_in_block + h]` for its h-th sixteen.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const auto* scales = reinterpret_cast<const std::int8_t*>(header);
        const __half d_all = *reinterpret_cast<const __half*>(header + QK_K / 16);
        return make_float2(__half2float(d_all) * static_cast<float>(scales[index]), 0.0f);
    }

    /// Units 0-4 cover the half's sixty-four `ql` bytes, 5-7 its thirty-two `qh`; the stager
    /// rounds each source down to sixteen and lands unit `u` at shared byte `16 u`, so the
    /// logical byte `i` of `ql` sits at `off + i` and of `qh` at `80 + off + i`.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        const int block = (tile / kTilesPerBlock) * kBlockBytes;
        const int half  = (tile % kTilesPerBlock) >> 1;
        return chunk < 5 ? block + offsetof(block_q6_K, ql) + 64 * half + 16 * chunk
                         : block + offsetof(block_q6_K, qh) + 32 * half + 16 * (chunk - 5);
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes + offsetof(block_q6_K, scales);
    }



    __device__ static __forceinline__ __nv_bfloat162 decode(const std::uint8_t* tile,
                                                            const std::uint8_t* header, int lane,
                                                            int tile_in_block) {
        const auto* scales = reinterpret_cast<const std::int8_t*>(header);
        const __half d_all = *reinterpret_cast<const __half*>(header + QK_K / 16);
        const std::uint8_t* ql = tile;
        const std::uint8_t* qh = tile + 80;
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

/// The prefill codec of a stored format: the three K-quants with hand codecs (and the int8
/// tensor-core route), everything else the generic BF16 route.
template <GgmlType type>
struct PrefillCodecFor {
    using Codec = GgmlBlockPrefill<type>;
    static constexpr bool kInt8Route = false;
};
template <> struct PrefillCodecFor<GgmlType::Q4_K> { using Codec = GgmlQ4KPrefill; static constexpr bool kInt8Route = true; };
template <> struct PrefillCodecFor<GgmlType::Q5_K> { using Codec = GgmlQ5KPrefill; static constexpr bool kInt8Route = true; };
template <> struct PrefillCodecFor<GgmlType::Q6_K> { using Codec = GgmlQ6KPrefill; static constexpr bool kInt8Route = true; };

} // namespace sinfer::ops::detail::ggml
