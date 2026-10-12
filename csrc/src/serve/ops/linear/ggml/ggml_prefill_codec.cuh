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
#include "ops/linear/ggml/ggml_vecdot.cuh"

#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

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
    /// Bytes one staging item moves. Two for the scalar route, sixteen for a `cp_async` one.
    static constexpr int kStageUnit   = 2;
    static constexpr bool kCover   = false;
    static constexpr int kMinBlocks   = 1;
    /// A row's K must be a whole number of this. A tile-staged format needs only the 64-wide
    /// tile; the 128- and 256-value blocks are staged per superblock and need the whole 256.
    static constexpr int kKMultiple = kTileStaged ? kPrefillTileK : QK_K;
    /// Bytes of one row of `K` values -- what a kernel strides rows by. Not
    /// `(K / 256) * kBlockBytes`: a 32-value format on a row whose width is not a multiple of
    /// 256 (Gemma 4's 704-wide experts) holds a whole number of its own blocks and no whole
    /// superblock at all, and that division would lose the remainder.
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / kValues) * kBytes;
    }
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

/// Q2_0 stores four consecutive signed 2-bit values per byte. A warp owns
/// the block's 32 pairs, so each lane reads exactly its two values instead of
/// calling the generic eight-value decoder and discarding six results.
struct GgmlQ20Prefill : GgmlBlockPrefill<GgmlType::Q2_0> {
    static constexpr int kTileStride = 20; // whole words, with nine halfword staging units
    static constexpr int kHeaderStride = 16;
    static constexpr int kScaleGroups = 2;
    static constexpr int kScaleWidth = 32;
    static constexpr int kScaleTiles = 1;
    static constexpr bool kScalesInTile = true;
    static constexpr bool kHasMin = false;

    __device__ static __forceinline__ unsigned quad(unsigned packed) {
        const unsigned lanes = (packed & 3) | ((packed & 12) << 6) |
                               ((packed & 48) << 12) | ((packed & 192) << 18);
        return __vsub4(lanes, 0x01010101u); // independent signed bytes: code - 1
    }
    __device__ static __forceinline__ void unpack(
        const std::uint8_t* tile, const std::uint8_t*, int, int q, unsigned& lo, unsigned& hi) {
        lo = quad(tile[2 + q]);
        hi = quad(tile[10 + q]);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* tile, int) {
        return make_float2(__half2float(*reinterpret_cast<const __half*>(tile)), 0.0F);
    }
    __device__ static __forceinline__ __nv_bfloat162 decode(
        const std::uint8_t* tile, const std::uint8_t*, int lane, int) {
        const float d = __half2float(*reinterpret_cast<const __half*>(tile));
        const int codes = tile[2 + lane / 2] >> (4 * (lane & 1));
        return __floats2bfloat162_rn(d * ((codes & 3) - 1), d * (((codes >> 2) & 3) - 1));
    }
};

/// Q2_0 on the int8 route, staged with four-byte `cp_async` copies. An 18-byte block starts
/// only two-byte aligned on an odd tile, so each tile stages the 20-byte, four-byte-aligned
/// window around its block, which then sits at offset 0 (even tile) or 2 (odd tile) of the
/// staged bytes -- `tile_shift`, which the int8 tile adds wherever it reads them. The windows
/// need every row to start four-byte aligned, a whole number of 128 values (an even number of
/// blocks); an even tile's window reads two bytes into the next block of its row, and a row's
/// last tile is odd, so no window leaves its row. The scalar two-byte route (GgmlQ20Prefill)
/// stays for rows of an odd block count; staging was more than half of a Q2_0 expert pair's
/// time there (ablated at 4,096 Flash-Next tokens: 37.9 -> 16.6 ms).
struct GgmlQ20I8Prefill : GgmlQ20Prefill {
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 4;
    static constexpr int kTileBytes   = 20; // the staged window, five units
    static constexpr int kHeaderBytes = 0;  // the scale travels in the tile
    static constexpr int kKMultiple   = 128;
    static_assert(kTileBytes <= kTileStride && kTileStride % 4 == 0);
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return 18 * tile - 2 * (tile & 1) + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int tile_shift(int tile) { return 2 * (tile & 1); }
};

/// The int8 codec a row of `K` values takes: one that needs more of K than its format does
/// (GgmlQ20I8Prefill: a whole number of 128 values) falls back to the format's own.
template <class Int8Codec, int K>
struct I8CodecForK {
    using type = Int8Codec;
};
template <int K>
struct I8CodecForK<GgmlQ20I8Prefill, K> {
    using type = std::conditional_t<K % GgmlQ20I8Prefill::kKMultiple == 0, GgmlQ20I8Prefill,
                                    GgmlQ20Prefill>;
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
    static constexpr int kStageUnit   = 16;
    static constexpr bool kCover     = false;
    static constexpr int kKMultiple   = QK_K;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }
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
    /// Tiles between two scale refreshes: a superblock's worth, since that is what one header
    /// covers. A format whose scales live in the tile itself refreshes every tile instead.
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
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
    static constexpr int kStageUnit  = 16;
    static constexpr bool kCover     = false;
    static constexpr int kKMultiple  = QK_K;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }
    static constexpr int kMinBlocks  = 2; // its header carries qh as well, so the tile costs more

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride   = 48;
    static constexpr int kHeaderStride = 48; // dm, scales, qh: already twelve words
    static constexpr int kScaleGroups  = 8;
    static constexpr int kScaleWidth   = 32;
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
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
    static constexpr int kStageUnit  = 16;
    static constexpr bool kCover     = true;
    static constexpr int kMinBlocks  = 2;
    static constexpr int kKMultiple  = QK_K;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride   = 128;
    static constexpr int kHeaderStride = 32;
    /// Sixteen scales of sixteen values each, so this codec's MMA groups are sixteen deep.
    static constexpr int kScaleGroups = 16;
    static constexpr int kScaleWidth  = 16;
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
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


/// Q8_0 on the int8 tensor-core route. Not a K-quant and not a superblock at all: 32 signed
/// int8 codes under one FP16 scale, `w = d*q`, which is already the affine form the route's
/// MMA wants -- so this codec unpacks nothing. It exists because `llama-quantize` cannot
/// K-quant a tensor whose reduction axis is not a multiple of 256 and emits Q8_0 instead; the
/// Gemma 4 experts are 704 wide, so their `ffn_down_exps` is Q8_0 beside a Q6_K
/// `ffn_gate_up_exps`. Without this the pair had no prefill kernel and every prompt ran on the
/// decode kernels in 46-token slices.
///
/// Three things differ from the K-quant codecs and the shared K loop names each:
///  - **The scales are in the tile.** A 64-wide tile is exactly two blocks, 68 bytes, and
///    carries both its codes and both its scales, so there is no header to stage and the scale
///    table is rebuilt every tile (`kScaleTiles == 1`) rather than every superblock.
///  - **Nothing is 16-byte aligned.** A row is `(K/32) * 34` bytes and tile `t` starts `68 t`
///    into it; both are multiples of four and neither of sixteen, so the tile is staged in
///    4-byte units. Seventeen of them cover it exactly -- unlike Q6_K, which is also unaligned
///    and covers its tile with eight 16-byte copies from rounded-down addresses, this reads
///    nothing outside the tile and needs no per-row offset. The cost is 17 staging
///    instructions a row where a 16-byte copy would take 5.
///  - **K need not be a whole superblock.** Only a whole 64-wide tile, which 704 is.
struct GgmlQ80Prefill {
    using Block = block_q8_0;
    static constexpr int kBlockValues = QK8_0;                  // 32
    static constexpr int kBlockBytes  = sizeof(block_q8_0);     // 34
    static constexpr int kTileBytes   = (kPrefillTileK / QK8_0) * kBlockBytes; // 68, exact
    /// No header: `scale_pair` reads the tile.
    static constexpr int kHeaderBytes = 0;
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 4;
    static constexpr bool kCover      = false;
    static constexpr int kMinBlocks   = 2;
    static constexpr int kKMultiple   = kPrefillTileK;          // 64, not 256
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / kBlockValues) * kBlockBytes;
    }
    static_assert(kTileBytes % kStageUnit == 0, "the tile is a whole number of staged units");

    // ---- the int8 tensor-core route ----
    /// 72 rather than 68: a multiple of four, which the staged 4-byte copies need of every
    /// row's start, and eighteen words, which spreads consecutive rows over sixteen banks
    /// rather than the four a 16-byte-multiple stride would reach.
    static constexpr int kTileStride   = 72;
    /// The header buffer is unused; one unit keeps it a well-formed array rather than a
    /// zero-length one.
    static constexpr int kHeaderStride  = 16;
    static constexpr int kScaleGroups   = 2;   // two blocks to a 64-wide tile
    static constexpr int kScaleWidth    = 32;
    static constexpr int kScaleTiles    = 1;   // the scales arrive with the tile
    static constexpr bool kScalesInTile = true;
    /// Symmetric: `w = d*q` with q already signed, so there is no min term and nothing to
    /// subtract from the codes.
    static constexpr bool kHasMin = false;

    /// A word at a shared address two bytes above a four-byte boundary -- which every code word
    /// is, because a block's scale precedes its codes. Two halfword loads rather than a
    /// misaligned word: the byte order is the same and it needs no assumption about `p`.
    __device__ static __forceinline__ unsigned load_code_word(const std::uint8_t* p) {
        const auto* h = reinterpret_cast<const std::uint16_t*>(p);
        return static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16);
    }
    /// The codes of quad `q` (values 4q..4q+3) of the tile's two blocks, each packed as four
    /// int8 in one word with value i in byte i -- which is exactly how the file stores them.
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        lo = load_code_word(tile + offsetof(block_q8_0, qs) + 4 * q);
        hi = load_code_word(tile + kBlockBytes + offsetof(block_q8_0, qs) + 4 * q);
    }
    /// The scale of block `index` of this tile. `tile` is the staged tile, not a header.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* tile, int index) {
        return make_float2(
            __half2float(*reinterpret_cast<const __half*>(tile + kBlockBytes * index)), 0.0f);
    }

    /// The tile is the row's `tile`-th run of 68 bytes: no superblock to step over, because a
    /// row is a flat sequence of 34-byte blocks.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return tile * kTileBytes + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) { return tile * kTileBytes; }
};

/// Spreads the four low bits of `b` to bit 4 of each of a word's four bytes: bit i of `b`
/// becomes bit 4 of byte i. The fifth bit of a Q5_0 / Q5_1 value lives in `qh`, one bit per
/// value in value order, so the four bits of quad `q` are `qh >> 4q`, and this puts each above
/// the nibble it belongs to.
__device__ __forceinline__ unsigned spread_fifth_bits(unsigned b) {
    return ((b | (b << 7) | (b << 14) | (b << 21)) & 0x01010101u) << 4;
}

/// Q5_0 on the int8 tensor-core route: 32 unsigned 5-bit codes under one FP16 scale, with
/// `w = d * (q - 16)`. The bias is folded into the codes -- per-byte `q - 16` with `__vsub4`,
/// which does not borrow across bytes -- so they arrive as signed int8 and there is no min
/// term, the way Q6_K folds its 32. `llama-quantize` writes this for `ffn_down_exps` where the
/// expert width is not a multiple of 256 and the mix asks for less than Q8_0: the Q3_K_M and
/// Q4_K_M artifacts of a 704-wide mixture.
///
/// Like Q8_0, and for the same reason: the block is 22 bytes and a tile two of them, so
/// nothing is 16-byte aligned and the tile is staged in eleven 4-byte units that cover it
/// exactly; its two scales travel in the tile, so there is no header and the scale table is
/// rebuilt every tile.
struct GgmlQ50Prefill {
    using Block = block_q5_0;
    static constexpr int kBlockValues = QK5_0;                  // 32
    static constexpr int kBlockBytes  = sizeof(block_q5_0);     // 22: d, qh[4], qs[16]
    static constexpr int kTileBytes   = (kPrefillTileK / QK5_0) * kBlockBytes; // 44, exact
    static constexpr int kHeaderBytes = 0;
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 4;
    static constexpr bool kCover      = false;
    static constexpr int kMinBlocks   = 2;
    static constexpr int kKMultiple   = kPrefillTileK;          // 64, not 256
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / kBlockValues) * kBlockBytes;
    }
    static_assert(kTileBytes % kStageUnit == 0, "the tile is a whole number of staged units");
    // A row of K values is 11K/16 bytes, a multiple of four exactly when K is one of 64: what
    // the 4-byte staging needs of every row's start, and what `kKMultiple` guarantees.
    static_assert(((kKMultiple / kBlockValues) * kBlockBytes) % 4 == 0,
                  "rows start on a four-byte boundary");

    // ---- the int8 tensor-core route ----
    /// 52 rather than the tile's 44: thirteen words. The unpack reads, per warp, four
    /// consecutive rows at the same eight byte offsets; at eleven words a row the first and
    /// fourth row land on the same four banks, at thirteen no two of the four overlap. A
    /// multiple of four, which the 4-byte staging needs of every row's start.
    static constexpr int kTileStride    = 52;
    static constexpr int kHeaderStride  = 16;  // unused, one unit keeps the array well-formed
    static constexpr int kScaleGroups   = 2;   // two blocks to a 64-wide tile
    static constexpr int kScaleWidth    = 32;
    static constexpr int kScaleTiles    = 1;   // the scales arrive with the tile
    static constexpr bool kScalesInTile = true;
    static constexpr bool kHasMin       = false;

    /// A word at a two-byte-aligned shared address, as two halfwords (the block's code and
    /// `qh` runs both sit two bytes above a four-byte boundary in the first block of a tile).
    __device__ static __forceinline__ unsigned load_word2(const std::uint8_t* p) {
        const auto* h = reinterpret_cast<const std::uint16_t*>(p);
        return static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16);
    }
    /// Values 4q..4q+3 of one block: the low nibbles of `qs[4q..]` for q < 4 and the high
    /// nibbles of `qs[4q-16..]` otherwise -- the first sixteen values are the low nibbles, the
    /// last sixteen the high ones, as the format's own dequantiser reads them -- plus the
    /// fifth bit from `qh`, then minus sixteen per byte.
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int q) {
        const unsigned w   = load_word2(block + offsetof(block_q5_0, qs) + 4 * (q & 3));
        const unsigned nib = (q < 4 ? w : (w >> 4)) & 0x0F0F0F0Fu;
        const unsigned qh  = load_word2(block + offsetof(block_q5_0, qh));
        return __vsub4(nib | spread_fifth_bits((qh >> (4 * q)) & 0xFu), 0x10101010u);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        lo = quad(tile, q);
        hi = quad(tile + kBlockBytes, q);
    }
    /// The scale of block `index` of this tile; `tile` is the staged tile, not a header.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* tile, int index) {
        return make_float2(
            __half2float(*reinterpret_cast<const __half*>(tile + kBlockBytes * index)), 0.0f);
    }

    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return tile * kTileBytes + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) { return tile * kTileBytes; }
};

/// Q5_1 on the int8 tensor-core route: 32 unsigned 5-bit codes under an FP16 scale and an
/// FP16 minimum, `w = d * q + m`. The codes stay unsigned and the minimum takes the route's
/// min term: the shared K loop computes `(d * x) * dot - m' * sum(x)` per 32-value group, so
/// `scale_pair` hands it `(d, -m)`. Written by `llama-quantize` for the down tensor of the
/// Q5_K_M mix (and one layer of Q3_K_M) of a 704-wide mixture.
///
/// Unlike Q8_0 and Q5_0, everything here is 16-byte aligned: a block is 24 bytes, a tile 48,
/// and a row of K values 48 * (K / 64) -- so the tile is staged in three 16-byte `cp.async`
/// and its shared stride is its own 48 bytes, which the unpack's read pattern (four rows of
/// eight quads per warp) walks without a bank conflict.
struct GgmlQ51Prefill {
    using Block = block_q5_1;
    static constexpr int kBlockValues = QK5_1;                  // 32
    static constexpr int kBlockBytes  = sizeof(block_q5_1);     // 24: dm, qh[4], qs[16]
    static constexpr int kTileBytes   = (kPrefillTileK / QK5_1) * kBlockBytes; // 48, exact
    static constexpr int kHeaderBytes = 0;
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 16;
    static constexpr bool kCover      = false;
    static constexpr int kMinBlocks   = 2;
    static constexpr int kKMultiple   = kPrefillTileK;          // 64, not 256
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / kBlockValues) * kBlockBytes;
    }
    static_assert(kTileBytes % kStageUnit == 0, "the tile is a whole number of staged units");
    static_assert(((kKMultiple / kBlockValues) * kBlockBytes) % 16 == 0,
                  "rows start on a sixteen-byte boundary");

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride    = kTileBytes;  // 48: a multiple of sixteen for cp.async
    static constexpr int kHeaderStride  = 16;          // unused
    static constexpr int kScaleGroups   = 2;
    static constexpr int kScaleWidth    = 32;
    static constexpr int kScaleTiles    = 1;
    static constexpr bool kScalesInTile = true;
    /// `w = d*q + m`: the min term carries `m`, negated for the loop's subtraction.
    static constexpr bool kHasMin       = true;

    /// Values 4q..4q+3 of one block, unsigned 0..31: nibbles as Q5_0, fifth bit from `qh`.
    /// Every run here is word-aligned (`dm` is four bytes, `qh` four, `qs` sixteen).
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int q) {
        const unsigned w   = *reinterpret_cast<const unsigned*>(block + offsetof(block_q5_1, qs) +
                                                                4 * (q & 3));
        const unsigned nib = (q < 4 ? w : (w >> 4)) & 0x0F0F0F0Fu;
        const unsigned qh  = *reinterpret_cast<const unsigned*>(block + offsetof(block_q5_1, qh));
        return nib | spread_fifth_bits((qh >> (4 * q)) & 0xFu);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        lo = quad(tile, q);
        hi = quad(tile + kBlockBytes, q);
    }
    /// `(d, -m)` of block `index` of this tile: the loop subtracts its second component times
    /// the activation sum, and the format adds `m`.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* tile, int index) {
        const __half2 dm = *reinterpret_cast<const __half2*>(tile + kBlockBytes * index);
        return make_float2(__low2float(dm), -__high2float(dm));
    }

    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return tile * kTileBytes + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) { return tile * kTileBytes; }
};

/// Q3_K on the int8 tensor-core route. Sixteen sub-blocks of sixteen under a six-bit scale
/// each, `w = d * (sc - 32) * code`, where the code is two bits from `qs` plus a third bit from
/// the `hmask` plane that is stored inverted: set means "add nothing", clear means "subtract
/// four". So a code is -4..3, already signed, and there is no min term. Written by
/// `llama-quantize` for `ffn_gate_up_exps` of a Q3_K_M mix.
///
/// The layout is Q6_K's stripes over a 128-value half: value `v = 128n + 32j + l` reads two
/// bits `2j` of `qs[32n + l]` and bit `4n + j` of `hmask[l]`. A 64-wide tile is one pair of
/// stripes of one half, so it needs that half's thirty-two `qs` bytes and all thirty-two
/// `hmask` bytes -- and the whole superblock is only 110 bytes, less than Q6_K stages for one
/// tile. So the superblock is staged whole, once per four tiles, as the "header": eight
/// covered sixteen-byte copies from the rounded-down address (the block is 110 bytes and
/// nothing is sixteen-byte aligned), the last of them bounded at the row's end the way Q6_K's
/// header is, and there is no per-tile stage at all (`kTileBytes = 0`; the tile buffer keeps
/// one unit's stride so the array is well-formed, as Q8_0's unused header does). Every span
/// inside the block -- `hmask` at 0, `qs` at 32, `scales` at 96 -- is a multiple of sixteen
/// from its start, so one row offset serves them all.
struct GgmlQ3KPrefill {
    using Block = block_q3_K;
    static constexpr int kTileBytes          = 0;    // nothing per tile: the readers use the header
    static constexpr int kHeaderBytes        = 128;  // the 110-byte block, covered
    static constexpr int kHeaderPayloadBytes = sizeof(block_q3_K);
    static constexpr int kBlockBytes         = sizeof(block_q3_K); // 110
    static constexpr bool kCpAsync           = true;
    static constexpr int kStageUnit          = 16;
    static constexpr bool kCover             = true;
    static constexpr int kMinBlocks          = 2;
    static constexpr int kKMultiple          = QK_K;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }
    static_assert(offsetof(block_q3_K, hmask) % 16 == 0 && offsetof(block_q3_K, qs) % 16 == 0 &&
                      offsetof(block_q3_K, scales) % 16 == 0,
                  "one cover offset serves every span of the block");

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride   = 16;
    /// Forty words: rows land eight words apart modulo thirty-two, which is conflict-free for
    /// the unpack's four rows of eight consecutive words per warp when the four rows share a
    /// cover offset. They need not (a row is 1,210 bytes for K = 2,816, so consecutive rows'
    /// offsets cycle through 0, 10, 4, 14, ...), and then it is at most 2-way on three banks --
    /// the property Q6_K's covered tile already has.
    static constexpr int kHeaderStride = 160;
    /// Sixteen scales of sixteen values each, so this codec's MMA groups are sixteen deep.
    static constexpr int kScaleGroups   = 16;
    static constexpr int kScaleWidth    = 16;
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
    static constexpr bool kHasMin       = false;

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
    /// Values 4q..4q+3 of stripe `j` of half `n`: two bits from `qs`, the third from `hmask`,
    /// then minus four per byte where the mask bit was clear. `(q2 | bit << 2) - 4` is that:
    /// 0..7 minus four, and `__vsub4` does not borrow across bytes.
    __device__ static __forceinline__ unsigned stripe(const std::uint8_t* block, int n, int j,
                                                      int q) {
        const unsigned qs = load_word(block + offsetof(block_q3_K, qs) + 32 * n + 4 * q);
        const unsigned hm = load_word(block + offsetof(block_q3_K, hmask) + 4 * q);
        const unsigned q2 = (qs >> (2 * j)) & 0x03030303u;
        const unsigned b3 = ((hm >> (4 * n + j)) & 0x01010101u) << 2;
        return __vsub4(q2 | b3, 0x04040404u);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        const int n  = tile_in_block >> 1;
        const int j0 = 2 * (tile_in_block & 1);
        lo = stripe(header, n, j0, q);
        hi = stripe(header, n, j0 + 1, q);
    }
    /// `index` counts sixteen-value groups within the superblock in value order, which is the
    /// order the format's sixteen scales are in: `is = 8n + 2j + (l >> 4)` for the value
    /// `128n + 32j + l`. The six bits of scale `is` are split over two of the twelve bytes.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const std::uint8_t* sc = header + offsetof(block_q3_K, scales);
        const int is           = index;
        const int us =
            is < 4    ? (sc[is] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4)
            : is < 8  ? (sc[is] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4)
            : is < 12 ? (sc[is - 8] >> 4) | (((sc[is] >> 4) & 3) << 4)
                      : (sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3) << 4);
        const __half d = *reinterpret_cast<const __half*>(header + offsetof(block_q3_K, d));
        return make_float2(__half2float(d) * static_cast<float>(us - 32), 0.0f);
    }

    /// Never reached (no per-tile units); kept so the codec has the full interface.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        (void)chunk;
        return (tile / kTilesPerBlock) * kBlockBytes;
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes;
    }
};

// ---------------------------------------------------------------------------------------------
// The i-quants on the int8 tensor-core route. Every one of them is `w = scale * code` with a
// small signed integer code per value and one scale per 32 (or 16) values, which is all the
// route asks: a codebook format's code is its grid level with the sign applied (IQ2/IQ3), or its
// table level (IQ4); IQ1_M's +-1/8 offset becomes an integer once the scale takes the eighth.
// The codes are what llama.cpp's MMQ feeds its int8 MMAs, decoded here the way this engine's
// decode-path dot products (ggml_vecdot.cuh) decode them. The BF16 route stays the generic
// eight-value decoder (GgmlBlockPrefill).
//
// The superblock formats are 82 to 110 bytes and only two-byte aligned, so they are staged as
// Q3_K is: the whole superblock as the "header", covered by sixteen-byte copies from the
// rounded-down address, once per four tiles, with no per-tile stage.

/// A word at a two-byte-aligned shared address: the two aligned words it straddles,
/// funnel-shifted (as Q6_K and Q3_K read their covered spans).
__device__ __forceinline__ unsigned iq_load_word(const std::uint8_t* p) {
    const auto addr     = reinterpret_cast<std::uintptr_t>(p);
    const auto* aligned = reinterpret_cast<const unsigned*>(addr & ~std::uintptr_t{3});
    const unsigned sh   = static_cast<unsigned>(addr & 3u) * 8u;
    const unsigned w0   = aligned[0];
    const unsigned w1   = sh ? aligned[1] : 0u;
    return __funnelshift_r(w0, w1, sh);
}
/// The four sign bits `n` (bit i for value i) as a byte mask, 0xFF where the value is negative.
__device__ __forceinline__ unsigned iq_sign_bytes(unsigned n) {
    // n * 0x00204081 puts bit i at bit 8i; the mask keeps exactly those four.
    return __vcmpne4((n * 0x00204081u) & 0x01010101u, 0u);
}
/// Four unsigned grid levels with the bytes `negative` marks negated: (g ^ 0xFF) - 0xFF = -g,
/// and __vsub4 does not borrow between bytes.
__device__ __forceinline__ unsigned iq_signed(unsigned grid, unsigned negative) {
    return __vsub4(grid ^ negative, negative);
}
/// The 32 codes of a sixteen-level table format (IQ4_NL, IQ4_XS) are sixteen `qs` bytes: low
/// nibbles the first sixteen values, high nibbles the last. Quad `q` of them, as table levels.
__device__ __forceinline__ unsigned iq4_quad(unsigned qs_word, int q) {
    const int2 levels = table16_levels(static_cast<int>(qs_word), kIq4nlValues);
    return static_cast<unsigned>(q < 4 ? levels.x : levels.y);
}

/// IQ4_NL: 32 values under one FP16 scale, four-bit codes into the sixteen-level table. Staged
/// as Q8_0 is -- two 18-byte blocks to a 64-wide tile, scales in the tile, 4-byte units (a tile
/// starts 36t into its row, a multiple of four and not of sixteen). This is the down projection
/// of the GSQ-RCO Flash-Next GGUFs, 640 wide, which no 256-value superblock divides.
struct GgmlIq4NlPrefill {
    using Block = block_iq4_nl;
    static constexpr int kBlockValues = QK4_NL;                                // 32
    static constexpr int kBlockBytes  = sizeof(block_iq4_nl);                  // 18
    static constexpr int kTileBytes   = (kPrefillTileK / QK4_NL) * kBlockBytes; // 36, exact
    static constexpr int kHeaderBytes = 0;
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 4;
    static constexpr bool kCover      = false;
    static constexpr int kMinBlocks   = 2;
    static constexpr int kKMultiple   = kPrefillTileK;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / kBlockValues) * kBlockBytes;
    }
    static_assert(kTileBytes % kStageUnit == 0, "the tile is a whole number of staged units");

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride    = 40; // ten words: a multiple of four past the 36 bytes
    /// Ten 64-wide K steps for a 640-wide down row: the loop's fill is a large part of it, and
    /// a deeper pipeline keeps the 36-byte tiles' loads ahead of their use (see i8_stages).
    static constexpr int kStages        = 4;
    static constexpr int kHeaderStride  = 16; // unused; one unit keeps the array well-formed
    static constexpr int kScaleGroups   = 2;
    static constexpr int kScaleWidth    = 32;
    static constexpr int kScaleTiles    = 1;
    static constexpr bool kScalesInTile = true;
    static constexpr bool kHasMin       = false;

    /// Block 0's `qs` sit two bytes above a word and block 1's on one: halfword loads read both.
    __device__ static __forceinline__ unsigned qs_word(const std::uint8_t* p) {
        const auto* h = reinterpret_cast<const std::uint16_t*>(p);
        return static_cast<unsigned>(h[0]) | (static_cast<unsigned>(h[1]) << 16);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        const int j = offsetof(block_iq4_nl, qs) + 4 * (q & 3);
        lo = iq4_quad(qs_word(tile + j), q);
        hi = iq4_quad(qs_word(tile + kBlockBytes + j), q);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* tile, int index) {
        return make_float2(
            __half2float(*reinterpret_cast<const __half*>(tile + kBlockBytes * index)), 0.0f);
    }
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return tile * kTileBytes + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) { return tile * kTileBytes; }
};

/// IQ4_XS: IQ4_NL's table over a 136-byte superblock, a six-bit scale per 32 (`ls - 32`, low
/// nibbles in `scales_l`, two high bits each in `scales_h`). A 64-wide tile is 32 `qs` bytes,
/// the first sixteen one sub-block's and the next sixteen the other's; the header is the eight
/// bytes ahead of `qs`. A row is a whole number of 136-byte superblocks, so every superblock,
/// its header and each tile's `qs` run start eight-byte aligned (half of them sixteen): both
/// stage as eight-byte `cp_async` copies. The scalar two-byte route this had first, synchronous
/// loads in the middle of the K loop, left IQ4_XS the slowest expert pair here (31.4 ms at 4,096
/// Flash-Next tokens against 22 to 25 for the other i-quants).
struct GgmlIq4XsPrefill {
    using Block = block_iq4_xs;
    static constexpr int kTileBytes   = 32;
    static constexpr int kHeaderBytes = offsetof(block_iq4_xs, qs); // d, scales_h, scales_l: 8
    static constexpr int kBlockBytes  = sizeof(block_iq4_xs);       // 136
    static constexpr bool kCpAsync    = true;
    static constexpr int kStageUnit   = 8;
    static constexpr int kHeaderUnit  = 8;
    static constexpr bool kCover      = false;
    static constexpr int kMinBlocks   = 2;
    static constexpr int kKMultiple   = QK_K;
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }

    // ---- the int8 tensor-core route ----
    static constexpr int kTileStride    = 48;
    static constexpr int kHeaderStride  = 16;
    static constexpr int kScaleGroups   = 8;
    static constexpr int kScaleWidth    = 32;
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
    static constexpr bool kHasMin       = false;

    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)header;
        (void)tile_in_block;
        const int j = 4 * (q & 3);
        lo = iq4_quad(*reinterpret_cast<const unsigned*>(tile + j), q);
        hi = iq4_quad(*reinterpret_cast<const unsigned*>(tile + 16 + j), q);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const float d         = __half2float(*reinterpret_cast<const __half*>(header));
        const unsigned high   = *reinterpret_cast<const std::uint16_t*>(header + offsetof(block_iq4_xs, scales_h));
        const std::uint8_t lo = header[offsetof(block_iq4_xs, scales_l) + index / 2];
        const int ls = ((lo >> (4 * (index & 1))) & 0x0F) | (((high >> (2 * index)) & 0x03) << 4);
        return make_float2(d * static_cast<float>(ls - 32), 0.0f);
    }
    /// Eight-byte unit `chunk` of the tile's 32 `qs` bytes.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        return (tile / kTilesPerBlock) * kBlockBytes + offsetof(block_iq4_xs, qs) +
               32 * (tile % kTilesPerBlock) + kStageUnit * chunk;
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes;
    }
};

/// The covered whole-superblock stage the two-byte-aligned i-quants share (Q3_K's): `Units`
/// sixteen-byte copies reach the `Payload` bytes from any even offset below sixteen.
template <class B, int Units>
struct GgmlIqCoveredStage {
    using Block = B;
    static constexpr int kTileBytes          = 0; // nothing per tile: the readers use the header
    static constexpr int kHeaderBytes        = 16 * Units;
    static constexpr int kHeaderPayloadBytes = sizeof(B);
    static constexpr int kBlockBytes         = sizeof(B);
    static constexpr bool kCpAsync           = true;
    static constexpr int kStageUnit          = 16;
    static constexpr bool kCover             = true;
    static constexpr int kMinBlocks          = 2;
    static constexpr int kKMultiple          = QK_K;
    static_assert(kHeaderBytes >= kHeaderPayloadBytes + 14, "the cover reaches from any even offset");
    __host__ __device__ static constexpr std::int64_t row_bytes(int K) {
        return static_cast<std::int64_t>(K / QK_K) * kBlockBytes;
    }
    static constexpr int kTileStride    = 16;  // unused, as Q3_K's
    /// The covered span and one unit more: Q3_K's forty words would not leave room for a third
    /// stage under the 48 KiB static limit (see i8_stages).
    static constexpr int kHeaderStride  = kHeaderBytes + 16;
    static constexpr int kStages        = 3;
    static constexpr int kScaleTiles    = kTilesPerBlock;
    static constexpr bool kScalesInTile = false;
    static constexpr bool kHasMin       = false;
    /// Never reached (no per-tile units); kept so the codec has the full interface.
    __device__ static __forceinline__ int chunk_offset(int tile, int chunk) {
        (void)chunk;
        return (tile / kTilesPerBlock) * kBlockBytes;
    }
    __device__ static __forceinline__ int header_offset(int tile) {
        return (tile / kTilesPerBlock) * kBlockBytes;
    }
    __device__ static __forceinline__ float block_scale(const std::uint8_t* block) {
        return __half2float(*reinterpret_cast<const __half*>(block));
    }
};

/// IQ3_XXS (98 bytes): per 32 values, eight one-byte indices into a grid of four-level rows, and
/// one word holding four seven-bit sign groups (the eighth sign is the parity) and a four-bit
/// scale, `(2 ls + 1) / 4`.
struct GgmlIq3XxsPrefill : GgmlIqCoveredStage<block_iq3_xxs, 7> {
    static constexpr int kScaleGroups = 8;
    static constexpr int kScaleWidth  = 32;
    static constexpr int kQs          = offsetof(block_iq3_xxs, qs);
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int sub, int q) {
        const unsigned grid    = kIq3xxsGrid[block[kQs + 8 * sub + q]];
        const unsigned aux32   = iq_load_word(block + kQs + QK_K / 4 + 4 * sub);
        const unsigned signs   = unpack_ksigns(static_cast<std::uint8_t>(aux32 >> (7 * (q >> 1))));
        const unsigned negative = __vcmpne4(signs & ((q & 1) ? 0x80402010u : 0x08040201u), 0u);
        return iq_signed(grid, negative);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        lo = quad(header, 2 * tile_in_block, q);
        hi = quad(header, 2 * tile_in_block + 1, q);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const unsigned aux32 = iq_load_word(header + kQs + QK_K / 4 + 4 * index);
        const int ls         = static_cast<int>(aux32 >> 28);
        return make_float2(block_scale(header) * static_cast<float>(2 * ls + 1) * 0.25f, 0.0f);
    }
};

/// IQ3_S (110 bytes): per 32 values, eight nine-bit indices (eight bits in `qs`, the ninth in
/// `qh`) into a grid of four-level rows, a sign bit per value, and a four-bit scale `1 + 2 s`.
struct GgmlIq3SPrefill : GgmlIqCoveredStage<block_iq3_s, 8> {
    static constexpr int kScaleGroups = 8;
    static constexpr int kScaleWidth  = 32;
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int sub, int q) {
        const int qh            = block[offsetof(block_iq3_s, qh) + sub];
        const int index         = block[offsetof(block_iq3_s, qs) + 8 * sub + q] | (((qh >> q) & 1) << 8);
        const unsigned grid     = kIq3sGrid[index];
        const unsigned signs    = block[offsetof(block_iq3_s, signs) + 4 * sub + (q >> 1)];
        return iq_signed(grid, iq_sign_bytes((signs >> (4 * (q & 1))) & 0x0Fu));
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        lo = quad(header, 2 * tile_in_block, q);
        hi = quad(header, 2 * tile_in_block + 1, q);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const int s = (header[offsetof(block_iq3_s, scales) + index / 2] >> (4 * (index & 1))) & 0x0F;
        return make_float2(block_scale(header) * static_cast<float>(1 + 2 * s), 0.0f);
    }
};

/// IQ2_S (82 bytes): per 32 values, four ten-bit indices (eight bits in the first half of `qs`,
/// two in `qh`) into a grid of eight-level rows, a sign byte per row (the second half of `qs`),
/// and two four-bit scales `(2 ls + 1) / 8`, one per sixteen values -- so sixteen-deep groups.
struct GgmlIq2SPrefill : GgmlIqCoveredStage<block_iq2_s, 6> {
    static constexpr int kScaleGroups = 16;
    static constexpr int kScaleWidth  = 16;
    static constexpr int kQs          = offsetof(block_iq2_s, qs);
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int sub, int q) {
        const int row         = q >> 1; // the grid row of eight values this quad is half of
        const int qh          = block[offsetof(block_iq2_s, qh) + sub];
        const int index       = block[kQs + 4 * sub + row] | (((qh >> (2 * row)) & 3) << 8);
        const std::uint64_t r = kIq2sGrid[index];
        const unsigned grid   = static_cast<unsigned>((q & 1) ? (r >> 32) : r);
        const unsigned signs  = block[kQs + QK_K / 8 + 4 * sub + row];
        return iq_signed(grid, iq_sign_bytes((signs >> (4 * (q & 1))) & 0x0Fu));
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        lo = quad(header, 2 * tile_in_block, q);
        hi = quad(header, 2 * tile_in_block + 1, q);
    }
    /// `index` counts sixteen-value groups in the superblock: sub-block index / 2, its half
    /// index % 2.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const int ls = (header[offsetof(block_iq2_s, scales) + index / 2] >> (4 * (index & 1))) & 0x0F;
        return make_float2(block_scale(header) * static_cast<float>(2 * ls + 1) * 0.125f, 0.0f);
    }
};

/// IQ2_XXS (66 bytes): per 32 values, four one-byte indices into a grid of eight-level rows and
/// one word of four seven-bit sign groups plus a four-bit scale `(2 ls + 1) / 8`.
struct GgmlIq2XxsPrefill : GgmlIqCoveredStage<block_iq2_xxs, 5> {
    static constexpr int kScaleGroups = 8;
    static constexpr int kScaleWidth  = 32;
    static constexpr int kQs          = offsetof(block_iq2_xxs, qs);
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int sub, int q) {
        const int row           = q >> 1;
        const std::uint64_t r   = kIq2xxsGrid[block[kQs + 8 * sub + row]];
        const unsigned grid     = static_cast<unsigned>((q & 1) ? (r >> 32) : r);
        const unsigned aux32    = iq_load_word(block + kQs + 8 * sub + 4);
        const unsigned signs    = unpack_ksigns(static_cast<std::uint8_t>(aux32 >> (7 * row)));
        const unsigned negative = __vcmpne4(signs & ((q & 1) ? 0x80402010u : 0x08040201u), 0u);
        return iq_signed(grid, negative);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        lo = quad(header, 2 * tile_in_block, q);
        hi = quad(header, 2 * tile_in_block + 1, q);
    }
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const unsigned aux32 = iq_load_word(header + kQs + 8 * index + 4);
        const int ls         = static_cast<int>(aux32 >> 27) | 1;
        return make_float2(block_scale(header) * static_cast<float>(ls) * 0.125f, 0.0f);
    }
};

/// IQ1_M (56 bytes, eight-byte aligned): per 32 values, four eleven-bit indices (eight bits in
/// `qs`, three in a nibble of `qh` whose fourth bit is the row's offset sign) into the 1-bit grid
/// of eight levels in {0, 1, 2} standing for -1..1, a row offset of +-1/8, and two three-bit
/// scales `2 s + 1` per 32 under a block scale spread over the scale words' top nibbles. A value
/// is `dl * (g - 1 +- 1/8)`; times eight that is `8 g - 7` or `8 g - 9`, an exact int8 code, and
/// the eighth goes to the scale.
struct GgmlIq1MPrefill : GgmlIqCoveredStage<block_iq1_m, 5> {
    static constexpr int kScaleGroups = 16;
    static constexpr int kScaleWidth  = 16;
    __device__ static __forceinline__ unsigned quad(const std::uint8_t* block, int sub, int q) {
        const int row     = q >> 1;
        const int nibble  = block[offsetof(block_iq1_m, qh) + 2 * sub + (row >> 1)] >> (4 * (row & 1));
        const int index   = block[offsetof(block_iq1_m, qs) + 4 * sub + row] | ((nibble & 7) << 8);
        const unsigned g  = (kIq1sGridGpu[index] >> (4 * (q & 1))) & 0x0F0F0F0Fu;
        return __vsub4(g << 3, (nibble & 8) ? 0x09090909u : 0x07070707u);
    }
    __device__ static __forceinline__ void unpack(const std::uint8_t* tile,
                                                  const std::uint8_t* header, int tile_in_block,
                                                  int q, unsigned& lo, unsigned& hi) {
        (void)tile;
        lo = quad(header, 2 * tile_in_block, q);
        hi = quad(header, 2 * tile_in_block + 1, q);
    }
    __device__ static __forceinline__ float block_scale(const std::uint8_t* block) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(block + offsetof(block_iq1_m, scales));
        const unsigned short u = static_cast<unsigned short>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                             ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        return __half2float(__ushort_as_half(u));
    }
    /// `index` counts sixteen-value groups: sub-block index / 2 picks the six bits of the scale
    /// word, index % 2 which three.
    __device__ static __forceinline__ float2 scale_pair(const std::uint8_t* header, int index) {
        const auto* sc = reinterpret_cast<const std::uint16_t*>(header + offsetof(block_iq1_m, scales));
        const int sub  = index / 2;
        const int s    = (sc[sub / 2] >> (6 * (sub % 2) + 3 * (index & 1))) & 0x07;
        return make_float2(block_scale(header) * static_cast<float>(2 * s + 1) * 0.125f, 0.0f);
    }
};

/// The prefill codec of a stored format: the three K-quants with hand codecs (and the int8
/// tensor-core route); Q3_K, Q8_0, Q5_0 and Q5_1 with a hand codec on the int8 route only;
/// everything else the generic BF16 route.
///
/// `Codec` is the BF16 route's and `Int8Codec` the int8 route's; they are the same type for
/// every format but the plain 32-value blocks, whose int8 codec reads the file's bytes directly
/// while their BF16 decode goes through the generic eight-value decoder like any other plain
/// block format.
template <GgmlType type>
struct PrefillCodecFor {
    using Codec     = GgmlBlockPrefill<type>;
    using Int8Codec = Codec;
    static constexpr bool kInt8Route = false;
};
template <> struct PrefillCodecFor<GgmlType::Q2_0> {
    using Codec = GgmlQ20Prefill;
    using Int8Codec = GgmlQ20I8Prefill; // via I8CodecForK: the scalar route for odd block counts
    static constexpr bool kInt8Route = true;
};
template <> struct PrefillCodecFor<GgmlType::Q4_K> { using Codec = GgmlQ4KPrefill; using Int8Codec = Codec; static constexpr bool kInt8Route = true; };
template <> struct PrefillCodecFor<GgmlType::Q5_K> { using Codec = GgmlQ5KPrefill; using Int8Codec = Codec; static constexpr bool kInt8Route = true; };
template <> struct PrefillCodecFor<GgmlType::Q6_K> { using Codec = GgmlQ6KPrefill; using Int8Codec = Codec; static constexpr bool kInt8Route = true; };
template <> struct PrefillCodecFor<GgmlType::Q8_0> {
    using Codec     = GgmlBlockPrefill<GgmlType::Q8_0>;
    using Int8Codec = GgmlQ80Prefill;
    static constexpr bool kInt8Route = true;
};
template <> struct PrefillCodecFor<GgmlType::Q5_0> {
    using Codec     = GgmlBlockPrefill<GgmlType::Q5_0>;
    using Int8Codec = GgmlQ50Prefill;
    static constexpr bool kInt8Route = true;
};
template <> struct PrefillCodecFor<GgmlType::Q5_1> {
    using Codec     = GgmlBlockPrefill<GgmlType::Q5_1>;
    using Int8Codec = GgmlQ51Prefill;
    static constexpr bool kInt8Route = true;
};
template <> struct PrefillCodecFor<GgmlType::Q3_K> {
    using Codec     = GgmlBlockPrefill<GgmlType::Q3_K>;
    using Int8Codec = GgmlQ3KPrefill;
    static constexpr bool kInt8Route = true;
};
// The i-quants: the generic BF16 decoder, and their own int8 codecs.
#define SINFER_IQ_PREFILL_CODEC(NAME, INT8)                                                        \
    template <> struct PrefillCodecFor<GgmlType::NAME> {                                           \
        using Codec     = GgmlBlockPrefill<GgmlType::NAME>;                                        \
        using Int8Codec = INT8;                                                                    \
        static constexpr bool kInt8Route = true;                                                   \
    };
SINFER_IQ_PREFILL_CODEC(IQ4_NL, GgmlIq4NlPrefill)
SINFER_IQ_PREFILL_CODEC(IQ4_XS, GgmlIq4XsPrefill)
SINFER_IQ_PREFILL_CODEC(IQ3_XXS, GgmlIq3XxsPrefill)
SINFER_IQ_PREFILL_CODEC(IQ3_S, GgmlIq3SPrefill)
SINFER_IQ_PREFILL_CODEC(IQ2_S, GgmlIq2SPrefill)
SINFER_IQ_PREFILL_CODEC(IQ2_XXS, GgmlIq2XxsPrefill)
SINFER_IQ_PREFILL_CODEC(IQ1_M, GgmlIq1MPrefill)
#undef SINFER_IQ_PREFILL_CODEC

} // namespace sinfer::ops::detail::ggml
