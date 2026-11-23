#pragma once

// The Inkling attention prologue's per-unit device code (ENGINE-INKLING lane c), shared by prologue_kernel and the decode
// attention kernel's in-kernel prologue (plain decode, W = 1): the same functions, so the same bits on both routes.
// Semantics: api/ops/inkling_attention.h; numerics: lanes/c/DESIGN-attention.md section 1 (P1-P8).

#include "ops/inkling_attention/inkling_attention_device.cuh"

#include <cstdint>

namespace sinfer::ops::inkling_detail {

struct PrologueParams {
    const __nv_bfloat16* q;
    const __nv_bfloat16* r;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const __nv_bfloat16* k_dyn;
    const __nv_bfloat16* v_dyn;
    std::int64_t q_stride, r_stride, k_stride, v_stride, k_dyn_stride, v_dyn_stride; // elements per column

    const __nv_bfloat16* q_scale; // [128, 24]
    const __nv_bfloat16* r_scale; // [16, 24]
    const __nv_bfloat16* k_conv;  // local [4, 1024] (taps of a channel contiguous); global [1024, 4]
    const __nv_bfloat16* v_conv;

    const std::int32_t* positions;     // [W, B]
    const std::int32_t* valid_columns; // [B] or null
    const std::int32_t* table_rows;    // [B]
    const std::int32_t* block_tables;  // [logical_pages, rows]
    std::int32_t table_stride;         // logical pages per table row

    __nv_bfloat16* states; // conv state: (c, tap, slot)
    std::int64_t state_tap_stride, state_slot_stride; // elements
    std::int32_t k_offset, v_offset;                  // state channel of K / V channel 0
    const std::int32_t* slots;                        // [B]
    bool record;
    __nv_bfloat16* records;                           // Record: (c, column, row)
    std::int64_t record_col_stride, record_row_stride; // elements
    std::int32_t capture_column, capture_slot;        // capture_column < 0: none

    std::uint8_t* k_pages;
    std::uint8_t* v_pages;

    __nv_bfloat16* q_out; // [128, 24, N]
    __nv_bfloat16* r_out; // [16, 24, N]

    std::int32_t width, batch;
    std::int32_t kv_chunk;      // columns of one K/V unit (>= 4, or the whole row)
    std::int32_t kv_chunks;     // units per (row, head, K|V)
    std::int64_t kv_units, q_units, r_units;

    std::int32_t* zero;         // CTA 0 zeroes these ints after the wait (the fused call's arrival counters); null: none
    std::int32_t zero_count;
};

inline constexpr int kPrologueThreads = 128;

__device__ __forceinline__ int valid_count(const PrologueParams& p, int b) {
    if (p.valid_columns == nullptr) { return p.width; }
    const int v = p.valid_columns[b];
    return v < 0 ? 0 : (v < p.width ? v : p.width);
}

__device__ __forceinline__ void load_bf16x4(const __nv_bfloat16* src, float (&out)[4]) {
    const uint2 raw = load_vec<uint2>(src);
    const float2 lo = bf16x2_bits_to_float2(raw.x);
    const float2 hi = bf16x2_bits_to_float2(raw.y);
    out[0] = lo.x;
    out[1] = lo.y;
    out[2] = hi.x;
    out[3] = hi.y;
}

__device__ __forceinline__ void store_bf16x4(__nv_bfloat16* dst, const float (&x)[4]) {
    store_vec(dst, make_uint2(pack_bf16x2(x[0], x[1]), pack_bf16x2(x[2], x[3])));
}

// ops/attention.py `_log_scaling_tau_kernel` (fusion off): ratio = max((pos + 1) * (1 / FLOOR), 1);
// tau = 1 + ALPHA * log(ratio), libdevice logf (this archive is built without fast math).
__device__ __forceinline__ float log_tau(std::int32_t position) {
    const float ratio = fmaxf(__fmul_rn(static_cast<float>(position + 1),
                                        1.0F / static_cast<float>(Geo::kTauFloor)),
                              1.0F);
    return __fadd_rn(1.0F, __fmul_rn(Geo::kTauAlpha, logf(ratio)));
}

// One warp: K or V (kv 0 / 1) of one KV head, columns [chunk * kv_chunk, ...) of row b, in order.
// Lane l owns channels head * 128 + 4 l .. + 3. Only chunk 0 reads or writes the row's slot (and the capture
// slot): it reads the initial window first and writes afterwards, so one call reads and replaces a slot.
// Record mode: each unit writes its own columns' raw inputs to the records; the slot is never written.
template <bool Global>
__device__ void kv_unit(const PrologueParams& p, std::int64_t unit) {
    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int chunk = static_cast<int>(unit % p.kv_chunks);
    std::int64_t t  = unit / p.kv_chunks;
    const int kv    = static_cast<int>(t % 2);
    t /= 2;
    const int head = static_cast<int>(t % Geo::kKvHeads);
    const int b    = static_cast<int>(t / Geo::kKvHeads);

    const int valid   = valid_count(p, b);
    const int j_begin = chunk * p.kv_chunk;
    if (j_begin >= valid) { return; }
    const int j_end = min(j_begin + p.kv_chunk, valid);

    const int ch                  = head * Geo::kHeadDim + lane * 4; // channel within the stream
    const int state_ch            = (kv ? p.v_offset : p.k_offset) + ch;
    const std::int64_t row_slot   = p.slots[b];
    // Row b's columns start at batch column W * b of the [*, N] input views.
    const std::int64_t col_base   = static_cast<std::int64_t>(p.width) * b;
    const std::int64_t x_stride   = kv ? p.v_stride : p.k_stride;
    const __nv_bfloat16* x        = (kv ? p.v : p.k) + col_base * x_stride;
    const __nv_bfloat16* conv     = kv ? p.v_conv : p.k_conv;
    const std::int64_t dyn_stride = kv ? p.v_dyn_stride : p.k_dyn_stride;
    const __nv_bfloat16* dyn      = Global ? (kv ? p.v_dyn : p.k_dyn) + col_base * dyn_stride : nullptr;
    std::uint8_t* pages           = kv ? p.v_pages : p.k_pages;
    const std::int32_t* positions = p.positions + static_cast<std::int64_t>(p.width) * b;
    const std::int32_t* table =
        p.block_tables + static_cast<std::int64_t>(p.table_rows[b]) * p.table_stride;
    const std::int32_t p0 = positions[0];

    // Taps u0, u1, u2 = the inputs at columns j - 3, j - 2, j - 1 of column j = j_begin. A column before
    // the row reads the initial window (tap 3 + column, oldest first); a position below 0 reads zero.
    float u[3][4];
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) {
        const int col = j_begin - 3 + tap;
        if (col >= 0) {
            load_bf16x4(x + static_cast<std::int64_t>(col) * x_stride + ch, u[tap]);
        } else if (p0 + col >= 0) {
            load_bf16x4(p.states + state_ch + static_cast<std::int64_t>(3 + col) * p.state_tap_stride +
                            row_slot * p.state_slot_stride,
                        u[tap]);
        } else {
            u[tap][0] = u[tap][1] = u[tap][2] = u[tap][3] = 0.0F;
        }
    }

    // Rewrite checkpoint (B = 1 prefill): the window before capture_column, from the inputs or the slot.
    if (chunk == 0 && p.capture_column >= 0) {
        __nv_bfloat16* dst = p.states + state_ch + static_cast<std::int64_t>(p.capture_slot) * p.state_slot_stride;
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) {
            const int col = p.capture_column - 3 + tap;
            float t4[4]   = {0.0F, 0.0F, 0.0F, 0.0F};
            if (col >= 0) {
                load_bf16x4(x + static_cast<std::int64_t>(col) * x_stride + ch, t4);
            } else if (p0 + col >= 0) {
                load_bf16x4(p.states + state_ch + static_cast<std::int64_t>(3 + col) * p.state_tap_stride +
                                row_slot * p.state_slot_stride,
                            t4);
            }
            store_bf16x4(dst + tap * p.state_tap_stride, t4);
        }
    }

    // Static: w[i][k] (channel i, tap k oldest first). Dynamic: w[k][i] = static_w (lag k, channel i).
    float w[4][4];
    if constexpr (!Global) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { load_bf16x4(conv + static_cast<std::int64_t>(ch + i) * 4, w[i]); }
    } else {
#pragma unroll
        for (int lag = 0; lag < 4; ++lag) { load_bf16x4(conv + lag * 1024 + ch, w[lag]); }
    }

    for (int j = j_begin; j < j_end; ++j) {
        float cur[4];
        load_bf16x4(x + static_cast<std::int64_t>(j) * x_stride + ch, cur);
        float y[4];
        if constexpr (!Global) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float acc = 0.0F;
                acc       = fmaf(u[0][i], w[i][0], acc);
                acc       = fmaf(u[1][i], w[i][1], acc);
                acc       = fmaf(u[2][i], w[i][2], acc);
                acc       = fmaf(cur[i], w[i][3], acc);
                y[i]      = round_bf16(__fmul_rn(__fadd_rn(acc, cur[i]), kStaticConvScale));
            }
        } else {
            float d4[4];
            load_bf16x4(dyn + static_cast<std::int64_t>(j) * dyn_stride + head * 4, d4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // Newest first: lag 0 = cur, lag 1 = u2, lag 2 = u1, lag 3 = u0.
                float acc = __fmul_rn(cur[i], __fadd_rn(w[0][i], d4[0]));
                acc       = fmaf(u[2][i], __fadd_rn(w[1][i], d4[1]), acc);
                acc       = fmaf(u[1][i], __fadd_rn(w[2][i], d4[2]), acc);
                acc       = fmaf(u[0][i], __fadd_rn(w[3][i], d4[3]), acc);
                y[i]      = round_bf16(__fmul_rn(__fadd_rn(acc, cur[i]), kDynamicConvScale));
            }
        }
        const float ss = sum_squares<32, 4>(y);
        float z[4];
        if (kv == 0) {
            const float f = l2_clip_factor(ss, Geo::kQkClip);
#pragma unroll
            for (int i = 0; i < 4; ++i) { z[i] = __fmul_rn(y[i], f); }
        } else {
            const float rs =
                ptx_rsqrt(__fadd_rn(__fdiv_rn(ss, static_cast<float>(Geo::kHeadDim)), Geo::kNormEps));
#pragma unroll
            for (int i = 0; i < 4; ++i) { z[i] = __fmul_rn(y[i], rs); }
        }
        const std::int32_t pos  = positions[j];
        const std::int32_t page = table[pos >> kPageShift];
        store_vec(pages + kv_page_offset(page, head, pos & (kPageTokens - 1)) + lane * 4,
                  e4m3x4(z[0], z[1], z[2], z[3]));

        if (p.record) {
            store_bf16x4(p.records + state_ch + static_cast<std::int64_t>(j) * p.record_col_stride +
                             static_cast<std::int64_t>(b) * p.record_row_stride,
                         cur);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[0][i] = u[1][i];
            u[1][i] = u[2][i];
            u[2][i] = cur[i];
        }
    }

    if (!p.record && chunk == 0) {
        // The window after the row's last valid column: still in registers when this unit ran every
        // column, else the raw inputs of the last three columns (all at columns >= kv_chunk >= 4).
        if (valid > j_end) {
#pragma unroll
            for (int tap = 0; tap < 3; ++tap) {
                load_bf16x4(x + static_cast<std::int64_t>(valid - 3 + tap) * x_stride + ch, u[tap]);
            }
        }
        __nv_bfloat16* dst = p.states + state_ch + row_slot * p.state_slot_stride;
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) { store_bf16x4(dst + tap * p.state_tap_stride, u[tap]); }
    }
}

// q of valid column n, head h: dims 4 lane .. + 3 of the whole warp's head (every lane of the warp calls it).
template <bool Global>
__device__ __forceinline__ void q_values(const PrologueParams& p, std::int64_t n, int h, int lane, float (&y)[4]) {
    const int d = lane * 4;
    float x[4], s[4];
    load_bf16x4(p.q + n * p.q_stride + h * Geo::kHeadDim + d, x);
    load_bf16x4(p.q_scale + h * Geo::kHeadDim + d, s);
#pragma unroll
    for (int i = 0; i < 4; ++i) { x[i] = __fmul_rn(x[i], s[i]); }
    const float f = l2_clip_factor(sum_squares<32, 4>(x), Geo::kQkClip);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[i] = __fmul_rn(x[i], f);
        if constexpr (Global) { y[i] = __fmul_rn(round_bf16(y[i]), log_tau(p.positions[n])); }
    }
}

// One warp: q of one (column, head): 4 dims a lane.
template <bool Global>
__device__ void q_unit(const PrologueParams& p, std::int64_t unit) {
    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int h         = static_cast<int>(unit % Geo::kQHeads);
    const std::int64_t n = unit / Geo::kQHeads;
    const int b         = static_cast<int>(n / p.width);
    const int j         = static_cast<int>(n % p.width);
    const int d         = lane * 4;
    __nv_bfloat16* dst  = p.q_out + (n * Geo::kQHeads + h) * Geo::kHeadDim + d;
    float y[4];
    if (j >= valid_count(p, b)) {
        y[0] = y[1] = y[2] = y[3] = 0.0F;
        store_bf16x4(dst, y);
        return;
    }
    q_values<Global>(p, n, h, lane, y);
    store_bf16x4(dst, y);
}

// r of valid column n, head h, dim d (a half warp per head: every lane of the 16 calls it).
template <bool Global>
__device__ __forceinline__ float r_value(const PrologueParams& p, std::int64_t n, int h, int d) {
    float x[1];
    x[0] = __fmul_rn(__bfloat162float(p.r[n * p.r_stride + h * Geo::kRelDim + d]),
                     __bfloat162float(p.r_scale[h * Geo::kRelDim + d]));
    const float f = l2_clip_factor(sum_squares<16, 1>(x), Geo::kRelClip);
    float y       = __fmul_rn(x[0], f);
    if constexpr (Global) { y = __fmul_rn(round_bf16(y), log_tau(p.positions[n])); }
    return y;
}

// One warp: r of two heads of one column; lanes 0-15 head 2g, 16-31 head 2g + 1, one dim a lane.
template <bool Global>
__device__ void r_unit(const PrologueParams& p, std::int64_t unit) {
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int g          = static_cast<int>(unit % (Geo::kQHeads / 2));
    const std::int64_t n = unit / (Geo::kQHeads / 2);
    const int b          = static_cast<int>(n / p.width);
    const int j          = static_cast<int>(n % p.width);
    const int h          = 2 * g + (lane >> 4);
    const int d          = lane & 15;
    __nv_bfloat16* dst   = p.r_out + (n * Geo::kQHeads + h) * Geo::kRelDim + d;
    if (j >= valid_count(p, b)) {
        *dst = __float2bfloat16_rn(0.0F);
        return;
    }
    *dst = __float2bfloat16_rn(r_value<Global>(p, n, h, d));
}


/// Host: validate the prologue's arguments and fill its parameters (inkling_prologue.cu). `out` null = no q / r outputs
/// (the decode attention's in-kernel prologue keeps them in shared memory).
PrologueParams prologue_params(const InklingProjections& projections, const InklingAttentionWeights& weights,
                               InklingLayerKind kind, const InklingAttentionRows& rows, const InklingConvStateArgs& conv,
                               const PagedKVBatchLayerView& cache, const InklingPrologueOutputs* out, const char* op);

} // namespace sinfer::ops::inkling_detail
