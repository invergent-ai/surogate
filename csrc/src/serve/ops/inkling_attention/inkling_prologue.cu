// Inkling-2-Mini attention prologue, log-n tau, relative-projection table and the dense relative
// logits (test route). ENGINE-INKLING lane c; semantics: api/ops/inkling_attention.h and
// lanes/c/DESIGN-attention.md sections 1 (P1-P8), 2 and 3. Reference: vLLM's
// ops/sconv.py `_attn_prologue_kernel` / `_kv_conv_tile` / `_tap_sum`, ops/norm.py `_l2_clip` /
// `qr_clip_rel_tokens`, ops/attention.py `compute_log_scaling_tau`.

#include "ops/inkling_attention/inkling_attention_common.h"
#include "ops/inkling_attention/inkling_attention_device.cuh"

#include "core/device.h"
#include "core/pdl.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace sinfer::ops {
namespace inkling_detail {
namespace {

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
    store_bf16x4(dst, y);
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
    float x[1];
    x[0] = __fmul_rn(__bfloat162float(p.r[n * p.r_stride + h * Geo::kRelDim + d]),
                     __bfloat162float(p.r_scale[h * Geo::kRelDim + d]));
    const float f = l2_clip_factor(sum_squares<16, 1>(x), Geo::kRelClip);
    float y       = __fmul_rn(x[0], f);
    if constexpr (Global) { y = __fmul_rn(round_bf16(y), log_tau(p.positions[n])); }
    *dst = __float2bfloat16_rn(y);
}

template <bool Global>
__global__ __launch_bounds__(kPrologueThreads) void prologue_kernel(PrologueParams p) {
    // The attention launched after this one may start (its own wait guards what it reads from us).
    pdl::trigger_dependents();
    pdl::wait_for_dependencies();
    if (p.zero != nullptr && blockIdx.x == 0) {
        for (int i = static_cast<int>(threadIdx.x); i < p.zero_count; i += kPrologueThreads) { p.zero[i] = 0; }
    }
    std::int64_t unit = static_cast<std::int64_t>(blockIdx.x) * (kPrologueThreads / 32) +
                        (threadIdx.x >> 5);
    if (unit < p.kv_units) {
        kv_unit<Global>(p, unit);
        return;
    }
    unit -= p.kv_units;
    if (unit < p.q_units) {
        q_unit<Global>(p, unit);
        return;
    }
    unit -= p.q_units;
    if (unit < p.r_units) { r_unit<Global>(p, unit); }
}

__global__ void log_tau_kernel(const std::int32_t* positions, float* tau, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    tau[i] = log_tau(positions[i]);
}

__global__ void rel_table_kernel(const __nv_bfloat16* rel_proj, __nv_bfloat16* table) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= Geo::kRelDim * Geo::kRelTableRows) { return; }
    const int j = i % Geo::kRelDim;
    const int d = i / Geo::kRelDim - Geo::kRelTablePad;
    table[i]    = (d >= 0 && d < Geo::kRelExtent) ? rel_proj[j * Geo::kRelExtent + d]
                                                  : __float2bfloat16_rn(0.0F);
}

// Test route: one warp per 16 consecutive (column, head) rows, every distance, on the k16 MMA the attention
// kernels use for their bias strips.
__global__ void rel_logits_kernel(const __nv_bfloat16* r_out, const __nv_bfloat16* table,
                                  __nv_bfloat16* rel, std::int64_t rows) {
    const int lane          = static_cast<int>(threadIdx.x) & 31;
    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const std::int64_t row0 = (static_cast<std::int64_t>(blockIdx.x) * (blockDim.x >> 5) +
                               (threadIdx.x >> 5)) * 16;
    if (row0 >= rows) { return; }
    const std::int64_t ra = row0 + gid, rb = row0 + gid + 8;
    const auto r_word = [&](std::int64_t row, int col) -> unsigned {
        return row < rows ? load_vec<unsigned>(r_out + row * Geo::kRelDim + col) : 0U;
    };
    const unsigned a0 = r_word(ra, 2 * lid), a1 = r_word(rb, 2 * lid);
    const unsigned a2 = r_word(ra, 2 * lid + 8), a3 = r_word(rb, 2 * lid + 8);
    for (int nt = 0; nt < Geo::kRelExtent / 8; ++nt) {
        const __nv_bfloat16* t = table + (Geo::kRelTablePad + nt * 8 + gid) * Geo::kRelDim + 2 * lid;
        float c0 = 0.0F, c1 = 0.0F, c2 = 0.0F, c3 = 0.0F;
        mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, load_vec<unsigned>(t), load_vec<unsigned>(t + 8));
        const int d = nt * 8 + 2 * lid;
        if (ra < rows) { store_vec(rel + ra * Geo::kRelExtent + d, pack_bf16x2(c0, c1)); }
        if (rb < rows) { store_vec(rel + rb * Geo::kRelExtent + d, pack_bf16x2(c2, c3)); }
    }
}

} // namespace
} // namespace inkling_detail

using inkling_detail::Geo;

void inkling_attention_prologue(const InklingProjections& in, const InklingAttentionWeights& wt,
                                InklingLayerKind kind, const InklingAttentionRows& rows,
                                const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                                InklingPrologueOutputs& out, cudaStream_t stream) {
    inkling_detail::prologue_launch(in, wt, kind, rows, conv, cache, out, stream, nullptr, 0);
}

void inkling_detail::prologue_launch(const InklingProjections& in, const InklingAttentionWeights& wt,
                                     InklingLayerKind kind, const InklingAttentionRows& rows,
                                     const InklingConvStateArgs& conv, const PagedKVBatchLayerView& cache,
                                     InklingPrologueOutputs& out, cudaStream_t stream, std::int32_t* zero,
                                     std::int32_t zero_count) {
    using namespace inkling_detail;
    const char* op       = "inkling_attention_prologue";
    const bool global    = kind == InklingLayerKind::Global;
    const RowsShape rs   = check_rows(rows, op);
    const std::int64_t n = rs.columns;
    check_kv_cache(cache, op);

    PrologueParams p{};
    p.q        = static_cast<const __nv_bfloat16*>(in.q.data);
    p.q_stride = column_stride_elements(in.q, DType::BF16, {Geo::kHeadDim, Geo::kQHeads}, n, 2, op, "q");
    p.r        = static_cast<const __nv_bfloat16*>(in.r.data);
    p.r_stride = column_stride_elements(in.r, DType::BF16, {Geo::kRelDim, Geo::kQHeads}, n, 2, op, "r", 2);
    p.k        = static_cast<const __nv_bfloat16*>(in.k.data);
    p.k_stride = column_stride_elements(in.k, DType::BF16, {1024}, n, 1, op, "k");
    p.v        = static_cast<const __nv_bfloat16*>(in.v.data);
    p.v_stride = column_stride_elements(in.v, DType::BF16, {1024}, n, 1, op, "v");
    if (global) {
        p.k_dyn        = static_cast<const __nv_bfloat16*>(in.k_dyn.data);
        p.k_dyn_stride = column_stride_elements(in.k_dyn, DType::BF16, {4, Geo::kKvHeads}, n, 2, op, "k_dyn");
        p.v_dyn        = static_cast<const __nv_bfloat16*>(in.v_dyn.data);
        p.v_dyn_stride = column_stride_elements(in.v_dyn, DType::BF16, {4, Geo::kKvHeads}, n, 2, op, "v_dyn");
    } else if (in.k_dyn.data != nullptr || in.v_dyn.data != nullptr) {
        fail(op, "a local layer takes no dynamic conv taps");
    }

    require_contiguous(wt.q_scale, DType::BF16, Geo::kQHeads * Geo::kHeadDim, op, "q_scale");
    require_contiguous(wt.r_scale, DType::BF16, Geo::kQHeads * Geo::kRelDim, op, "r_scale");
    require_contiguous(wt.k_conv, DType::BF16, 4 * 1024, op, "k_conv");
    require_contiguous(wt.v_conv, DType::BF16, 4 * 1024, op, "v_conv");
    for (const Tensor* w : {&wt.k_conv, &wt.v_conv}) {
        if (global ? (w->ne[0] != 1024 || w->ne[1] != 4) : (w->ne[0] != 4 || w->ne[1] != 1024)) {
            fail(op, "conv weights are [4, 1024] (local attn/k_conv) or [1024, 4] (global attn/k_conv_static)");
        }
    }
    p.q_scale = static_cast<const __nv_bfloat16*>(wt.q_scale.data);
    p.r_scale = static_cast<const __nv_bfloat16*>(wt.r_scale.data);
    p.k_conv  = static_cast<const __nv_bfloat16*>(wt.k_conv.data);
    p.v_conv  = static_cast<const __nv_bfloat16*>(wt.v_conv.data);

    p.positions     = static_cast<const std::int32_t*>(rows.positions.data);
    p.valid_columns = rs.masked ? static_cast<const std::int32_t*>(rows.valid_columns.data) : nullptr;
    p.table_rows    = static_cast<const std::int32_t*>(rows.kv_table_rows.data);
    p.block_tables  = static_cast<const std::int32_t*>(cache.block_tables.data);
    p.table_stride  = cache.block_tables.ne[0];

    // Conv state: the visit's slots [C, 3, S]; K at k_offset, V at v_offset, 4 channels a lane (8-byte loads).
    const std::int32_t channels_needed = std::max(conv.k_offset, conv.v_offset) + 1024;
    if (conv.states.data == nullptr || conv.states.dtype != DType::BF16 || conv.states.ne[0] < channels_needed ||
        conv.states.ne[1] != 3 || conv.states.nb[0] != 2 || conv.states.nb[1] % 16 != 0 ||
        conv.states.nb[2] % 16 != 0 || reinterpret_cast<std::uintptr_t>(conv.states.data) % 16 != 0 ||
        conv.k_offset < 0 || conv.v_offset < 0 || conv.k_offset % 4 != 0 || conv.v_offset % 4 != 0) {
        fail(op, "conv states are BF16 [C, 3, S] (dim 0 contiguous, 16-byte strides) holding K at k_offset and "
                 "V at v_offset (multiples of 4)");
    }
    require_contiguous(conv.slots, DType::I32, rs.batch, op, "conv slots");
    p.states            = static_cast<__nv_bfloat16*>(conv.states.data);
    p.state_tap_stride  = conv.states.nb[1] / 2;
    p.state_slot_stride = conv.states.nb[2] / 2;
    p.k_offset          = conv.k_offset;
    p.v_offset          = conv.v_offset;
    p.slots             = static_cast<const std::int32_t*>(conv.slots.data);
    p.record            = conv.mode == InklingConvMode::Record;
    p.capture_column    = -1;
    if (p.record) {
        const Tensor& rec = conv.records;
        if (rec.data == nullptr || rec.dtype != DType::BF16 || rec.ne[0] < channels_needed ||
            rec.ne[1] < rs.width || rec.ne[2] < rs.batch || rec.nb[0] != 2 || rec.nb[1] % 8 != 0 ||
            rec.nb[2] % 8 != 0 || reinterpret_cast<std::uintptr_t>(rec.data) % 8 != 0) {
            fail(op, "Record mode needs records BF16 [C, W_max >= W, >= B], dim 0 contiguous, 8-byte strides");
        }
        p.records           = static_cast<__nv_bfloat16*>(rec.data);
        p.record_col_stride = rec.nb[1] / 2;
        p.record_row_stride = rec.nb[2] / 2;
        if (conv.capture_column >= 0) { fail(op, "a rewrite checkpoint capture runs in InPlace mode only"); }
    } else if (conv.capture_column >= 0) {
        if (rs.batch != 1 || conv.capture_column > rs.width || conv.capture_slot < 0 ||
            conv.capture_slot >= conv.states.ne[2]) {
            fail(op, "a capture needs B = 1, capture_column in [0, W] and a capture slot of the pool");
        }
        p.capture_column = conv.capture_column;
        p.capture_slot   = conv.capture_slot;
    }

    p.k_pages = static_cast<std::uint8_t*>(cache.k_pages.data);
    p.v_pages = static_cast<std::uint8_t*>(cache.v_pages.data);

    require_contiguous(out.q, DType::BF16, n * Geo::kQHeads * Geo::kHeadDim, op, "q_out");
    require_contiguous(out.r, DType::BF16, n * Geo::kQHeads * Geo::kRelDim, op, "r_out");
    p.q_out = static_cast<__nv_bfloat16*>(out.q.data);
    p.r_out = static_cast<__nv_bfloat16*>(out.r.data);

    p.width = rs.width;
    p.batch = rs.batch;
    // Decode / verify rows run in one K/V unit; a prompt row is cut into 16-column units (>= 4 columns, so
    // only unit 0 reaches back into the slot).
    p.kv_chunk  = rs.width <= Geo::kMaxSmallColumns ? rs.width : 16;
    p.kv_chunks = (rs.width + p.kv_chunk - 1) / p.kv_chunk;
    p.kv_units  = static_cast<std::int64_t>(rs.batch) * Geo::kKvHeads * 2 * p.kv_chunks;
    p.q_units   = n * Geo::kQHeads;
    p.r_units   = n * (Geo::kQHeads / 2);

    p.zero       = zero_count > 0 ? zero : nullptr;
    p.zero_count = zero_count;

    const std::int64_t warps = p.kv_units + p.q_units + p.r_units;
    const pdl::LaunchConfig launch{
        dim3(static_cast<unsigned>((warps + kPrologueThreads / 32 - 1) / (kPrologueThreads / 32))),
        dim3(kPrologueThreads), 0, stream};
    if (global) {
        CUDA_CHECK(pdl::launch_dependent(launch, prologue_kernel<true>, p));
    } else {
        CUDA_CHECK(pdl::launch_dependent(launch, prologue_kernel<false>, p));
    }
}

void inkling_log_tau(const Tensor& positions, Tensor& tau, cudaStream_t stream) {
    using namespace inkling_detail;
    const std::int64_t n = positions.numel();
    if (positions.dtype != DType::I32 || !positions.is_contiguous() || tau.dtype != DType::FP32 ||
        !tau.is_contiguous() || tau.numel() != n) {
        throw std::invalid_argument("inkling_log_tau: positions contiguous I32, tau contiguous FP32, same count");
    }
    if (n == 0) { return; }
    log_tau_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, stream>>>(
        static_cast<const std::int32_t*>(positions.data), static_cast<float*>(tau.data), n);
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rel_proj_table(const Tensor& rel_proj, Tensor& table, cudaStream_t stream) {
    using namespace inkling_detail;
    require_contiguous(rel_proj, DType::BF16, Geo::kRelDim * Geo::kRelExtent, "inkling_rel_proj_table", "rel_proj");
    require_contiguous(table, DType::BF16, Geo::kRelDim * Geo::kRelTableRows, "inkling_rel_proj_table", "table");
    constexpr int total = Geo::kRelDim * Geo::kRelTableRows;
    rel_table_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(rel_proj.data), static_cast<__nv_bfloat16*>(table.data));
    CUDA_CHECK(cudaGetLastError());
}

void inkling_rel_logits(const Tensor& r_out, const Tensor& rel_table, Tensor& rel, cudaStream_t stream) {
    using namespace inkling_detail;
    const char* op = "inkling_rel_logits";
    if (r_out.dtype != DType::BF16 || !r_out.is_contiguous() || r_out.numel() % (Geo::kRelDim * Geo::kQHeads) != 0) {
        throw std::invalid_argument(std::string(op) + ": r_out is contiguous BF16 [16, 24, N]");
    }
    const std::int64_t rows = r_out.numel() / Geo::kRelDim;
    require_contiguous(rel_table, DType::BF16, Geo::kRelDim * Geo::kRelTableRows, op, "rel_table");
    require_contiguous(rel, DType::BF16, rows * Geo::kRelExtent, op, "rel");
    if (rows == 0) { return; }
    const std::int64_t warps = (rows + 15) / 16;
    rel_logits_kernel<<<static_cast<unsigned>((warps + 3) / 4), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(r_out.data), static_cast<const __nv_bfloat16*>(rel_table.data),
        static_cast<__nv_bfloat16*>(rel.data), rows);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace sinfer::ops
