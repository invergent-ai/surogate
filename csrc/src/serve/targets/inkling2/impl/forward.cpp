// Inkling-2-Mini: the visit schedule (lane e1, DESIGN-target.md 1.2 / 1.4). `Variant::run_visits` walks the
// visits of `visits.h` over the AttnRes stack the residual holds (7 slots x 2304 per column). Per visit:
// attention inputs (AttnRes mix, or for a looped visit the mix blended with the cross-pass message), the
// projections, attention (prologue with the K / V convs, rel-bias attention, gate / sink; NAIVE until lane
// c lands), wo_ud, the attention-output conv, the MLP input mix (+ blend), the MoE (NAIVE until lane d
// lands) and the MLP-output conv, which writes the next partial block in place.

#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"
#include "targets/inkling2/impl/naive/naive.h"

#include "api/ops/inkling_mix.h"
#include "core/device.h"
#include "ops/linear/bf16/bf16_cublaslt.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace sinfer::targets::inkling2::detail {
namespace {

using C                         = TextConfig;
constexpr int D                 = C::hidden;
constexpr std::int64_t kColumn  = C::residual; ///< stack elements per column
constexpr int kTileColumns      = 1024;        ///< column tile of the fp32 blend transients
constexpr int kConvChannels     = ops::InklingMixGeometry::kConvChannels;
using G                         = ops::InklingMixGeometry;

std::uint16_t* bf16_ptr(const Tensor& t) { return static_cast<std::uint16_t*>(t.data); }

/// Stack slot `slot` of the residual from column `column` on.
ops::InklingPlane slot_plane(const Tensor& x, int slot, int column = 0) {
    return {bf16_ptr(x) + static_cast<std::int64_t>(column) * kColumn + static_cast<std::int64_t>(slot) * D,
            DType::BF16, kColumn};
}

/// A packed [width, T] buffer from column `column` on.
ops::InklingPlane packed(const Tensor& t, int column = 0, std::int64_t width = D, std::int64_t offset = 0) {
    const std::size_t element = t.dtype == DType::FP32 ? 4 : 2;
    return {static_cast<std::byte*>(t.data) + (static_cast<std::int64_t>(column) * width + offset) * element,
            t.dtype, width};
}

/// A [2304, T] view of one stack slot as a strided Tensor (feature capture).
Tensor slot_tensor(const Tensor& x, int slot, int columns) {
    Tensor t;
    t.data  = bf16_ptr(x) + static_cast<std::int64_t>(slot) * D;
    t.dtype = DType::BF16;
    t.ne[0] = D;
    t.ne[1] = columns;
    t.ne[2] = t.ne[3] = 1;
    t.nb[0] = 2;
    t.nb[1] = kColumn * 2;
    t.nb[2] = t.nb[3] = t.nb[1] * columns;
    return t;
}

void gemm_bf16(const Weight& w, std::int32_t row_begin, std::int32_t rows, const void* x, std::int32_t tokens,
               void* out, std::int32_t ldc, cudaStream_t s) {
    if (w.qtype != QType::BF16_CTRL || w.qdata == nullptr) {
        throw std::logic_error("inkling2: only the BF16 attention / controller weights are wired (FP8 attention: "
                               "lanes b / c, DESIGN 10.1)");
    }
    const auto* base = static_cast<const std::uint16_t*>(w.qdata) + static_cast<std::int64_t>(row_begin) * w.k;
    ops::detail::bf16_cublaslt_gemm_raw(base, rows, w.k, x, tokens, out, ldc, 0.0F, s);
}

/// One attention / conv call: a prompt segment (B = 1) or the decode / verify block.
struct Group {
    std::int32_t column = 0, width = 0, batch = 0;
    Tensor positions;  ///< I32 [W, B]
    Tensor valid;      ///< I32 [B] or empty
    Tensor table_rows; ///< I32 [B]
    Tensor slots;      ///< I32 [B]
    const family::VisitSegment* segment = nullptr;
    bool record = false;

    [[nodiscard]] ops::InklingConvRows rows() const { return {width, batch, positions, valid, slots}; }
};

struct Round {
    const RuntimeModelView& model;
    family::VisitRound& r;
    const InklingModel& im;
    const DerivedQueries& q;
    WorkspaceArena& work;
    cudaStream_t s;
    std::int32_t T;
    std::vector<Group> groups;
    ops::InklingConvRecords records{};
    ops::InklingConvPool pool{};
};

ops::InklingConvStateView state_view(const Round& rd, int visit) {
    const Tensor& conv = rd.r.state->conv.at(static_cast<std::size_t>(visit));
    return {conv.data, static_cast<std::int64_t>(kConvChannels) * 3, kConvChannels};
}

/// The fp32 AttnRes mix + message blend of a looped visit, column-tiled (DESIGN-target.md 3.4).
/// streams = {0, 1, 2} (attention q / k / v inputs) or {3} (MLP input); `out` BF16 [D, T, S].
void blended_inputs(Round& rd, const VisitInfo& vi, const load::InklingLayerWeights& L,
                    const ops::InklingAttnResSources& sources, const Tensor& dest_queries, int first_stream,
                    int streams, const Tensor& norm_weight, Tensor& out) {
    const int core_index = vi.layer - C::core_start;
    const Tensor& messages = rd.q.message.at(static_cast<std::size_t>(core_index));
    const int dq = dest_queries.ne[1];
    for (int c0 = 0; c0 < rd.T; c0 += kTileColumns) {
        const int n = std::min(kTileColumns, rd.T - c0);
        auto tile   = rd.work.scope();
        ops::InklingAttnResSources src = sources;
        const auto shift = [&](ops::InklingPlane& p) {
            if (p.data == nullptr) { return; }
            const std::size_t element = p.dtype == DType::FP32 ? 4 : 2;
            p.data = static_cast<std::byte*>(p.data) + static_cast<std::int64_t>(c0) * p.column_stride * element;
        };
        for (int i = 0; i < src.stack_count; ++i) { shift(src.stack[i]); }
        shift(src.last);
        shift(src.addend);
        Tensor dest = rd.work.alloc(DType::FP32, {D, n, dq});
        ops::inkling_attn_res(src, n, dest_queries, nullptr, 0.0F, dest, rd.s);
        // The message: the bank's two entries (the previous pass's B6 and P12) with this layer's queries.
        ops::InklingAttnResSources bank;
        bank.stack_count = 1;
        bank.stack[0]    = slot_plane(rd.r.x, vi.bank[0], c0);
        bank.last        = slot_plane(rd.r.x, vi.bank[1], c0);
        Tensor msg_queries(static_cast<float*>(messages.data) + static_cast<std::int64_t>(first_stream) * D,
                           DType::FP32, {D, streams});
        Tensor msg = rd.work.alloc(DType::FP32, {D, n, streams});
        ops::inkling_attn_res(bank, n, msg_queries, nullptr, 0.0F, msg, rd.s);
        Tensor pair = rd.work.alloc(DType::BF16, {2 * D, n, streams});
        ops::inkling_pair_rms_norm(dest, msg, pair, rd.s);
        Tensor h  = rd.work.alloc(DType::BF16, {C::gate_rank, n, streams});
        Tensor gf = rd.work.alloc(DType::FP32, {2 * D, n, streams});
        for (int j = 0; j < streams; ++j) {
            const int stream_index = first_stream + j;
            const auto* down = bf16_ptr(L.loop_down) + static_cast<std::int64_t>(stream_index) * C::gate_rank * 2 * D;
            const auto* gfw  = bf16_ptr(L.loop_gain_forget) + static_cast<std::int64_t>(stream_index) * 2 * D * C::gate_rank;
            ops::detail::bf16_cublaslt_gemm_raw(down, C::gate_rank, 2 * D,
                                                bf16_ptr(pair) + static_cast<std::int64_t>(j) * n * 2 * D, n,
                                                bf16_ptr(h) + static_cast<std::int64_t>(j) * n * C::gate_rank,
                                                C::gate_rank, 0.0F, rd.s);
            ops::inkling_gemm_bf16_f32(gfw, 2 * D, C::gate_rank,
                                       bf16_ptr(h) + static_cast<std::int64_t>(j) * n * C::gate_rank, n,
                                       static_cast<float*>(gf.data) + static_cast<std::int64_t>(j) * n * 2 * D,
                                       2 * D, rd.s);
        }
        Tensor blended = rd.work.alloc(DType::BF16, {D, n, streams});
        ops::inkling_blend_rms_norm(dest, msg, gf, norm_weight, C::rms_epsilon, blended, rd.s);
        for (int j = 0; j < streams; ++j) {
            CUDA_CHECK(cudaMemcpyAsync(bf16_ptr(out) + (static_cast<std::int64_t>(j) * rd.T + c0) * D,
                                       bf16_ptr(blended) + static_cast<std::int64_t>(j) * n * D,
                                       static_cast<std::size_t>(n) * D * 2, cudaMemcpyDeviceToDevice, rd.s));
        }
    }
}

Tensor vector_as_rows(const Tensor& t, int rows) { return Tensor(t.data, t.dtype, {D, rows}); }

void attention_inputs(Round& rd, int v, const VisitInfo& vi, const load::InklingLayerWeights& L, Tensor& attn_in) {
    if (vi.stack_count == 0 && vi.partial == 0) { // visit 0: attn_norm of the embedding, no mix
        ops::inkling_rms_norm(slot_plane(rd.r.x, 0), rd.T, L.attn_input_norm, C::rms_epsilon, attn_in, rd.s);
        return;
    }
    ops::InklingAttnResSources src;
    src.stack_count = vi.stack_count;
    for (int i = 0; i < vi.stack_count; ++i) { src.stack[i] = slot_plane(rd.r.x, vi.stack[i]); }
    src.last = slot_plane(rd.r.x, vi.partial);
    const Tensor& queries = rd.q.attn.at(static_cast<std::size_t>(vi.layer));
    if (!vi.looped) {
        ops::inkling_attn_res(src, rd.T, queries, &L.attn_input_norm, C::rms_epsilon, attn_in, rd.s);
        return;
    }
    (void)v;
    blended_inputs(rd, vi, L, src, queries, 0, 3, L.attn_input_norm, attn_in);
}

void mlp_input(Round& rd, const VisitInfo& vi, const load::InklingLayerWeights& L, const Tensor& attn_out,
               Tensor& mlp_in) {
    ops::InklingAttnResSources src;
    src.stack_count = vi.stack_count;
    for (int i = 0; i < vi.stack_count; ++i) { src.stack[i] = slot_plane(rd.r.x, vi.stack[i]); }
    if (vi.first) {
        src.stack[src.stack_count++] = slot_plane(rd.r.x, vi.partial);
        src.last                     = packed(attn_out);
    } else {
        src.last   = slot_plane(rd.r.x, vi.partial);
        src.addend = packed(attn_out);
    }
    const Tensor& queries = rd.q.mlp.at(static_cast<std::size_t>(vi.layer));
    const Tensor norm     = vector_as_rows(L.moe_input_norm, 1);
    if (!vi.looped) {
        ops::inkling_attn_res(src, rd.T, queries, &norm, C::rms_epsilon, mlp_in, rd.s);
        return;
    }
    blended_inputs(rd, vi, L, src, queries, 3, 1, norm, mlp_in);
}

/// The visit's KV plane for a group: the round's batch view, or the single sequence's own view (row 0).
struct PlaneView {
    const std::int32_t* block_tables = nullptr;
    std::int64_t table_stride        = 0;
    void* k_pages                    = nullptr;
    void* v_pages                    = nullptr;
    bool fp8                         = true;
};

PlaneView plane_view(const Round& rd, int visit) {
    PlaneView out;
    DType dtype = DType::BF16;
    if (rd.r.sequence_kv.valid()) {
        const PagedKVLayerView view = rd.r.sequence_kv.layer_view(static_cast<std::uint32_t>(visit));
        out.block_tables = static_cast<const std::int32_t*>(view.block_table.data);
        out.table_stride = view.block_table.ne[0];
        out.k_pages      = view.k_pages.data;
        out.v_pages      = view.v_pages.data;
        dtype            = view.dtype;
    } else {
        const PagedKVBatchLayerView view = rd.r.text_kv->batch_layer_view(static_cast<std::uint32_t>(visit));
        out.block_tables = static_cast<const std::int32_t*>(view.block_tables.data);
        out.table_stride = view.block_tables.ne[0];
        out.k_pages      = view.k_pages.data;
        out.v_pages      = view.v_pages.data;
        dtype            = view.dtype;
    }
    if (dtype != DType::FP8_E4M3FN && dtype != DType::BF16) {
        throw std::logic_error("inkling2: KV planes must be FP8 e4m3 (unit scale) or BF16");
    }
    out.fp8 = dtype == DType::FP8_E4M3FN;
    return out;
}

naive::AttentionLayer attention_layer(const load::InklingLayerWeights& L) {
    naive::AttentionLayer out;
    out.global   = L.global;
    out.q_scale  = L.q_scale.data;
    out.r_scale  = L.r_scale.data;
    out.rel_proj = L.rel_proj.data;
    out.k_conv   = L.k_conv.data;
    out.v_conv   = L.v_conv.data;
    out.k_static = L.k_conv_static.data;
    out.v_static = L.v_conv_static.data;
    return out;
}

naive::ExpertBankView bank_view(const load::ExpertBank& b) {
    if (!b.gate_up_interleaved) {
        throw std::logic_error("inkling2: the naive MoE reads the interleaved w13 row order only");
    }
    naive::ExpertBankView out;
    out.nvfp4                   = b.format == load::ExpertFormat::Nvfp4;
    out.experts                 = b.experts;
    out.rows_per_expert         = b.rows_per_expert;
    out.columns                 = b.columns;
    out.codes                   = b.codes;
    out.scales                  = b.scales;
    out.amax                    = b.amax;
    out.outer_scale_denominator = b.outer_scale_denominator_value;
    return out;
}

naive::MoeLayer moe_layer(const load::InklingLayerWeights& L) {
    naive::MoeLayer out;
    out.router      = L.router.qdata;
    out.router_bias = L.router_bias;
    out.w13         = bank_view(L.w13);
    out.w2          = bank_view(L.w2);
    return out;
}

void run_visit(Round& rd, int v) {
    const VisitInfo& vi                 = kVisits[static_cast<std::size_t>(v)];
    const load::InklingLayerWeights& L  = *rd.model.full_layers.at(static_cast<std::size_t>(v)).projection.layer;
    const int T                         = rd.T;
    cudaStream_t s                      = rd.s;
    auto scope                          = rd.work.scope();
    if (vi.feature >= 0 && rd.r.features.active()) {
        rd.r.features.capture(rd.r.features.context, vi.feature, slot_tensor(rd.r.x, vi.partial, T), s);
    }
    // 1. attention inputs
    const int Q    = (vi.global || vi.looped) ? 3 : 1;
    Tensor attn_in = rd.work.alloc(DType::BF16, {D, T, Q});
    attention_inputs(rd, v, vi, L, attn_in);
    // 2. projections: one GEMM over the fused rows, or three over their row ranges (q-side / K / V)
    Tensor qkvr = rd.work.alloc(DType::BF16, {C::qkvr_rows, T});
    const auto in = [&](int stream) { return bf16_ptr(attn_in) + static_cast<std::int64_t>(stream) * T * D; };
    if (Q == 1) {
        gemm_bf16(L.qkvr, 0, C::qkvr_rows, in(0), T, qkvr.data, C::qkvr_rows, s);
    } else {
        gemm_bf16(L.qkvr, 0, C::qkvr_query_end, in(0), T, qkvr.data, C::qkvr_rows, s);
        gemm_bf16(L.qkvr, C::qkvr_k_begin, C::qkvr_v_begin - C::qkvr_k_begin, in(1), T,
                  bf16_ptr(qkvr) + C::qkvr_k_begin, C::qkvr_rows, s);
        gemm_bf16(L.qkvr, C::qkvr_v_begin, C::qkvr_rows - C::qkvr_v_begin, in(2), T,
                  bf16_ptr(qkvr) + C::qkvr_v_begin, C::qkvr_rows, s);
    }
    Tensor k_dyn, v_dyn;
    if (vi.global) {
        k_dyn = rd.work.alloc(DType::BF16, {32, T});
        v_dyn = rd.work.alloc(DType::BF16, {32, T});
        gemm_bf16(*L.k_conv_proj, 0, 32, in(1), T, k_dyn.data, 32, s);
        gemm_bf16(*L.v_conv_proj, 0, 32, in(2), T, v_dyn.data, 32, s);
    }
    // 3. attention per group (NAIVE)
    Tensor q_out = rd.work.alloc(DType::BF16, {C::attn_width, T});
    Tensor r_out = rd.work.alloc(DType::BF16, {C::query_heads * C::d_rel, T});
    Tensor gated = rd.work.alloc(DType::BF16, {C::attn_width, T});
    const ops::InklingConvStateView state = state_view(rd, v);
    const PlaneView plane                 = plane_view(rd, v);
    const auto kv_input = [&](int column, int row_base) {
        return ops::InklingPlane{bf16_ptr(qkvr) + static_cast<std::int64_t>(column) * C::qkvr_rows + row_base,
                                 DType::BF16, C::qkvr_rows};
    };
    for (const Group& g : rd.groups) {
        const auto rows = g.rows();
        const family::VisitSegment* seg = g.segment;
        if (seg != nullptr && seg->checkpoint_slot >= 0) {
            ops::inkling_conv_capture_window(kv_input(g.column, C::qkvr_k_begin), 1024, rows, state, G::kStreamK,
                                             seg->checkpoint_column, seg->checkpoint_slot, s);
            ops::inkling_conv_capture_window(kv_input(g.column, C::qkvr_v_begin), 1024, rows, state, G::kStreamV,
                                             seg->checkpoint_column, seg->checkpoint_slot, s);
        }
        naive::AttentionCall call;
        call.qkvr         = bf16_ptr(qkvr) + static_cast<std::int64_t>(g.column) * C::qkvr_rows;
        call.qkvr_stride  = C::qkvr_rows;
        call.width        = g.width;
        call.batch        = g.batch;
        call.positions    = static_cast<const std::int32_t*>(g.positions.data);
        call.valid        = static_cast<const std::int32_t*>(g.valid.data);
        call.table_rows   = static_cast<const std::int32_t*>(g.table_rows.data);
        call.block_tables = plane.block_tables;
        call.table_stride = plane.table_stride;
        call.k_pages      = plane.k_pages;
        call.v_pages      = plane.v_pages;
        call.fp8          = plane.fp8;
        call.layer        = attention_layer(L);
        if (vi.global) {
            call.k_dyn      = bf16_ptr(k_dyn) + static_cast<std::int64_t>(g.column) * 32;
            call.v_dyn      = bf16_ptr(v_dyn) + static_cast<std::int64_t>(g.column) * 32;
            call.dyn_stride = 32;
        }
        call.state       = state.base;
        call.slot_stride = state.slot_stride;
        call.tap_stride  = state.tap_stride;
        call.slots       = static_cast<const std::int32_t*>(g.slots.data);
        call.q_out       = bf16_ptr(q_out) + static_cast<std::int64_t>(g.column) * C::attn_width;
        call.r_out       = bf16_ptr(r_out) + static_cast<std::int64_t>(g.column) * C::query_heads * C::d_rel;
        call.out         = bf16_ptr(gated) + static_cast<std::int64_t>(g.column) * C::attn_width;
        naive::attention(call, s);
        if (g.record) {
            ops::inkling_conv_record(kv_input(g.column, C::qkvr_k_begin), 1024, g.width, g.batch, v, G::kStreamK,
                                     rd.records, s);
            ops::inkling_conv_record(kv_input(g.column, C::qkvr_v_begin), 1024, g.width, g.batch, v, G::kStreamV,
                                     rd.records, s);
        }
        ops::inkling_conv_update_window(kv_input(g.column, C::qkvr_k_begin), 1024, rows, state, G::kStreamK, s);
        ops::inkling_conv_update_window(kv_input(g.column, C::qkvr_v_begin), 1024, rows, state, G::kStreamV, s);
        if (seg != nullptr && seg->capture_slot >= 0) {
            ops::inkling_conv_copy_slot(state, G::kStreamK, 2048, seg->state_slot, seg->capture_slot, s);
        }
    }
    // 4. wo_ud
    Tensor attn_partial = rd.work.alloc(DType::BF16, {D, T});
    gemm_bf16(L.output, 0, D, gated.data, T, attn_partial.data, D, s);
    // 5. attention-output conv
    Tensor attn_out = rd.work.alloc(DType::BF16, {D, T});
    for (const Group& g : rd.groups) {
        const family::VisitSegment* seg = g.segment;
        if (g.record) {
            ops::inkling_conv_record(packed(attn_partial, g.column), D, g.width, g.batch, v, G::kStreamAttn, rd.records, s);
        }
        ops::inkling_short_conv(packed(attn_partial, g.column), D, L.attn_out_conv, g.rows(), state, G::kStreamAttn,
                                nullptr, nullptr, packed(attn_out, g.column), s,
                                seg != nullptr ? seg->checkpoint_column : -1, seg != nullptr ? seg->checkpoint_slot : -1);
        if (seg != nullptr && seg->capture_slot >= 0) {
            ops::inkling_conv_copy_slot(state, G::kStreamAttn, D, seg->state_slot, seg->capture_slot, s);
        }
    }
    // 6. MLP input
    Tensor mlp_in = rd.work.alloc(DType::BF16, {D, T, 1});
    mlp_input(rd, vi, L, attn_out, mlp_in);
    // 7. MoE (NAIVE), routed experts captured per VISIT
    Tensor moe_out = rd.work.alloc(DType::BF16, {D, T});
    naive::MoeCall moe;
    moe.x      = mlp_in.data;
    moe.tokens = T;
    moe.layer  = moe_layer(L);
    moe.out    = moe_out.data;
    if (rd.r.routing.ids != nullptr && T <= rd.r.routing.columns) {
        moe.capture = rd.r.routing.ids + static_cast<std::int64_t>(v) * rd.r.routing.columns * rd.r.routing.top_k;
    }
    naive::moe(moe, rd.work, s);
    // 8. MLP-output conv: (residual + attn_out) + conv(moe_out) into the next partial (in place when not first)
    for (const Group& g : rd.groups) {
        const family::VisitSegment* seg = g.segment;
        if (g.record) {
            ops::inkling_conv_record(packed(moe_out, g.column), D, g.width, g.batch, v, G::kStreamMlp, rd.records, s);
        }
        const ops::InklingPlane residual = slot_plane(rd.r.x, vi.partial, g.column);
        const ops::InklingPlane addend   = packed(attn_out, g.column);
        ops::inkling_short_conv(packed(moe_out, g.column), D, L.mlp_out_conv, g.rows(), state, G::kStreamMlp,
                                vi.first ? nullptr : &residual, &addend, slot_plane(rd.r.x, vi.next_partial, g.column), s,
                                seg != nullptr ? seg->checkpoint_column : -1, seg != nullptr ? seg->checkpoint_slot : -1);
        if (seg != nullptr && seg->capture_slot >= 0) {
            ops::inkling_conv_copy_slot(state, G::kStreamMlp, D, seg->state_slot, seg->capture_slot, s);
        }
    }
}

const InklingModel& model_of(const RuntimeModelView& model) {
    if (model.full_layers.empty() || model.full_layers.front().projection.model == nullptr) {
        throw std::logic_error("inkling2: the model view carries no visits");
    }
    return *model.full_layers.front().projection.model;
}

} // namespace

void Variant::run_visits(const ModelView& model, family::VisitRound& r) {
    if (r.work == nullptr || r.state == nullptr || r.x.data == nullptr || r.positions.data == nullptr) {
        throw std::logic_error("inkling2: incomplete visit round");
    }
    if (r.x.ne[0] != C::residual || r.state->conv.size() != static_cast<std::size_t>(C::layers)) {
        throw std::logic_error("inkling2: the residual or the conv-state pool does not have the visit layout");
    }
    if (!r.sequence_kv.valid() && r.text_kv == nullptr) {
        throw std::logic_error("inkling2: a visit round needs a KV view");
    }
    const InklingModel& im = model_of(model);
    Round rd{model, r, im, *im.queries, *r.work, r.stream, r.x.ne[1], {}, {}, {}};
    auto round_scope = rd.work.scope();
    // Groups: every segment (B = 1) and the decode / verify block.
    for (const family::VisitSegment& seg : r.segments) {
        Group g;
        g.column    = seg.column;
        g.width     = seg.count;
        g.batch     = 1;
        g.positions = Tensor(static_cast<std::int32_t*>(r.positions.data) + seg.column, DType::I32, {seg.count, 1});
        Tensor ints = rd.work.alloc(DType::I32, {2});
        if (seg.table_row.data != nullptr && !r.sequence_kv.valid()) {
            g.table_rows = seg.table_row.view({1});
        } else {
            if (!r.sequence_kv.valid() && seg.kv_table_row < 0) {
                throw std::logic_error("inkling2: a segment without a KV row");
            }
            naive::fill_i32(static_cast<std::int32_t*>(ints.data), 1, r.sequence_kv.valid() ? 0 : seg.kv_table_row, rd.s);
            g.table_rows = Tensor(ints.data, DType::I32, {1});
        }
        naive::fill_i32(static_cast<std::int32_t*>(ints.data) + 1, 1, seg.state_slot, rd.s);
        g.slots      = Tensor(static_cast<std::int32_t*>(ints.data) + 1, DType::I32, {1});
        g.segment    = &seg;
        rd.groups.push_back(g);
    }
    if (r.block != nullptr && r.block->batch > 0) {
        const family::VisitBlock& b = *r.block;
        Group g;
        g.column     = b.column;
        g.width      = b.width;
        g.batch      = b.batch;
        g.positions  = b.positions;
        g.valid      = b.valid_columns;
        g.table_rows = b.kv_table_rows;
        g.slots      = b.state_slots;
        g.record     = b.mode == family::VisitStateMode::Record;
        rd.groups.push_back(g);
    }
    if (rd.groups.empty()) { throw std::logic_error("inkling2: a visit round with no columns"); }
    const auto& conv = r.state->conv;
    rd.pool = ops::InklingConvPool{conv[0].data,
                                   (static_cast<const std::byte*>(conv[1].data) - static_cast<const std::byte*>(conv[0].data)) / 2,
                                   static_cast<std::int64_t>(kConvChannels) * 3, C::layers};
    for (const Group& g : rd.groups) {
        if (!g.record) { continue; }
        if (r.records.old == nullptr || g.width > r.records.width_max || g.batch > r.records.batch_max) {
            throw std::logic_error("inkling2: a verify round without room in the conv records");
        }
        rd.records = ops::InklingConvRecords{r.records.old, r.records.inputs, r.records.visits, r.records.width_max,
                                             r.records.batch_max};
        ops::inkling_conv_backup(rd.pool, g.slots, g.batch, rd.records, rd.s);
    }
    for (int v = r.first_visit; v < r.last_visit; ++v) { run_visit(rd, v); }
}

std::size_t Variant::visit_workspace_capacity_bytes(const family::TextGeometry&, family::TextPhase, std::int32_t batch_size,
                                                    std::int32_t max_width, std::int32_t max_columns, std::uint32_t) {
    const auto up = [](std::size_t bytes) { return (bytes + 255U) & ~std::size_t{255}; };
    const std::size_t T    = static_cast<std::size_t>(std::max<std::int32_t>(max_columns, std::max(batch_size, 1) * std::max(max_width, 1)));
    const std::size_t tile = std::min<std::size_t>(T, kTileColumns);
    std::size_t bytes = 0;
    bytes += up(64 * 8);                                     // segment ints
    bytes += up(T * D * 3 * 2);                              // attn_in
    bytes += up(T * C::qkvr_rows * 2);                       // qkvr
    bytes += 2 * up(T * 32 * 2);                             // dynamic taps
    bytes += up(T * C::attn_width * 2) * 2 + up(T * 384 * 2); // q_out, gated, r_out
    bytes += up(T * D * 2) * 4;                              // attn_partial, attn_out, mlp_in, moe_out
    // the looped blend tile: dest + msg fp32, pair, h, gf fp32, blended
    bytes += up(tile * D * 3 * 4) + up(tile * D * 3 * 4) + up(tile * 2 * D * 3 * 2) + up(tile * C::gate_rank * 3 * 2) +
             up(tile * 2 * D * 3 * 4) + up(tile * D * 3 * 2);
    bytes += naive::moe_workspace_bytes(static_cast<std::int32_t>(T));
    return bytes + (1U << 20);
}

} // namespace sinfer::targets::inkling2::detail
