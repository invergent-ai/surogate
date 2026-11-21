// Inkling-2-Mini: the visit schedule (lane e1, DESIGN-target.md 1.2 / 1.4). `Variant::run_visits` walks the
// visits of `visits.h` over the AttnRes stack the residual holds (7 slots x 2304 per column). Per visit:
// attention inputs (AttnRes mix, or for a looped visit the mix blended with the cross-pass message), the
// projections, attention (prologue with the K / V convs, rel-bias attention, gate / sink; NAIVE until lane
// c lands), wo_ud, the attention-output conv, the MLP input mix (+ blend), the MoE (NAIVE until lane d
// lands) and the MLP-output conv, which writes the next partial block in place.

#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"
#include "targets/inkling2/impl/naive/naive.h"

#include "api/ops/inkling_attention.h"
#include "api/ops/inkling_mix.h"
#include "api/ops/inkling_moe.h"
#include "api/ops/lora_store.h"
#include "family/impl/lora_bind_inkling.h"
#include "targets/inkling2/impl/lora_attention.h"
#include "core/device.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
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

/// SUROGATE_SERVE_INKLING_NAIVE=1: the naive attention / MoE kernels instead of lanes c / d's ops (A/B checks only).
bool use_naive() {
    static const bool naive = std::getenv("SUROGATE_SERVE_INKLING_NAIVE") != nullptr;
    return naive;
}

/// A view whose leading dims are contiguous and whose LAST dim (the columns) steps by `column_elems` elements.
Tensor column_view(void* data, DType dtype, std::initializer_list<std::int32_t> dims, std::int64_t column_elems) {
    Tensor t(data, dtype, dims);
    const std::int64_t element = dtype == DType::FP32 ? 4 : 2;
    const int last             = static_cast<int>(dims.size()) - 1;
    t.nb[last]                 = column_elems * element;
    for (int i = last + 1; i < 4; ++i) { t.nb[i] = t.nb[i - 1] * t.ne[i - 1]; }
    return t;
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

/// The FP32 scratch a split shape of the invariant GEMM needs (none for one chunk), from the round's arena.
void* gemm_scratch(WorkspaceArena& work, std::size_t bytes) {
    if (bytes == 0) { return nullptr; }
    return work.alloc(DType::FP32, {static_cast<std::int32_t>((bytes + 3) / 4)}).data;
}

/// Every projection runs the batch- and path-invariant GEMM (agent-EN 10:2x): a column's bits depend on the weights
/// and that column only, never on the round's width or on the other columns.
void gemm_bf16(WorkspaceArena& work, const Weight& w, std::int32_t row_begin, std::int32_t rows, const void* x,
               std::int32_t tokens, void* out, std::int32_t ldc, cudaStream_t s) {
    if (w.qtype != QType::BF16_CTRL || w.qdata == nullptr) {
        throw std::logic_error("inkling2: only the BF16 attention / controller weights are wired (FP8 attention: "
                               "lanes b / c, DESIGN 10.1)");
    }
    const auto* base   = static_cast<const std::uint16_t*>(w.qdata) + static_cast<std::int64_t>(row_begin) * w.k;
    auto scope         = work.scope();
    const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(rows, w.k, tokens, 1);
    ops::inkling_gemm_inv(base, rows, w.k, x, tokens, out, DType::BF16, ldc, {}, gemm_scratch(work, bytes), bytes, s);
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
    /// The segments' ints went in one host copy this round: everything the step wrote before is complete.
    bool seg_barrier = false;
};

/// S10 attention LoRA: the round's adapter selection on the bank table registered under `key` (a layer's qkvr or
/// wo_ud weight; package.cpp), or no table (no adapters, a base round, no published round).
struct AttnLoraSelection {
    const ops::LoraBank* table     = nullptr;
    const std::int32_t* token_slot = nullptr;
    const std::int32_t* uniform    = nullptr;
    std::int32_t rank              = 0;
};

AttnLoraSelection attn_lora_selection(const void* key, std::int32_t tokens) {
    AttnLoraSelection sel;
    if (!ops::lora_active()) { return sel; }
    const ops::LoraRound& round = ops::lora_current_round();
    if (!round.valid()) { return sel; }
    const ops::LoraStore& store = ops::lora_store_for_current_device();
    sel.table = store.bank_table(key);
    if (sel.table == nullptr) { return sel; }
    if (round.slots != nullptr) {
        if (round.slots->numel() < static_cast<std::int64_t>(tokens)) {
            throw std::logic_error("inkling2: the round's adapter slots are narrower than the round");
        }
        sel.token_slot = static_cast<const std::int32_t*>(round.slots->data);
    } else {
        sel.uniform = round.uniform_cell;
    }
    sel.rank = store.max_rank();
    return sel;
}

void apply_attn_lora(WorkspaceArena& work, const AttnLoraSelection& sel, const ops::InklingLoraSlice* slices,
                     int count, std::int32_t tokens, cudaStream_t s) {
    const std::size_t bytes = ops::inkling_lora_delta_workspace_bytes(count, sel.rank, tokens);
    Tensor low = work.alloc(DType::FP32, {static_cast<std::int32_t>(bytes / sizeof(float))});
    ops::inkling_lora_delta(sel.table, slices, count, sel.rank, tokens, sel.token_slot, sel.uniform, low.data, bytes, s);
}

/// Lane c's key-group cut, passed explicitly (engine and trainer must pass the same value; lane h froze P3).
constexpr ops::InklingKeyGroups kKeyGroups = ops::InklingKeyGroups::P3;

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
        // The message: the bank's two entries (the previous pass's B6 and P12) with this layer's queries.
        ops::InklingAttnResSources bank;
        bank.stack_count = 1;
        bank.stack[0]    = slot_plane(rd.r.x, vi.bank[0], c0);
        bank.last        = slot_plane(rd.r.x, vi.bank[1], c0);
        Tensor msg_queries(static_cast<float*>(messages.data) + static_cast<std::int64_t>(first_stream) * D,
                           DType::FP32, {D, streams});
        Tensor msg  = rd.work.alloc(DType::FP32, {D, n, streams});
        Tensor pair = rd.work.alloc(DType::BF16, {2 * D, n, streams});
        // dest mix + message mix + pair norm: one launch, bit-identical to the three ops (invariance test I3).
        ops::inkling_blend_inputs(src, dest_queries, bank, msg_queries, n, dest, msg, pair, rd.s);
        const auto* down = bf16_ptr(L.loop_down) + static_cast<std::int64_t>(first_stream) * C::gate_rank * 2 * D;
        const auto* gfw  = bf16_ptr(L.loop_gain_forget) + static_cast<std::int64_t>(first_stream) * 2 * D * C::gate_rank;
        const ops::InklingGemmBatch down_batch{streams, static_cast<std::int64_t>(C::gate_rank) * 2 * D,
                                               static_cast<std::int64_t>(n) * 2 * D,
                                               static_cast<std::int64_t>(n) * C::gate_rank};
        // Decode / verify widths: the `down` partials, then reduce + gain/forget + blend norm in ONE cluster launch
        // (bit-identical to the three ops; invariance test I6), straight into `out` (one tile).
        if (n == rd.T && ops::inkling_controller_tail_supported(n)) {
            auto gemm_scope         = rd.work.scope();
            const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(C::gate_rank, 2 * D, n, streams);
            auto* partials          = static_cast<float*>(gemm_scratch(rd.work, bytes));
            ops::inkling_gemm_inv_partials(down, C::gate_rank, 2 * D, pair.data, n, down_batch, partials, bytes, rd.s);
            ops::inkling_controller_tail(partials, gfw, dest, msg, norm_weight, C::rms_epsilon, out, rd.s);
            continue;
        }
        Tensor h  = rd.work.alloc(DType::BF16, {C::gate_rank, n, streams});
        Tensor gf = rd.work.alloc(DType::FP32, {2 * D, n, streams});
        // The controllers of the `streams` streams, one invariant launch each (down: a split fixed by its 64 x 4608
        // shape; gain / forget: FP32 out).
        {
            const ops::InklingGemmBatch gf_batch{streams, static_cast<std::int64_t>(2) * D * C::gate_rank,
                                                 static_cast<std::int64_t>(n) * C::gate_rank,
                                                 static_cast<std::int64_t>(n) * 2 * D};
            auto gemm_scope         = rd.work.scope();
            const std::size_t bytes = ops::inkling_gemm_inv_workspace_bytes(C::gate_rank, 2 * D, n, streams);
            ops::inkling_gemm_inv(down, C::gate_rank, 2 * D, pair.data, n, h.data, DType::BF16, C::gate_rank,
                                  down_batch, gemm_scratch(rd.work, bytes), bytes, rd.s);
            ops::inkling_gemm_inv(gfw, 2 * D, C::gate_rank, h.data, n, gf.data, DType::FP32, 2 * D, gf_batch, nullptr,
                                  0, rd.s);
        }
        // One tile (every decode / verify round and every prompt round <= kTileColumns): straight into `out`.
        if (n == rd.T) {
            ops::inkling_blend_rms_norm(dest, msg, gf, norm_weight, C::rms_epsilon, out, rd.s);
            continue;
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

/// The MLP input mix's sources (stack..., partial, attn_out as `last` on a block's first visit, else as `addend`).
ops::InklingAttnResSources mlp_sources(const Round& rd, const VisitInfo& vi, const Tensor& attn_out) {
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
    return src;
}

/// SUROGATE_SERVE_INKLING_NO_FUSE=1: the unfused kernels of the (d) fusions (A/B; same bits either way).
bool fusions_off() {
    static const bool off = [] {
        const char* v = std::getenv("SUROGATE_SERVE_INKLING_NO_FUSE");
        return v != nullptr && std::string_view(v) == "1";
    }();
    return off;
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

/// SUROGATE_SERVE_INKLING_DUMP=<dir> (parity work only; synchronises the stream): the first eager forward of the
/// process writes, per visit, its attention inputs, qkvr, gated attention, attn_out, mlp_in, moe_out and the
/// partial it wrote, as `<dir>/v<visit>_<name>.bin` = header {u32 'INK1', u32 dtype (0 bf16, 1 fp32), i32 rows,
/// i32 columns, i32 planes} + the values, column-major [rows, columns, planes] (lane a's `record_visits`, T7).
struct Dump {
    const char* dir = nullptr;
    bool active     = false;
};

Dump& dump_state() {
    static Dump d = [] {
        Dump out;
        out.dir = std::getenv("SUROGATE_SERVE_INKLING_DUMP");
        return out;
    }();
    return d;
}

void dump(const char* name, int visit, const void* data, DType dtype, std::int64_t column_stride, int rows,
          int columns, int planes, cudaStream_t s) {
    Dump& d = dump_state();
    if (!d.active) { return; }
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(s, &capturing));
    if (capturing != cudaStreamCaptureStatusNone) { return; }
    const std::size_t element = dtype == DType::FP32 ? 4 : 2;
    std::vector<std::byte> host(static_cast<std::size_t>(rows) * columns * planes * element);
    CUDA_CHECK(cudaStreamSynchronize(s));
    for (int p = 0; p < planes; ++p) {
        for (int c = 0; c < columns; ++c) {
            const auto* src = static_cast<const std::byte*>(data) +
                              (static_cast<std::int64_t>(p) * columns * column_stride + static_cast<std::int64_t>(c) * column_stride) * element;
            CUDA_CHECK(cudaMemcpy(host.data() + (static_cast<std::size_t>(p) * columns + c) * rows * element, src,
                                  static_cast<std::size_t>(rows) * element, cudaMemcpyDeviceToHost));
        }
    }
    const std::string path = std::string(d.dir) + "/v" + std::to_string(visit) + "_" + name + ".bin";
    if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
        const std::int32_t header[5] = {0x314b4e49, dtype == DType::FP32 ? 1 : 0, rows, columns, planes};
        std::fwrite(header, sizeof(header), 1, f);
        std::fwrite(host.data(), 1, host.size(), f);
        std::fclose(f);
    }
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

ops::InklingAttentionWeights attention_weights(const Round& rd, const load::InklingLayerWeights& L) {
    ops::InklingAttentionWeights w;
    w.q_scale   = L.q_scale;
    w.r_scale   = L.r_scale;
    w.k_conv    = L.global ? L.k_conv_static : L.k_conv;
    w.v_conv    = L.global ? L.v_conv_static : L.v_conv;
    w.rel_table = rd.q.rel_table.at(static_cast<std::size_t>(L.layer));
    return w;
}

/// The visit's KV plane as lane c's batch view (a single-sequence prefill's one block table as a 1-row matrix).
PagedKVBatchLayerView kv_batch_view(const Round& rd, int visit) {
    if (!rd.r.sequence_kv.valid()) { return rd.r.text_kv->batch_layer_view(static_cast<std::uint32_t>(visit)); }
    const PagedKVLayerView v = rd.r.sequence_kv.layer_view(static_cast<std::uint32_t>(visit));
    PagedKVBatchLayerView out;
    out.k_pages       = v.k_pages;
    out.v_pages       = v.v_pages;
    out.k_scale_pages = v.k_scale_pages;
    out.v_scale_pages = v.v_scale_pages;
    out.block_tables  = Tensor(v.block_table.data, DType::I32, {static_cast<std::int32_t>(v.block_table.ne[0]), 1});
    out.head_dim      = v.head_dim;
    out.num_kv_heads  = v.num_kv_heads;
    out.dtype         = v.dtype;
    out.quant_group   = v.quant_group;
    return out;
}

/// Visit v's slice of the verify records' inputs: BF16 [6656, W_max, B] ([b][visit][j][channel] in memory).
Tensor records_view(const Round& rd, int visit, int batch) {
    Tensor t(static_cast<std::uint16_t*>(rd.records.inputs) +
                 static_cast<std::int64_t>(visit) * rd.records.width_max * kConvChannels,
             DType::BF16, {kConvChannels, rd.records.width_max, batch});
    t.nb[1] = static_cast<std::int64_t>(kConvChannels) * 2;
    t.nb[2] = static_cast<std::int64_t>(rd.records.visits) * rd.records.width_max * kConvChannels * 2;
    t.nb[3] = t.nb[2] * batch;
    return t;
}

ops::InklingMoeWeights moe_weights(const load::InklingLayerWeights& L) {
    const auto bank = [](const load::ExpertBank& b) {
        if (!b.gate_up_interleaved) { throw std::logic_error("inkling2: the MoE reads the interleaved w13 rows only"); }
        ops::InklingExpertBank out;
        out.codes     = b.codes;
        out.scales    = b.scales;
        out.amax      = b.amax;
        out.outer_den = b.outer_scale_denominator;
        return out;
    };
    ops::InklingMoeWeights w;
    w.format      = L.w13.format == load::ExpertFormat::Mxfp8 ? ops::InklingExpertFormat::Mxfp8
                                                               : ops::InklingExpertFormat::Nvfp4;
    w.router      = L.router.qdata;
    w.router_bias = L.router_bias;
    w.w13         = bank(L.w13);
    w.w2          = bank(L.w2);
    return w;
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
    dump("attn_in", v, attn_in.data, DType::BF16, D, D, T, Q, s);
    // 2. projections: one GEMM over the fused rows, or three over their row ranges (q-side / K / V)
    Tensor qkvr = rd.work.alloc(DType::BF16, {C::qkvr_rows, T});
    const auto in = [&](int stream) { return bf16_ptr(attn_in) + static_cast<std::int64_t>(stream) * T * D; };
    Tensor k_dyn, v_dyn;
    if (L.fp8_attention) {
        // FP8 attention weights (the artifact's per-channel / 1x128 objects, SUROGATE_SERVE_INKLING_ATTENTION=fp8-*):
        // lane c's inkling_attn_linear, every row range of the visit in ONE launch; the dynamic taps (BF16) ride the
        // grouped invariant GEMM.
        Tensor x0(in(0), DType::BF16, {D, T});
        Tensor x1(in(Q == 3 ? 1 : 0), DType::BF16, {D, T});
        Tensor x2(in(Q == 3 ? 2 : 0), DType::BF16, {D, T});
        if (Q == 1) {
            if (vi.global) { throw std::logic_error("inkling2: a global visit has three attention inputs"); }
            Tensor out(qkvr.data, DType::BF16, {C::qkvr_rows, T});
            const ops::InklingProjection pr[] = {{&x0, &L.qkvr, 0, &out}};
            ops::inkling_attn_linear(pr, &rd.work, s);
        } else {
            Tensor out_q = column_view(qkvr.data, DType::BF16, {C::qkvr_query_end, T}, C::qkvr_rows);
            Tensor out_k = column_view(bf16_ptr(qkvr) + C::qkvr_k_begin, DType::BF16,
                                       {C::qkvr_v_begin - C::qkvr_k_begin, T}, C::qkvr_rows);
            Tensor out_v = column_view(bf16_ptr(qkvr) + C::qkvr_v_begin, DType::BF16,
                                       {C::qkvr_rows - C::qkvr_v_begin, T}, C::qkvr_rows);
            const ops::InklingProjection pr[] = {{&x0, &L.qkvr, 0, &out_q},
                                                 {&x1, &L.qkvr, C::qkvr_k_begin, &out_k},
                                                 {&x2, &L.qkvr, C::qkvr_v_begin, &out_v}};
            ops::inkling_attn_linear(pr, &rd.work, s);
        }
        if (vi.global) {
            if (!L.k_conv_proj.has_value() || !L.v_conv_proj.has_value()) {
                throw std::logic_error("inkling2: a global visit without its dynamic-conv projections");
            }
            k_dyn = rd.work.alloc(DType::BF16, {32, T});
            v_dyn = rd.work.alloc(DType::BF16, {32, T});
            const ops::InklingGemmProblem taps[] = {{L.k_conv_proj->qdata, 32, in(1), k_dyn.data, 32},
                                                    {L.v_conv_proj->qdata, 32, in(2), v_dyn.data, 32}};
            ops::inkling_gemm_inv_group(taps, static_cast<std::int32_t>(L.k_conv_proj->k), T, DType::BF16, s);
        }
    } else if (Q == 1) {
        if (vi.global) { throw std::logic_error("inkling2: a global visit has three attention inputs"); }
        gemm_bf16(rd.work, L.qkvr, 0, C::qkvr_rows, in(0), T, qkvr.data, C::qkvr_rows, s);
    } else {
        // The q-side / K / V row ranges from their own inputs: one grouped launch, each element the chain of
        // gemm_bf16.
        if (L.qkvr.qtype != QType::BF16_CTRL || L.qkvr.qdata == nullptr) {
            throw std::logic_error("inkling2: only the BF16 attention weights are wired (FP8 attention: lanes b / c)");
        }
        const auto* w = static_cast<const std::uint16_t*>(L.qkvr.qdata);
        const std::int64_t k = L.qkvr.k;
        ops::InklingGemmProblem problems[5] = {
            {w, C::qkvr_query_end, in(0), qkvr.data, C::qkvr_rows},
            {w + C::qkvr_k_begin * k, C::qkvr_v_begin - C::qkvr_k_begin, in(1), bf16_ptr(qkvr) + C::qkvr_k_begin,
             C::qkvr_rows},
            {w + C::qkvr_v_begin * k, C::qkvr_rows - C::qkvr_v_begin, in(2), bf16_ptr(qkvr) + C::qkvr_v_begin,
             C::qkvr_rows},
            {},
            {},
        };
        int count = 3;
        if (vi.global) { // the dynamic K / V taps (32 rows each) ride in the same launch, from x_k / x_v
            if (!L.k_conv_proj.has_value() || !L.v_conv_proj.has_value() || L.k_conv_proj->k != k ||
                L.v_conv_proj->k != k || L.k_conv_proj->qtype != QType::BF16_CTRL ||
                L.v_conv_proj->qtype != QType::BF16_CTRL) {
                throw std::logic_error("inkling2: a global visit without its dynamic-conv projections");
            }
            k_dyn       = rd.work.alloc(DType::BF16, {32, T});
            v_dyn       = rd.work.alloc(DType::BF16, {32, T});
            problems[3] = {L.k_conv_proj->qdata, 32, in(1), k_dyn.data, 32};
            problems[4] = {L.v_conv_proj->qdata, 32, in(2), v_dyn.data, 32};
            count       = 5;
        }
        ops::inkling_gemm_inv_group(std::span<const ops::InklingGemmProblem>(problems, static_cast<std::size_t>(count)),
                                    static_cast<std::int32_t>(k), T, DType::BF16, s);
    }
    // 2b. S10 attention LoRA on the six qkvr slices, each from its own input (q-side from x_q, K from x_k, V from x_v on a
    //     3-input visit, as vLLM's _project), after the base projection and before anything reads qkvr.
    if (const AttnLoraSelection sel = attn_lora_selection(L.qkvr.qdata, T); sel.table != nullptr) {
        ops::InklingLoraSlice slices[6];
        for (int i = 0; i < 6; ++i) {
            const AttnLoraSlice& d = kQkvrLoraSlices[i];
            const int stream       = Q == 3 ? (i < 4 ? 0 : i - 3) : 0;
            slices[i]              = {i, d.rows, in(stream), D, bf16_ptr(qkvr) + d.begin, C::qkvr_rows};
        }
        apply_attn_lora(rd.work, sel, slices, 6, T, s);
    }
    // 3. attention per group: lane c's invariant op (prologue: K / V conv + cache write; relative attention; gate and
    //    sink). The K / V conv state is its: in place, recorded (verify), or captured at a rewrite checkpoint.
    Tensor gated = rd.work.alloc(DType::BF16, {C::attn_width, T});
    const ops::InklingConvStateView state = state_view(rd, v);
    if (use_naive()) {
        Tensor q_out = rd.work.alloc(DType::BF16, {C::attn_width, T});
        Tensor r_out = rd.work.alloc(DType::BF16, {C::query_heads * C::d_rel, T});
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
    } else {
        const ops::InklingAttentionWeights aw = attention_weights(rd, L);
        const ops::InklingLayerKind kind      = vi.global ? ops::InklingLayerKind::Global : ops::InklingLayerKind::Local;
        const PagedKVBatchLayerView cache     = kv_batch_view(rd, v);
        Tensor states                         = rd.r.state->conv.at(static_cast<std::size_t>(v));
        using AG                              = ops::InklingAttentionGeometry;
        for (const Group& g : rd.groups) {
            const family::VisitSegment* seg = g.segment;
            const std::int32_t n            = g.width * g.batch;
            auto* base = bf16_ptr(qkvr) + static_cast<std::int64_t>(g.column) * C::qkvr_rows;
            ops::InklingProjections pr;
            pr.q    = column_view(base + AG::kRowsQ, DType::BF16, {C::head_dim, C::query_heads, n}, C::qkvr_rows);
            pr.r    = column_view(base + AG::kRowsR, DType::BF16, {C::d_rel, C::query_heads, n}, C::qkvr_rows);
            pr.gate = column_view(base + AG::kRowsGate, DType::BF16, {C::attn_width, n}, C::qkvr_rows);
            pr.sink = column_view(base + AG::kRowsSink, DType::BF16, {C::query_heads, n}, C::qkvr_rows);
            pr.k    = column_view(base + AG::kRowsK, DType::BF16, {1024, n}, C::qkvr_rows);
            pr.v    = column_view(base + AG::kRowsV, DType::BF16, {1024, n}, C::qkvr_rows);
            if (vi.global) {
                pr.k_dyn = column_view(bf16_ptr(k_dyn) + static_cast<std::int64_t>(g.column) * 32, DType::BF16,
                                       {4, C::kv_heads, n}, 32);
                pr.v_dyn = column_view(bf16_ptr(v_dyn) + static_cast<std::int64_t>(g.column) * 32, DType::BF16,
                                       {4, C::kv_heads, n}, 32);
            }
            ops::InklingAttentionRows rows;
            rows.positions     = g.positions;
            rows.valid_columns = g.valid;
            rows.kv_table_rows = g.table_rows;
            rows.key_groups    = kKeyGroups;
            // Lane c's early metadata read (c2's metadata_static, e1's Q3 answer 12:41): segments when their ints went
            // in the round's host copy (a full barrier after everything the step wrote before); the ordinary decode
            // block (ingress copy, the step's first node) or any non-verify block behind that barrier; never verify.
            rows.metadata_static = seg != nullptr ? rd.seg_barrier
                                                  : !g.record && (rd.r.block->metadata_static || rd.seg_barrier);
            if (rd.r.block != nullptr && seg == nullptr) {
                rows.envelope = rd.r.block->envelope;
            } else {
                rows.envelope.max_visible_keys = rd.r.max_visible_keys;
            }
            ops::InklingConvStateArgs conv;
            conv.states = states;
            conv.slots  = g.slots;
            conv.mode   = g.record ? ops::InklingConvMode::Record : ops::InklingConvMode::InPlace;
            if (g.record) { conv.records = records_view(rd, v, g.batch); }
            if (seg != nullptr && seg->checkpoint_slot >= 0) {
                conv.capture_column = seg->checkpoint_column;
                conv.capture_slot   = seg->checkpoint_slot;
            }
            conv.k_offset = G::kStreamK;
            conv.v_offset = G::kStreamV;
            Tensor out(bf16_ptr(gated) + static_cast<std::int64_t>(g.column) * C::attn_width, DType::BF16,
                       {C::attn_width, n});
            ops::inkling_attention(pr, aw, kind, rows, conv, cache, rd.work, out, s);
            if (seg != nullptr && seg->capture_slot >= 0) {
                ops::inkling_conv_copy_slot(state, G::kStreamK, 2048, seg->state_slot, seg->capture_slot, s);
            }
        }
    }
    dump("qkvr", v, qkvr.data, DType::BF16, C::qkvr_rows, C::qkvr_rows, T, 1, s);
    dump("gated", v, gated.data, DType::BF16, C::attn_width, C::attn_width, T, 1, s);
    // 4. wo_ud
    Tensor attn_partial = rd.work.alloc(DType::BF16, {D, T});
    if (L.fp8_attention) {
        const ops::InklingProjection pr[] = {{&gated, &L.output, 0, &attn_partial}};
        ops::inkling_attn_linear(pr, &rd.work, s);
    } else {
        gemm_bf16(rd.work, L.output, 0, D, gated.data, T, attn_partial.data, D, s);
    }
    if (const AttnLoraSelection sel = attn_lora_selection(L.output.qdata, T); sel.table != nullptr) {
        const ops::InklingLoraSlice wo[] = {{0, D, gated.data, C::attn_width, attn_partial.data, D}};
        apply_attn_lora(rd.work, sel, wo, 1, T, s);
    }
    // 5. attention-output conv (+ 6. the MLP input mix: one fused launch on a decode round of a non-looped visit)
    Tensor attn_out = rd.work.alloc(DType::BF16, {D, T});
    Tensor mlp_in   = rd.work.alloc(DType::BF16, {D, T, 1});
    const bool fuse_conv_mix = !fusions_off() && !vi.looped && rd.groups.size() == 1 &&
                               rd.groups[0].segment == nullptr && !rd.groups[0].record && rd.groups[0].width == 1 &&
                               rd.groups[0].column == 0 && rd.groups[0].batch == T;
    if (fuse_conv_mix) {
        const Group& g                         = rd.groups[0];
        const ops::InklingAttnResSources src   = mlp_sources(rd, vi, attn_out);
        const Tensor& queries                  = rd.q.mlp.at(static_cast<std::size_t>(vi.layer));
        const Tensor norm                      = vector_as_rows(L.moe_input_norm, 1);
        ops::inkling_conv_attn_res(packed(attn_partial, 0), L.attn_out_conv, g.rows(), state, G::kStreamAttn,
                                   packed(attn_out, 0), src, queries, &norm, C::rms_epsilon, mlp_in, s);
    }
    for (const Group& g : rd.groups) {
        if (fuse_conv_mix) { break; }
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
    dump("attn_out", v, attn_out.data, DType::BF16, D, D, T, 1, s);
    // 6. MLP input
    if (!fuse_conv_mix) { mlp_input(rd, vi, L, attn_out, mlp_in); }
    dump("mlp_in", v, mlp_in.data, DType::BF16, D, D, T, 1, s);
    // 7. MoE: lane d's op (routing, NVFP4 four-over-six / MXFP8 activations, experts, combine); routed experts
    //    captured per VISIT
    Tensor moe_out = rd.work.alloc(DType::BF16, {D, T});
    // S10: the round's adapter selection (published by the family schedule; none in a base round) on this
    // physical layer's bank table (lane d's family/impl/lora_bind_inkling.h) -- every pass of a core layer shares it.
    ops::InklingMoeAdapters adapter_storage;
    ops::InklingMoeWeights moe_key;
    moe_key.router = L.router.qdata;
    const ops::InklingMoeAdapters* adapters = family::inkling_moe_round_adapters(moe_key, adapter_storage);
    if (adapters != nullptr) {
        const ops::LoraRound& round = ops::lora_current_round();
        if (round.slots != nullptr && round.slots->numel() < static_cast<std::int64_t>(T)) {
            throw std::logic_error("inkling2: the round's adapter slots are narrower than the round");
        }
    }
    if (use_naive()) {
        if (adapters != nullptr) { throw std::logic_error("inkling2: the naive MoE serves no adapters"); }
        naive::MoeCall moe;
        moe.x      = mlp_in.data;
        moe.tokens = T;
        moe.layer  = moe_layer(L);
        moe.out    = moe_out.data;
        if (rd.r.routing.ids != nullptr && T <= rd.r.routing.columns) {
            moe.capture = rd.r.routing.ids + static_cast<std::int64_t>(v) * rd.r.routing.columns * rd.r.routing.top_k;
        }
        naive::moe(moe, rd.work, s);
    } else {
        ops::InklingMoeCapture capture;
        ops::InklingMoeOptions options;
        if (rd.r.routing.ids != nullptr && T <= rd.r.routing.columns) {
            capture.ids             = rd.r.routing.ids + static_cast<std::int64_t>(v) * rd.r.routing.columns * rd.r.routing.top_k;
            capture.capacity_tokens = rd.r.routing.columns;
            capture.first_token     = 0;
            capture.token_stride    = rd.r.routing.top_k;
            options.capture         = &capture;
        }
        options.adapters = adapters;
        // Lane d's narrow-round router + select fold: the stream's persistent counters (zeroed once at startup).
        options.state = static_cast<std::int32_t*>(rd.r.target_state);
        const Tensor x(mlp_in.data, DType::BF16, {D, T});
        ops::inkling_moe(x, moe_weights(L), moe_out, rd.work, s, options);
    }
    dump("moe_out", v, moe_out.data, DType::BF16, D, D, T, 1, s);
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
    dump("partial", v, bf16_ptr(rd.r.x) + static_cast<std::int64_t>(vi.next_partial) * D, DType::BF16, kColumn, D, T, 1, s);
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
    // Groups: every segment (B = 1) and the decode / verify block. The segments' KV rows and state slots are staged
    // with ONE host copy before the round's first visit kernel (so they are complete before any attention kernel:
    // lane c's metadata_static); under a stream capture (never for a visit target's prompt rounds today) the copy
    // from pageable memory is not allowed and they are written by kernels instead.
    cudaStreamCaptureStatus seg_capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(rd.s, &seg_capture));
    std::vector<std::int32_t> seg_ints;
    Tensor seg_dev;
    // 8 ints per segment: the KV row at [8 i], the state slot at [8 i + 4] -- each 16-byte aligned (lane c's ops
    // require aligned I32 metadata).
    constexpr int kSegInts = 8;
    if (!r.segments.empty()) {
        seg_dev = rd.work.alloc(DType::I32, {static_cast<std::int32_t>(kSegInts * r.segments.size())});
    }
    for (std::size_t i = 0; i < r.segments.size(); ++i) {
        const family::VisitSegment& seg = r.segments[i];
        Group g;
        g.column    = seg.column;
        g.width     = seg.count;
        g.batch     = 1;
        g.positions = Tensor(static_cast<std::int32_t*>(r.positions.data) + seg.column, DType::I32, {seg.count, 1});
        auto* ints  = static_cast<std::int32_t*>(seg_dev.data) + kSegInts * i;
        std::int32_t row = 0; // unused when the segment's own table row tensor is taken
        if (seg.table_row.data != nullptr && !r.sequence_kv.valid()) {
            g.table_rows = seg.table_row.view({1});
        } else {
            if (!r.sequence_kv.valid() && seg.kv_table_row < 0) {
                throw std::logic_error("inkling2: a segment without a KV row");
            }
            row          = r.sequence_kv.valid() ? 0 : seg.kv_table_row;
            g.table_rows = Tensor(ints, DType::I32, {1});
        }
        const std::int32_t block[kSegInts] = {row, 0, 0, 0, seg.state_slot, 0, 0, 0};
        seg_ints.insert(seg_ints.end(), block, block + kSegInts);
        g.slots   = Tensor(ints + 4, DType::I32, {1});
        g.segment = &seg;
        rd.groups.push_back(g);
    }
    if (!seg_ints.empty()) {
        if (seg_capture == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaMemcpyAsync(seg_dev.data, seg_ints.data(), seg_ints.size() * sizeof(std::int32_t),
                                       cudaMemcpyHostToDevice, rd.s));
            rd.seg_barrier = true;
        } else {
            for (std::size_t i = 0; i < seg_ints.size(); ++i) {
                naive::fill_i32(static_cast<std::int32_t*>(seg_dev.data) + i, 1, seg_ints[i], rd.s);
            }
        }
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
    if (!r.segments.empty() && r.max_visible_keys == 0) {
        throw std::logic_error("inkling2: prompt segments without a visible-key promise (VisitRound.max_visible_keys)");
    }
    if (r.sequence_kv.valid() && r.block != nullptr && r.block->batch > 0) {
        throw std::logic_error("inkling2: a single-sequence round cannot carry a decode / verify block");
    }
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
    Dump& d = dump_state();
    static bool dumped = false;
    cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(rd.s, &capturing));
    // A prompt round only (the graph warmup's dummy decode rows come first and are not a parity prompt).
    // SUROGATE_SERVE_INKLING_DUMP_MIN_COLUMNS (default 64): the engine's warmup prompt is a few tokens and must not
    // take the one-shot dump meant for the parity prompt.
    static const int min_columns = [] {
        const char* raw = std::getenv("SUROGATE_SERVE_INKLING_DUMP_MIN_COLUMNS");
        return raw != nullptr ? std::atoi(raw) : 64;
    }();
    d.active = d.dir != nullptr && !dumped && capturing == cudaStreamCaptureStatusNone && !r.segments.empty() &&
               rd.T >= min_columns;
    for (int v = r.first_visit; v < r.last_visit; ++v) { run_visit(rd, v); }
    if (d.active) {
        dumped   = true;
        d.active = false;
    }
}

std::size_t Variant::visit_target_state_bytes() { return ops::inkling_moe_state_bytes(); }

std::size_t Variant::visit_workspace_capacity_bytes(const family::TextGeometry&, family::TextPhase, std::int32_t batch_size,
                                                    std::int32_t max_width, std::int32_t max_columns,
                                                    std::uint32_t max_visible_keys, bool lora_enabled) {
    const auto up = [](std::size_t bytes) { return (bytes + 255U) & ~std::size_t{255}; };
    const std::size_t T    = static_cast<std::size_t>(std::max<std::int32_t>(max_columns, std::max(batch_size, 1) * std::max(max_width, 1)));
    const std::size_t tile = std::min<std::size_t>(T, kTileColumns);
    std::size_t bytes = 0;
    bytes += up(64 * 8 * 4);                                 // segment ints (8 per segment, 64 segments)
    bytes += up(T * D * 3 * 2);                              // attn_in
    bytes += up(T * C::qkvr_rows * 2);                       // qkvr
    bytes += 2 * up(T * 32 * 2);                             // dynamic taps
    bytes += up(T * C::attn_width * 2) * 2 + up(T * 384 * 2); // q_out, gated, r_out
    bytes += up(T * D * 2) * 4;                              // attn_partial, attn_out, mlp_in, moe_out
    // the looped blend tile: dest + msg fp32, pair, h, gf fp32, blended
    bytes += up(tile * D * 3 * 4) + up(tile * D * 3 * 4) + up(tile * 2 * D * 3 * 2) + up(tile * C::gate_rank * 3 * 2) +
             up(tile * 2 * D * 3 * 4) + up(tile * D * 3 * 2);
    // the invariant GEMM's split partials: the controller `down` of a blend tile (the taps run unsplit in the group)
    bytes += up(ops::inkling_gemm_inv_workspace_bytes(C::gate_rank, 2 * D, static_cast<std::int32_t>(tile), 3));
    // lanes c / d's ops (and the naive kernels behind SUROGATE_SERVE_INKLING_NAIVE); attention and MoE both counted
    ops::GqaExecutionEnvelope envelope;
    envelope.max_visible_keys = max_visible_keys;
    std::size_t attention     = 0;
    for (const auto kind : {ops::InklingLayerKind::Local, ops::InklingLayerKind::Global}) {
        if (max_columns > 0) {
            attention = std::max(attention, ops::inkling_attention_workspace_capacity_bytes(kind, 1, 1, max_columns, envelope,
                                                                                            kKeyGroups));
        }
        if (batch_size > 0 && max_width > 0) {
            attention = std::max(attention, ops::inkling_attention_workspace_capacity_bytes(kind, batch_size, 1, max_width,
                                                                                            envelope, kKeyGroups));
        }
    }
    // An engine started with adapters (--enable-lora / --lora-modules) books the expert LoRA buffers.
    const std::size_t moe = std::max({ops::inkling_moe_workspace_bytes(ops::InklingExpertFormat::Nvfp4,
                                                                       static_cast<std::int32_t>(T), lora_enabled),
                                      ops::inkling_moe_workspace_bytes(ops::InklingExpertFormat::Mxfp8,
                                                                       static_cast<std::int32_t>(T), lora_enabled),
                                      naive::moe_workspace_bytes(static_cast<std::int32_t>(T))});
    bytes += up(attention) + up(moe);
    // S10 attention LoRA: the fp32 low-rank buffers of the qkvr (6 slices) and wo_ud calls, at the kernels' rank bound 64
    if (lora_enabled) {
        bytes += up(ops::inkling_lora_delta_workspace_bytes(6, 64, static_cast<std::int32_t>(T))) +
                 up(ops::inkling_lora_delta_workspace_bytes(1, 64, static_cast<std::int32_t>(T)));
    }
    return bytes + (1U << 20);
}

} // namespace sinfer::targets::inkling2::detail
