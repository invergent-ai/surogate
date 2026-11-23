// ENGINE-INKLING lane T8 (agent-T8): COPY of lane T's inkling_train_shim.cpp (6a9c4a18) with ONE addition, marked "T8:": the S10
// attention LoRA (lane e1, targets/inkling2/impl/lora_attention.h) on the six qkvr slices and wo_ud, from bank tables the trainer passes
// EXPLICITLY (forward.cpp reads them from the engine's LoRA store). Same op, same slices, same order as forward.cpp 2b and step 4.
// Built as libsinfer_inkling_train_t8.so; lane T's shim is unchanged. ABI 3 (ift_run_visit takes one more pointer).
//
// ENGINE-INKLING lane T (agent-EN-T): the engine-op trainer's FAST route (lanes/T/NOTES.md PLAN).
//
// libsinfer_inkling_train.so: an extern "C" shim over the SAME libsinfer.so the rollout engine runs, so a torch
// training loop (surogate/train/inkling_fastpath/) can drive the engine's own Inkling forward and get the engine's
// bits (DESIGN.md 11: one shared, batch- and path-invariant forward for engine and trainer).
//
// What it does:
// - ift_load: loads a `.sinfer` artifact with the engine's own loader (lane b's bind + lane e1's derived queries and
//   rel tables; the DSpark drafter is not bound) into engine-owned device memory.
// - ift_run_visit: ONE visit of the 48, a copy of `targets/inkling2/impl/forward.cpp::run_visit` restricted to one
//   prompt segment (B = 1, column 0, positions 0..T-1) -- the same op calls with the same arguments, in the same
//   order (attention inputs / looped blend, qkvr projections, lane c's attention, wo_ud, attention-output conv, MLP
//   input, lane d's MoE with the routed-expert capture, MLP-output conv). Copied, not re-derived: every difference
//   from forward.cpp is marked "T:". The M2 test (engine dump per visit + prompt log-probs) proves the copy.
// - ift_embed / ift_final_hidden / ift_head_logprob: the Variant's embed and final-mix hooks and lane e1's
//   `inkling_head_logprob` (= the value of every log-prob the engine returns).
// - ift_tensor: device views of the loaded weights for the torch backward (no copy).
// Every entry returns 0 on success, else -1 with the reason in ift_last_error() (no exception crosses the C ABI).
// Nothing here is linked into the engine; libsinfer.so is unchanged.

#include "targets/inkling2/impl/model.h"
#include "targets/inkling2/impl/variant.h"
#include "targets/inkling2/impl/visits.h"
#include "targets/inkling2/impl/lora_attention.h"

#include "api/ops/inkling_attention.h"
#include "api/ops/inkling_mix.h"
#include "api/ops/inkling_moe.h"
#include "api/ops/lora_store.h"
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include <api/family/startup_features.h>
#include <api/targets/inkling2/package.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sinfer::targets::inkling2::fastpath {
namespace {

using namespace ::sinfer::targets::inkling2::detail;
using C                         = TextConfig;
constexpr int D                 = C::hidden;
constexpr std::int64_t kColumn  = C::residual;
constexpr int kTileColumns      = 1024;
constexpr int kConvChannels     = ops::InklingMixGeometry::kConvChannels;
using G                         = ops::InklingMixGeometry;
/// Lane c's key-group cut, passed explicitly (forward.cpp kKeyGroups; lane h froze P3).
constexpr ops::InklingKeyGroups kKeyGroups = ops::InklingKeyGroups::P3;

thread_local std::string t_error;

/// The loaded model: the engine's own LoadedModelData (weights, derived queries, rel tables, the ModelView).
struct Handle {
    int device = 0;
    std::unique_ptr<DeviceContext> context;
    std::unique_ptr<artifact::Reader> reader;
    std::unique_ptr<LoadedModelData> data;
    bool fp8_attention = false;
};

/// forward.cpp's attention_choice() (package.cpp, anonymous namespace there): SUROGATE_SERVE_INKLING_ATTENTION =
/// bf16 (default) | fp8-<scheme>. T: copied, so the trainer and the engine read the same variable.
load::AttentionChoice attention_choice() {
    load::AttentionChoice choice;
    const char* raw = std::getenv("SUROGATE_SERVE_INKLING_ATTENTION");
    if (raw == nullptr || std::string_view(raw).empty() || std::string_view(raw) == "bf16") { return choice; }
    const std::string_view value(raw);
    if (value.substr(0, 4) != "fp8-") {
        throw std::invalid_argument("SUROGATE_SERVE_INKLING_ATTENTION must be bf16 or fp8-<scheme>");
    }
    choice.fp8    = true;
    choice.scheme = std::string(value.substr(4));
    return choice;
}

// ---------------------------------------------------------------------------------------------------------------
// forward.cpp helpers, verbatim
// ---------------------------------------------------------------------------------------------------------------
std::uint16_t* bf16_ptr(const Tensor& t) { return static_cast<std::uint16_t*>(t.data); }

ops::InklingPlane slot_plane(const Tensor& x, int slot, int column = 0) {
    return {bf16_ptr(x) + static_cast<std::int64_t>(column) * kColumn + static_cast<std::int64_t>(slot) * D,
            DType::BF16, kColumn};
}

ops::InklingPlane packed(const Tensor& t, int column = 0, std::int64_t width = D, std::int64_t offset = 0) {
    const std::size_t element = t.dtype == DType::FP32 ? 4 : 2;
    return {static_cast<std::byte*>(t.data) + (static_cast<std::int64_t>(column) * width + offset) * element,
            t.dtype, width};
}

Tensor column_view(void* data, DType dtype, std::initializer_list<std::int32_t> dims, std::int64_t column_elems) {
    Tensor t(data, dtype, dims);
    const std::int64_t element = dtype == DType::FP32 ? 4 : 2;
    const int last             = static_cast<int>(dims.size()) - 1;
    t.nb[last]                 = column_elems * element;
    for (int i = last + 1; i < 4; ++i) { t.nb[i] = t.nb[i - 1] * t.ne[i - 1]; }
    return t;
}

void* gemm_scratch(WorkspaceArena& work, std::size_t bytes) {
    if (bytes == 0) { return nullptr; }
    return work.alloc(DType::FP32, {static_cast<std::int32_t>((bytes + 3) / 4)}).data;
}

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

/// T: forward.cpp's Group for the one prompt segment of a trainer forward (B = 1, column 0, width T).
struct Group {
    std::int32_t column = 0, width = 0, batch = 0;
    Tensor positions;  ///< I32 [W, B]
    Tensor valid;      ///< empty
    Tensor table_rows; ///< I32 [1]
    Tensor slots;      ///< I32 [1]
    bool record = false;

    [[nodiscard]] ops::InklingConvRows rows() const { return {width, batch, positions, valid, slots}; }
};

/// T8: one visit's attention LoRA, passed by the trainer: the S10 bank tables of the visit's physical layer (qkvr: 6 LoraBank, slice order of
/// kQkvrLoraSlices; wo_ud: 1 LoraBank), slot = *uniform_slot for every token, rank = the banks' max_rank (<= 64, the kernels' bound).
struct AttnAdapters {
    const void* qkvr_table          = nullptr;
    const void* wo_table            = nullptr;
    const std::int32_t* uniform_slot = nullptr;
    std::int32_t rank               = 0;
};

/// T8: forward.cpp's apply_attn_lora with an explicit table (forward.cpp reads it from the store).
void apply_attn_lora_t8(WorkspaceArena& work, const void* table, const std::int32_t* uniform, std::int32_t rank,
                        const ops::InklingLoraSlice* slices, int count, std::int32_t tokens, cudaStream_t s) {
    const std::size_t bytes = ops::inkling_lora_delta_workspace_bytes(count, rank, tokens);
    Tensor low = work.alloc(DType::FP32, {static_cast<std::int32_t>(bytes / sizeof(float))});
    ops::inkling_lora_delta(table, slices, count, rank, tokens, nullptr, uniform, low.data, bytes, s);
}

/// T: the parts of forward.cpp's Round + VisitRound a trainer segment uses.
struct Round {
    const RuntimeModelView& model;
    const InklingModel& im;
    const DerivedQueries& q;
    WorkspaceArena& work;
    cudaStream_t s;
    std::int32_t T;
    Tensor x;                       ///< the residual BF16 [7 x 2304, T]
    std::vector<Group> groups;
    Tensor conv;                    ///< this visit's conv state BF16 [6656, 3, slots]
    PagedKVBatchLayerView kv;       ///< this visit's plane (identity block table over trainer memory)
    void* target_state = nullptr;   ///< V::visit_target_state_bytes(), zeroed once
    std::int32_t* routing_ids = nullptr; ///< I32 [T, 8] for this visit, or null
    const ops::InklingMoeAdapters* adapters = nullptr; ///< T: explicit (null = base)
    const struct AttnAdapters* attn = nullptr;          ///< T8: explicit attention LoRA of this visit's layer (null = base)
    std::uint32_t max_visible_keys = 0;
    bool seg_barrier = true;        ///< T: the segment ints were written before the forward (torch)
};

std::size_t moe_state_span() { return (ops::inkling_moe_state_bytes() + 255U) & ~std::size_t{255}; }

ops::InklingConvStateView state_view(const Round& rd) {
    return {rd.conv.data, static_cast<std::int64_t>(kConvChannels) * 3, kConvChannels};
}

std::int32_t moe_workspace_tokens(const WorkspaceArena& work, ops::InklingExpertFormat format, bool adapters,
                                  std::int32_t tokens) {
    constexpr std::size_t kSlack = std::size_t{1} << 16;
    const std::size_t used = work.used(), capacity = work.capacity();
    if (capacity <= used + kSlack) { return 0; }
    const std::size_t free = capacity - used - kSlack;
    if (ops::inkling_moe_workspace_bytes(format, tokens, adapters) > free) { return 0; }
    std::int32_t lo = tokens, hi = std::max<std::int32_t>(tokens, 1) * 2;
    while (hi <= (1 << 20) && ops::inkling_moe_workspace_bytes(format, hi, adapters) <= free) {
        lo = hi;
        hi *= 2;
    }
    while (hi - lo > 1) {
        const std::int32_t mid = lo + (hi - lo) / 2;
        (ops::inkling_moe_workspace_bytes(format, mid, adapters) <= free ? lo : hi) = mid;
    }
    return lo;
}

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
        ops::InklingAttnResSources bank;
        bank.stack_count = 1;
        bank.stack[0]    = slot_plane(rd.x, vi.bank[0], c0);
        bank.last        = slot_plane(rd.x, vi.bank[1], c0);
        Tensor msg_queries(static_cast<float*>(messages.data) + static_cast<std::int64_t>(first_stream) * D,
                           DType::FP32, {D, streams});
        Tensor msg  = rd.work.alloc(DType::FP32, {D, n, streams});
        Tensor pair = rd.work.alloc(DType::BF16, {2 * D, n, streams});
        ops::inkling_blend_inputs(src, dest_queries, bank, msg_queries, n, dest, msg, pair, rd.s);
        const auto* down = bf16_ptr(L.loop_down) + static_cast<std::int64_t>(first_stream) * C::gate_rank * 2 * D;
        const auto* gfw  = bf16_ptr(L.loop_gain_forget) + static_cast<std::int64_t>(first_stream) * 2 * D * C::gate_rank;
        const ops::InklingGemmBatch down_batch{streams, static_cast<std::int64_t>(C::gate_rank) * 2 * D,
                                               static_cast<std::int64_t>(n) * 2 * D,
                                               static_cast<std::int64_t>(n) * C::gate_rank};
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

void attention_inputs(Round& rd, const VisitInfo& vi, const load::InklingLayerWeights& L, Tensor& attn_in) {
    if (vi.stack_count == 0 && vi.partial == 0) {
        ops::inkling_rms_norm(slot_plane(rd.x, 0), rd.T, L.attn_input_norm, C::rms_epsilon, attn_in, rd.s);
        return;
    }
    ops::InklingAttnResSources src;
    src.stack_count = vi.stack_count;
    for (int i = 0; i < vi.stack_count; ++i) { src.stack[i] = slot_plane(rd.x, vi.stack[i]); }
    src.last = slot_plane(rd.x, vi.partial);
    const Tensor& queries = rd.q.attn.at(static_cast<std::size_t>(vi.layer));
    if (!vi.looped) {
        ops::inkling_attn_res(src, rd.T, queries, &L.attn_input_norm, C::rms_epsilon, attn_in, rd.s);
        return;
    }
    blended_inputs(rd, vi, L, src, queries, 0, 3, L.attn_input_norm, attn_in);
}

void mlp_input(Round& rd, const VisitInfo& vi, const load::InklingLayerWeights& L, const Tensor& attn_out,
               Tensor& mlp_in) {
    ops::InklingAttnResSources src;
    src.stack_count = vi.stack_count;
    for (int i = 0; i < vi.stack_count; ++i) { src.stack[i] = slot_plane(rd.x, vi.stack[i]); }
    if (vi.first) {
        src.stack[src.stack_count++] = slot_plane(rd.x, vi.partial);
        src.last                     = packed(attn_out);
    } else {
        src.last   = slot_plane(rd.x, vi.partial);
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

ops::InklingAttentionWeights attention_weights(const Round& rd, const load::InklingLayerWeights& L) {
    ops::InklingAttentionWeights w;
    w.q_scale   = L.q_scale;
    w.r_scale   = L.r_scale;
    w.k_conv    = L.global ? L.k_conv_static : L.k_conv;
    w.v_conv    = L.global ? L.v_conv_static : L.v_conv;
    w.rel_table = rd.q.rel_table.at(static_cast<std::size_t>(L.layer));
    return w;
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

/// T: optional per-visit taps for the torch backward (null pointers = not written). Device buffers, column-major
/// like the engine's (a [rows, T] buffer = T columns of `rows` contiguous elements).
struct VisitTaps {
    void* attn_in   = nullptr; ///< BF16 [2304, T, Q] (Q = 3 on global / looped visits, else 1)
    void* qkvr      = nullptr; ///< BF16 [8600, T]
    void* k_dyn     = nullptr; ///< BF16 [32, T] (global visits)
    void* v_dyn     = nullptr; ///< BF16 [32, T] (global visits)
    void* gated     = nullptr; ///< BF16 [3072, T]
    void* attn_partial = nullptr; ///< BF16 [2304, T] (wo_ud output)
    void* attn_out  = nullptr; ///< BF16 [2304, T]
    void* mlp_in    = nullptr; ///< BF16 [2304, T]
    void* moe_out   = nullptr; ///< BF16 [2304, T]
    // lane c's training call (when any of q / r / o / lse is set; else the production call)
    void* att_q     = nullptr; ///< BF16 [128, 24, T]
    void* att_r     = nullptr; ///< BF16 [16, 24, T]
    void* att_tau   = nullptr; ///< FP32 [T] (global visits)
    void* att_o     = nullptr; ///< FP32 [128, 24, T]
    void* att_lse   = nullptr; ///< FP32 [24, T]
    // lane d's MoE taps (InklingMoeTaps)
    void* moe_gammas  = nullptr; ///< FP32 [T * 8]
    void* moe_gate_up = nullptr; ///< BF16 [T * 8][2304]
    void* moe_y       = nullptr; ///< BF16 [T * 8][1152]
    void* moe_down    = nullptr; ///< BF16 [T * 8][2304]
    void* moe_logits  = nullptr; ///< FP32 [T][128]
};

void copy_tap(void* dst, const Tensor& src, cudaStream_t s) {
    if (dst == nullptr) { return; }
    CUDA_CHECK(cudaMemcpyAsync(dst, src.data, src.bytes(), cudaMemcpyDeviceToDevice, s));
}

void run_visit(Round& rd, int v, const VisitTaps& taps) {
    const VisitInfo& vi                 = kVisits[static_cast<std::size_t>(v)];
    const load::InklingLayerWeights& L  = *rd.model.full_layers.at(static_cast<std::size_t>(v)).projection.layer;
    const int T                         = rd.T;
    cudaStream_t s                      = rd.s;
    auto scope                          = rd.work.scope();
    // T: no DSpark feature capture (the trainer has no drafter).
    // 1. attention inputs
    const int Q    = (vi.global || vi.looped) ? 3 : 1;
    Tensor attn_in = rd.work.alloc(DType::BF16, {D, T, Q});
    attention_inputs(rd, vi, L, attn_in);
    copy_tap(taps.attn_in, attn_in, s);
    // 2. projections
    Tensor qkvr = rd.work.alloc(DType::BF16, {C::qkvr_rows, T});
    const auto in = [&](int stream) { return bf16_ptr(attn_in) + static_cast<std::int64_t>(stream) * T * D; };
    Tensor k_dyn, v_dyn;
    if (L.fp8_attention) {
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
            const ops::InklingGemmProblem taps_p[] = {{L.k_conv_proj->qdata, 32, in(1), k_dyn.data, 32},
                                                      {L.v_conv_proj->qdata, 32, in(2), v_dyn.data, 32}};
            ops::inkling_gemm_inv_group(taps_p, static_cast<std::int32_t>(L.k_conv_proj->k), T, DType::BF16, s);
        }
    } else if (Q == 1) {
        if (vi.global) { throw std::logic_error("inkling2: a global visit has three attention inputs"); }
        gemm_bf16(rd.work, L.qkvr, 0, C::qkvr_rows, in(0), T, qkvr.data, C::qkvr_rows, s);
    } else {
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
        if (vi.global) {
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
    // 2b. T8: S10 attention LoRA on the six qkvr slices, each from its own input (q-side from x_q, K from x_k, V from x_v on a
    //     3-input visit), after the base projection and before anything reads qkvr (forward.cpp 2b, with the table passed explicitly).
    if (rd.attn != nullptr && rd.attn->qkvr_table != nullptr) {
        if (rd.attn->rank <= 0 || rd.attn->rank > 64) { throw std::invalid_argument("T8: attention LoRA rank must be 1..64"); }
        ops::InklingLoraSlice slices[6];
        for (int i = 0; i < 6; ++i) {
            const AttnLoraSlice& d = kQkvrLoraSlices[i];
            const int stream       = Q == 3 ? (i < 4 ? 0 : i - 3) : 0;
            slices[i]              = {i, d.rows, in(stream), D, bf16_ptr(qkvr) + d.begin, C::qkvr_rows};
        }
        apply_attn_lora_t8(rd.work, rd.attn->qkvr_table, rd.attn->uniform_slot, rd.attn->rank, slices, 6, T, s);
    }
    copy_tap(taps.qkvr, qkvr, s);
    if (vi.global) {
        copy_tap(taps.k_dyn, k_dyn, s);
        copy_tap(taps.v_dyn, v_dyn, s);
    }
    // 3. attention: lane c's invariant op (the training entry when the caller asks for q / r / tau / o / lse; it is
    //    the production call's arithmetic on every route, lanes/c/NOTES.md TRAINING ENTRY).
    Tensor gated = rd.work.alloc(DType::BF16, {C::attn_width, T});
    const ops::InklingConvStateView state = state_view(rd);
    {
        const ops::InklingAttentionWeights aw = attention_weights(rd, L);
        const ops::InklingLayerKind kind      = vi.global ? ops::InklingLayerKind::Global : ops::InklingLayerKind::Local;
        const PagedKVBatchLayerView& cache    = rd.kv;
        Tensor states                         = rd.conv;
        using AG                              = ops::InklingAttentionGeometry;
        for (const Group& g : rd.groups) {
            const std::int32_t n = g.width * g.batch;
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
            if (rd.target_state != nullptr) {
                rows.state = reinterpret_cast<std::int32_t*>(static_cast<std::byte*>(rd.target_state) + moe_state_span());
            }
            // T: a prompt segment (forward.cpp: seg != nullptr -> rd.seg_barrier).
            rows.metadata_static            = rd.seg_barrier;
            rows.envelope.max_visible_keys  = rd.max_visible_keys;
            ops::InklingConvStateArgs conv;
            conv.states   = states;
            conv.slots    = g.slots;
            conv.mode     = ops::InklingConvMode::InPlace;
            conv.k_offset = G::kStreamK;
            conv.v_offset = G::kStreamV;
            Tensor out(bf16_ptr(gated) + static_cast<std::int64_t>(g.column) * C::attn_width, DType::BF16,
                       {C::attn_width, n});
            const bool train = taps.att_q != nullptr || taps.att_r != nullptr || taps.att_o != nullptr ||
                               taps.att_lse != nullptr;
            if (!train) {
                ops::inkling_attention(pr, aw, kind, rows, conv, cache, rd.work, out, s);
            } else {
                // The training entry writes q / r / o / lse into caller tensors; the missing ones go to scratch.
                auto keep = rd.work.scope();
                ops::InklingAttentionTrainOutputs tr;
                tr.q   = Tensor(taps.att_q ? taps.att_q : rd.work.alloc(DType::BF16, {C::head_dim, C::query_heads, n}).data,
                                DType::BF16, {C::head_dim, C::query_heads, n});
                tr.r   = Tensor(taps.att_r ? taps.att_r : rd.work.alloc(DType::BF16, {C::d_rel, C::query_heads, n}).data,
                                DType::BF16, {C::d_rel, C::query_heads, n});
                if (vi.global) {
                    tr.tau = Tensor(taps.att_tau ? taps.att_tau : rd.work.alloc(DType::FP32, {n}).data, DType::FP32, {n});
                }
                tr.o   = Tensor(taps.att_o ? taps.att_o : rd.work.alloc(DType::FP32, {C::head_dim, C::query_heads, n}).data,
                                DType::FP32, {C::head_dim, C::query_heads, n});
                tr.lse = Tensor(taps.att_lse ? taps.att_lse : rd.work.alloc(DType::FP32, {C::query_heads, n}).data,
                                DType::FP32, {C::query_heads, n});
                ops::inkling_attention_train(pr, aw, kind, rows, conv, cache, rd.work, out, tr, s);
            }
        }
    }
    copy_tap(taps.gated, gated, s);
    // 4. wo_ud
    Tensor attn_partial = rd.work.alloc(DType::BF16, {D, T});
    if (L.fp8_attention) {
        const ops::InklingProjection pr[] = {{&gated, &L.output, 0, &attn_partial}};
        ops::inkling_attn_linear(pr, &rd.work, s);
    } else {
        gemm_bf16(rd.work, L.output, 0, D, gated.data, T, attn_partial.data, D, s);
    }
    if (rd.attn != nullptr && rd.attn->wo_table != nullptr) {  // T8: S10 on wo_ud (forward.cpp step 4)
        if (rd.attn->rank <= 0 || rd.attn->rank > 64) { throw std::invalid_argument("T8: attention LoRA rank must be 1..64"); }
        const ops::InklingLoraSlice wo[] = {{0, D, gated.data, C::attn_width, attn_partial.data, D}};
        apply_attn_lora_t8(rd.work, rd.attn->wo_table, rd.attn->uniform_slot, rd.attn->rank, wo, 1, T, s);
    }
    copy_tap(taps.attn_partial, attn_partial, s);
    // 5. attention-output conv
    Tensor attn_out = rd.work.alloc(DType::BF16, {D, T});
    for (const Group& g : rd.groups) {
        ops::inkling_short_conv(packed(attn_partial, g.column), D, L.attn_out_conv, g.rows(), state, G::kStreamAttn,
                                nullptr, nullptr, packed(attn_out, g.column), s, -1, -1);
    }
    copy_tap(taps.attn_out, attn_out, s);
    // 6. MLP input
    Tensor mlp_in = rd.work.alloc(DType::BF16, {D, T, 1});
    mlp_input(rd, vi, L, attn_out, mlp_in);
    copy_tap(taps.mlp_in, mlp_in, s);
    // 7. MoE: lane d's op, routed experts captured per VISIT
    Tensor moe_out = rd.work.alloc(DType::BF16, {D, T});
    {
        ops::InklingMoeCapture capture;
        ops::InklingMoeOptions options;
        if (rd.routing_ids != nullptr) {
            capture.ids             = rd.routing_ids;
            capture.capacity_tokens = T;
            capture.first_token     = 0;
            capture.token_stride    = C::experts_per_token;
            options.capture         = &capture;
        }
        options.adapters = rd.adapters;
        options.state    = static_cast<std::int32_t*>(rd.target_state);
        options.workspace_tokens = moe_workspace_tokens(rd.work, moe_weights(L).format, rd.adapters != nullptr, T);
        ops::InklingMoeTaps moe_taps;
        moe_taps.gammas  = static_cast<float*>(taps.moe_gammas);
        moe_taps.gate_up = taps.moe_gate_up;
        moe_taps.y       = taps.moe_y;
        moe_taps.down    = taps.moe_down;
        moe_taps.logits  = static_cast<float*>(taps.moe_logits);
        if (moe_taps.gammas || moe_taps.gate_up || moe_taps.y || moe_taps.down || moe_taps.logits) {
            options.taps = &moe_taps; // T: lane d's taps (test / debug / trainer; the bits are the call's)
        }
        const Tensor x(mlp_in.data, DType::BF16, {D, T});
        ops::inkling_moe(x, moe_weights(L), moe_out, rd.work, s, options);
    }
    copy_tap(taps.moe_out, moe_out, s);
    // 8. MLP-output conv: (residual + attn_out) + conv(moe_out) into the next partial (in place when not first)
    for (const Group& g : rd.groups) {
        const ops::InklingPlane residual = slot_plane(rd.x, vi.partial, g.column);
        const ops::InklingPlane addend   = packed(attn_out, g.column);
        ops::inkling_short_conv(packed(moe_out, g.column), D, L.mlp_out_conv, g.rows(), state, G::kStreamMlp,
                                vi.first ? nullptr : &residual, &addend, slot_plane(rd.x, vi.next_partial, g.column), s,
                                -1, -1);
    }
}

Handle* as_handle(void* h) {
    if (h == nullptr) { throw std::invalid_argument("null handle"); }
    return static_cast<Handle*>(h);
}

template <class F>
int guarded(F&& f) {
    try {
        f();
        return 0;
    } catch (const std::exception& e) {
        t_error = e.what();
    } catch (...) {
        t_error = "unknown exception";
    }
    return -1;
}

} // namespace
} // namespace sinfer::targets::inkling2::fastpath

using namespace sinfer;
using namespace sinfer::targets::inkling2;
using namespace sinfer::targets::inkling2::fastpath;

extern "C" {

/// Version of this C ABI (bump on any signature change; the Python binding checks it).
int ift_abi_version() { return 3; }

const char* ift_last_error() { return t_error.c_str(); }

/// Load a `.sinfer` artifact on `device` with the engine's loader (no drafter). Returns a handle or null.
void* ift_load(const char* artifact_path, int device) {
    Handle* out = nullptr;
    const int rc = guarded([&] {
        CUDA_CHECK(cudaSetDevice(device));
        auto h       = std::make_unique<Handle>();
        h->device    = device;
        h->context   = std::make_unique<DeviceContext>(device);
        h->reader    = std::make_unique<artifact::Reader>(artifact_path);
        // Package::resolve_weights refuses an artifact the target does not serve (and reads the attention choice).
        (void)Package::resolve_weights(h->reader->identity());
        artifact::Binder binder(*h->reader);
        load::BindOptions bind;
        bind.attention   = attention_choice();
        bind.draft       = false;
        h->fp8_attention = bind.attention.fp8;
        EngineOptions options;
        ArtifactLoadPlan plan;
        plan.bindings.plan     = load::bind_artifact(binder, bind);
        plan.bindings.geometry = visit_geometry(*h->reader);
        plan.bindings.features = family::startup_features(options);
        plan.materialization   = binder.finish();
        auto materialized      = artifact::materialize(*h->reader, plan.materialization, *h->context);
        h->data = std::make_unique<LoadedModelData>(std::move(plan.bindings), std::move(materialized));
        CUDA_CHECK(cudaDeviceSynchronize());
        out = h.release();
    });
    return rc == 0 ? out : nullptr;
}

void ift_free(void* handle) { delete static_cast<Handle*>(handle); }

/// Geometry constants, in this order: hidden, residual, layers (visits), physical_layers, output_rows, token_domain,
/// qkvr_rows, attn_width, conv_channels, experts, top_k, intermediate, fp8_attention (of the loaded handle or -1).
int ift_constants(void* handle, std::int32_t* out, int n) {
    const std::int32_t v[] = {C::hidden,      C::residual,     C::layers,     C::physical_layers, C::output_rows,
                              C::token_domain, C::qkvr_rows,   C::attn_width, kConvChannels,      C::experts,
                              C::experts_per_token, C::intermediate,
                              handle ? (static_cast<Handle*>(handle)->fp8_attention ? 1 : 0) : -1};
    const int count = static_cast<int>(sizeof(v) / sizeof(v[0]));
    for (int i = 0; i < n && i < count; ++i) { out[i] = v[i]; }
    return count;
}

/// Visit v's schedule row: layer, pass, stack_count, stack[4], partial, next_partial, bank[2], feature, first, global,
/// core, looped (16 ints).
int ift_visit_info(int v, std::int32_t* out16) {
    return guarded([&] {
        if (v < 0 || v >= C::layers) { throw std::out_of_range("visit"); }
        const VisitInfo& vi = kVisits[static_cast<std::size_t>(v)];
        const std::int32_t r[16] = {vi.layer,    vi.pass,    vi.stack_count, vi.stack[0], vi.stack[1], vi.stack[2],
                                    vi.stack[3], vi.partial, vi.next_partial, vi.bank[0], vi.bank[1],  vi.feature,
                                    vi.first,    vi.global,  vi.core,        vi.looped};
        std::memcpy(out16, r, sizeof(r));
    });
}

/// Workspace bytes of one trainer forward of T columns (Variant::visit_workspace_capacity_bytes for a prompt round
/// of T columns, every key visible; with expert adapters when `adapters` != 0). The head's: ift_head_workspace_bytes.
std::size_t ift_workspace_bytes(std::int32_t T, int adapters) {
    std::size_t bytes = 0;
    (void)guarded([&] {
        family::TextGeometry geometry{};
        bytes = Variant::visit_workspace_capacity_bytes(geometry, family::TextPhase::Prefill, 0, 0, T,
                                                        static_cast<std::uint32_t>(T), adapters != 0);
    });
    return bytes;
}

/// Workspace of one ift_head_logprob call of T columns (its fp32 logits: ~0.8 MB per column; the caller chunks the
/// columns -- every column's value is fixed by that column alone).
std::size_t ift_head_workspace_bytes(std::int32_t T) {
    return ops::inkling_head_logprob_workspace_bytes(C::output_rows, T) + 4096;
}

/// Bytes of the persistent per-stream state (lane d's MoE counters + lane c's fold counters); zero it ONCE.
std::size_t ift_state_bytes() { return Variant::visit_target_state_bytes(); }

/// residual slot 0 <- embed_norm(embed[ids]). x: BF16 [7 x 2304, T] (column-major: column t at x + t * 16128).
int ift_embed(void* handle, const std::int32_t* ids, std::int32_t T, void* x, void* ws, std::size_t ws_bytes,
              void* stream) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        WorkspaceArena work(DeviceSpan{ws, ws_bytes});
        Tensor ids_t(const_cast<std::int32_t*>(ids), DType::I32, {T});
        Tensor residual(x, DType::BF16, {C::residual, T});
        Variant::embed_residual(h->data->runtime, ids_t, residual, work, static_cast<cudaStream_t>(stream));
    });
}

/// One visit over a T-column prompt segment at positions 0..T-1. meta: device I32, 16-byte aligned, [8]:
/// [0] = the KV table row (0), [4] = the conv state slot (0) (forward.cpp's segment ints). positions: device I32 [T].
/// k_pages / v_pages: e4m3 [128, 64, 8, pages]; block_table: I32 [pages] (identity). conv: BF16 [6656, 3, 1].
/// routing_ids: I32 [T, 8] or null. taps: a VisitTaps (or null). adapters: an ops::InklingMoeAdapters (or null).
int ift_run_visit(void* handle, int v, void* x, std::int32_t T, const std::int32_t* positions, const std::int32_t* meta,
                  void* k_pages, void* v_pages, const std::int32_t* block_table, std::int32_t pages, void* conv,
                  void* target_state, std::int32_t* routing_ids, const void* taps, const void* adapters, const void* attn,
                  void* ws, std::size_t ws_bytes, void* stream) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        if (v < 0 || v >= C::layers) { throw std::out_of_range("visit"); }
        if (ops::lora_active()) {
            throw std::logic_error("ift_run_visit: an engine LoRA store is active in this process; the trainer passes "
                                   "its expert and attention adapters explicitly (T8)");
        }
        WorkspaceArena work(DeviceSpan{ws, ws_bytes});
        if (h->data->model.queries == nullptr) { throw std::logic_error("ift_run_visit: no derived queries"); }
        const InklingModel& im = h->data->model;
        Round rd{h->data->runtime, im, *im.queries, work, static_cast<cudaStream_t>(stream), T,
                 Tensor(x, DType::BF16, {C::residual, T})};
        Group g;
        g.column     = 0;
        g.width      = T;
        g.batch      = 1;
        g.positions  = Tensor(const_cast<std::int32_t*>(positions), DType::I32, {T, 1});
        g.table_rows = Tensor(const_cast<std::int32_t*>(meta), DType::I32, {1});
        g.slots      = Tensor(const_cast<std::int32_t*>(meta) + 4, DType::I32, {1});
        rd.groups.push_back(g);
        rd.conv = Tensor(conv, DType::BF16, {kConvChannels, 3, 1});
        PagedKVBatchLayerView kv;
        kv.k_pages      = Tensor(k_pages, DType::FP8_E4M3FN, {128, 64, 8, pages});
        kv.v_pages      = Tensor(v_pages, DType::FP8_E4M3FN, {128, 64, 8, pages});
        kv.block_tables = Tensor(const_cast<std::int32_t*>(block_table), DType::I32, {pages, 1});
        kv.head_dim     = C::head_dim;
        kv.num_kv_heads = C::kv_heads;
        kv.dtype        = DType::FP8_E4M3FN;
        rd.kv           = kv;
        rd.target_state = target_state;
        rd.routing_ids  = routing_ids;
        rd.adapters     = static_cast<const ops::InklingMoeAdapters*>(adapters);
        rd.attn         = static_cast<const AttnAdapters*>(attn);
        rd.max_visible_keys = static_cast<std::uint32_t>(T);
        static const VisitTaps kNoTaps{};
        run_visit(rd, v, taps != nullptr ? *static_cast<const VisitTaps*>(taps) : kNoTaps);
    });
}

/// The final output mix (stack 0-3 + the coda partial, output query, final norm) -> hidden BF16 [2304, T].
int ift_final_hidden(void* handle, void* x, std::int32_t T, void* hidden, void* ws, std::size_t ws_bytes, void* stream) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        WorkspaceArena work(DeviceSpan{ws, ws_bytes});
        Tensor residual(x, DType::BF16, {C::residual, T});
        Tensor out(hidden, DType::BF16, {C::hidden, T});
        Variant::final_residual_mix(h->data->runtime, residual, out, work, static_cast<cudaStream_t>(stream));
    });
}

/// out[t] = log p(ids[t] | hidden[:, t]) exactly as the engine returns it (lane e1's inkling_head_logprob).
int ift_head_logprob(void* handle, const void* hidden, std::int32_t T, const std::int32_t* ids, float* out, void* ws,
                     std::size_t ws_bytes, void* stream) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        const std::size_t need = ops::inkling_head_logprob_workspace_bytes(C::output_rows, T);
        if (ws_bytes < need) { throw std::invalid_argument("ift_head_logprob: workspace too small"); }
        ops::inkling_head_logprob(h->data->runtime.output_head.qdata, C::output_rows, C::hidden, C::token_domain,
                                  C::logit_softcap, hidden, T, ids, out, ws, ws_bytes,
                                  static_cast<cudaStream_t>(stream));
    });
}

/// A device view of a loaded weight for the torch backward (no copy). Names: "embedding", "output_head",
/// "embedding_norm", "final_norm", "query.output", "L<i>.<field>" with field in attn_input_norm, qkvr, output,
/// q_scale, r_scale, rel_proj, k_conv, v_conv, k_conv_static, v_conv_static, k_conv_proj, v_conv_proj, attn_out_conv,
/// mlp_out_conv, moe_input_norm, router, router_bias, loop_down, loop_gain_forget, rel_table, query.attn, query.mlp,
/// query.message, w13.codes, w13.scales, w13.amax, w2.codes, w2.scales, w2.amax.
/// desc out: [0] data pointer, [1] dtype code (0 bf16, 1 fp32, 2 u8, 3 e4m3, 4 i32), [2] numel or bytes, [3..6] ne,
/// [7] format extra (expert banks: 0 nvfp4 / 1 mxfp8; qkvr / output: qtype), [8] outer_scale_denominator bits.
int ift_tensor(void* handle, const char* name, std::int64_t* desc) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        const auto& W = h->data->weights;
        const auto& Q = h->data->queries;
        std::memset(desc, 0, 9 * sizeof(std::int64_t));
        const auto put = [&](const Tensor& t) {
            if (t.data == nullptr) { throw std::invalid_argument(std::string("tensor not resident: ") + name); }
            desc[0] = reinterpret_cast<std::int64_t>(t.data);
            desc[1] = t.dtype == DType::BF16 ? 0 : t.dtype == DType::FP32 ? 1 : t.dtype == DType::I32 ? 4 : 2;
            desc[2] = t.numel();
            for (int i = 0; i < 4; ++i) { desc[3 + i] = t.ne[i]; }
        };
        const auto put_weight = [&](const Weight& w) {
            desc[0] = reinterpret_cast<std::int64_t>(w.qdata);
            desc[1] = w.qtype == QType::BF16_CTRL ? 0 : 3;
            desc[2] = static_cast<std::int64_t>(w.n) * w.k;
            desc[3] = w.k;
            desc[4] = w.n;
            desc[7] = static_cast<std::int64_t>(w.qtype);
        };
        const std::string n(name);
        if (n == "embedding") { put_weight(W.globals.embedding); return; }
        if (n == "output_head") { put_weight(W.globals.output_head); return; }
        if (n == "embedding_norm") { put(W.globals.embedding_norm); return; }
        if (n == "final_norm") { put(W.globals.final_norm); return; }
        if (n == "query.output") { put(Q.output); return; }
        if (n.size() < 3 || n[0] != 'L') { throw std::invalid_argument("unknown tensor " + n); }
        const std::size_t dot = n.find('.');
        const int i = std::stoi(n.substr(1, dot - 1));
        if (i < 0 || i >= C::physical_layers) { throw std::out_of_range("layer"); }
        const std::string f = n.substr(dot + 1);
        const auto& L = W.layers.at(static_cast<std::size_t>(i));
        if (f == "attn_input_norm") { put(L.attn_input_norm); return; }
        if (f == "qkvr") { put_weight(L.qkvr); return; }
        if (f == "output") { put_weight(L.output); return; }
        if (f == "q_scale") { put(L.q_scale); return; }
        if (f == "r_scale") { put(L.r_scale); return; }
        if (f == "rel_proj") { put(L.rel_proj); return; }
        if (f == "k_conv") { put(L.k_conv); return; }
        if (f == "v_conv") { put(L.v_conv); return; }
        if (f == "k_conv_static") { put(L.k_conv_static); return; }
        if (f == "v_conv_static") { put(L.v_conv_static); return; }
        if (f == "k_conv_proj" && L.k_conv_proj) { put_weight(*L.k_conv_proj); return; }
        if (f == "v_conv_proj" && L.v_conv_proj) { put_weight(*L.v_conv_proj); return; }
        if (f == "attn_out_conv") { put(L.attn_out_conv); return; }
        if (f == "mlp_out_conv") { put(L.mlp_out_conv); return; }
        if (f == "moe_input_norm") { put(L.moe_input_norm); return; }
        if (f == "router") { put_weight(L.router); return; }
        if (f == "router_bias") {
            desc[0] = reinterpret_cast<std::int64_t>(L.router_bias);
            desc[1] = 1;
            desc[2] = C::experts;
            desc[3] = C::experts;
            return;
        }
        if (f == "loop_down") { put(L.loop_down); return; }
        if (f == "loop_gain_forget") { put(L.loop_gain_forget); return; }
        if (f == "rel_table") { put(Q.rel_table.at(static_cast<std::size_t>(i))); return; }
        if (f == "query.attn") { put(Q.attn.at(static_cast<std::size_t>(i))); return; }
        if (f == "query.mlp") { put(Q.mlp.at(static_cast<std::size_t>(i))); return; }
        if (f == "query.message") {
            if (i < C::core_start || i >= C::core_end) { throw std::invalid_argument("not a core layer"); }
            put(Q.message.at(static_cast<std::size_t>(i - C::core_start)));
            return;
        }
        const auto bank_field = [&](const load::ExpertBank& b, const std::string& which) {
            desc[3] = b.experts;
            desc[4] = b.rows_per_expert;
            desc[5] = b.columns;
            desc[7] = b.format == load::ExpertFormat::Mxfp8 ? 1 : 0;
            float den = b.outer_scale_denominator_value;
            std::int32_t bits = 0;
            std::memcpy(&bits, &den, 4);
            desc[8] = bits;
            if (which == "codes") { desc[0] = reinterpret_cast<std::int64_t>(b.codes); desc[1] = 2; return; }
            if (which == "scales") { desc[0] = reinterpret_cast<std::int64_t>(b.scales); desc[1] = 2; return; }
            if (which == "amax") { desc[0] = reinterpret_cast<std::int64_t>(b.amax); desc[1] = 1; return; }
            throw std::invalid_argument("unknown bank field " + which);
        };
        if (f.rfind("w13.", 0) == 0) { bank_field(L.w13, f.substr(4)); return; }
        if (f.rfind("w2.", 0) == 0) { bank_field(L.w2, f.substr(3)); return; }
        throw std::invalid_argument("unknown tensor " + n);
    });
}

/// Lane d's exact unit decode of one physical layer's expert bank (the backward's dX operand, STE): `which` 0 = w13
/// (k 2304, rows interleaved gate / up), 1 = w2 (k 1152). unit: BF16 [count * 2304][k]; factor: FP32 [count]
/// (NVFP4 amax[e] / den; MXFP8 1). dX = sum_slots (dY . unit[e]) * factor[e].
int ift_decode_experts(void* handle, int layer, int which, std::int32_t first, std::int32_t count, void* unit,
                       float* factor, void* stream) {
    return guarded([&] {
        Handle* h = as_handle(handle);
        if (layer < 0 || layer >= C::physical_layers) { throw std::out_of_range("layer"); }
        const auto& L                 = h->data->weights.layers.at(static_cast<std::size_t>(layer));
        const ops::InklingMoeWeights w = moe_weights(L);
        const std::int32_t k          = which == 0 ? C::hidden : C::intermediate;
        Tensor u(unit, DType::BF16, {k, count * 2304});
        Tensor f(factor, DType::FP32, {count});
        ops::inkling_decode_expert_units(which == 0 ? w.w13 : w.w2, w.format, k, first, count, u, f,
                                         static_cast<cudaStream_t>(stream));
    });
}

/// Bytes of a VisitTaps / InklingMoeAdapters / LoraBank (the Python binding checks its struct layouts against these).
std::size_t ift_sizeof(int which) {
    switch (which) {
        case 0: return sizeof(VisitTaps);
        case 1: return sizeof(ops::InklingMoeAdapters);
        case 2: return sizeof(ops::LoraBank);
        case 3: return sizeof(AttnAdapters);
        default: return 0;
    }
}

} // extern "C"
